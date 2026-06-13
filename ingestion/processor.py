"""Extract text from PDFs in blob storage using Azure Document Intelligence.

Given a blob name, this downloads the PDF from the ``raw-documents`` container
and runs the ``prebuilt-read`` model to OCR/extract its text. The result is
returned page by page so chunking can record accurate page numbers.

Authentication to both Storage and Document Intelligence uses
``DefaultAzureCredential`` — no API keys.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from functools import lru_cache

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RAW_DOCUMENTS_CONTAINER = os.environ.get("RAW_DOCUMENTS_CONTAINER", "raw-documents")

# Durable OCR cache. Document Intelligence is the slow/paid step, so its output
# is persisted here — in the SAME storage account (stakgloucesterdev) — keyed by
# the source blob_name. This makes prebuilt-read a one-time cost per document
# that SURVIVES an AI Search index recreate: the existing source_url skip lives
# only inside the index, so wiping/recreating the index re-OCRs everything; this
# cache does not, because it is an independent, durable record in blob storage.
EXTRACTED_TEXT_CONTAINER = os.environ.get("EXTRACTED_TEXT_CONTAINER", "extracted-text")


@dataclass
class ExtractedPage:
    """Text extracted from a single page of a document."""

    page_number: int  # 1-based, as reported by Document Intelligence
    text: str


def _blob_service_client() -> BlobServiceClient:
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    account_url = f"https://{account}.blob.core.windows.net"
    return BlobServiceClient(account_url, credential=DefaultAzureCredential())


def _document_intelligence_client() -> DocumentIntelligenceClient:
    endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
    return DocumentIntelligenceClient(endpoint, credential=DefaultAzureCredential())


def _download_blob(blob_name: str) -> bytes:
    container = _blob_service_client().get_container_client(RAW_DOCUMENTS_CONTAINER)
    return container.download_blob(blob_name).readall()


@lru_cache(maxsize=1)
def _cache_container():
    """Return the OCR-cache container client, creating it if it doesn't exist.

    Mirrors :func:`indexer.ensure_index`'s existence guard: check for the
    container, create it only on a miss. The existing Storage Blob Data
    Contributor RBAC already covers read/write here — no new role assignment.

    Memoized (like the other client singletons) so the existence check and
    client construction happen once per process, not once per document during a
    backfill.
    """
    container = _blob_service_client().get_container_client(EXTRACTED_TEXT_CONTAINER)
    try:
        container.get_container_properties()
    except ResourceNotFoundError:
        logger.info("Creating OCR cache container '%s'", EXTRACTED_TEXT_CONTAINER)
        container.create_container()
    return container


def _cache_blob_name(blob_name: str) -> str:
    """Cache object name for a source blob.

    Keyed on the source ``blob_name`` ONLY — no content hash. This is correct
    because document identity is source-ID-based: revisions arrive under a new
    ADID/fileId, which yields a new ``blob_name`` (and thus a new cache key), so
    changed content never collides with stale cached text. KNOWN LIMITATION: if
    the SAME ADID/fileId were ever re-published with different bytes (not how
    Gloucester's Archive/Drive sources behave today), this cache would serve the
    old OCR text until the cache object is deleted. Documented here so that edge
    case is a known limitation, not a silent bug.

    Slashes in ``blob_name`` are preserved — blob names may contain ``/``, so the
    cache mirrors the source path under the cache container (e.g.
    ``extracted-text/School-Committee/2026-01-14/123_minutes.pdf.json``).
    """
    return f"{blob_name}.json"


def _serialize_pages(pages: list[ExtractedPage]) -> bytes:
    """Serialize extracted pages to JSON bytes for the cache."""
    return json.dumps([asdict(p) for p in pages]).encode("utf-8")


def _deserialize_pages(data: bytes) -> list[ExtractedPage]:
    """Rebuild ``list[ExtractedPage]`` from cached JSON bytes."""
    return [
        ExtractedPage(page_number=item["page_number"], text=item["text"])
        for item in json.loads(data)
    ]


def _page_text(result, page) -> str:
    """Slice the per-page text out of the document's full content.

    The ``prebuilt-read`` model returns one flat ``content`` string plus, for
    every page, the character spans of that page within it. Slicing by span is
    the supported way to recover page-scoped text in the v4 SDK.
    """
    content = result.content or ""
    parts = []
    for span in page.spans or []:
        start = span.offset
        end = span.offset + span.length
        parts.append(content[start:end])
    return "".join(parts).strip()


def extract_text(blob_name: str) -> list[ExtractedPage]:
    """Run Document Intelligence over a blob and return its text, page by page.

    Args:
        blob_name: Name (path) of the PDF blob in the raw-documents container.

    Returns:
        A list of :class:`ExtractedPage`, ordered by page number. Pages with no
        extractable text are omitted.

    Document Intelligence output is cached durably in the ``extracted-text``
    container, keyed by ``blob_name``. On a cache hit the download and the
    prebuilt-read call are both skipped; on a miss the result is written to the
    cache before returning. The cache lives outside the AI Search index, so
    re-OCR is a one-time cost per document even across an index recreate.
    """
    cache = _cache_container()
    cache_blob_name = _cache_blob_name(blob_name)

    # Cache read: return the stored extraction if we've OCR'd this blob before.
    # Fail-open — only a genuine "blob not found" is a clean miss. ANY other
    # error (transient storage failure, truncated/corrupt entry, bad JSON) is
    # logged at warning and ALSO treated as a miss, so we fall through to the
    # normal download + OCR path. A bad cache entry must never raise or poison a
    # run. (`is not None` keeps a legitimately empty `[]` extraction as a hit.)
    cached: list[ExtractedPage] | None = None
    try:
        raw = cache.download_blob(cache_blob_name).readall()
        cached = _deserialize_pages(raw)
    except ResourceNotFoundError:
        cached = None  # never OCR'd this blob — clean miss, no warning
    except Exception as exc:  # noqa: BLE001 - fail-open on any other cache error
        logger.warning(
            "OCR cache read failed for %s (%s); treating as miss and re-extracting",
            cache_blob_name,
            exc,
        )
        cached = None
    if cached is not None:
        logger.info(
            "OCR cache hit for %s (%d page(s)); skipping download + Document Intelligence",
            blob_name,
            len(cached),
        )
        return cached

    # Cache miss: run the original download + prebuilt-read path unchanged.
    logger.info("OCR cache miss for %s; extracting text", blob_name)
    pdf_bytes = _download_blob(blob_name)

    client = _document_intelligence_client()
    poller = client.begin_analyze_document(
        "prebuilt-read",
        AnalyzeDocumentRequest(bytes_source=pdf_bytes),
    )
    result = poller.result()

    pages: list[ExtractedPage] = []
    for page in result.pages or []:
        text = _page_text(result, page)
        if text:
            pages.append(ExtractedPage(page_number=page.page_number, text=text))

    logger.info("Extracted %d page(s) of text from %s", len(pages), blob_name)

    # Cache write: best-effort. Persist the extraction so future runs (and
    # post-index-recreate runs) skip Document Intelligence entirely for this
    # blob. A write failure must NOT fail an otherwise-successful extraction —
    # log a warning and return the freshly-OCR'd pages anyway (next run retries).
    try:
        cache.upload_blob(
            name=cache_blob_name,
            data=_serialize_pages(pages),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        logger.info("OCR cache write for %s -> %s (%d page(s))", blob_name, cache_blob_name, len(pages))
    except Exception as exc:  # noqa: BLE001 - best-effort cache; never fail a good extraction
        logger.warning(
            "OCR cache write failed for %s (%s); returning OCR result uncached",
            cache_blob_name,
            exc,
        )

    return pages


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Extract text from a blob PDF.")
    parser.add_argument("blob_name", help="Path of the PDF blob in raw-documents.")
    args = parser.parse_args()

    for page in extract_text(args.blob_name):
        print(f"--- page {page.page_number} ---")
        print(page.text)
