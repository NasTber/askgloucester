"""Extract text from PDFs in blob storage using Azure Document Intelligence.

Given a blob name, this downloads the PDF from the ``raw-documents`` container
and runs the ``prebuilt-read`` model to OCR/extract its text. The result is
returned page by page so chunking can record accurate page numbers.

Authentication to both Storage and Document Intelligence uses
``DefaultAzureCredential`` — no API keys.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RAW_DOCUMENTS_CONTAINER = os.environ.get("RAW_DOCUMENTS_CONTAINER", "raw-documents")


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
    """
    logger.info("Extracting text from %s", blob_name)
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
