"""Ingest Gloucester's city-services content pages (trash / recycling).

This is a SEPARATE source, a sibling of :mod:`directory_source` and
:mod:`calendar_source`. Like them, the fetch+parse half is **PURE** — no Azure,
no blob storage, no Document Intelligence, no embedding, no AI Search index — so
it can be verified in isolation against the live site. A later step will consume
the parsed :class:`CityServicePage` records and route them into indexing (HTML
text straight in; PDF brochures through the existing Document Intelligence path).

Source shape (confirmed by probing — no auth, no special headers)
-----------------------------------------------------------------
The trash/recycling pages are ordinary CivicPlus "content" pages on
``https://gloucester-ma.gov``. Unlike the Directory listing template (whose
roster lives in ``<ul id="contentarea">``), these pages put the visible body in
a Froala rich-text view: ``<div class="fr-view">`` nested inside
``<div id="page"> → <div id="moduleContent">``. The page title is the
``<h1 id="versionHeadLine">`` headline.

Two parsing hazards, both handled here:

  * **Themed 404s.** A dead path (e.g. ``/308/Trash-Recycling``) returns a full
    ~106 KB themed HTML page with HTTP 404 — a "did I get HTML?" check would
    happily ingest a not-found page. :func:`fetch_html_page` therefore gates on
    ``status_code == 200`` explicitly and refuses to parse anything else.
  * **The trailing "Loading" token.** A script-injected placeholder element
    inside ``fr-view`` leaves the literal word ``Loading`` at the very end of the
    captured text. We strip that trailing token deliberately.

Targets are MVP-trash only for now (see :data:`TARGETS`). PDF targets are carried
in the list but their OCR path is Azure (Document Intelligence) and out of scope
for the pure HTML extractor verified by :func:`verify`.
"""

from __future__ import annotations

import html
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser

import requests
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

# Reuse the existing chunker's token-splitting (cl100k_base, 500/50) AND its key
# sanitizer so city-services chunks are windowed identically to the docs pipeline
# and get the same valid Azure-key coercion. chunker is Azure-free (tiktoken only).
import chunker

# Reuse the existing embedder (text-embedding-3-small, token-bucket pacing), the
# indexer's search-client construction (DefaultAzureCredential) + batched upload +
# vector-config constants, and the processor's Document Intelligence OCR + durable
# cache (for the brochure PDF) — all pointed at our own index, not the docs one.
import embedder
import indexer
import processor

# Reuse the scraper's browser User-Agent so requests look like a real browser
# (some CivicPlus endpoints reject a bare client), AND its blob-upload idiom
# (_blob_service_client, RAW_DOCUMENTS_CONTAINER, ContentSettings, _sanitize) for
# the brochure PDF. Mirrors directory_source.
import scraper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Targets (MVP — trash). Each entry is a content page or a brochure PDF.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CityServiceTarget:
    """One source to ingest: a CivicPlus content page or a brochure PDF."""

    url: str
    kind: str  # "html" | "pdf"
    service_category: str  # e.g. "trash"
    label: str  # human-readable


TARGETS: list[CityServiceTarget] = [
    CityServiceTarget(
        url="https://gloucester-ma.gov/1031/Special-Collections",
        kind="html",
        service_category="trash",
        label="Special Collections",
    ),
    CityServiceTarget(
        url="https://gloucester-ma.gov/1028/Trash-Collection",
        kind="html",
        service_category="trash",
        label="Trash Collection",
    ),
    CityServiceTarget(
        url="https://gloucester-ma.gov/DocumentCenter/View/11960/Trash-2026",
        kind="pdf",
        service_category="trash",
        label="2026 Recycling Brochure",
    ),
    # NB: /308/Trash-Recycling is intentionally OMITTED — it is a dead URL that
    # serves a full themed HTTP 404 page (see fetch_html_page's status gate).
    #
    # --- Building / permits (same fr-view content-page shape as the trash pages) ---
    CityServiceTarget(
        url="https://www.gloucester-ma.gov/230/Inspectional-Services",
        kind="html",
        service_category="permits",
        label="Inspectional Services",
    ),
    CityServiceTarget(
        url="https://www.gloucester-ma.gov/895/Online-Permitting",
        kind="html",
        service_category="permits",
        label="Online Permitting",
    ),
    CityServiceTarget(
        url="https://www.gloucester-ma.gov/231/Building-Inspector",
        kind="html",
        service_category="permits",
        label="Building Inspector",
    ),
    # NB: the ViewPoint/OpenGov filing portal (gloucesterma.viewpointcloud.com) is
    # intentionally OMITTED — it is the dynamic filing app, not prose; its link
    # comes through inside the /895 Online-Permitting page text. /839 (bulk
    # permit-data CSV/Excel records) is also omitted — wrong shape for this parser.
]


@dataclass
class CityServicePage:
    """Parsed output for one source — the consumable a later indexer reads.

    For HTML targets, ``title``/``text`` come from the rich-text body. For PDF
    targets these stay empty here (their content arrives via the Azure OCR path,
    out of scope for the pure HTML extractor).
    """

    url: str
    kind: str
    service_category: str
    label: str
    title: str
    text: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CityServiceFetchError(RuntimeError):
    """A page could not be fetched (non-200 status, network error)."""


class CityServiceParseError(RuntimeError):
    """A fetched page had no recognizable content container."""


# ---------------------------------------------------------------------------
# Parsing (PURE — no network, no Azure)
# ---------------------------------------------------------------------------
def _clean_text(value: str) -> str:
    """Unescape HTML entities and collapse runs of whitespace to single spaces.

    Mirrors :func:`directory_source._clean_text`; used for the single-line title.
    """
    return " ".join(html.unescape(value).split())


# Page title from the CivicPlus headline. Mirrors directory_source's
# _PAGE_HEADER_RE regex idiom for a single structured scalar field.
_TITLE_RE = re.compile(r'<h1[^>]*id="versionHeadLine"[^>]*>(.*?)</h1>', re.I | re.S)

# Block-level tags whose boundaries should become whitespace, so adjacent blocks
# don't run together when their text nodes are concatenated.
_BLOCK_TAGS = frozenset(
    {"p", "br", "li", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol"}
)

# Container candidates, in priority order. The Froala rich-text view is the
# tightest wrapper around exactly the body content; id="page" / id="moduleContent"
# are stable fallbacks that also enclose the title/feature column. Each matcher
# takes (tag, attrs_dict) and returns True for the container's opening tag.
def _match_fr_view(tag: str, attrs: dict[str, str]) -> bool:
    return tag == "div" and "fr-view" in (attrs.get("class") or "").split()


def _match_id(target_id: str):
    def _matcher(tag: str, attrs: dict[str, str]) -> bool:
        return tag == "div" and (attrs.get("id") or "") == target_id

    return _matcher


_CONTAINER_MATCHERS = [
    ("fr-view", _match_fr_view),
    ("id=page", _match_id("page")),
    ("id=moduleContent", _match_id("moduleContent")),
]


class _ContainerTextParser(HTMLParser):
    """Capture text inside the FIRST element matching ``match_start``.

    A depth/boolean state machine in the directory_source idiom: once the
    container's opening tag is seen we set ``capturing`` and track the depth of
    the container's own tag so nested same-name tags (e.g. ``<div>`` inside the
    body) don't end capture early. ``<script>``/``<style>`` text is suppressed.
    Block-tag boundaries emit a newline so blocks stay word-separated.
    """

    def __init__(self, match_start) -> None:
        super().__init__()
        self._match_start = match_start
        self._capturing = False
        self._container_tag: str | None = None
        self._depth = 0
        self._suppress = 0  # >0 while inside <script>/<style>
        self._parts: list[str] = []
        self.found = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        if not self._capturing:
            if self._match_start(tag, attr_map):
                self._capturing = True
                self.found = True
                self._container_tag = tag
                self._depth = 1
            return
        # Already capturing.
        if tag in ("script", "style"):
            self._suppress += 1
        if tag == self._container_tag:
            self._depth += 1
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing tags (e.g. <br/>) — only the block-boundary newline matters.
        if self._capturing and tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._capturing:
            return
        if tag in ("script", "style") and self._suppress:
            self._suppress -= 1
        if tag == self._container_tag:
            self._depth -= 1
            if self._depth == 0:
                self._capturing = False  # container closed — stop capturing

    def handle_data(self, data: str) -> None:
        if self._capturing and not self._suppress:
            self._parts.append(data)

    def get_text(self) -> str:
        """Normalized body text: unescape, strip per line, drop blank lines."""
        raw = html.unescape("".join(self._parts))
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _strip_trailing_loading(text: str) -> str:
    """Drop the script-injected ``Loading`` placeholder if it's the last token.

    A widget inside ``fr-view`` leaves the literal word ``Loading`` at the very
    end of the captured text. Remove it only when it stands alone as the final
    line, so a genuine sentence merely ending with the word is left intact.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "Loading":
        lines.pop()
    return "\n".join(lines)


def extract_title(page_html: str) -> str:
    """Return the page headline from ``<h1 id="versionHeadLine">`` (may be empty)."""
    match = _TITLE_RE.search(page_html)
    if not match:
        return ""
    # Strip any residual inline tags inside the headline, then collapse whitespace.
    inner = re.sub(r"<[^>]+>", " ", match.group(1))
    return _clean_text(inner)


def extract_body_text(page_html: str) -> str:
    """Extract the visible body text from a CivicPlus content page (PURE).

    Tries the container candidates in priority order (``fr-view`` → ``id=page`` →
    ``id=moduleContent``); the first that matches wins. Raises
    :class:`CityServiceParseError` if NONE match — we never silently fall back to
    scraping the whole page (which would pull in nav/header/footer chrome).
    """
    for name, matcher in _CONTAINER_MATCHERS:
        parser = _ContainerTextParser(matcher)
        parser.feed(page_html)
        if parser.found:
            text = _strip_trailing_loading(parser.get_text())
            if name != "fr-view":
                logger.warning(
                    "city_services: fr-view not found; fell back to %s container", name
                )
            return text
    raise CityServiceParseError("no content container (fr-view / page / moduleContent)")


def parse_html_page(target: CityServiceTarget, page_html: str) -> CityServicePage:
    """Parse a fetched content page into a :class:`CityServicePage` (PURE)."""
    return CityServicePage(
        url=target.url,
        kind=target.kind,
        service_category=target.service_category,
        label=target.label,
        title=extract_title(page_html),
        text=extract_body_text(page_html),
    )


# ---------------------------------------------------------------------------
# Fetch (network, but PURE of Azure)
# ---------------------------------------------------------------------------
def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": scraper.BROWSER_USER_AGENT})
    return session


def fetch_html_page(url: str, session: requests.Session | None = None) -> str:
    """Fetch a content page, gating on HTTP 200.

    CivicPlus serves a full themed HTML body for 404s, so a content-type/"got
    HTML?" check is not enough — we reject any non-200 status outright and never
    parse it. Raises :class:`CityServiceFetchError` on non-200 or transport error.
    """
    session = session or _new_session()
    try:
        response = session.get(url, timeout=60)
    except requests.RequestException as exc:
        raise CityServiceFetchError(f"{url}: request failed: {exc}") from exc
    if response.status_code != 200:
        raise CityServiceFetchError(f"{url}: HTTP {response.status_code} (refusing to parse)")
    return response.text


def fetch_pdf_bytes(url: str, session: requests.Session | None = None) -> bytes:
    """Download a PDF target, gating on HTTP 200 AND real-PDF validation.

    Same defensive spirit as :func:`fetch_html_page`: CivicPlus serves a full
    themed HTML body for 404s, so a content-type check alone is not enough. We
    reject any non-200 status, and additionally require the payload to actually BE
    a PDF — accepted when the bytes start with the ``%PDF`` magic OR the
    Content-Type is ``application/pdf``. A themed-404 HTML page (text/html, no
    magic) is refused. Raises :class:`CityServiceFetchError` on non-200, non-PDF,
    or transport error, so the caller can log-and-skip without crashing the run.
    """
    session = session or _new_session()
    try:
        response = session.get(url, allow_redirects=True, timeout=60)
    except requests.RequestException as exc:
        raise CityServiceFetchError(f"{url}: request failed: {exc}") from exc
    if response.status_code != 200:
        raise CityServiceFetchError(f"{url}: HTTP {response.status_code} (refusing to ingest)")
    body = response.content
    content_type = (response.headers.get("Content-Type") or "").lower()
    looks_like_pdf = body[:5] == b"%PDF-" or "application/pdf" in content_type
    if not looks_like_pdf:
        raise CityServiceFetchError(
            f"{url}: not a PDF (content-type={content_type!r}, first-bytes={body[:5]!r})"
        )
    return body


def fetch_and_extract(
    target: CityServiceTarget, session: requests.Session | None = None
) -> CityServicePage:
    """Fetch + parse one HTML target. NETWORK ONLY — never touches Azure."""
    if target.kind != "html":
        raise ValueError(f"fetch_and_extract is HTML-only; got kind={target.kind!r}")
    session = session or _new_session()
    page_html = fetch_html_page(target.url, session=session)
    return parse_html_page(target, page_html)


# ---------------------------------------------------------------------------
# Chunking + metadata prefix (PURE — no network, no Azure)
# ---------------------------------------------------------------------------
# service_category -> display name used in the BM25-findable chunk prefix. A
# dict so adding "permits" / "water" later is a one-line change, not new logic.
CATEGORY_DISPLAY: dict[str, str] = {
    "trash": "Trash & Recycling",
    "permits": "Permits & Inspections",
}

# The leading marker of every city-services chunk prefix (used as a presence
# check in verify and as documentation of the format).
_PREFIX_LEAD = "Gloucester City Services — "


@dataclass
class CityServiceChunk:
    """One index-ready chunk: prefixed content plus the metadata to cite it.

    Source-agnostic — the same record shape serves HTML-extracted text now and
    OCR'd PDF text later. ``content`` already carries the city-services prefix.
    """

    id: str
    content: str
    source_url: str
    title: str
    service_category: str
    page_number: int
    # Embedding of ``content``, populated in place by :mod:`embedder` (1536-dim,
    # text-embedding-3-small). Empty until the embed step runs; an empty list is a
    # valid value for the index's nullable vector field. Mirrors chunker.Chunk.
    content_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _docs_neutral_prefix() -> str:
    """Reconstruct chunk_pages' docs prefix for NEUTRAL (empty) metadata.

    chunk_pages always prepends ``"{body} {type} — {date}\\n\\n"`` to each window.
    We call it with empty body/type/date, so that prefix collapses to a fixed,
    content-free string we strip back off — leaving exactly the cl100k 500/50
    window text. Built with chunker's OWN formula + date helper so it tracks any
    change to that format (and :func:`_token_windows` guards with startswith).
    """
    body = document_type = document_date = ""
    return (
        f"{body} {document_type} — "
        f"{chunker._format_date_human(document_date)}\n\n"
    )


def _token_windows(text: str, page_number: int) -> list[tuple[int, str]]:
    """Reuse chunker.chunk_pages for cl100k 500/50 windowing (no reimplement).

    Returns ``(page_number, window_text)`` pairs with the docs-specific prefix
    removed, so the caller can apply the city-services prefix instead. Windowing,
    overlap and page attribution are exactly the docs pipeline's.
    """
    pages = [{"page_number": page_number, "text": text}]
    docs_chunks = chunker.chunk_pages(
        pages,
        source_url="",  # provenance is re-attached by the caller, not here
        document_date="",
        meeting_body="",
        document_type="",
        meeting_category="",
        base_id="cs",
    )
    neutral_prefix = _docs_neutral_prefix()
    windows: list[tuple[int, str]] = []
    for chunk in docs_chunks:
        if not chunk.content.startswith(neutral_prefix):
            raise RuntimeError(
                "chunker prefix format drifted; city_services strip is stale"
            )
        windows.append((chunk.page_number, chunk.content[len(neutral_prefix):]))
    return windows


def chunk_city_service_text(
    text: str,
    *,
    source_url: str,
    title: str,
    service_category: str,
    label: str,
    page_number: int,
) -> list[CityServiceChunk]:
    """Chunk extracted text into prefixed, index-ready records (PURE).

    Source-agnostic: identical for HTML body text and (later) OCR'd PDF pages.
    Each chunk is prefixed with
    ``"Gloucester City Services — {category_display} — {label}\\n\\n"`` so service
    terms are BM25-findable. IDs are deterministic — ``{sanitized_source}::{page}
    ::{idx}`` run through chunker's key sanitizer (which turns the ``::`` into
    ``__`` for a valid Azure key, mirroring the docs pipeline) — so a re-run
    overwrites by key instead of duplicating.
    """
    category_display = CATEGORY_DISPLAY.get(service_category, service_category)
    prefix = f"{_PREFIX_LEAD}{category_display} — {label}\n\n"
    sanitized_source = chunker._sanitize_key(source_url)

    records: list[CityServiceChunk] = []
    for idx, (page_no, window_text) in enumerate(_token_windows(text, page_number)):
        raw_id = f"{sanitized_source}::{page_no}::{idx}"
        records.append(
            CityServiceChunk(
                id=chunker._sanitize_key(raw_id),
                content=prefix + window_text,
                source_url=source_url,
                title=title,
                service_category=service_category,
                page_number=page_no,
            )
        )
    return records


def chunk_page(page: CityServicePage) -> list[CityServiceChunk]:
    """Chunk an HTML-extracted :class:`CityServicePage` (page_number = 1)."""
    return chunk_city_service_text(
        page.text,
        source_url=page.url,
        title=page.title,
        service_category=page.service_category,
        label=page.label,
        page_number=1,
    )


# ---------------------------------------------------------------------------
# Azure AI Search — dedicated city-services index (separate from the docs index)
# ---------------------------------------------------------------------------
# Its own index on the SAME search service as the docs index. Env-overridable so
# a throwaway/test index can be targeted without code changes.
CITY_SERVICES_INDEX_NAME = os.environ.get(
    "CITY_SERVICES_INDEX_NAME", "gloucester-city-services"
)


def _build_city_services_index(index_name: str) -> SearchIndex:
    """Build the reduced city-services index schema.

    A minimal field set — no meeting_body / document_date / meeting_category — but
    the vector + hybrid config is MIRRORED from :func:`indexer._build_index`: the
    same HNSW algorithm/profile names, the same 1536 dimensions
    (text-embedding-3-small), and a parallel semantic config over ``content`` —
    so hybrid (BM25 + vector, RRF) behaves identically to the docs index.
    """
    fields = [
        # Key — filterable, mirroring the docs index's key field.
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        # The prefixed chunk text — BM25-searchable.
        SearchableField(name="content", type=SearchFieldDataType.String),
        # Provenance for citations — searchable + retrievable (not filtered: the
        # index is wiped+rebuilt each run, so there's no source_url skip-set need).
        SearchableField(name="source_url", type=SearchFieldDataType.String),
        # Page title — searchable so its terms help BM25.
        SearchableField(name="title", type=SearchFieldDataType.String),
        # The one field we actually scope on — filterable only.
        SimpleField(
            name="service_category",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=False,
        ),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        # Vector field — same shape as the docs index: searchable via the profile,
        # not retrievable (keep result payloads small).
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=False,
            vector_search_dimensions=indexer.VECTOR_DIMENSIONS,
            vector_search_profile_name=indexer.VECTOR_PROFILE_NAME,
        ),
    ]

    # Vector search config — names reused verbatim from indexer so the two indexes
    # are structurally identical.
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=indexer.HNSW_CONFIG_NAME)],
        profiles=[
            VectorSearchProfile(
                name=indexer.VECTOR_PROFILE_NAME,
                algorithm_configuration_name=indexer.HNSW_CONFIG_NAME,
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=indexer.SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def ensure_city_services_index(index_name: str = CITY_SERVICES_INDEX_NAME) -> None:
    """WIPE-AND-REBUILD the city-services index: delete if present, then create.

    Deliberately NOT create-if-not-exists (unlike :func:`indexer.ensure_index`).
    Rationale: this corpus is tiny (a handful of trash/recycling pages) and is
    fully re-ingested on every run, so a clean rebuild is cheap and sidesteps
    revision-orphan chunks — a page that drops a paragraph, or a target removed
    from :data:`TARGETS`, leaves no stale chunks behind because the whole index is
    recreated from the current crawl. The brief read-gap during the weekly cron is
    acceptable for this low-traffic, non-critical content. Kept SEPARATE from the
    docs ensure_index so the docs index is never delete+recreated by this path.
    """
    client = SearchIndexClient(indexer._search_endpoint(), credential=indexer._credential())
    try:
        client.delete_index(index_name)
        logger.info("city_services: deleted existing index '%s'", index_name)
    except ResourceNotFoundError:
        logger.info("city_services: no existing index '%s' to delete", index_name)
    index = _build_city_services_index(index_name)
    client.create_or_update_index(index)
    logger.info("city_services: created index '%s'", index_name)


# ---------------------------------------------------------------------------
# PDF brochure branch (download → blob → OCR → chunk)
# ---------------------------------------------------------------------------
def _upload_pdf_blob(blob_name: str, pdf_bytes: bytes, target: CityServiceTarget) -> None:
    """Upload a brochure PDF to the raw-documents container (mirrors scraper).

    Uses a STABLE ``blob_name`` (derived from the target's constant label) so the
    downstream OCR cache — keyed on ``blob_name`` — is stable across runs:
    Document Intelligence runs once for this PDF and every later run is a cache
    hit. Reuses scraper's blob-client + container + ContentSettings idiom.
    """
    container = scraper._blob_service_client().get_container_client(
        scraper.RAW_DOCUMENTS_CONTAINER
    )
    # Provisioned by Bicep, but create-if-missing keeps a fresh account runnable.
    try:
        container.create_container()
    except Exception:  # noqa: BLE001 - "already exists" is the common case
        pass
    metadata = {
        "source_url": target.url,
        "service_category": target.service_category,
        # Blob metadata must be ASCII; the label already is, but stay defensive.
        "title": target.label.encode("ascii", "ignore").decode("ascii"),
    }
    container.upload_blob(
        name=blob_name,
        data=pdf_bytes,
        overwrite=True,
        metadata=metadata,
        content_settings=scraper.ContentSettings(content_type="application/pdf"),
    )
    logger.info("city_services: uploaded %s (%d bytes)", blob_name, len(pdf_bytes))


def _pdf_target_chunks(
    target: CityServiceTarget, session: requests.Session
) -> list[CityServiceChunk]:
    """Download → blob → OCR → chunk one PDF target. Returns [] on any skip.

    Network download + PDF validation, then a stable-name blob upload, then OCR
    via the shared :func:`processor.extract_text` (which carries the durable OCR
    cache), then the SAME city-services chunking/prefix as the HTML path — only
    the page-text source differs (OCR pages instead of one HTML body). A bad
    download (non-200 / non-PDF) or an empty OCR result is logged and skipped so
    one bad target can't sink the rebuild.
    """
    try:
        pdf_bytes = fetch_pdf_bytes(target.url, session=session)
    except CityServiceFetchError as exc:
        logger.warning("city_services: skipping PDF %s: %s", target.url, exc)
        return []

    # Stable blob name → stable OCR cache key (scan once, cached forever).
    blob_name = f"city-services/{scraper._sanitize(target.label)}.pdf"
    _upload_pdf_blob(blob_name, pdf_bytes, target)

    pages = processor.extract_text(blob_name)
    if not pages:
        logger.warning("city_services: no OCR text from PDF %s; skipping", target.url)
        return []

    # The PDF has no HTML <h1>; its label IS its title. Chunk per OCR page so each
    # chunk carries its real page_number (HTML pages always used page_number=1).
    chunks: list[CityServiceChunk] = []
    for page in pages:
        chunks.extend(
            chunk_city_service_text(
                page.text,
                source_url=target.url,
                title=target.label,
                service_category=target.service_category,
                label=target.label,
                page_number=page.page_number,
            )
        )
    logger.info(
        "city_services: %s -> %d chunk(s) from %d OCR page(s)",
        target.url, len(chunks), len(pages),
    )
    return chunks


# ---------------------------------------------------------------------------
# Orchestrator (HTML pages + brochure PDF)
# ---------------------------------------------------------------------------
def fetch_and_index_city_services(
    index_name: str = CITY_SERVICES_INDEX_NAME,
) -> int:
    """Fetch + chunk + embed + (wipe-rebuild) index the city-services sources.

    Mirrors the calendar/directory orchestration shape. HTML targets: fetch →
    :class:`CityServicePage` → :func:`chunk_page`. PDF targets: download → blob →
    OCR → chunk via :func:`_pdf_target_chunks`. All chunks join one list, then a
    single pass embeds everything, wipe-and-rebuilds the index, and uploads.

    Returns the number of chunks indexed (0 if nothing was produced — in which
    case the existing index is left untouched rather than wiped to empty).
    """
    session = _new_session()

    all_chunks: list[CityServiceChunk] = []
    for target in TARGETS:
        if target.kind == "pdf":
            # Brochure PDF: download → blob → OCR → chunk. Its chunks join the
            # HTML chunks below before the single embed/wipe-rebuild/upload pass.
            all_chunks.extend(_pdf_target_chunks(target, session))
            continue
        if target.kind != "html":
            logger.warning("city_services: unknown kind %r for %s; skipping", target.kind, target.url)
            continue
        try:
            page = fetch_and_extract(target, session=session)
        except (CityServiceFetchError, CityServiceParseError) as exc:
            # Isolate a bad page so one dead URL can't sink the whole rebuild.
            logger.warning("city_services: skipping %s: %s", target.url, exc)
            continue
        chunks = chunk_page(page)
        logger.info("city_services: %s -> %d chunk(s)", target.url, len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("city_services: no chunks produced; leaving index untouched")
        return 0

    # Embed FIRST (populates content_vector in place), so we only wipe the live
    # index once we actually have vectors ready to upload — a failed embed leaves
    # the current index serving.
    embedder.embed_chunks(all_chunks)

    # Wipe-and-rebuild, then upload. Reuse indexer.upload_chunks (batched, AAD
    # search client) pointed at OUR index.
    ensure_city_services_index(index_name)
    indexed = indexer.upload_chunks(all_chunks, index_name=index_name)
    logger.info(
        "city_services: %d chunk(s) from %d target(s) indexed into '%s'",
        indexed,
        len(TARGETS),
        index_name,
    )
    return indexed


# ---------------------------------------------------------------------------
# Verify mode (Azure-free, read-only) — HTML targets only
# ---------------------------------------------------------------------------
_DEAD_URL = "https://gloucester-ma.gov/308/Trash-Recycling"
_SPECIAL_COLLECTIONS_URL = "https://gloucester-ma.gov/1031/Special-Collections"

# Content assertions are PER-PAGE: the holiday-shift rule appears on every trash
# page, but "Christmas Tree" / "Yard Waste" / "Household Hazardous" are unique to
# the Special Collections page. Applying those to Trash Collection (purple-bag
# rules, bulk stickers) would be a wrong test, not a parser failure.
_MUST_CONTAIN_COMMON = ["Holiday Schedule", "one day later"]
_MUST_CONTAIN_BY_URL = {
    "https://gloucester-ma.gov/1031/Special-Collections": [
        "Christmas Tree",
        "Yard Waste",
        "Household Hazardous",
    ],
    "https://gloucester-ma.gov/1028/Trash-Collection": [
        "purple bags",
        "Compost Facility",
    ],
}
_MUST_EXCLUDE = ["Quick Links", "Powered by", "Site Map"]


def verify() -> bool:
    """Fetch + extract every HTML target and assert the content boundaries.

    Returns True iff every check passes. Prints a PASS/FAIL line per check. No
    Azure, no index — verifiable in isolation against the live site.
    """
    session = _new_session()
    all_ok = True

    def check(label: str, ok: bool) -> None:
        nonlocal all_ok
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    html_targets = [t for t in TARGETS if t.kind == "html"]
    for target in html_targets:
        print(f"\n=== {target.label} <{target.url}> ===")
        try:
            page = fetch_and_extract(target, session=session)
        except (CityServiceFetchError, CityServiceParseError) as exc:
            check(f"fetch+extract ({exc})", False)
            continue

        print(f"--- title: {page.title!r}")
        print("--- extracted text ---")
        print(page.text)
        print("--- end text ---")

        expected = _MUST_CONTAIN_COMMON + _MUST_CONTAIN_BY_URL.get(target.url, [])
        for needle in expected:
            check(f"contains {needle!r}", needle in page.text)
        for needle in _MUST_EXCLUDE:
            check(f"excludes {needle!r}", needle not in page.text)
        # Trailing placeholder must be gone (and not lurking anywhere as a line).
        check(
            "trailing 'Loading' token stripped",
            "Loading" not in page.text.split("\n"),
        )

    # The dead URL must be rejected by the status gate, not returned as content.
    print(f"\n=== dead-URL status gate <{_DEAD_URL}> ===")
    try:
        fetch_html_page(_DEAD_URL, session=session)
        check("status gate rejects 404 (no content returned)", False)
    except CityServiceFetchError as exc:
        print(f"  rejected as expected: {exc}")
        check("status gate rejects 404 (no content returned)", True)

    print(f"\n==== EXTRACTION OVERALL: {'PASS' if all_ok else 'FAIL'} ====")
    return all_ok


def verify_chunking() -> bool:
    """Fetch+extract the HTML targets, chunk them, and assert chunk invariants.

    Azure-free: no embedding, no index, no upload. Returns True iff every check
    passes; prints a PASS/FAIL line per check.
    """
    session = _new_session()
    all_ok = True

    def check(label: str, ok: bool) -> None:
        nonlocal all_ok
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    html_targets = [t for t in TARGETS if t.kind == "html"]
    for target in html_targets:
        print(f"\n=== chunking {target.label} <{target.url}> ===")
        page = fetch_and_extract(target, session=session)
        chunks = chunk_page(page)

        for c in chunks:
            prefix_line, _, body = c.content.partition("\n\n")
            print(f"  - id={c.id}")
            print(f"    source_url={c.source_url}")
            print(f"    service_category={c.service_category}  title={c.title!r}")
            print(f"    prefix_line={prefix_line!r}")
            print(f"    body[:80]={body[:80]!r}")

        print(f"  total chunks for this page: {len(chunks)}")
        check("at least one chunk produced", len(chunks) >= 1)
        check(
            "prefix line present on EVERY chunk",
            all(c.content.startswith(_PREFIX_LEAD) for c in chunks),
        )

        # Deterministic + unique ids: a second identical run must match exactly.
        ids_run1 = [c.id for c in chunks]
        ids_run2 = [c.id for c in chunk_page(page)]
        check("ids identical across two runs (deterministic)", ids_run1 == ids_run2)
        check("ids unique within page", len(ids_run1) == len(set(ids_run1)))

        if target.url == _SPECIAL_COLLECTIONS_URL:
            blob = " ".join(c.content for c in chunks).lower()
            check("'holiday' findable in Special Collections chunks", "holiday" in blob)
            check("'trash' findable in Special Collections chunks", "trash" in blob)

    print(f"\n==== CHUNKING OVERALL: {'PASS' if all_ok else 'FAIL'} ====")
    return all_ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    extraction_ok = verify()
    chunking_ok = verify_chunking()
    print(
        f"\n#### FINAL: extraction={'PASS' if extraction_ok else 'FAIL'}, "
        f"chunking={'PASS' if chunking_ok else 'FAIL'} ####"
    )
    sys.exit(0 if (extraction_ok and chunking_ok) else 1)
