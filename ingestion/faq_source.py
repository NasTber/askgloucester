"""Ingest Gloucester's published FAQ into a dedicated AI Search vector index.

A SEPARATE source that mirrors :mod:`city_services_source` exactly — a vector
index + a retrieval tool, NOT a Table. The city publishes ~174 Q&A pairs across
22 categories (assessing, recycling, beach parking, clerk, retirement, veterans,
police, fire, …). A single GET of the flat all-content FAQ view returns every
Q&A, which we parse, chunk (cl100k 500/50, reusing the docs chunker), embed, and
load into a wipe-and-rebuilt ``gloucester-faq`` index.

Source shape (decoded by probing — see the verification harness)
----------------------------------------------------------------
ONE GET returns the whole FAQ: ``https://www.gloucester-ma.gov/m/faq`` (where
``/faq.aspx`` redirects — the flat view). It is server-rendered HTML:

  * the FAQ lives inside ``<div class="… faq-container …">`` (NOT ``#contentarea``
    as older pages used);
  * each category is a ``<div id="faq-category-{N}">`` whose header carries the
    category NAME in ``<h2 class="m-0">`` and its COUNT in a
    ``<span class="badge … rounded-pill ms-2">{count}</span>`` (the per-category
    completeness check);
  * each Q&A is a ``<li id="question-{QID}">`` — the QID is STABLE and citable as
    ``faq.aspx?QID={QID}``. The question text is in the
    ``<button class="accordion-button …">`` and the answer in
    ``<div class="accordion-text … fr-view">`` (visible text, link text
    preserved, hrefs dropped — like the city-services parser).

Category is taken from the STRUCTURAL section the question sits in (the section's
``<h2>``), NOT the per-question ``category-pill`` span — 7 questions carry no pill,
but every question is inside exactly one section, so the structural assignment is
total and matches the header counts exactly.

Architecture (mirrors :mod:`city_services_source`)
--------------------------------------------------
Own index ``gloucester-faq`` (``FAQ_INDEX_NAME`` env) on the SAME Basic search
service. Wipe-and-rebuild ensure (delete-if-exists → create), separate from the
docs / city-services indexes. Code-created (no Bicep) — adds nothing to the IaC
reconciliation landmine and needs no new RBAC (the UAMI's service-scoped search
role already covers it). The orchestrator embeds BEFORE the wipe so a failed embed
leaves the live index serving, and an empty parse never wipes the index to empty.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import re
from dataclasses import dataclass, field
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

# Reuse the docs chunker's cl100k 500/50 windowing + key sanitizer (same as
# city_services), the embedder (token-bucket pacing), and the indexer's
# search-client construction + batched upload + vector-config constants.
import chunker
import embedder
import indexer

# Reuse the scraper's browser User-Agent so the request looks like a real browser.
import scraper

logger = logging.getLogger(__name__)

# One GET of this flat view returns the entire FAQ (every category + Q&A).
FAQ_PAGE_URL = os.environ.get("FAQ_PAGE_URL", "https://www.gloucester-ma.gov/m/faq")
# Per-question citation URL (the canonical single-question permalink).
FAQ_QUESTION_TEMPLATE = "https://www.gloucester-ma.gov/faq.aspx?QID={qid}"

# Own index on the SAME search service as the docs/city-services indexes.
FAQ_INDEX_NAME = os.environ.get("FAQ_INDEX_NAME", "gloucester-faq")

# The leading marker of every FAQ chunk prefix (presence check in verify).
_PREFIX_LEAD = "Gloucester FAQ — "


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class FaqFetchError(RuntimeError):
    """The FAQ page could not be fetched (non-200 status, network error)."""


class FaqParseError(RuntimeError):
    """The fetched page had no recognizable FAQ container."""


# ---------------------------------------------------------------------------
# Parsed (Azure-free) data shapes
# ---------------------------------------------------------------------------
@dataclass
class FaqEntry:
    """One parsed Q&A (the parse output)."""

    qid: int
    category: str
    question: str
    answer: str

    @property
    def source_url(self) -> str:
        return FAQ_QUESTION_TEMPLATE.format(qid=self.qid)


@dataclass
class FaqChunk:
    """One index-ready chunk: prefixed content plus the metadata to cite it."""

    id: str
    content: str
    source_url: str
    title: str  # the question text
    category: str
    qid: int
    # Embedding of ``content``, populated in place by :mod:`embedder`.
    content_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class FaqParseResult:
    """Pure (Azure-free) parse output for one sweep."""

    entries: list[FaqEntry] = field(default_factory=list)
    # category name -> the count shown in that category's header badge.
    category_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing (PURE — no network, no Azure)
# ---------------------------------------------------------------------------
def _clean_inline(value: str) -> str:
    """Unescape entities + collapse all whitespace to single spaces (one line)."""
    return " ".join(html.unescape(value).split())


def _clean_block(value: str) -> str:
    """Unescape + strip per line + drop blank lines (keeps paragraph breaks)."""
    text = html.unescape(value)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


# Block-level tags inside an answer whose boundaries become a newline, so adjacent
# blocks (paragraphs, list items) don't run together when text nodes concatenate.
_ANSWER_BLOCK_TAGS = frozenset(
    {"p", "br", "li", "ul", "ol", "div", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
)

_QID_RE = re.compile(r"^question-(\d+)$")


class _FaqParser(HTMLParser):
    """Extract every (qid, category, question, answer) from the flat FAQ page.

    A state machine in the sibling idiom (cf. ``_RosterParser`` /
    ``_ContainerTextParser``), scoped to the ``faq-container``. Category comes from
    each section's ``<h2 class="m-0">`` (structural, total); the per-category count
    badge is captured for the completeness check. Question text comes from the
    ``accordion-button``; answer text from the ``accordion-text`` div (visible text
    only — link hrefs are dropped, link text kept).
    """

    def __init__(self) -> None:
        super().__init__()
        self.found_container = False
        self.entries: list[FaqEntry] = []
        self.category_counts: dict[str, int] = {}

        self._current_category: str | None = None
        self._cap_category = False
        self._category_parts: list[str] = []

        self._cap_count = False
        self._count_parts: list[str] = []

        self._current_qid: int | None = None
        self._in_question = False
        self._question_parts: list[str] = []
        self._answer_depth = 0  # >0 while inside the accordion-text div
        self._answer_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        classes = attr.get("class", "").split()

        if not self.found_container:
            if tag == "div" and "faq-container" in classes:
                self.found_container = True
            return

        # Category name — the section's <h2 class="m-0">.
        if tag == "h2" and "m-0" in classes:
            self._cap_category = True
            self._category_parts = []
            return

        # Category count badge — a header-level badge (NOT the per-question pill),
        # which appears before any question li in the section.
        if (
            tag == "span"
            and self._current_qid is None
            and "rounded-pill" in classes
            and "category-pill" not in classes
        ):
            self._cap_count = True
            self._count_parts = []
            return

        # A Q&A item — <li id="question-{QID}">.
        if tag == "li":
            match = _QID_RE.match(attr.get("id", ""))
            if match:
                self._current_qid = int(match.group(1))
                self._question_parts = []
                self._answer_parts = []
                return
            # else: a list item INSIDE an answer — fall through to block handling.

        if self._current_qid is None:
            return

        if tag == "button" and "accordion-button" in classes:
            self._in_question = True
            return

        if tag == "div" and "accordion-text" in classes:
            self._answer_depth = 1
            return

        # Inside the answer: track nested div depth + emit block separators.
        if self._answer_depth:
            if tag == "div":
                self._answer_depth += 1
            if tag in _ANSWER_BLOCK_TAGS:
                self._answer_parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._answer_depth and tag in _ANSWER_BLOCK_TAGS:
            self._answer_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cap_category:
            self._category_parts.append(data)
        elif self._cap_count:
            self._count_parts.append(data)
        elif self._in_question:
            self._question_parts.append(data)
        elif self._answer_depth:
            self._answer_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.found_container:
            return
        if self._cap_category and tag == "h2":
            self._cap_category = False
            self._current_category = _clean_inline("".join(self._category_parts)) or None
        elif self._cap_count and tag == "span":
            self._cap_count = False
            text = _clean_inline("".join(self._count_parts))
            if text.isdigit() and self._current_category:
                self.category_counts[self._current_category] = int(text)
        elif self._in_question and tag == "button":
            self._in_question = False
        elif self._answer_depth and tag == "div":
            self._answer_depth -= 1
        elif self._current_qid is not None and tag == "li":
            self.entries.append(
                FaqEntry(
                    qid=self._current_qid,
                    category=self._current_category or "",
                    question=_clean_inline("".join(self._question_parts)),
                    answer=_clean_block("".join(self._answer_parts)),
                )
            )
            self._current_qid = None


def parse_faq(page_html: str) -> FaqParseResult:
    """Parse the flat FAQ page into Q&A entries + per-category header counts (PURE).

    Raises :class:`FaqParseError` if the ``faq-container`` is absent — we never
    silently scrape the whole page.
    """
    parser = _FaqParser()
    parser.feed(page_html)
    if not parser.found_container:
        raise FaqParseError("FAQ page has no 'faq-container' — refusing to scrape whole page")
    return FaqParseResult(entries=parser.entries, category_counts=parser.category_counts)


# ---------------------------------------------------------------------------
# Chunking (PURE) — reuse the docs chunker's cl100k 500/50 windowing
# ---------------------------------------------------------------------------
def _docs_neutral_prefix() -> str:
    """Reconstruct chunk_pages' docs prefix for NEUTRAL (empty) metadata.

    Mirrors :func:`city_services_source._token_windows`: chunk_pages prepends
    ``"{body} {type} — {date}\\n\\n"``; with empty metadata that's a fixed,
    content-free string we strip back off, leaving exactly the cl100k 500/50
    window text. Built with chunker's own formula so it tracks any format change.
    """
    body = document_type = document_date = ""
    return f"{body} {document_type} — {chunker._format_date_human(document_date)}\n\n"


def _token_windows(text: str) -> list[str]:
    """Reuse chunker.chunk_pages for cl100k 500/50 windowing (no reimplement)."""
    docs_chunks = chunker.chunk_pages(
        [{"page_number": 1, "text": text}],
        source_url="",
        document_date="",
        meeting_body="",
        document_type="",
        meeting_category="",
        base_id="faq",
    )
    neutral_prefix = _docs_neutral_prefix()
    windows: list[str] = []
    for chunk in docs_chunks:
        if not chunk.content.startswith(neutral_prefix):
            raise RuntimeError("chunker prefix format drifted; faq strip is stale")
        windows.append(chunk.content[len(neutral_prefix):])
    return windows


def chunk_faq_entry(entry: FaqEntry) -> list[FaqChunk]:
    """Chunk one Q&A into prefixed, index-ready records (PURE).

    Each chunk is prefixed ``"Gloucester FAQ — {category} — {question}\\n\\n"`` so
    the category/question are BM25-findable. IDs are deterministic —
    ``{qid}::{idx}`` run through chunker's key sanitizer (``::`` → ``__``) — so a
    re-run overwrites by key. Most answers are one chunk; long ones (e.g. the
    recycling essays) split into overlapping windows.
    """
    prefix = f"{_PREFIX_LEAD}{entry.category} — {entry.question}\n\n"
    records: list[FaqChunk] = []
    for idx, window in enumerate(_token_windows(entry.answer)):
        records.append(
            FaqChunk(
                id=chunker._sanitize_key(f"{entry.qid}::{idx}"),
                content=prefix + window,
                source_url=entry.source_url,
                title=entry.question,
                category=entry.category,
                qid=entry.qid,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Azure AI Search — dedicated FAQ index (wipe-and-rebuild)
# ---------------------------------------------------------------------------
def _build_faq_index(index_name: str) -> SearchIndex:
    """Build the reduced FAQ index schema.

    Minimal fields — id, content, source_url, title (=question), category, qid,
    content_vector — but the vector + hybrid config is MIRRORED from
    :func:`indexer._build_index` (same HNSW algorithm/profile names, same 1536
    dims, parallel semantic config), so hybrid (BM25 + vector, RRF) behaves
    identically to the docs / city-services indexes.
    """
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="source_url", type=SearchFieldDataType.String),
        # The question — searchable so its terms help BM25 and it can surface as
        # the source label in the UI.
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(
            name="category", type=SearchFieldDataType.String, filterable=True, sortable=False
        ),
        SimpleField(name="qid", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=False,
            vector_search_dimensions=indexer.VECTOR_DIMENSIONS,
            vector_search_profile_name=indexer.VECTOR_PROFILE_NAME,
        ),
    ]
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


def ensure_faq_index(index_name: str = FAQ_INDEX_NAME) -> None:
    """WIPE-AND-REBUILD the FAQ index: delete if present, then create.

    Same rationale as :func:`city_services_source.ensure_city_services_index`: the
    corpus is tiny and fully re-ingested every run, so a clean rebuild sidesteps
    revision-orphan chunks (a removed/renamed Q&A leaves no stale chunk). Kept
    SEPARATE from the docs / city-services ensures so neither is ever wiped here.
    """
    client = SearchIndexClient(indexer._search_endpoint(), credential=indexer._credential())
    try:
        client.delete_index(index_name)
        logger.info("faq: deleted existing index '%s'", index_name)
    except ResourceNotFoundError:
        logger.info("faq: no existing index '%s' to delete", index_name)
    client.create_or_update_index(_build_faq_index(index_name))
    logger.info("faq: created index '%s'", index_name)


# ---------------------------------------------------------------------------
# Fetch + orchestrator
# ---------------------------------------------------------------------------
def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": scraper.BROWSER_USER_AGENT})
    return session


def fetch_faq(session: requests.Session | None = None) -> FaqParseResult:
    """Fetch + parse the whole FAQ. NETWORK ONLY — never touches Azure.

    One 200-gated GET returns the entire FAQ; raises :class:`FaqFetchError` on a
    non-200 / transport error and :class:`FaqParseError` if the container is absent.
    """
    session = session or _new_session()
    try:
        response = session.get(FAQ_PAGE_URL, timeout=60)
    except requests.RequestException as exc:
        raise FaqFetchError(f"{FAQ_PAGE_URL}: request failed: {exc}") from exc
    if response.status_code != 200:
        raise FaqFetchError(f"{FAQ_PAGE_URL}: HTTP {response.status_code} (refusing to parse)")
    return parse_faq(response.text)


def fetch_and_index_faq(index_name: str = FAQ_INDEX_NAME) -> int:
    """Fetch → parse → chunk → embed → (wipe-rebuild) index the FAQ.

    Mirrors :func:`city_services_source.fetch_and_index_city_services`. Embeds
    BEFORE the wipe (a failed embed leaves the live index serving); an empty parse
    leaves the index untouched rather than wiping it to empty. Returns the number
    of chunks indexed.
    """
    result = fetch_faq()
    entries = result.entries

    if not entries:
        logger.warning("faq: parsed zero Q&As; leaving index untouched")
        return 0

    # Per-category completeness: log any drift from the header badge counts (the
    # structural parse matches them exactly, so this should be silent).
    from collections import Counter

    parsed_by_cat = Counter(e.category for e in entries)
    for category, expected in result.category_counts.items():
        got = parsed_by_cat.get(category, 0)
        if got != expected:
            logger.warning(
                "faq: category %r parsed %d but header says %d", category, got, expected
            )

    all_chunks: list[FaqChunk] = []
    for entry in entries:
        all_chunks.extend(chunk_faq_entry(entry))
    logger.info("faq: %d Q&A -> %d chunk(s)", len(entries), len(all_chunks))

    # Embed FIRST (populates content_vector in place), then wipe + upload.
    embedder.embed_chunks(all_chunks)
    ensure_faq_index(index_name)
    indexed = indexer.upload_chunks(all_chunks, index_name=index_name)
    logger.info(
        "faq: %d chunk(s) from %d Q&A indexed into '%s'", indexed, len(entries), index_name
    )
    return indexed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    argparse.ArgumentParser(
        description="Fetch, chunk, embed and (wipe-)rebuild the Gloucester FAQ index."
    ).parse_args()
    print(fetch_and_index_faq())
