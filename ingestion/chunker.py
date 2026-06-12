"""Chunk extracted document text into overlapping, token-sized segments.

Takes the page-by-page output of :mod:`processor` and produces ~500-token
chunks with 50 tokens of overlap, using a sliding window over the document's
token stream. Each chunk carries the metadata needed to cite it later:
``source_url``, ``document_date``, ``meeting_body``, ``document_type``,
``page_number`` and a stable ``chunk_id``.

Token counts use tiktoken's ``cl100k_base`` encoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date as _date

import tiktoken

# cl100k_base is the encoding used by current OpenAI/Azure embedding and chat
# models; it's a reasonable, widely-available proxy for "tokens" here.
_ENCODING = tiktoken.get_encoding("cl100k_base")

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


@dataclass
class Chunk:
    """A single searchable chunk of text plus its provenance metadata."""

    id: str
    content: str
    source_url: str
    document_date: str
    meeting_body: str
    document_type: str
    meeting_category: str
    page_number: int
    chunk_id: str
    # Embedding of ``content`` produced by :mod:`embedder` (1536-dim,
    # text-embedding-3-small). Empty until the embedding step runs; an empty
    # list is a valid value for Azure AI Search's nullable vector field.
    content_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _sanitize_key(value: str) -> str:
    """Coerce a string into a valid Azure AI Search document key.

    Keys may only contain letters, digits, underscore, dash and equals.
    """
    return "".join(c if c.isalnum() or c in "_-=" else "_" for c in value)


def _format_date_human(iso_date: str) -> str:
    """Return 'January 14, 2026 (2026-01-14)' from '2026-01-14'."""
    try:
        d = _date.fromisoformat(iso_date)
        return f"{d.strftime('%B %-d, %Y')} ({iso_date})"
    except (ValueError, AttributeError):
        return iso_date


def chunk_pages(
    pages,
    source_url: str,
    document_date: str,
    meeting_body: str,
    document_type: str,
    meeting_category: str,
    base_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a document's pages into overlapping token windows.

    Args:
        pages: Iterable of objects/dicts with ``page_number`` and ``text``
            (e.g. :class:`processor.ExtractedPage`).
        source_url: Original URL of the document, for citations.
        document_date: ISO date of the meeting (``YYYY-MM-DD``).
        meeting_body: Committee name.
        document_type: "agenda" or "minutes".
        meeting_category: "full_committee", "subcommittee" or "negotiations".
        base_id: Stable prefix identifying the source document (e.g. the blob
            name). Used to build unique chunk ids.
        chunk_size: Target tokens per chunk.
        overlap: Tokens shared between consecutive chunks.

    Returns:
        A list of :class:`Chunk` in document order.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Flatten the document into a single token stream, remembering which page
    # each token came from so we can attribute every chunk to a page.
    token_ids: list[int] = []
    token_pages: list[int] = []
    for page in pages:
        page_number = getattr(page, "page_number", None)
        text = getattr(page, "text", None)
        if page_number is None:  # support plain dicts too
            page_number, text = page["page_number"], page["text"]
        ids = _ENCODING.encode(text)
        token_ids.extend(ids)
        token_pages.extend([page_number] * len(ids))

    prefix = (
        f"{meeting_body} {document_type} — "
        f"{_format_date_human(document_date)}\n\n"
    )

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    total = len(token_ids)
    index = 0
    start = 0

    while start < total:
        window = token_ids[start : start + chunk_size]
        text = _ENCODING.decode(window).strip()
        if text:
            chunk_id = f"{base_id}::{index}"
            chunks.append(
                Chunk(
                    id=_sanitize_key(chunk_id),
                    content=prefix + text,
                    source_url=source_url,
                    document_date=document_date,
                    meeting_body=meeting_body,
                    document_type=document_type,
                    meeting_category=meeting_category,
                    # Attribute the chunk to the page its first token came from.
                    page_number=token_pages[start],
                    chunk_id=chunk_id,
                )
            )
            index += 1
        if start + chunk_size >= total:
            break
        start += step

    return chunks


if __name__ == "__main__":
    # Tiny self-test with synthetic pages.
    sample = [
        type("P", (), {"page_number": 1, "text": "word " * 800})(),
        type("P", (), {"page_number": 2, "text": "other " * 400})(),
    ]
    out = chunk_pages(
        sample,
        source_url="https://example.org/doc.pdf",
        document_date="2026-01-01",
        meeting_body="School Committee",
        document_type="minutes",
        meeting_category="full_committee",
        base_id="example/doc.pdf",
    )
    print(f"{len(out)} chunks")
    for c in out:
        print(c.chunk_id, "page", c.page_number, "tokens~", len(_ENCODING.encode(c.content)))
