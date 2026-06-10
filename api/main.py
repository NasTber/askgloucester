"""FastAPI app for AskGloucester.

A thin HTTP layer over the RAG loop in ``query.ask``. The endpoint owns no
retrieval or generation logic of its own — it validates input, calls ``ask``,
and shapes the result into JSON. Keeping ``ask`` as the single source of truth
means the API and the CLI can never answer the same question differently
(including the deterministic "no documents" decline, which ``ask`` returns with
an empty source list).

Run locally:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Import the module (not the bare name) so the shared RAG entry point and its
# Azure clients are initialised exactly once, lazily, on first request.
from . import query

app = FastAPI(
    title="AskGloucester",
    description="Civic AI assistant answering questions about Gloucester, MA municipal documents.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    """Body for POST /ask."""

    # min_length=1 rejects empty/whitespace-only questions at the edge so we
    # never spend an embedding call on nothing.
    question: str = Field(..., min_length=1, description="The question to ask.")


class Source(BaseModel):
    """One retrieved chunk, surfaced so the client can show provenance.

    Field names and order mirror the citation numbering produced by
    ``build_context`` — the Nth item here is what the answer cites as [N].
    """

    # Citation number matching the [n] markers in the answer text; 1-based.
    n: int
    meeting_body: str | None = None
    document_type: str | None = None
    document_date: str | None = None
    page_number: int | None = None
    source_url: str | None = None
    chunk_id: str | None = None


class AskResponse(BaseModel):
    """Response for POST /ask."""

    answer: str
    # Empty whenever the answer is a decline (no matching documents) — the same
    # signal ``ask`` gives its callers.
    sources: list[Source]


def _to_source(chunk: dict, n: int) -> Source:
    """Project a raw search chunk down to the public Source shape.

    Only whitelisted fields are exposed; ``content`` and any vector data stay
    server-side. Going through the Pydantic model drops anything unexpected.

    ``n`` is the 1-based citation number matching the [n] markers in the answer.
    """
    return Source(
        n=n,
        meeting_body=chunk.get("meeting_body"),
        document_type=chunk.get("document_type"),
        document_date=chunk.get("document_date"),
        page_number=chunk.get("page_number"),
        source_url=chunk.get("source_url"),
        chunk_id=chunk.get("chunk_id"),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Container Apps — no Azure calls, always cheap."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """Answer a question about Gloucester civic documents via RAG.

    Delegates the whole retrieval-and-generation pass to ``query.ask`` and only
    reshapes its (answer, chunks) tuple into JSON.
    """
    answer_text, chunks = query.ask(request.question)
    return AskResponse(
        answer=answer_text,
        sources=[_to_source(c, i) for i, c in enumerate(chunks, start=1)],
    )
