"""Azure AI Search retrieval primitives for AskGloucester.

The reusable building blocks of a retrieval-augmented-generation (RAG) pass over
the Gloucester civic documents indexed in Azure AI Search:

1. ``embed``         — turn the question into a vector with Azure OpenAI.
2. ``retrieve``      — hybrid (keyword + vector) search over the index.
3. ``build_context`` — assemble a numbered, grounded source block.

Generation and orchestration live in ``api.agent`` (a LangChain tool-using
agent that calls these primitives via the ``doc_search`` tool). ``ask`` here is a
thin delegator to that agent, kept as the single shared entry point so the CLI
(``main``) and the FastAPI ``/ask`` endpoint can never answer differently. Only
``main()`` knows about argparse and printing.

Authentication is ``DefaultAzureCredential`` end to end; no API keys are read
or stored anywhere. The Azure OpenAI client authenticates with an AAD bearer
token provider rather than a key.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from functools import lru_cache

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --- Configuration (exact env names + defaults requested) -------------------
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "gloucester-documents")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini"
)
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Number of chunks to retrieve and feed to the model as grounding.
TOP_K = 10

# Index fields safe to return. content_vector is intentionally absent — it is
# not retrievable in the index schema, so requesting it would error.
SELECT_FIELDS = [
    "content",
    "source_url",
    "document_date",
    "meeting_body",
    "document_type",
    "page_number",
    "chunk_id",
]

# The grounding contract: answer only from sources, cite with [n], admit gaps.
SYSTEM_PROMPT = (
    "You are AskGloucester, an assistant that answers questions about the City of Gloucester, MA\n"
    "using ONLY numbered source excerpts returned by your tools. For any civic factual question\n"
    "you MUST call doc_search to retrieve sources before you answer OR decline — never judge from\n"
    "your own knowledge whether the indexed documents cover a topic.\n"
    "\n"
    "Follow these rules strictly:\n"
    "\n"
    "1. GROUNDING. Base every statement solely on the provided sources. Do not use outside\n"
    "   knowledge or guess. Cite each claim with its source number(s), e.g. [1] or [2][3].\n"
    "\n"
    "2. NOT IN SOURCES. If — after calling doc_search — the returned sources do not cover the\n"
    "   meeting body, topic, or time period asked about, your ENTIRE answer must be a brief\n"
    "   statement that the indexed Gloucester documents don't cover it. Never make this statement\n"
    "   without having searched first. Do NOT summarize what the sources DO contain. Do NOT answer\n"
    "   a related or substitute question.\n"
    "\n"
    "3. AGENDAS vs MINUTES. Each source is labeled \"agenda\" or \"minutes\". An agenda lists items\n"
    "   PLANNED for a meeting — it does not prove anything was discussed, voted on, or approved.\n"
    "   Minutes are the official record of what actually happened. For questions about what was\n"
    "   decided, approved, or voted, rely on minutes. If only an agenda supports a point, say the\n"
    "   item was SCHEDULED or ON THE AGENDA, not that it was decided or approved.\n"
    "\n"
    "4. DATES. These are historical records. A meeting dated before today is in the PAST — never\n"
    "   describe a past meeting as \"upcoming\" or \"next\". Only call a meeting upcoming if its date\n"
    "   is after today's date, stated above. Always state meeting dates explicitly.\n"
    "\n"
    "5. COMPLETENESS. The sources are a limited set retrieved for this question, not the complete\n"
    "   record. Do not imply your answer is exhaustive. Phrase accordingly (e.g. \"based on the\n"
    "   records I found\") and note the resident can check the cited documents for more.\n"
    "\n"
    "6. VOICE & FORMAT. Write for a resident in plain, warm language — not bureaucratic.\n"
    "   - Lead with a direct one- or two-sentence answer to exactly what was asked, then\n"
    "     add the supporting detail.\n"
    "   - Separate ideas into short paragraphs with a blank line between them. When listing\n"
    "     multiple items (votes, agenda items, dates), put each on its own line starting\n"
    "     with \"• \".\n"
    "   - Plain text only — do NOT use Markdown (**bold**, #headings, or tables); it renders\n"
    "     as literal characters. Blank lines and \"• \" bullets are the only formatting the\n"
    "     interface displays.\n"
    "   - Keep each [n] citation immediately after the claim it supports.\n"
    "   - Be concise but complete: synthesize across the sources into a genuinely useful\n"
    "     summary, not a terse one-liner or a wall of quotes."
)

# The exact meeting_body values stored in the index, keyed by a canonical
# lowercase form. These are the controlled constants the OData filter is built
# from (never raw user/LLM text). ``api.agent`` reuses the values to normalize an
# LLM-provided body string to the right casing/spelling. Extend this as new
# bodies are ingested.
BODY_KEYWORDS = {
    "city council": "City Council",
    "school committee": "School Committee",
    "planning board": "Planning Board",
    "conservation commission": "Conservation Commission",
    "zoning board of appeals": "Zoning Board of Appeals",
}


def _required(name: str, value: str | None) -> str:
    """Return a required config value or raise a clear error if it's unset."""
    if not value:
        raise RuntimeError(
            f"{name} is required but not set. Add it to your .env or environment."
        )
    return value


@lru_cache(maxsize=1)
def _credential() -> DefaultAzureCredential:
    """Single shared credential (token caching) for all Azure clients."""
    return DefaultAzureCredential()


@lru_cache(maxsize=1)
def _openai_client() -> AzureOpenAI:
    """Azure OpenAI client authenticated with an AAD bearer token provider.

    ``get_bearer_token_provider`` wraps the credential so the SDK fetches and
    refreshes tokens automatically for the Cognitive Services scope — no API
    key is ever used.
    """
    token_provider = get_bearer_token_provider(
        _credential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=_required("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
        api_version=AZURE_OPENAI_API_VERSION,
        azure_ad_token_provider=token_provider,
    )


@lru_cache(maxsize=1)
def _search_client() -> SearchClient:
    """Azure AI Search client for the configured index (AAD auth)."""
    return SearchClient(
        endpoint=_required("AZURE_SEARCH_ENDPOINT", AZURE_SEARCH_ENDPOINT),
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=_credential(),
    )


def embed(question: str) -> list[float]:
    """Embed the question into a vector with the Azure OpenAI embedding model.

    The returned vector is compared against the index's ``content_vector`` field
    during the vector half of the hybrid search.
    """
    client = _openai_client()
    response = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=question,
    )
    return response.data[0].embedding


def resolve_latest_meeting_date(body: str) -> str | None:
    """Return the date of the newest past *minutes* for a body, or None.

    Finds the most recent full-committee ``minutes`` document for ``body`` dated
    on or before today, and returns its ``document_date`` string. Restricting to
    minutes (not agendas) and to ``document_date le today`` means a future-dated
    agenda can never masquerade as "the latest meeting" — we anchor on the newest
    meeting that has an official record. The ``meeting_category eq
    'full_committee'`` clause mirrors what :func:`retrieve` applies on the recency
    path: a subcommittee or negotiations session whose minutes are dated later
    than the last full meeting must not become the anchor date, or retrieve's
    own full_committee filter would then find nothing and the user would get a
    decline despite earlier full-committee minutes existing. Returns None when
    the body has no such minutes, letting the caller fall back to the normal
    retrieval path.

    ``body`` is a controlled constant from BODY_KEYWORDS, never raw user text, so
    it is safe to interpolate into the OData filter.
    """
    today = date.today().isoformat()
    search_filter = (
        f"meeting_body eq '{body}' "
        f"and document_type eq 'minutes' "
        f"and meeting_category eq 'full_committee' "
        f"and document_date le '{today}'"
    )
    results = _search_client().search(
        search_text="*",
        filter=search_filter,
        order_by=["document_date desc"],  # newest first
        top=1,
        select=["document_date"],
    )
    # At most one result; return its date or None if the body has no minutes yet.
    for r in results:
        return r.get("document_date")
    return None


def retrieve(
    question: str,
    vector: list[float],
    meeting_body: str | None = None,
    date_eq: str | None = None,
    meeting_category: str | None = None,
) -> list[dict]:
    """Hybrid search the index: keyword + vector in one request.

    Passing both ``search_text`` (BM25 keyword matching) and ``vector_queries``
    (approximate nearest-neighbour over the embeddings) makes Azure AI Search
    fuse the two rankings — keyword matching catches exact terms/names while the
    vector side catches semantic paraphrases. ``select`` restricts the returned
    fields to retrievable ones (never content_vector).

    If ``meeting_body`` is given, an OData ``filter`` restricts the candidate set
    to that body BEFORE scoring (Azure's default preFilter mode applies it to
    both the keyword and vector halves). This is the hard guarantee: a City
    Council question can only return City Council chunks — or nothing — so the
    model is never handed mismatched text to summarise. When the filter yields
    no hits, this returns an empty list and the caller short-circuits.

    ``meeting_category`` is only passed on the "latest meeting" path (see
    :func:`ask`), where it pins retrieval to ``full_committee`` so a later
    subcommittee or negotiations session can't be returned as "the last
    meeting". General queries leave it None and search every category. Like
    ``meeting_body``, it is a controlled constant, never raw user text, so it is
    safe to interpolate into the OData filter.

    Semantic ranker (query_type="semantic") is intentionally NOT enabled — the
    Free search tier may not support it; we can layer it on later.
    """
    # Build the OData filter from parts so body, date and category constraints
    # compose: any combination, or none. With no parts this stays None (search
    # everything) — the original unchanged behaviour.
    filter_parts = []
    if meeting_body:
        filter_parts.append(f"meeting_body eq '{meeting_body}'")
    if date_eq:
        filter_parts.append(f"document_date eq '{date_eq}'")
    if meeting_category:
        filter_parts.append(f"meeting_category eq '{meeting_category}'")
    search_filter = " and ".join(filter_parts) if filter_parts else None

    # When pinned to one meeting's date, lift the cap so the whole meeting comes
    # back (a single meeting can run to dozens of chunks), not just the top 10.
    top = 50 if date_eq else TOP_K

    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=TOP_K,
        fields="content_vector",
    )
    results = _search_client().search(
        search_text=question,            # keyword (BM25) half of the hybrid query
        vector_queries=[vector_query],   # vector (ANN) half of the hybrid query
        select=SELECT_FIELDS,
        filter=search_filter,            # None = search everything (unchanged behaviour)
        top=top,
    )
    # Materialise the lazy paged iterator into a plain list of dicts.
    return [dict(r) for r in results]


def build_context(chunks: list[dict], start: int = 1) -> str:
    """Assemble retrieved chunks into a numbered, grounded source block.

    Each chunk becomes one numbered source. The numbering ([1], [2], ...)
    follows the chunk order and is what the model is told to cite. ``start`` lets
    a caller continue the numbering across multiple retrieval calls — the agent's
    ``doc_search`` tool passes a running offset so a second search produces
    ``[11], [12], ...`` instead of colliding back at ``[1]``.
    """
    blocks = []
    for i, c in enumerate(chunks, start=start):
        # A short provenance header helps the model attribute claims correctly.
        header_bits = [
            c.get("meeting_body"),
            c.get("document_type"),
            c.get("document_date"),
        ]
        header = " — ".join(b for b in header_bits if b)
        source_url = c.get("source_url", "")
        content = (c.get("content") or "").strip()
        blocks.append(
            f"[{i}] {header}\n"
            f"Source URL: {source_url}\n"
            f"{content}"
        )
    return "\n\n".join(blocks)


def _format_sources(chunks: list[tuple[int, dict]]) -> str:
    """Render the numbered source list for CLI display (matches citation order)."""
    lines = []
    for n, c in chunks:
        header_bits = [
            c.get("meeting_body"),
            c.get("document_type"),
            c.get("document_date"),
        ]
        header = " — ".join(b for b in header_bits if b) or "(untitled source)"
        url = c.get("source_url", "")
        page = c.get("page_number")
        page_str = f", p.{page}" if page is not None else ""
        lines.append(f"[{n}] {header}{page_str}\n    {url}")
    return "\n".join(lines)


def ask(question: str, history: list[dict] | None = None) -> tuple[str, list[tuple[int, dict]]]:
    """Run one agent pass and return ``(answer_text, source_chunks)``.

    Thin delegator to :func:`api.agent.ask` — the single shared entry point for
    both the CLI (:func:`main`) and the FastAPI ``/ask`` endpoint, so they can
    never answer the same question differently. The agent plans retrieval (via the
    ``doc_search`` tool wrapping the primitives above) and writes a cited answer.
    ``source_chunks`` holds exactly the chunks the answer cited, each with its
    stable ``[n]``; it is empty whenever the answer cites nothing (a decline) —
    the signal callers rely on.

    Imported lazily so this module has no import-time dependency on ``api.agent``
    (which imports the retrieval primitives from here).
    """
    from .agent import ask as _agent_ask

    return _agent_ask(question, history=history)


def main() -> None:
    """CLI entry point: thin wrapper over ask() for terminal use."""
    parser = argparse.ArgumentParser(
        description="Ask a question about Gloucester civic documents (RAG)."
    )
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args()

    result, chunks = ask(args.question)

    print(result)
    if chunks:
        print("\nSources:")
        print(_format_sources(chunks))


if __name__ == "__main__":
    main()