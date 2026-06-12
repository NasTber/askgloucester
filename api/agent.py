"""LangChain 1.x tool-using agent for AskGloucester.

Replaces the hand-rolled generate step that used to live in ``query.ask``. A
``create_agent`` (LangGraph-backed) agent plans its own retrieval by calling the
``doc_search`` tool, then writes a grounded, cited answer. The Azure retrieval
primitives (embed / hybrid search / OData filters) are reused unchanged from
``query`` — only the orchestration and generation move here.

Authentication is ``DefaultAzureCredential`` end to end: the chat model uses an
AAD bearer-token provider, never an API key (matching the rest of the codebase).

Contract preserved: :func:`ask` returns ``(answer_text, [(n, chunk)])`` where the
chunks are exactly the ones the answer cited, carrying their stable ``[n]``
numbers. Memory is stateless — the client carries ``history`` and we rebuild the
agent's message list each request (no checkpointer; see :func:`ask`).
"""

from __future__ import annotations

import contextvars
import re
import uuid
from datetime import date
from functools import lru_cache

from azure.identity import get_bearer_token_provider
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from .query import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    BODY_KEYWORDS,
    SYSTEM_PROMPT,
    _credential,
    _required,
    build_context,
    embed,
    resolve_latest_meeting_date,
    retrieve,
)

# --- Canonical meeting bodies ----------------------------------------------
# The exact meeting_body constants stored in the index, keyed by their lowercased
# form for normalization. BODY_KEYWORDS already holds the canonical spellings as
# its values; we reuse them so there is one source of truth for "what bodies
# exist". Used ONLY to snap an LLM-provided body string to a known constant — not
# to infer intent (the agent decides whether a body applies).
_CANONICAL_BODIES = {body.lower(): body for body in BODY_KEYWORDS.values()}

# Per-request citation accumulator. doc_search appends (n, chunk) pairs and bumps
# the running counter so a SECOND tool call continues numbering ([11], [12], ...)
# instead of colliding at [1]. ask() sets a fresh dict at the start of every
# request, so concurrent requests never share state.
_CITATION_STATE: contextvars.ContextVar[dict] = contextvars.ContextVar("citation_state")


def _normalize_body(body: str | None) -> str | None:
    """Snap an LLM-provided body string to the exact index constant, or None.

    Normalization only, never intent detection: the model decides whether a body
    applies; here we just map its (possibly differently-cased) string to the
    canonical spelling actually stored in the index, so the OData filter is built
    from a controlled constant and never from raw model text. An unrecognised
    non-empty string returns None, which the caller treats as "body not indexed".
    """
    if not body or not body.strip():
        return None
    return _CANONICAL_BODIES.get(body.strip().lower())


def _normalize_date(value: str | None) -> str | None:
    """Validate an LLM-supplied date down to an exact YYYY-MM-DD literal, or None.

    Only a strictly-formatted ISO date is allowed to reach the OData filter, so
    the model can never inject arbitrary text through ``target_date``.
    """
    if not value or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


@tool(response_format="content_and_artifact")
def doc_search(
    query: str,
    body: str | None = None,
    recency: bool = False,
    target_date: str | None = None,
) -> tuple[str, list[dict]]:
    """Search indexed Gloucester, MA civic meeting documents (agendas and minutes).

    Call this to ground every factual claim — never answer civic questions from
    your own knowledge. Returns numbered source excerpts ([1], [2], ...) for you
    to cite.

    Args:
        query: A focused natural-language search query for the documents.
        body: The meeting body when the user named or clearly implied one. Must be
            exactly one of "City Council", "School Committee", or "Planning Board".
            Omit to search across all bodies.
        recency: True when the user asks about the most recent / last / latest
            meeting of a body. Requires ``body``; pins the search to that body's
            newest past full-committee minutes.
        target_date: A single meeting date as YYYY-MM-DD when the user (or the
            prior turn) refers to one specific meeting. Omit otherwise.
    """
    state = _CITATION_STATE.get()

    # --- Deterministic, server-side filter construction ---------------------
    # Every OData clause is built from a validated constant, never from raw LLM
    # text: body is normalized to a canonical constant, target_date is validated
    # to an ISO literal, meeting_category is a fixed string.
    resolved_body: str | None = None
    if body is not None and body.strip():
        resolved_body = _normalize_body(body)
        if resolved_body is None:
            # The model named a body we do not index — decline rather than
            # silently searching everything and substituting a related body.
            return ("No documents are indexed for that meeting body.", [])

    date_eq = _normalize_date(target_date)
    meeting_category: str | None = None

    # Recency path: pin to the body's newest past full-committee minutes so a
    # later subcommittee/agenda can't masquerade as "the last meeting".
    if recency and resolved_body:
        latest = resolve_latest_meeting_date(resolved_body)
        if latest:
            date_eq = latest
            meeting_category = "full_committee"

    vector = embed(query)
    chunks = retrieve(
        query,
        vector,
        meeting_body=resolved_body,
        date_eq=date_eq,
        meeting_category=meeting_category,
    )

    if not chunks:
        # Deterministic empty signal — no fabricated sources.
        return ("No matching documents were found for that search.", [])

    # --- Global, stable [n] numbering across tool calls ---------------------
    start = state["next_n"]
    numbered = build_context(chunks, start=start)
    state["pairs"].extend((start + i, c) for i, c in enumerate(chunks))
    state["next_n"] = start + len(chunks)

    # content (what the model reads + cites) , artifact (raw chunks for callers)
    return numbered, chunks


# --- Router seam ------------------------------------------------------------
# The tools the agent may call. ``doc_search`` is the document-retrieval tool
# today. Append future tools (calendar, contacts, ...) to this list — create_agent
# routes to them automatically once registered. This is the single place new
# capabilities plug in.
TOOLS = [doc_search]


# Tool-use guidance appended to the ported grounding rules. Tells the agent when
# to set each doc_search argument and reinforces the "don't substitute an
# unindexed body" rule at the planning (tool-call) layer, not just at write time.
TOOL_GUIDANCE = (
    "\n\nTOOLS\n"
    "You have one tool, doc_search. Use it to ground every factual claim; never\n"
    "answer civic questions from your own knowledge.\n"
    "\n"
    "INDEXED BODIES (ALLOWLIST). The ONLY meeting bodies with indexed documents\n"
    "are:\n"
    "  - School Committee\n"
    "  - City Council\n"
    "If the user asks about ANY other named body — for example the Parking\n"
    "Commission, Conservation Commission, Zoning Board of Appeals, Licensing\n"
    "Board, Planning Board, or any commission/board/authority not on the\n"
    "allowlist above — you MUST decline: state briefly that that body's documents\n"
    "are not indexed, and stop. Do NOT search on its behalf. Another body's\n"
    "documents are NEVER a substitute for the named body: e.g. City Council\n"
    "parking ordinances are NOT an answer about the Parking Commission, and must\n"
    "not be offered. Only when a question names no body, or names an allowlisted\n"
    "body, may you search.\n"
    "- Set the body argument when the user names or clearly implies an\n"
    "  allowlisted body; omit it for general questions so the search spans all\n"
    "  indexed bodies.\n"
    "- Set recency=true (together with body) for 'last / latest / most recent\n"
    "  meeting' questions.\n"
    "- For a follow-up question, use the conversation so far to choose the body,\n"
    "  and when the previous answer was about one specific meeting, pass that\n"
    "  meeting's date as target_date so the follow-up stays on the same meeting.\n"
    "- You may call doc_search more than once. Cite sources by their bracketed\n"
    "  number exactly as shown, e.g. [1] or [2][3]. If a search returns no\n"
    "  documents, say so plainly — never invent sources or citation numbers."
)


@lru_cache(maxsize=1)
def _chat_model() -> AzureChatOpenAI:
    """Azure OpenAI chat model authenticated with an AAD bearer-token provider.

    No API key is used: ``get_bearer_token_provider`` wraps the shared
    ``DefaultAzureCredential`` so the SDK fetches and refreshes tokens for the
    Cognitive Services scope automatically.
    """
    token_provider = get_bearer_token_provider(
        _credential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureChatOpenAI(
        azure_endpoint=_required("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_ad_token_provider=token_provider,
        temperature=0,  # deterministic: stay faithful to the sources, no sampling
    )


def _system_prompt() -> str:
    """Ported grounding rules + today's date + tool-use guidance.

    The date is injected fresh per call (the agent is built per request) so the
    model reasons correctly about past vs. upcoming meetings, exactly as the old
    answer() did.
    """
    dated = f"Today's date is {date.today():%A, %B %d, %Y}.\n\n{SYSTEM_PROMPT}"
    return dated + TOOL_GUIDANCE


def _build_agent():
    """Construct the tool-using agent with the dated system prompt.

    Built per request rather than cached so today's date stays current; the
    expensive client (_chat_model) is cached, graph compilation is cheap.
    """
    return create_agent(
        model=_chat_model(),
        tools=TOOLS,
        system_prompt=_system_prompt(),
    )


def _to_messages(history: list[dict] | None) -> list:
    """Rebuild LangChain messages from the client-carried history array."""
    messages: list = []
    for m in history or []:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content))
        elif role == "assistant":
            messages.append(AIMessage(content))
    return messages


def ask(
    question: str, history: list[dict] | None = None
) -> tuple[str, list[tuple[int, dict]]]:
    """Run one agent pass and return ``(answer_text, [(n, chunk)])``.

    The single shared entry point for both the CLI and ``POST /ask`` so they can
    never drift. Stateless memory: the agent's message list is rebuilt from the
    client-carried ``history`` each call and invoked with NO persistent
    checkpointer. The invoke is still structured around LangGraph's
    config/thread_id pattern, so a persistent checkpointer can be dropped in later
    without reworking callers.

    ``source_chunks`` contains exactly the chunks the answer cited (only-cited),
    each with the stable ``[n]`` it was given across all doc_search calls. When the
    answer cites nothing (a decline), the list is empty — the same signal the old
    implementation gave its callers.
    """
    # Fresh per-request citation state (replaces the old retrieval-order list).
    _CITATION_STATE.set({"pairs": [], "next_n": 1})

    messages = _to_messages(history) + [HumanMessage(question)]
    # thread_id is unused without a checkpointer, but wiring it now means adding
    # one later (e.g. for server-side sessions) needs no caller changes.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = _build_agent().invoke({"messages": messages}, config=config)

    last = result["messages"][-1]
    answer_text = last.content if isinstance(last.content, str) else str(last.content)

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    pairs = [(n, c) for (n, c) in _CITATION_STATE.get()["pairs"] if n in cited]
    return answer_text, pairs
