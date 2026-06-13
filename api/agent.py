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

from . import calendar
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
        recency: Set recency=True ONLY for questions explicitly about the most
            recent / last / latest / upcoming meeting of a body (e.g. "what
            happened at the last City Council meeting", "when does the School
            Committee next meet"). Leave False for topical/subject questions —
            anything about a topic, project, budget, vote, or decision ("what's
            the budget", "the library project", "the FY27 appropriations") — even
            if recent documents seem most relevant. recency pins retrieval to a
            single most-recent-meeting date; using it on a topical question hides
            relevant content from other dates. Requires ``body``.
        target_date: A single meeting date as YYYY-MM-DD, set ONLY when the user
            (or the prior turn) names ONE specific meeting day, e.g. "the May 13
            meeting" or "the March 4th agenda". This is an exact-date match, so a
            date with no meeting returns nothing. OMIT it for month-granular,
            seasonal, range, or relative-time questions ("June", "this spring",
            "between March and May", "lately") — those are NOT a single date.
            When you omit it, KEEP the temporal words in the query text: source
            content is date-prefixed (e.g. "City Council agenda — June 9, 2026
            (2026-06-09) ..."), so "June" and date words match in hybrid search.
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


@tool
def schedule_lookup(
    body: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Look up WHEN Gloucester public bodies meet — dates and times of meetings.

    Use for "when does X meet", "next / upcoming meeting", "what's on the
    schedule". Do NOT use for what was discussed, decided, or voted at a meeting —
    that is ``doc_search``. Covers the city's full calendar roster, not just the
    bodies with indexed documents.

    Args:
        body: The public body, when the user named one. Must be on the city
            calendar roster: City Council, School Committee, Conservation
            Commission, Council on Aging, Board of Assessors, Affordable Housing
            Trust, Board of Registrars, Community Preservation, Fisheries
            Commission, Licensing Board, or City-Owned Cemeteries Advisory
            Committee. Omit to span every body.
        start_date: Window start as YYYY-MM-DD. Omit for upcoming meetings (today).
            For "last / previous meeting" questions, pass a past date.
        end_date: Window end as YYYY-MM-DD. Omit to default to ~60 days out.
    """
    # Deterministic body normalization against the calendar roster; decline a body
    # the calendar does not cover rather than silently spanning everything.
    resolved_body: str | None = None
    if body is not None and body.strip():
        resolved_body = calendar.normalize_body(body)
        if resolved_body is None:
            return "That body isn't on the Gloucester public-meeting calendar."

    # Dates are ISO-validated here; the calendar module applies Eastern day
    # boundaries and the default look-ahead window.
    start_utc, end_utc = calendar.window_from_dates(
        _normalize_date(start_date), _normalize_date(end_date)
    )
    events = calendar.get_events(resolved_body, start_utc, end_utc)
    return calendar.render_events(resolved_body, events, start_utc, end_utc)


# --- Router seam ------------------------------------------------------------
# The tools the agent may call. ``doc_search`` answers "what was discussed /
# decided" from indexed documents (SC + City Council); ``schedule_lookup``
# answers "when do they meet" from the city calendar (full roster). Append future
# tools (contacts, ...) here — create_agent routes to them automatically once
# registered. This is the single place new capabilities plug in.
TOOLS = [doc_search, schedule_lookup]


# Tool-use guidance appended to the ported grounding rules. Routes between the two
# tools by INTENT and SCOPE, and reinforces the allowlists at the planning
# (tool-call) layer, not just at write time. The two tools cover different bodies
# on purpose — keep their scopes distinct.
TOOL_GUIDANCE = (
    "\n\nTOOLS — route by what the question is asking.\n"
    "Never answer civic questions from your own knowledge; use a tool.\n"
    "This applies to declines too: never state that the documents don't cover something unless\n"
    "doc_search has already returned nothing relevant for it.\n"
    "\n"
    "1) doc_search — what was DISCUSSED, DECIDED, VOTED, or SAID at a meeting\n"
    "   (the content of agendas and minutes).\n"
    "   DOCUMENT ALLOWLIST: the ONLY bodies with indexed documents are\n"
    "     - School Committee\n"
    "     - City Council\n"
    "   For a content question about ANY other body (Parking Commission,\n"
    "   Conservation Commission, Zoning Board of Appeals, Licensing Board,\n"
    "   Planning Board, etc.) you MUST decline: say that body's documents are not\n"
    "   indexed, and stop. Do NOT search on its behalf and NEVER substitute\n"
    "   another body's documents (e.g. City Council parking ordinances are not an\n"
    "   answer about the Parking Commission). Set body only for an allowlisted\n"
    "   body; omit it to span both. Set recency=true (with body) ONLY for\n"
    "   'last / latest / most recent / next meeting' questions; leave it false for\n"
    "   topical/subject questions (a topic, project, budget, vote, or decision —\n"
    "   'what's the budget', 'the FY27 appropriations') even if recent docs seem\n"
    "   most relevant, since recency pins retrieval to one meeting date and hides\n"
    "   other dates. For a follow-up, reuse the conversation to\n"
    "   pick the body and pass the prior meeting's date as target_date. Cite\n"
    "   sources by their bracketed number exactly as shown, e.g. [1] or [2][3];\n"
    "   if a search returns nothing, say so — never invent sources.\n"
    "\n"
    "2) schedule_lookup — WHEN a body meets (dates/times: next, upcoming, past,\n"
    "   or 'what's on the schedule'). This uses the city's FULL calendar roster,\n"
    "   which is broader than the document allowlist: City Council, School\n"
    "   Committee, Conservation Commission, Council on Aging, Board of Assessors,\n"
    "   Affordable Housing Trust, Board of Registrars, Community Preservation,\n"
    "   Fisheries Commission, Licensing Board, City-Owned Cemeteries Advisory\n"
    "   Committee. Set body when one is named; omit to span all. Pass start_date/\n"
    "   end_date (YYYY-MM-DD) only when the user gives a specific window; omit for\n"
    "   upcoming meetings, and pass a past start_date for 'last meeting' timing.\n"
    "   Schedule answers are PLAIN PROSE: include the relevant calendar link(s)\n"
    "   inline — each event carries a 'Details:' permalink — so residents can\n"
    "   click through to the official calendar entry.\n"
    "\n"
    "CITATIONS. Bracketed numbers like [1] or [2][3] are ONLY for doc_search\n"
    "document sources. NEVER put a tool name in brackets — no [schedule_lookup],\n"
    "no [doc_search], no [toolname] of any kind. Schedule answers carry NO [n]\n"
    "markers; they use plain prose plus the calendar links above.\n"
    "\n"
    "SCOPE CROSS-CASE (important): the two rosters differ.\n"
    " - Conservation Commission: a CONTENT question ('what did they decide') must\n"
    "   DECLINE (not in the document allowlist), but a SCHEDULE question ('when do\n"
    "   they meet') is ANSWERABLE via schedule_lookup.\n"
    " - A body on NEITHER the document allowlist NOR the calendar roster (e.g.\n"
    "   Parking Commission, or any made-up board): for ANY question — content OR\n"
    "   schedule — decline cleanly. Say that neither the indexed documents nor the\n"
    "   city meeting calendar cover that body. Do NOT offer to 'check the calendar'\n"
    "   for it and do NOT call schedule_lookup — it is not on the roster.\n"
    " When in doubt which tool: 'what/why/who decided' → doc_search;\n"
    " 'when/next/upcoming' → schedule_lookup."
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
