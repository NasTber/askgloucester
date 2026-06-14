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

from . import boards, calendar, city_services, directory
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


@tool
def directory_lookup(query: str) -> str:
    """Look up CURRENT Gloucester city staff and officials — who holds a role and how to reach them.

    Use for "who is the IT director", "who runs Public Works", "how do I contact
    the Assessor's office". Covers the PUBLISHED city staff directory (name,
    title, department, department phone, directory page) — NOT what someone said
    or did in a meeting (that is ``doc_search``) and NOT meeting dates/times (that
    is ``schedule_lookup``).

    Args:
        query: A free-text role, name, or department to search for (e.g. "IT
            director", "Public Works", "city clerk").
    """
    matches = directory.search_officials(query)
    return directory.render_officials(query, matches)


@tool
def board_lookup(query: str) -> str:
    """Look up APPOINTED Gloucester board / commission / committee members and their terms.

    Use for who SITS ON or CHAIRS an appointed board, when a member's TERM
    EXPIRES, or who holds a DESIGNATED SEAT (e.g. "who's on the Zoning Board of
    Appeals", "when does the Conservation Commission chair's term end", "who
    represents the Planning Board on the Community Preservation Committee"). Covers
    the PUBLISHED board & commission appointments (name, position, designated
    seat, term-expiration date, board appointments page).

    DISAMBIGUATION — do NOT confuse with directory_lookup: directory_lookup is
    PAID city STAFF and departments (the IT director, the Public Works office);
    board_lookup is APPOINTED, mostly-volunteer BOARD MEMBERS and their terms.
    Also NOT what a body discussed/decided in a meeting (doc_search) and NOT
    meeting dates/times (schedule_lookup).

    Answers are PLAIN PROSE with each board's appointments page inline — there is
    NO [n] citation channel for boards. If board_lookup returns nothing, you MAY
    fall back to doc_search for context before declining.

    Args:
        query: A board name (for a whole-board roster) or a person/role/seat to
            search for (e.g. "Zoning Board of Appeals", "Harry Hoglander",
            "Planning Board representative").
    """
    result = boards.search_boards(query)
    return boards.render_boards(query, result)


@tool
def city_services_search(query: str) -> str:
    """Look up Gloucester city-SERVICE info: trash, recycling, yard waste, compost, special/bulk/hazardous collections.

    Use for how the city's PUBLISHED service pages describe routine operations:
    trash & recycling collection rules, the holiday-shift schedule, yard-waste /
    leaf weeks, the compost facility, special collections (Christmas trees,
    household hazardous waste), and bulk-item / sticker rules. Covers the published
    city-service information pages — NOT what was said or decided in a meeting
    (that is doc_search), NOT meeting dates/times (schedule_lookup), and NOT the
    staff directory (directory_lookup). Returns numbered source excerpts ([n]) to
    cite, exactly like doc_search.

    For LIVE operational status — a one-off suspension or delay, or "is recycling
    running this week" — the ingested page text can lag reality: do not present it
    as current ground truth; point the resident to the city website / AlertCenter
    for the live status.

    Args:
        query: A free-text city-services question (e.g. "holiday trash schedule",
            "yard waste pickup weeks", "household hazardous waste day").
    """
    chunks = city_services.search_city_services(query)
    if not chunks:
        # Deterministic empty signal — no fabricated sources. The agent may then
        # fall back to doc_search (see TOOL_GUIDANCE) before declining.
        return "No matching city-service pages were found for that search."

    # Reuse doc_search's EXACT citation mechanism: append (n, chunk) pairs to the
    # shared per-request state and advance the same running counter, so [n] numbers
    # never collide when city_services_search and doc_search both fire in one turn.
    state = _CITATION_STATE.get()
    start = state["next_n"]
    numbered = build_context(chunks, start=start)
    state["pairs"].extend((start + i, c) for i, c in enumerate(chunks))
    state["next_n"] = start + len(chunks)
    return numbered


# --- Router seam ------------------------------------------------------------
# The tools the agent may call. ``doc_search`` answers "what was discussed /
# decided" from indexed documents (SC + City Council); ``schedule_lookup``
# answers "when do they meet" from the city calendar (full roster);
# ``directory_lookup`` answers "who holds this role / how do I reach them" from
# the published staff directory. Append future tools here — create_agent routes
# to them automatically once registered. This is the single place new
# capabilities plug in.
TOOLS = [doc_search, schedule_lookup, directory_lookup, city_services_search, board_lookup]


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
    "3) directory_lookup — WHO currently holds a role or works in a department, and\n"
    "   HOW to reach them (name, title, department, department phone, directory\n"
    "   page). Use for current staff/official IDENTITY & CONTACT questions: 'who is\n"
    "   the IT director', 'who runs Public Works', 'how do I contact the Assessor's\n"
    "   office'. It covers the PUBLISHED city staff directory — NOT what someone\n"
    "   said or did in a meeting (that is doc_search) and NOT meeting dates/times\n"
    "   (that is schedule_lookup). Directory answers are PLAIN PROSE with the\n"
    "   person's directory page inline (plus any department phone / contact-form\n"
    "   link). If directory_lookup returns nothing, you MAY fall back to doc_search\n"
    "   for context before declining.\n"
    "\n"
    "4) city_services_search — published CITY-SERVICE information pages, covering\n"
    "   TWO categories today:\n"
    "   - Trash & recycling: trash / recycling / yard waste / leaf collection /\n"
    "     compost / special collections (Christmas trees, household hazardous\n"
    "     waste) / bulk-item & sticker / holiday-collection-shift questions.\n"
    "   - Permits & inspections: building / electrical / plumbing / gas / sign /\n"
    "     demolition permits, inspection requests, the building inspector's role,\n"
    "     building-inspector fees, how/where to file, and 'do I need a permit'\n"
    "     questions.\n"
    "   These are PUBLISHED service info pages — NOT meetings, schedules, or the\n"
    "   staff directory: not what was said or decided at a meeting (doc_search),\n"
    "   not meeting dates/times (schedule_lookup), not who holds a role\n"
    "   (directory_lookup). It returns numbered [n] sources — cite them by number\n"
    "   exactly like doc_search. If city_services_search returns nothing, you MAY\n"
    "   fall back to doc_search for context before declining.\n"
    "   LIVE-STATUS CAVEAT: the ingested page text can LAG reality — do NOT assert\n"
    "   current operational status as ground truth. Give what the page says (with\n"
    "   its date if present) and point the resident to the live source:\n"
    "   - Trash/recycling: for a one-off suspension or delay ('is recycling running\n"
    "     this week'), point to the city website / AlertCenter.\n"
    "   - Permits: these pages describe HOW to file (the city's online-permitting\n"
    "     portal) and fee/process info — they do NOT carry the live status of any\n"
    "     specific permit or application. For filing, or to check an application,\n"
    "     point the resident to the city's Online Permitting page / portal (linked\n"
    "     from the cited page), and to AlertCenter for service alerts. Do not state\n"
    "     a permit's current status as fact.\n"
    "\n"
    "5) board_lookup — APPOINTED board / commission / committee MEMBERS and their\n"
    "   TERMS. Route: who SITS ON or CHAIRS a board, when a member's TERM EXPIRES,\n"
    "   who holds a DESIGNATED SEAT ('who represents the Planning Board on the\n"
    "   Community Preservation Committee'), and whole-board rosters ('who's on the\n"
    "   ZBA'). This is the PUBLISHED board & commission appointments list.\n"
    "   CRITICAL — board_lookup vs directory_lookup: directory_lookup = PAID city\n"
    "   STAFF and departments (the IT director, the Public Works office);\n"
    "   board_lookup = APPOINTED, mostly-VOLUNTEER board members + their term dates.\n"
    "   A 'who's on board X' / 'when does their term end' question is board_lookup;\n"
    "   a 'who runs department Y / how do I reach staff' question is\n"
    "   directory_lookup. board_lookup is also NOT what a body discussed or decided\n"
    "   (doc_search) and NOT meeting dates/times (schedule_lookup). Answers are\n"
    "   PLAIN PROSE with each board's appointments page inline (NO [n] markers). For\n"
    "   an ELECTED body (School Committee, City Council) board_lookup will say so —\n"
    "   defer those to doc_search. If board_lookup returns nothing, you MAY fall\n"
    "   back to doc_search for context before declining.\n"
    "\n"
    "CITATIONS. Bracketed numbers like [1] or [2][3] tag doc_search AND\n"
    "city_services_search sources — cite those by number. NEVER put a tool name in\n"
    "brackets — no [schedule_lookup], no [directory_lookup], no [city_services_search],\n"
    "no [board_lookup], no [doc_search], no [toolname] of any kind. Schedule,\n"
    "directory, and board answers carry NO [n] markers; they use plain prose plus\n"
    "the links above.\n"
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
