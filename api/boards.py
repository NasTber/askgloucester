"""Read-only Azure Table Storage access for Gloucester board/commission appointments.

API-local and deliberately self-contained: this module talks to the ``boards``
table DIRECTLY (``TableServiceClient`` over ``DefaultAzureCredential``) and does
NOT import anything from ``ingestion/`` — the deployed API image ships only
``api/`` (see the Dockerfile); importing ``ingestion`` would crash-loop the
container. The ingestion side (``ingestion/boards_source.py``) owns WRITES to the
table; this module owns the read path used by the ``board_lookup`` agent tool,
plus the presentation helper that turns rows into resident-facing prose.

The table holds TWO row types, distinguished by RowKey (written by ingestion):
  * PERSON rows (RowKey = the numeric appointment ``rid``) — one per appointment.
  * BOARD rows  (RowKey = ``"__board__"``) — one per board, carrying
    ``has_roster`` and the official active count, so the reader can answer
    board-level questions honestly for the boards whose individual members the
    city's appointments system doesn't publish (empty target / elected bodies).

Auth is ``DefaultAzureCredential`` — no account keys. The API managed identity's
**Storage Table Data Reader** role is account-scoped, so it already covers this
table alongside ``events`` and ``officials``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential

# Dedicated table holding the normalized board appointments (written by ingestion).
BOARDS_TABLE = os.environ.get("BOARDS_TABLE_NAME", "boards")

# RowKey sentinel for the per-board (non-person) row (mirror ingestion's enum).
BOARD_ROW_KEY = "__board__"

# Status value for a row that left the source (mirror ingestion's enum). Excluded
# from every read so departed appointments never surface in an answer.
STATUS_REMOVED = "removed"

# The BCC public index, linked when a result set is capped or for board-level info.
BCC_INDEX_URL = "https://appext.gloucester-ma.gov/bcc/view.php?a=BCCs"

# Boards whose members are ELECTED, not appointed — the appointments system has no
# roster for them, so we defer the resident to doc_search instead of a roster.
ELECTED_BODIES = {"School Committee", "City Council"}

# The whole boards corpus is tiny (~180 person rows + 38 board rows), so the read
# path pulls everything and filters in Python — no per-query Table filters. The
# cap bounds a pathological fuzzy match; when a result set is capped the rendering
# says so and links the full index, so a partial list is never shown as complete.
MAX_RESULTS = 30


@dataclass
class BoardMember:
    """One appointed board member as read back from the table (the tool's shape)."""

    rid: int
    board: str
    full_name: str
    position_type: str
    representation: str | None
    expiration_iso: str | None
    is_voting: bool
    source_url: str


@dataclass
class BoardInfo:
    """One board's row — board-level facts for the no-roster / count messaging."""

    board: str
    active_count: int | None
    has_roster: bool
    source_url: str


@dataclass
class BoardsQueryResult:
    """What :func:`search_boards` returns: matched people + matched board rows."""

    members: list[BoardMember]
    boards: list[BoardInfo]


@lru_cache(maxsize=1)
def _table_client():
    """TableClient for the boards table (AAD auth, read-only use).

    ``TableServiceClient`` authenticates with ``DefaultAzureCredential`` against
    ``https://{account}.table.core.windows.net``. We only ever read here, so the
    table is assumed to already exist (created by ingestion); no create call.
    """
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    endpoint = f"https://{account}.table.core.windows.net"
    service = TableServiceClient(endpoint=endpoint, credential=DefaultAzureCredential())
    return service.get_table_client(BOARDS_TABLE)


def _to_member(entity) -> BoardMember:
    """Rebuild a :class:`BoardMember` from a stored PERSON entity."""
    return BoardMember(
        rid=int(entity["RowKey"]),
        board=entity.get("board", entity.get("PartitionKey", "")),
        full_name=entity.get("full_name", ""),
        position_type=entity.get("position_type", ""),
        representation=entity.get("representation"),
        expiration_iso=entity.get("expiration_iso"),
        is_voting=bool(entity.get("is_voting")),
        source_url=entity.get("source_url", ""),
    )


def _to_board(entity) -> BoardInfo:
    """Rebuild a :class:`BoardInfo` from a stored BOARD entity."""
    count = entity.get("board_active_count")
    return BoardInfo(
        board=entity.get("board", entity.get("PartitionKey", "")),
        active_count=int(count) if count is not None else None,
        has_roster=bool(entity.get("has_roster")),
        source_url=entity.get("source_url", ""),
    )


def _load_active() -> tuple[list[BoardMember], list[BoardInfo]]:
    """Read every active row, split into (people, boards).

    The table is small enough to pull whole and filter in Python, so there is no
    OData query to maintain; ``removed`` rows are dropped here so no read path can
    surface them. Rows are split by RowKey: ``"__board__"`` → board, else person.
    """
    table = _table_client()
    members: list[BoardMember] = []
    boards: list[BoardInfo] = []
    for entity in table.list_entities():
        if entity.get("status") == STATUS_REMOVED:
            continue
        if entity.get("RowKey") == BOARD_ROW_KEY:
            boards.append(_to_board(entity))
        else:
            members.append(_to_member(entity))
    return members, boards


def search_boards(query: str) -> BoardsQueryResult:
    """Find board appointments matching ``query`` (people) + matched board rows.

    Two paths in one pass:
      * WHOLE-BOARD listing — when the query names a board (substring either way),
        every person row for that board is returned ("who's on the ZBA"), and the
        board row is captured so the renderer can give a board-level message for a
        no-roster board.
      * PERSON substring — case-insensitive across full_name + board +
        representation + position_type ("who is Harry Hoglander", "who represents
        the Planning Board on CPC").

    Results are sorted by board then name for a stable rendering. Capping is
    applied at render time (so the renderer knows the true total). An empty/blank
    query returns nothing (the tool declines).
    """
    needle = (query or "").strip().lower()
    if not needle:
        return BoardsQueryResult([], [])

    members, boards = _load_active()

    # Boards named by the query (substring either direction) → whole-board listing
    # + board-level rows (for the no-roster / elected messaging).
    matched_boards = [
        b for b in boards if needle in b.board.lower() or b.board.lower() in needle
    ]
    matched_board_names = {b.board for b in matched_boards}

    matched_members = [
        m
        for m in members
        if m.board in matched_board_names
        or needle in f"{m.full_name} {m.board} {m.representation or ''} {m.position_type}".lower()
    ]
    matched_members.sort(key=lambda m: (m.board.lower(), m.full_name.lower()))
    matched_boards.sort(key=lambda b: b.board.lower())
    return BoardsQueryResult(matched_members, matched_boards)


# --- Presentation -----------------------------------------------------------
def render_boards(query: str, result: BoardsQueryResult, limit: int = MAX_RESULTS) -> str:
    """Render board matches as plain prose for the agent.

    People are grouped by board with their position, designated seat
    (representation), and term-expiration date; each board's BCC appointments page
    (``source_url``) is rendered INLINE — there is no ``[n]`` citation channel for
    boards (that belongs to doc_search only). For a matched board that has NO
    roster published (empty target / elected body), an honest board-level message
    is given instead of pretending the members are unknown-because-missing. On no
    match, a clean decline. When more than ``limit`` people match, the first
    ``limit`` are shown with the true total and the full index link — never a
    silent trim. NO vacancy claim is made (the roster count and official active
    count legitimately differ; we don't reverse-engineer "open seats").
    """
    members = result.members
    # Boards named by the query that have no listable roster → board-level message.
    noroster = [b for b in result.boards if not b.has_roster]

    if not members and not noroster:
        return f'I don\'t have board-appointment information matching "{query}".'

    lines: list[str] = []

    if members:
        total = len(members)
        shown = members[:limit]
        lines.append(f'Board & commission appointments matching "{query}":')
        lines.append("")
        current_board: str | None = None
        for m in shown:
            if m.board != current_board:
                if current_board is not None:
                    lines.append("")
                lines.append(f"{m.board}:")
                current_board = m.board
            position = m.position_type or ("Voting Member" if m.is_voting else "Member")
            seat = f", {m.representation} seat" if m.representation else ""
            term = f", term expires {m.expiration_iso}" if m.expiration_iso else ""
            lines.append(f"- {m.full_name} — {position}{seat}{term}")
            lines.append(f"  Appointments: {m.source_url}")
        if total > limit:
            lines.append("")
            lines.append(
                f"Showing {limit} of {total} appointments; full BCC index: {BCC_INDEX_URL}"
            )

    # Honest board-level messages for matched boards with no published roster.
    for b in noroster:
        if members:
            lines.append("")
        if b.board in ELECTED_BODIES:
            lines.append(
                f"{b.board} members are elected, not appointed, so the city's "
                f"appointments system has no roster for them. Ask what the "
                f"{b.board} discussed or decided, or see {b.source_url}."
            )
        else:
            count = b.active_count if b.active_count is not None else "some"
            lines.append(
                f"The {b.board} has {count} active appointment(s) per the city "
                f"index, but its individual members aren't published in the "
                f"appointments system. See {b.source_url}."
            )

    return "\n".join(lines).rstrip()
