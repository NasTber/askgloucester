"""Ingest Gloucester's Boards / Commissions / Committees appointments.

A SEPARATE, structured source — a sibling of :mod:`directory_source` and
:mod:`calendar_source`. Like them, the fetch+parse half is **PURE** (no Azure) and
the source upserts one normalized row per appointment into a dedicated ``boards``
table, so a ``board_lookup`` tool can answer "who's on / who chairs board X, when
does a member's term expire, who holds a designated seat" from structured data.

Source shape (decoded by probing — see the verification harness)
----------------------------------------------------------------
The city's BCC app at ``https://appext.gloucester-ma.gov/bcc/`` is a **JavaScript
SPA**, NOT server-rendered HTML tables. Every ``view.php`` page ships the COMPLETE
client-side cache inline as a ``var data = {…};`` JSON blob; jQuery DataTables then
renders it. So a SINGLE raw HTTP GET returns everything — no per-board fetching, no
``html.parser`` on tables (the tables are empty scaffolds). We extract the blob,
``json.loads`` it, and work from two arrays:

  * ``data.cache.BCCs`` — the 38 boards. Numeric-keyed fields:
    ``"6"`` canonical name, ``"41"`` board target id (EMPTY for some boards),
    ``"56"`` official active count, ``"9"`` min voting, ``"10"`` max voting.
  * ``data.cache.Appointments`` — every appointment row. ``"64"`` full name,
    ``"94"`` position type, ``"52"`` representation, ``"9"`` expiration (epoch-ms
    or ``""``), ``"155"`` board target (joins to a board's ``"41"``), ``rid``
    stable unique id.

Ingest rules (agreed; see the module verification)
--------------------------------------------------
1. REAL BOARDS = ``BCCs`` entries with a non-empty integer ``"41"``.
2. MEMBERSHIP = ``Appointments`` whose ``"155"`` joins a real board (mirrors the
   app's own ``e[155] === target`` roster filter). This structurally drops the
   off-list "Magnolia Pier" orphans and any non-board row.
3. TEST DROP = after rule 2, drop any row whose name contains "TEST"
   (case-insensitive), logged per drop (never silent — like the directory's
   titleless drop).
4. RowKey = ``rid`` (survives renames). PartitionKey = canonical board name
   (joined ``155`` → ``41`` → ``"6"`` — ONE canonical source).
5. NO vacancy reconstruction. The roster sum (~179) differs from the index's
   official 177; that's accepted — we mirror the source faithfully and store the
   counts as fields rather than reverse-engineer the server's "active" rule.

Two row types live in the ``boards`` table, distinguished by RowKey:
  * PERSON row — RK = ``str(rid)``; one per roster appointment.
  * BOARD row  — RK = ``"__board__"`` (non-numeric, cannot collide with a rid);
    one per board INCLUDING the no-roster ones (empty target, or elected (0)
    bodies), carrying ``has_roster`` so the reader can answer board-level
    questions honestly.

Reconciliation mirrors :mod:`directory_source`: a stored row now ABSENT from a
fresh sweep is marked ``status=removed`` (never hard-deleted). One GET is the whole
cache, so on a SUCCESSFUL fetch the reconcile scope is the entire table; a FAILED
fetch skips reconcile (so a transient error can't wrongly "remove" everyone).

Auth is ``DefaultAzureCredential`` end to end (Table endpoint
``https://{account}.table.core.windows.net``) — no account keys.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import requests
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Reuse the scraper's browser User-Agent so the request looks like a real browser
# (mirrors directory_source — some CivicPlus/city endpoints reject a bare client).
import scraper

load_dotenv()

logger = logging.getLogger(__name__)

# The BCC SPA. One GET of the index view returns the full client-side cache.
BCC_BASE_URL = os.environ.get("BCC_BASE_URL", "https://appext.gloucester-ma.gov/bcc")
BCC_INDEX_URL = f"{BCC_BASE_URL}/view.php?a=BCCs"
# Per-board appointments page (the citable roster URL), keyed by board target id.
BCC_APPOINTMENTS_TEMPLATE = BCC_BASE_URL + "/view.php?a=appointments&target={target}"

# Dedicated table for the structured boards data. Created on first use.
BOARDS_TABLE = os.environ.get("BOARDS_TABLE_NAME", "boards")

# RowKey sentinel for the per-board (non-person) row. Non-numeric, so it can never
# collide with a person row's RowKey (which is the numeric appointment rid).
BOARD_ROW_KEY = "__board__"

# --- Status enum (string values stored in the table) ------------------------
STATUS_ACTIVE = "active"
STATUS_REMOVED = "removed"  # was in the source, now absent from a fresh sweep

# Cache field keys (the BCC app's numeric column ids — see module docstring).
_B_NAME, _B_TARGET, _B_ACTIVE, _B_MIN, _B_MAX = "6", "41", "56", "9", "10"
_A_NAME, _A_POSITION, _A_REPR, _A_EXPIRATION, _A_TARGET = "64", "94", "52", "9", "155"


# ---------------------------------------------------------------------------
# Parsed (Azure-free) data shapes
# ---------------------------------------------------------------------------
@dataclass
class Board:
    """One board (BOARD row source). target is None for the no-roster bodies."""

    name: str
    target: int | None
    active_count: int
    min_voting: int | None
    max_voting: int | None
    has_roster: bool = False  # set True once any person row joins to it


@dataclass
class BoardMember:
    """One parsed appointment (PERSON row source)."""

    rid: int
    board: str
    board_target: int
    full_name: str
    position_type: str
    representation: str | None
    expiration_iso: str | None
    is_voting: bool
    board_min_voting: int | None
    board_max_voting: int | None
    board_active_count: int
    source_url: str


@dataclass
class BoardsFetchResult:
    """Pure (Azure-free) fetch+parse output for one sweep."""

    boards: list[Board] = field(default_factory=list)
    members: list[BoardMember] = field(default_factory=list)
    dropped_test: list[tuple[str, str]] = field(default_factory=list)  # (board, name)
    dropped_orphans: list[tuple[Any, str]] = field(default_factory=list)  # (target, name)
    fetch_ok: bool = False


@dataclass
class BoardsSweepResult:
    """Counters returned by :func:`fetch_and_upsert_boards` for reporting."""

    boards: int = 0
    members: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failures: int = 0
    dropped_test: int = 0
    dropped_orphans: int = 0


# ---------------------------------------------------------------------------
# Parsing (PURE — no network, no Azure)
# ---------------------------------------------------------------------------
def _clean_text(value: str) -> str:
    """Unescape HTML entities and collapse whitespace. Mirrors directory_source."""
    return " ".join(html.unescape(str(value)).split())


def _as_int(value: Any) -> int | None:
    """Return an int for a real integer value, else None (handles ""/None/str)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _epoch_ms_to_iso(value: Any) -> str | None:
    """Convert an epoch-millisecond expiration to ISO ``YYYY-MM-DD``, else None.

    The BCC app stores expirations as epoch-ms (e.g. ``1802649600000``) and uses
    ``""`` for "no expiration" (permanent / some non-voting roles). Anything that
    isn't a positive number maps to None — null-safe by construction.
    """
    ms = _as_int(value)
    if not ms or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def extract_data_blob(page_html: str) -> dict:
    """Extract and parse the ``var data = {…};`` JSON blob from a BCC page (PURE).

    Scans for the first ``{`` after ``var data =`` and walks the string tracking
    brace depth (string-literal aware) to find the matching ``}``, then
    ``json.loads`` that slice. Stdlib only — no bs4, no JS engine. Raises
    ValueError if the blob is absent or unbalanced.
    """
    marker = page_html.find("var data =")
    if marker == -1:
        raise ValueError("BCC page has no 'var data =' blob")
    start = page_html.find("{", marker)
    if start == -1:
        raise ValueError("BCC 'var data' has no opening brace")

    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(page_html)):
        ch = page_html[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(page_html[start : i + 1])
    raise ValueError("BCC 'var data' blob never closed")


def parse_boards_cache(cache: dict) -> BoardsFetchResult:
    """Parse ``data.cache`` into Board + BoardMember rows (PURE, Azure-free).

    Applies the agreed rules: real boards = integer ``"41"``; membership =
    appointments joining a real board's target; TEST rows dropped (logged);
    PartitionKey/board name taken from the JOINED board (``155`` → ``41`` →
    ``"6"``), the single canonical source.
    """
    result = BoardsFetchResult(fetch_ok=True)

    bccs = cache.get("BCCs") or []
    appointments = cache.get("Appointments") or []

    # Rule 1 — real boards (non-empty integer target). Build target -> Board.
    boards_by_target: dict[int, Board] = {}
    for raw in bccs:
        name = _clean_text(raw.get(_B_NAME, ""))
        if not name:
            continue
        target = _as_int(raw.get(_B_TARGET))
        board = Board(
            name=name,
            target=target,
            active_count=_as_int(raw.get(_B_ACTIVE)) or 0,
            min_voting=_as_int(raw.get(_B_MIN)),
            max_voting=_as_int(raw.get(_B_MAX)),
        )
        result.boards.append(board)
        if target is not None:
            # Last writer wins on a duplicate target (none observed); keep simple.
            boards_by_target[target] = board

    # Rules 2-4 — membership join + TEST drop.
    for raw in appointments:
        target = _as_int(raw.get(_A_TARGET))
        board = boards_by_target.get(target) if target is not None else None
        if board is None:
            # Rule 2: "155" doesn't join a real board → off-list orphan, dropped.
            result.dropped_orphans.append(
                (raw.get(_A_TARGET), _clean_text(raw.get(_A_NAME, "")))
            )
            continue

        full_name = _clean_text(raw.get(_A_NAME, ""))
        if "test" in full_name.lower():
            # Rule 3: structural join already passed, but a TEST name slipped
            # through (the app seeds fake boards/people). Drop + log, never store.
            logger.warning(
                "Boards: dropping TEST row on %r: rid=%s name=%r",
                board.name, raw.get("rid"), full_name,
            )
            result.dropped_test.append((board.name, full_name))
            continue

        rid = _as_int(raw.get("rid"))
        if rid is None:
            logger.warning(
                "Boards: dropping row with no rid on %r: name=%r", board.name, full_name
            )
            continue

        position_type = _clean_text(raw.get(_A_POSITION, ""))
        representation = _clean_text(raw.get(_A_REPR, "")) or None
        member = BoardMember(
            rid=rid,
            board=board.name,
            board_target=board.target,  # int by construction (board joined by target)
            full_name=full_name,
            position_type=position_type,
            representation=representation,
            expiration_iso=_epoch_ms_to_iso(raw.get(_A_EXPIRATION)),
            is_voting=position_type.startswith("Voting"),
            board_min_voting=board.min_voting,
            board_max_voting=board.max_voting,
            board_active_count=board.active_count,
            source_url=BCC_APPOINTMENTS_TEMPLATE.format(target=board.target),
        )
        result.members.append(member)
        board.has_roster = True

    return result


# ---------------------------------------------------------------------------
# Fetch + parse (network, but PURE of Azure — verifiable without Table access)
# ---------------------------------------------------------------------------
def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": scraper.BROWSER_USER_AGENT})
    return session


def fetch_boards(session: requests.Session | None = None) -> BoardsFetchResult:
    """Fetch + parse the BCC cache. NETWORK ONLY — never touches Azure.

    One GET of the index page returns the whole client-side cache. On any
    fetch/parse failure, returns an empty result with ``fetch_ok=False`` so the
    caller skips reconciliation (and never "removes" rows it couldn't read).
    """
    session = session or _new_session()
    try:
        response = session.get(BCC_INDEX_URL, timeout=60)
        response.raise_for_status()
        data = extract_data_blob(response.text)
        cache = data.get("cache") or {}
        return parse_boards_cache(cache)
    except Exception as exc:  # noqa: BLE001 - a bad sweep must not reconcile
        logger.warning("Boards: failed to fetch/parse the BCC cache: %s", exc)
        return BoardsFetchResult(fetch_ok=False)


# ---------------------------------------------------------------------------
# Table Storage
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _table_client():
    """TableClient for the boards table (AAD auth, created if missing).

    Mirrors :func:`directory_source._table_client`: ``TableServiceClient`` with
    ``DefaultAzureCredential`` against ``https://{account}.table.core.windows.net``
    — managed identity in Azure, developer creds locally, never an account key.
    ``create_table_if_not_exists`` keeps the pipeline runnable on a fresh account
    even though Bicep also declares the table.
    """
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    endpoint = f"https://{account}.table.core.windows.net"
    service = TableServiceClient(endpoint=endpoint, credential=DefaultAzureCredential())
    return service.create_table_if_not_exists(BOARDS_TABLE)


# Fields compared to decide whether a stored row actually changed (everything
# except the always-moving ingested_at_utc). Union of BOTH row types' fields; an
# absent field reads as None on both sides, so the comparison stays correct.
_COMPARE_KEYS = (
    "row_type",
    "board",
    "board_target",
    "full_name",
    "position_type",
    "representation",
    "expiration_iso",
    "is_voting",
    "board_min_voting",
    "board_max_voting",
    "board_active_count",
    "has_roster",
    "status",
    "source_url",
)


def _put(entity: dict, key: str, value: Any) -> None:
    """Set a Table field, OMITTING None (absent == None, like directory_source)."""
    if value is not None:
        entity[key] = value


def _person_entity(member: BoardMember, now_utc: datetime, status: str = STATUS_ACTIVE) -> dict:
    """Project a BoardMember onto a PERSON Table entity (RK = rid)."""
    entity: dict = {
        "PartitionKey": member.board,
        "RowKey": str(member.rid),
        "row_type": "person",
        "board": member.board,
        "board_target": member.board_target,
        "full_name": member.full_name,
        "position_type": member.position_type,
        "is_voting": member.is_voting,
        "board_active_count": member.board_active_count,
        "status": status,
        "source_url": member.source_url,
        "ingested_at_utc": now_utc,
    }
    _put(entity, "representation", member.representation)
    _put(entity, "expiration_iso", member.expiration_iso)
    _put(entity, "board_min_voting", member.board_min_voting)
    _put(entity, "board_max_voting", member.board_max_voting)
    return entity


def _board_entity(board: Board, now_utc: datetime, status: str = STATUS_ACTIVE) -> dict:
    """Project a Board onto a BOARD Table entity (RK = ``__board__``)."""
    entity: dict = {
        "PartitionKey": board.name,
        "RowKey": BOARD_ROW_KEY,
        "row_type": "board",
        "board": board.name,
        "board_active_count": board.active_count,
        "has_roster": board.has_roster,
        "status": status,
        # No appointments page for a targetless board — cite the BCC index instead.
        "source_url": (
            BCC_APPOINTMENTS_TEMPLATE.format(target=board.target)
            if board.target is not None
            else BCC_INDEX_URL
        ),
        "ingested_at_utc": now_utc,
    }
    _put(entity, "board_target", board.target)
    _put(entity, "board_min_voting", board.min_voting)
    _put(entity, "board_max_voting", board.max_voting)
    return entity


def _normalized(entity: dict) -> dict:
    """Comparable view of an entity over the change-detection keys."""
    return {key: entity.get(key) for key in _COMPARE_KEYS}


def _upsert(table, entity: dict) -> str:
    """Idempotently upsert one entity keyed on (PartitionKey, RowKey).

    Returns "inserted", "updated", or "unchanged". Unchanged rows are left
    untouched (not even ingested_at_utc) so a second sweep is a verifiable no-op —
    exactly like :func:`directory_source._upsert`.
    """
    pk, rk = entity["PartitionKey"], entity["RowKey"]
    try:
        existing = table.get_entity(pk, rk)
    except ResourceNotFoundError:
        existing = None

    if existing is None:
        try:
            table.create_entity(entity)
        except ResourceExistsError:
            table.update_entity(entity, mode=UpdateMode.REPLACE)
            return "updated"
        return "inserted"

    if _normalized(entity) != _normalized(existing):
        table.update_entity(entity, mode=UpdateMode.REPLACE)
        return "updated"
    return "unchanged"


def _reconcile(table, present_keys: set[tuple[str, str]], now_utc: datetime) -> int:
    """Flag stored rows absent from this sweep as ``removed`` (never hard-delete).

    Every stored row not already ``removed`` whose (PartitionKey, RowKey) is NOT
    in ``present_keys`` has left the source — mark it ``removed``. Returns the
    count flagged. Called ONLY after a SUCCESSFUL fetch (one GET = whole cache, so
    the present set is authoritative for the entire table).
    """
    removed = 0
    query = "status ne @removed"
    parameters = {"removed": STATUS_REMOVED}
    for entity in table.query_entities(query, parameters=parameters):
        key = (entity["PartitionKey"], entity["RowKey"])
        if key in present_keys:
            continue
        entity["status"] = STATUS_REMOVED
        entity["ingested_at_utc"] = now_utc
        table.update_entity(entity, mode=UpdateMode.MERGE)
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_and_upsert_boards() -> BoardsSweepResult:
    """Sweep the BCC cache and upsert board + person rows into Table Storage.

    Fetch + parse is delegated to :func:`fetch_boards` (Azure-free). Each board
    emits one BOARD row; each surviving appointment one PERSON row. Rows are
    idempotently upserted; then — ONLY if the fetch succeeded — reconciliation
    marks any stored row absent from this sweep ``removed``. A failed fetch upserts
    nothing and skips reconcile, so a transient error can't wipe the table.
    """
    now_utc = datetime.now(timezone.utc)
    fetched = fetch_boards()

    result = BoardsSweepResult()
    result.dropped_test = len(fetched.dropped_test)
    result.dropped_orphans = len(fetched.dropped_orphans)

    if not fetched.fetch_ok:
        logger.warning("Boards: fetch failed; no upsert, no reconcile this run.")
        result.failures += 1
        return result

    table = _table_client()
    present_keys: set[tuple[str, str]] = set()

    # BOARD rows (every board, including the no-roster ones).
    for board in fetched.boards:
        entity = _board_entity(board, now_utc)
        present_keys.add((entity["PartitionKey"], entity["RowKey"]))
        try:
            outcome = _upsert(table, entity)
        except Exception as exc:  # noqa: BLE001 - isolate a bad write
            logger.warning("Boards: failed to upsert board %r: %s", board.name, exc)
            result.failures += 1
            continue
        setattr(result, outcome, getattr(result, outcome) + 1)
    result.boards = len(fetched.boards)

    # PERSON rows.
    for member in fetched.members:
        result.members += 1
        entity = _person_entity(member, now_utc)
        present_keys.add((entity["PartitionKey"], entity["RowKey"]))
        try:
            outcome = _upsert(table, entity)
        except Exception as exc:  # noqa: BLE001 - isolate a bad write
            logger.warning("Boards: failed to upsert rid %s: %s", member.rid, exc)
            result.failures += 1
            continue
        setattr(result, outcome, getattr(result, outcome) + 1)

    # Reconcile the WHOLE table against this sweep (fetch succeeded).
    try:
        result.removed = _reconcile(table, present_keys, now_utc)
    except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
        logger.warning("Boards: reconciliation failed: %s", exc)
        result.failures += 1

    logger.info(
        "Boards sweep: %d board(s), %d member(s), %d inserted, %d updated, "
        "%d unchanged, %d removed, %d dropped-TEST, %d dropped-orphan, %d failure(s)",
        result.boards, result.members, result.inserted, result.updated,
        result.unchanged, result.removed, result.dropped_test,
        result.dropped_orphans, result.failures,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    argparse.ArgumentParser(
        description="Sweep the Gloucester BCC appointments into the boards table."
    ).parse_args()
    print(fetch_and_upsert_boards())
