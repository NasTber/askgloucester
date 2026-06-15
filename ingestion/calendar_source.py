"""Ingest Gloucester public-meeting calendar events into Azure Table Storage.

This is a SEPARATE, structured source. Unlike :mod:`scraper` / :mod:`drive_source`,
it does **not** touch blob storage, Document Intelligence, chunking, embedding, or
the Azure AI Search index. It reads the city's machine-readable iCalendar feeds and
upserts one normalized row per event into a dedicated ``events`` table, so a future
"when does X meet" tool can answer from structured data instead of RAG.

Feed
----
Per-category iCalendar only (RSS is ignored — it carries no category and disregards
the CID filter). One feed per category id (CID)::

    https://www.gloucester-ma.gov/common/modules/iCalendar/iCalendar.aspx?feed=calendar&catID=<CID>

Because we fetch *per CID*, the CID we requested IS the body — there is no
title-based body guessing. :data:`CID_BODY` is the allowlist of CIDs we ingest;
everything else (holidays, special events, unknown categories) is excluded by
omission.

Status (correctness-critical)
-----------------------------
The feed exposes no STATUS property, so a meeting's status is **derived and
fail-safe**, never blindly "scheduled":

  * token parsing of SUMMARY+DESCRIPTION (CANCEL / RESCHEDUL / POSTPON / REVISED:),
    with any *cancel-ish but unrecognized* token mapped to ``uncertain``; and
  * **reconciliation** — on each sweep, a future-dated event we already stored that
    is now ABSENT from its feed was pulled, so its status is set to ``removed``
    (never hard-deleted). This closes the gap where a quietly-cancelled meeting
    would otherwise linger as ``scheduled``.

Auth is ``DefaultAzureCredential`` end to end (Table endpoint
``https://{account}.table.core.windows.net``) — no account keys.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Iterable
from zoneinfo import ZoneInfo

import requests
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from icalendar import Calendar

# Reuse the scraper's browser User-Agent and — critically — its canonical
# meeting_body strings, so calendar events join the document index on identical
# body values without retyping (and risking divergent casing/spelling).
import scraper

load_dotenv()

logger = logging.getLogger(__name__)

# Base host for the calendar feeds and per-event permalinks.
CALENDAR_BASE_URL = os.environ.get("CALENDAR_BASE_URL", "https://www.gloucester-ma.gov")
ICAL_FEED_TEMPLATE = (
    CALENDAR_BASE_URL
    + "/common/modules/iCalendar/iCalendar.aspx?feed=calendar&catID={cid}"
)

# Dedicated table for structured events. Created on first use.
EVENTS_TABLE = os.environ.get("EVENTS_TABLE_NAME", "events")

# All DTSTART/DTEND values are local Gloucester time. The feed tags them
# TZID=America/New_York; LAST-MODIFIED is tagged too but icalendar hands it back
# naive, so we localize anything naive to this zone before converting to UTC.
EASTERN = ZoneInfo("America/New_York")


def _canonical_body(name: str) -> str:
    """Return the exact meeting_body string the document index uses, if this is
    one of the indexed bodies; otherwise the name unchanged.

    ``scraper.ARCHIVE_SOURCES`` is the single source of truth for the two indexed
    bodies ("City Council", "School Committee"). Snapping through it means calendar
    rows and document chunks share byte-identical meeting_body values even if this
    map is later typed with different casing.
    """
    for body, _document_type in scraper.ARCHIVE_SOURCES.values():
        if body.lower() == name.lower():
            return body
    return name


# Allowlist: category id (CID) -> canonical meeting_body. We fetch each feed by
# CID, so the CID alone determines the body. The first two map to the existing
# indexed-document bodies (via _canonical_body); the rest are calendar-only
# bodies that have no documents indexed.
#
# WRITE-SIDE SOURCE OF TRUTH for the calendar body roster. The deployed API reads
# this table but cannot import ingestion/, so it mirrors these canonical body
# names in api/calendar.py:CALENDAR_BODIES — keep the two in sync when a CID is
# added/removed/renamed here.
#
# Excluded by omission: 14 (holidays), 20 (special events).
CID_BODY: dict[int, str] = {
    23: _canonical_body("City Council"),
    65: _canonical_body("School Committee"),
    48: "Conservation Commission",
    58: "Planning Board",
    37: "Zoning Board of Appeals",
    24: "Council on Aging",
    25: "Board of Assessors",
    38: "Affordable Housing Trust",
    41: "Board of Registrars",
    47: "Community Preservation",
    51: "Fisheries Commission",
    55: "Licensing Board",
    81: "City-Owned Cemeteries Advisory Committee",
}

# --- Status enum (string values stored in the table) ------------------------
STATUS_SCHEDULED = "scheduled"
STATUS_CANCELLED = "cancelled"
STATUS_RESCHEDULED = "rescheduled"
STATUS_REVISED = "revised"
STATUS_UNCERTAIN = "uncertain"  # cancel-ish but unrecognized — never "scheduled"
STATUS_REMOVED = "removed"  # was in the table, now absent from the feed

# Cancel-ish hints that are NOT one of the recognized tokens above. Their presence
# forces "uncertain" rather than "scheduled", so an oddly-worded cancellation can
# never silently read as a live meeting.
# TODO(before launch): confirm the EXACT cancelled token against a real cancelled
# event in a live feed; the tokens below are conservative guesses.
AMBIGUOUS_CANCEL_TOKENS = (
    "CXL",
    "CANC",  # e.g. "CANC." — a real "CANCEL(LED)" is caught earlier and wins
    "NO MEETING",
    "NOT MEET",
    "WILL NOT MEET",
    "MEETING OFF",
    "NO SESSION",
)

# Leading clock time in a SUMMARY, e.g. "6:30 PM City Council Meeting".
_LEADING_TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*[AaPp][Mm]\s*")
# An EID embedded in a UID or DESCRIPTION URL, used as a fallback id source.
_EID_RE = re.compile(r"EID=(\d+)", re.IGNORECASE)


@dataclass
class Event:
    """One normalized calendar event (the table row, and the read-helper return)."""

    eid: int
    meeting_body: str
    raw_category_cid: int
    title: str
    start_utc: datetime
    end_utc: datetime | None
    all_day: bool
    location: str | None
    status: str
    status_note: str | None
    source_url: str
    last_modified_utc: datetime | None
    ingested_at_utc: datetime


@dataclass
class CalendarSweepResult:
    """Counters returned by :func:`ingest_calendar_events` for reporting."""

    feeds: int = 0
    parsed: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failures: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _as_utc(value: date | datetime) -> datetime:
    """Convert an icalendar date/datetime to a tz-aware UTC datetime.

    A naive datetime (the feed's LAST-MODIFIED comes back naive despite its TZID)
    is assumed to be America/New_York. A bare ``date`` becomes midnight Eastern.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=EASTERN)
        return value.astimezone(timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=EASTERN).astimezone(
        timezone.utc
    )


def _local(value: date | datetime) -> datetime | None:
    """Return the value as a local (Eastern) datetime for time-of-day heuristics,
    or None for a date-only value (no clock time to inspect)."""
    if isinstance(value, datetime):
        return value if value.tzinfo is None else value.astimezone(EASTERN)
    return None


def _strip_leading_time(summary: str) -> str:
    """Drop a leading clock time from a SUMMARY ("6:30 PM City Council Meeting")."""
    return _LEADING_TIME_RE.sub("", summary).strip()


def _extract_eid(uid: str, description: str) -> int | None:
    """The event id is the integer UID; fall back to an EID= in the description."""
    if uid.isdigit():
        return int(uid)
    match = _EID_RE.search(f"{uid} {description}")
    return int(match.group(1)) if match else None


def _derive_status(summary: str, description: str) -> str:
    """Fail-safe status from SUMMARY+DESCRIPTION tokens (never blindly scheduled).

    Cancellation wins over everything; an unrecognized cancel-ish token yields
    ``uncertain`` rather than ``scheduled``; "REVISED:" alone yields ``revised``.
    """
    blob = f"{summary} {description}".upper()
    if "CANCEL" in blob:  # CANCELLED / CANCELED / CANCEL
        return STATUS_CANCELLED
    if "RESCHEDUL" in blob or "POSTPON" in blob:
        return STATUS_RESCHEDULED
    if any(token in blob for token in AMBIGUOUS_CANCEL_TOKENS):
        return STATUS_UNCERTAIN
    if "REVISED:" in description.upper():
        return STATUS_REVISED
    return STATUS_SCHEDULED


def _status_note(description: str) -> str | None:
    """Raw posting/revision note from DESCRIPTION, or None.

    DESCRIPTION is normally just the event's calendar.aspx URL; events with a
    posting history carry extra text like "POSTED ... || REVISED: ...". Strip the
    URL(s) and return whatever remains.
    """
    note = re.sub(r"https?://\S+", "", description).strip()
    return note or None


def parse_event(component, cid: int, ingested_at_utc: datetime) -> Event | None:
    """Normalize one icalendar VEVENT into an :class:`Event`, or None to skip.

    ``cid`` is the feed category we fetched, which determines the meeting_body
    (and is recorded as ``raw_category_cid``). Returns None when the event has no
    usable id or no start.
    """
    uid = str(component.get("uid") or "").strip()
    description = str(component.get("description") or "")
    eid = _extract_eid(uid, description)
    if eid is None:
        logger.debug("Calendar: skipping VEVENT with no usable id (uid=%r)", uid)
        return None

    dtstart = component.get("dtstart")
    if dtstart is None:
        logger.debug("Calendar: skipping event %s with no DTSTART", eid)
        return None
    start_value = dtstart.dt
    start_local = _local(start_value)
    start_utc = _as_utc(start_value)
    date_only = isinstance(start_value, date) and not isinstance(start_value, datetime)

    # DTEND of 23:59 is the feed's "no real end time" placeholder.
    dtend = component.get("dtend")
    end_value = dtend.dt if dtend is not None else None
    end_local = _local(end_value) if end_value is not None else None
    end_is_placeholder = (
        end_local is not None and end_local.hour == 23 and end_local.minute == 59
    )
    end_utc = (
        None if (end_value is None or end_is_placeholder) else _as_utc(end_value)
    )

    # all-day: a date-only start, or a midnight start paired with the 23:59 end.
    all_day = date_only or (
        start_local is not None
        and start_local.hour == 0
        and start_local.minute == 0
        and (end_value is None or end_is_placeholder)
    )

    summary = str(component.get("summary") or "").strip()
    location_raw = component.get("location")
    location = str(location_raw).strip() if location_raw else None
    location = location or None

    last_modified = component.get("last-modified")
    last_modified_utc = _as_utc(last_modified.dt) if last_modified is not None else None

    return Event(
        eid=eid,
        meeting_body=CID_BODY[cid],
        raw_category_cid=cid,
        title=_strip_leading_time(summary),
        start_utc=start_utc,
        end_utc=end_utc,
        all_day=all_day,
        location=location,
        status=_derive_status(summary, description),
        status_note=_status_note(description),
        source_url=f"{CALENDAR_BASE_URL}/calendar.aspx?EID={eid}",
        last_modified_utc=last_modified_utc,
        ingested_at_utc=ingested_at_utc,
    )


def _fetch_vevents(session: requests.Session, cid: int) -> list:
    """Fetch one CID's iCalendar feed and return its VEVENT components."""
    url = ICAL_FEED_TEMPLATE.format(cid=cid)
    response = session.get(url, timeout=60)
    response.raise_for_status()
    calendar = Calendar.from_ical(response.text)
    return list(calendar.walk("VEVENT"))


# ---------------------------------------------------------------------------
# Table Storage
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _table_client():
    """TableClient for the events table (AAD auth, created if missing).

    ``TableServiceClient`` authenticates with ``DefaultAzureCredential`` against
    ``https://{account}.table.core.windows.net`` — managed identity in Azure,
    developer credentials locally, never an account key.
    ``create_table_if_not_exists`` makes the pipeline runnable against a fresh
    account.
    """
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    endpoint = f"https://{account}.table.core.windows.net"
    service = TableServiceClient(endpoint=endpoint, credential=DefaultAzureCredential())
    return service.create_table_if_not_exists(EVENTS_TABLE)


# Fields compared to decide whether a stored row actually changed (everything
# except the always-moving ingested_at_utc), so a re-sweep of unchanged data is a
# true no-op rather than a churn of identical writes.
_COMPARE_KEYS = (
    "eid",
    "raw_category_cid",
    "title",
    "start_utc",
    "end_utc",
    "all_day",
    "location",
    "status",
    "status_note",
    "source_url",
    "last_modified_utc",
)


def _entity(event: Event) -> dict:
    """Project an Event onto a Table entity. None values are omitted (= absent)."""
    entity: dict = {
        "PartitionKey": event.meeting_body,
        "RowKey": str(event.eid),
        "eid": event.eid,
        "meeting_body": event.meeting_body,
        "raw_category_cid": event.raw_category_cid,
        "title": event.title,
        "start_utc": event.start_utc,
        "all_day": event.all_day,
        "status": event.status,
        "source_url": event.source_url,
        "ingested_at_utc": event.ingested_at_utc,
    }
    if event.end_utc is not None:
        entity["end_utc"] = event.end_utc
    if event.location:
        entity["location"] = event.location
    if event.status_note:
        entity["status_note"] = event.status_note
    if event.last_modified_utc is not None:
        entity["last_modified_utc"] = event.last_modified_utc
    return entity


def _normalized(entity: dict) -> dict:
    """Comparable view of an entity: datetimes to second-precision UTC ISO strings.

    Table Storage round-trips datetimes at sub-second precision and as tz-aware
    UTC; normalizing both sides avoids false "changed" verdicts from formatting.
    """
    out: dict = {}
    for key in _COMPARE_KEYS:
        value = entity.get(key)
        if isinstance(value, datetime):
            value = value.astimezone(timezone.utc).replace(microsecond=0).isoformat()
        out[key] = value
    return out


def _upsert(table, event: Event) -> str:
    """Idempotently upsert one event keyed on (meeting_body, eid).

    Returns "inserted", "updated", or "unchanged". Unchanged rows are left
    untouched (not even ingested_at_utc) so a second sweep is a verifiable no-op.
    """
    entity = _entity(event)
    try:
        existing = table.get_entity(event.meeting_body, str(event.eid))
    except ResourceNotFoundError:
        existing = None

    if existing is None:
        try:
            table.create_entity(entity)
        except ResourceExistsError:
            # Lost a race; fall through to an update so the sweep stays idempotent.
            table.update_entity(entity, mode=UpdateMode.REPLACE)
            return "updated"
        return "inserted"

    if _normalized(entity) != _normalized(existing):
        table.update_entity(entity, mode=UpdateMode.REPLACE)
        return "updated"
    return "unchanged"


def _reconcile(table, meeting_body: str, present_eids: set[int], now_utc: datetime) -> int:
    """Flag future events that vanished from the feed as ``removed`` (not deleted).

    For ``meeting_body``, every stored event dated in the future and not already
    ``removed`` whose eid is absent from the current feed was pulled — mark it
    ``removed``. Past events are ignored: they age out of the feed naturally.
    Returns the number of events flagged.
    """
    removed = 0
    query = "PartitionKey eq @pk and start_utc gt @now and status ne @removed"
    parameters = {"pk": meeting_body, "now": now_utc, "removed": STATUS_REMOVED}
    for entity in table.query_entities(query, parameters=parameters):
        try:
            eid = int(entity["eid"])
        except (KeyError, TypeError, ValueError):
            continue
        if eid in present_eids:
            continue
        entity["status"] = STATUS_REMOVED
        entity["ingested_at_utc"] = now_utc
        table.update_entity(entity, mode=UpdateMode.MERGE)
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ingest_calendar_events(cids: Iterable[int] | None = None) -> CalendarSweepResult:
    """Sweep the allowlisted calendar feeds and upsert events into Table Storage.

    For each CID: fetch its iCalendar feed, normalize every VEVENT, and
    idempotently upsert it. After all feeds, reconcile per body so future events
    missing from a feed are flagged ``removed``. A failure on one feed (or one
    event) is logged and skipped — it never aborts the sweep.

    Args:
        cids: CIDs to sweep. Defaults to every key in :data:`CID_BODY`.

    Returns:
        A :class:`CalendarSweepResult` with per-outcome counters.
    """
    cids = list(cids) if cids is not None else list(CID_BODY)
    now_utc = datetime.now(timezone.utc)
    table = _table_client()

    session = requests.Session()
    session.headers.update({"User-Agent": scraper.BROWSER_USER_AGENT})

    result = CalendarSweepResult()
    present_by_body: dict[str, set[int]] = {}

    for cid in cids:
        body = CID_BODY.get(cid)
        if body is None:
            logger.warning("Calendar: skipping CID %s (not in allowlist)", cid)
            continue

        try:
            components = _fetch_vevents(session, cid)
        except Exception as exc:  # noqa: BLE001 - isolate a bad feed
            logger.warning("Calendar: failed to fetch CID %s (%s): %s", cid, body, exc)
            result.failures += 1
            continue
        result.feeds += 1

        for component in components:
            try:
                event = parse_event(component, cid, now_utc)
            except Exception as exc:  # noqa: BLE001 - isolate a bad event
                logger.warning("Calendar: failed to parse a VEVENT in CID %s: %s", cid, exc)
                result.failures += 1
                continue
            if event is None:
                continue

            result.parsed += 1
            present_by_body.setdefault(body, set()).add(event.eid)
            result.status_counts[event.status] = (
                result.status_counts.get(event.status, 0) + 1
            )

            try:
                outcome = _upsert(table, event)
            except Exception as exc:  # noqa: BLE001 - isolate a bad write
                logger.warning("Calendar: failed to upsert event %s: %s", event.eid, exc)
                result.failures += 1
                continue
            setattr(result, outcome, getattr(result, outcome) + 1)

    # Reconcile only the bodies we actually swept (so a skipped feed can't wrongly
    # "remove" that body's events).
    for body, present in present_by_body.items():
        try:
            result.removed += _reconcile(table, body, present, now_utc)
        except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
            logger.warning("Calendar: reconciliation failed for %s: %s", body, exc)
            result.failures += 1

    logger.info(
        "Calendar sweep: %d feed(s), %d parsed, %d inserted, %d updated, "
        "%d unchanged, %d removed, %d failure(s); status=%s",
        result.feeds, result.parsed, result.inserted, result.updated,
        result.unchanged, result.removed, result.failures, result.status_counts,
    )
    return result


def _ensure_utc(value: datetime) -> datetime:
    """Coerce a datetime to tz-aware UTC (a naive value is assumed already UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _entity_to_event(entity: dict) -> Event:
    """Rebuild an :class:`Event` from a stored table entity."""

    def to_utc(value):
        return value.astimezone(timezone.utc) if isinstance(value, datetime) else value

    return Event(
        eid=int(entity["eid"]),
        meeting_body=entity["PartitionKey"],
        raw_category_cid=int(entity.get("raw_category_cid", 0)),
        title=entity.get("title", ""),
        start_utc=to_utc(entity.get("start_utc")),
        end_utc=to_utc(entity["end_utc"]) if entity.get("end_utc") else None,
        all_day=bool(entity.get("all_day")),
        location=entity.get("location"),
        status=entity.get("status", STATUS_SCHEDULED),
        status_note=entity.get("status_note"),
        source_url=entity.get("source_url", ""),
        last_modified_utc=(
            to_utc(entity["last_modified_utc"]) if entity.get("last_modified_utc") else None
        ),
        ingested_at_utc=to_utc(entity.get("ingested_at_utc")),
    )


def get_events(
    meeting_body: str | None, start: datetime, end: datetime
) -> list[Event]:
    """Return events whose start falls in ``[start, end]``, optionally one body.

    Minimal read helper for the future calendar tool. Filters on PartitionKey
    (when ``meeting_body`` is given) and a ``start_utc`` range. Events are returned
    WITH their status and WITHOUT any status filtering — the tool layer decides how
    to treat non-``scheduled`` events (cancelled / rescheduled / removed / etc.).
    """
    table = _table_client()
    start_utc, end_utc = _ensure_utc(start), _ensure_utc(end)

    conditions = ["start_utc ge @start", "start_utc le @end"]
    parameters: dict = {"start": start_utc, "end": end_utc}
    if meeting_body:
        conditions.insert(0, "PartitionKey eq @pk")
        parameters["pk"] = meeting_body

    entities = table.query_entities(" and ".join(conditions), parameters=parameters)
    events = [_entity_to_event(e) for e in entities]
    events.sort(key=lambda e: e.start_utc)
    return events


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Sweep Gloucester calendar feeds into the events table."
    )
    parser.add_argument(
        "--cid",
        type=int,
        action="append",
        dest="cids",
        help="Category id to sweep (repeatable). Defaults to the full allowlist.",
    )
    args = parser.parse_args()
    print(ingest_calendar_events(cids=args.cids))
