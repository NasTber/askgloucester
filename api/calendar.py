"""Read-only Azure Table Storage access for the Gloucester meeting calendar.

API-local and deliberately self-contained: this module talks to the ``events``
table DIRECTLY (``TableServiceClient`` over ``DefaultAzureCredential``) and does
NOT import anything from ``ingestion/`` — the deployed API image ships only
``api/`` (see the Dockerfile). The ingestion side
(``ingestion/calendar_source.py``) owns WRITES to the table; this module owns the
read path used by the ``schedule_lookup`` agent tool, plus the presentation
helpers that turn rows into resident-facing text.

Auth is ``DefaultAzureCredential`` (managed identity in Azure, developer creds
locally) — no account keys. The API managed identity needs the **Storage Table
Data Reader** role on the account (see ``infra/modules/storage.bicep``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential

# Dedicated table holding normalized calendar events (written by ingestion).
EVENTS_TABLE = os.environ.get("EVENTS_TABLE_NAME", "events")

# All event times are stored in UTC; the public sees Gloucester-local time.
EASTERN = ZoneInfo("America/New_York")

# Default look-ahead for a dateless query, so "next meeting" works without dates.
DEFAULT_WINDOW_DAYS = 60

# Status values stored in the table (mirror ingestion/calendar_source.py).
STATUS_SCHEDULED = "scheduled"
STATUS_CANCELLED = "cancelled"
STATUS_RESCHEDULED = "rescheduled"
STATUS_REVISED = "revised"
STATUS_UNCERTAIN = "uncertain"
STATUS_REMOVED = "removed"

# The bodies the city calendar covers (the table's PartitionKey values). This is
# the READ-side roster used to normalize/decline a body name for schedule_lookup.
#
# SOURCE OF TRUTH: ingestion/calendar_source.py:CID_BODY (the WRITE side, keyed by
# feed CID). Keep these in sync — if a CID is added/removed/renamed there, mirror
# the canonical body name here.
CALENDAR_BODIES: tuple[str, ...] = (
    "City Council",
    "School Committee",
    "Conservation Commission",
    "Planning Board",
    "Zoning Board of Appeals",
    "Council on Aging",
    "Board of Assessors",
    "Affordable Housing Trust",
    "Board of Registrars",
    "Community Preservation",
    "Fisheries Commission",
    "Licensing Board",
    "City-Owned Cemeteries Advisory Committee",
)
_CANONICAL_BODIES = {body.lower(): body for body in CALENDAR_BODIES}


@dataclass
class CalendarEvent:
    """One calendar event as read back from the table (the tool's working shape)."""

    eid: int
    meeting_body: str
    title: str
    start_utc: datetime
    end_utc: datetime | None
    all_day: bool
    location: str | None
    status: str
    status_note: str | None
    source_url: str


def normalize_body(body: str | None) -> str | None:
    """Snap an LLM-provided body name to a canonical roster constant, or None.

    Normalization only (not intent detection): the agent decides whether a body
    applies; here we just map its (possibly differently-cased) string to a name
    on :data:`CALENDAR_BODIES`. An unrecognized non-empty string returns None,
    which the caller treats as "not on the calendar roster" and declines.
    """
    if not body or not body.strip():
        return None
    return _CANONICAL_BODIES.get(body.strip().lower())


def _parse_iso(value: str | None) -> date | None:
    """Parse a YYYY-MM-DD string to a date, or None if absent/malformed."""
    if not value or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def window_from_dates(
    start_date: str | None, end_date: str | None
) -> tuple[datetime, datetime]:
    """Build a UTC ``[start, end]`` window from optional YYYY-MM-DD strings.

    Dates are interpreted as Gloucester-local (Eastern) day boundaries, then
    converted to UTC for the table query. Defaults make a dateless call cover
    upcoming meetings: start = today (Eastern), end = today + DEFAULT_WINDOW_DAYS.
    """
    today_local = datetime.now(EASTERN).date()
    start = _parse_iso(start_date) or today_local
    end = _parse_iso(end_date) or (today_local + timedelta(days=DEFAULT_WINDOW_DAYS))
    start_utc = datetime(start.year, start.month, start.day, 0, 0, tzinfo=EASTERN).astimezone(
        timezone.utc
    )
    end_utc = datetime(
        end.year, end.month, end.day, 23, 59, 59, tzinfo=EASTERN
    ).astimezone(timezone.utc)
    return start_utc, end_utc


@lru_cache(maxsize=1)
def _table_client():
    """TableClient for the events table (AAD auth, read-only use).

    ``TableServiceClient`` authenticates with ``DefaultAzureCredential`` against
    ``https://{account}.table.core.windows.net``. We only ever read here, so the
    table is assumed to already exist (created by ingestion); no create call.
    """
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    endpoint = f"https://{account}.table.core.windows.net"
    service = TableServiceClient(endpoint=endpoint, credential=DefaultAzureCredential())
    return service.get_table_client(EVENTS_TABLE)


def _ensure_utc(value: datetime) -> datetime:
    """Coerce a datetime to tz-aware UTC (a naive value is assumed already UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_event(entity) -> CalendarEvent:
    """Rebuild a :class:`CalendarEvent` from a stored table entity."""

    def to_utc(value):
        return value.astimezone(timezone.utc) if isinstance(value, datetime) else value

    return CalendarEvent(
        eid=int(entity["eid"]),
        meeting_body=entity["PartitionKey"],
        title=entity.get("title", ""),
        start_utc=to_utc(entity.get("start_utc")),
        end_utc=to_utc(entity["end_utc"]) if entity.get("end_utc") else None,
        all_day=bool(entity.get("all_day")),
        location=entity.get("location"),
        status=entity.get("status", STATUS_SCHEDULED),
        status_note=entity.get("status_note"),
        source_url=entity.get("source_url", ""),
    )


def get_events(
    meeting_body: str | None, start: datetime, end: datetime
) -> list[CalendarEvent]:
    """Return events whose ``start_utc`` falls in ``[start, end]``, optionally one body.

    Read-only port of ingestion's ``get_events``: a ``PartitionKey`` filter (when
    a body is given) plus a ``start_utc`` range. Events come back sorted by
    ``start_utc`` with their status included and NOT filtered — the tool layer
    decides what to show.
    """
    table = _table_client()
    start_utc, end_utc = _ensure_utc(start), _ensure_utc(end)

    conditions = ["start_utc ge @start", "start_utc le @end"]
    parameters: dict = {"start": start_utc, "end": end_utc}
    if meeting_body:
        conditions.insert(0, "PartitionKey eq @pk")
        parameters["pk"] = meeting_body

    entities = table.query_entities(" and ".join(conditions), parameters=parameters)
    events = [_to_event(e) for e in entities]
    events.sort(key=lambda e: e.start_utc)
    return events


# --- Presentation -----------------------------------------------------------
def _format_when(event: CalendarEvent) -> str:
    """Human date/time in America/New_York.

    All-day -> date only (no clock time); a null end -> start only; a same-day
    end -> just the end time; a multi-day end -> the full end date+time.
    """
    start_local = event.start_utc.astimezone(EASTERN)
    if event.all_day:
        return f"{start_local:%A, %B %-d, %Y} (all day)"
    when = f"{start_local:%A, %B %-d, %Y at %-I:%M %p}"
    if event.end_utc is not None:
        end_local = event.end_utc.astimezone(EASTERN)
        if end_local.date() == start_local.date():
            when += f"–{end_local:%-I:%M %p}"
        else:
            when += f" to {end_local:%A, %B %-d, %Y at %-I:%M %p}"
    return when


def render_events(
    body: str | None,
    events: list[CalendarEvent],
    start: datetime,
    end: datetime,
) -> str:
    """Render events as plain text for the agent, applying the presentation rules.

    Rules: drop ``cancelled`` and ``removed`` entirely (keyed on the status
    field); surface ``revised``/``rescheduled`` with their note; flag
    ``uncertain`` as unconfirmed (never as a confirmed meeting) with a verify
    caution; include each event's ``source_url`` inline.
    """
    window = (
        f"{start.astimezone(EASTERN):%B %-d, %Y} to "
        f"{end.astimezone(EASTERN):%B %-d, %Y}"
    )
    scope = body or "all bodies on the city calendar"

    # Exclude cancelled and removed entirely (status-keyed; accuracy depends on
    # the ingestion status parser, which is a separate launch gate).
    visible = [e for e in events if e.status not in (STATUS_CANCELLED, STATUS_REMOVED)]
    if not visible:
        return f"No scheduled meetings found for {scope} ({window})."

    lines = [f"Meetings for {scope} ({window}):", ""]
    for event in visible:
        lines.append(f"- {event.meeting_body}: {event.title or 'Meeting'} — {_format_when(event)}")
        if event.location:
            lines.append(f"  Location: {event.location}")
        if event.status == STATUS_UNCERTAIN:
            caution = "  Status unconfirmed — verify on the city calendar"
            if event.status_note:
                caution += f" (feed note: {event.status_note})"
            lines.append(caution)
        elif event.status in (STATUS_REVISED, STATUS_RESCHEDULED):
            note = f"  Note: this meeting was {event.status}"
            if event.status_note:
                note += f" ({event.status_note})"
            lines.append(note)
        lines.append(f"  Details: {event.source_url}")
    return "\n".join(lines)
