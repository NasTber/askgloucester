"""Ingest Gloucester's city staff directory into Azure Table Storage.

This is a SEPARATE, structured source — a sibling of :mod:`calendar_source`.
Like the calendar, it does **not** touch blob storage, Document Intelligence,
chunking, embedding, or the Azure AI Search index. It scrapes the city's
CivicEngage staff directory and upserts one normalized row per (department,
person) into a dedicated ``officials`` table, so a future "who runs X / how do
I reach department Y" tool can answer from structured data instead of RAG.

Source (confirmed by probing — no auth, no special headers needed)
-------------------------------------------------------------------
Two HTML endpoints on ``https://www.gloucester-ma.gov``:

  * **Department index** — ``/Directory.aspx``. Department links are
    ``<a href="Directory.aspx?did=N">Department Name</a>``. We use this ONLY to
    enumerate the (DID) list (~31 departments); the canonical department name is
    taken from each listing page's header, not from here.
  * **Department listing** — ``/Directory/Home/DepartmentListing?DID=N``. A
    jQuery-Mobile HTML page. The roster lives in ``<ul id="contentarea">``:
      - the FIRST ``<li>`` is the department block (not a person): a
        ``Physical Address:`` label, a ``Phone:`` label, and a "Contact Us"
        link ``<a href="/forms.aspx?fid=N">``;
      - each subsequent ``<li>`` is one person, an
        ``<a href="/Directory/Home/SingleStaff?EID=N">`` wrapping
        ``<div class="staffName"><strong>Name</strong><br/><em>Title</em></div>``.

We deliberately do NOT fetch the per-person ``SingleStaff`` pages (contact
enrichment is out of scope for step 1); ``source_url`` is constructed from the
EID instead.

Reconciliation (mirrors :mod:`calendar_source`)
-----------------------------------------------
On each sweep, a stored official whose row is now ABSENT from its department's
freshly-fetched roster is marked ``status=removed`` (never hard-deleted).
Crucially this is scoped to departments fetched **successfully** this run: a
department whose fetch failed is logged and skipped, and its people are left
untouched (so a transient HTTP error can't wrongly "remove" a whole department).

A person can appear under two departments (EID is global). Because the row key
is ``(department, eid)``, that naturally yields two rows — intended, not deduped.

Auth is ``DefaultAzureCredential`` end to end (Table endpoint
``https://{account}.table.core.windows.net``) — no account keys.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from html.parser import HTMLParser
from typing import Iterable

import requests
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Reuse the scraper's browser User-Agent so the directory requests look like a
# real browser (some CivicPlus endpoints reject a bare client).
import scraper

load_dotenv()

logger = logging.getLogger(__name__)

# Base host for the directory pages and per-person permalinks.
DIRECTORY_BASE_URL = os.environ.get("DIRECTORY_BASE_URL", "https://www.gloucester-ma.gov")
DEPARTMENT_INDEX_URL = DIRECTORY_BASE_URL + "/Directory.aspx"
DEPARTMENT_LISTING_TEMPLATE = DIRECTORY_BASE_URL + "/Directory/Home/DepartmentListing?DID={did}"
SINGLE_STAFF_TEMPLATE = DIRECTORY_BASE_URL + "/Directory/Home/SingleStaff?EID={eid}"

# Dedicated table for the structured staff directory. Created on first use.
OFFICIALS_TABLE = os.environ.get("OFFICIALS_TABLE_NAME", "officials")

# --- Status enum (string values stored in the table) ------------------------
STATUS_ACTIVE = "active"
STATUS_REMOVED = "removed"  # was in the directory, now absent from its listing

# Department-index links: <a href="Directory.aspx?did=N">. Case-insensitive on
# the param name so "DID=" works too.
_DEPT_LINK_RE = re.compile(r"Directory\.aspx\?did=(\d+)", re.I)
# Per-person links inside a listing: .../SingleStaff?EID=N.
_SINGLE_STAFF_RE = re.compile(r"SingleStaff\?EID=(\d+)", re.I)
# Canonical department name from the listing header.
_PAGE_HEADER_RE = re.compile(r'<h2 id="pageHeader">(.*?)</h2>', re.I | re.S)
# Department-block contact fields (the first <li> of the roster).
_PHONE_RE = re.compile(r"<strong>\s*Phone:\s*</strong>\s*([^<]*)", re.I)
_ADDRESS_RE = re.compile(
    r"<strong>\s*Physical Address:\s*</strong>(.*?)(?:<strong>|<a\s|</li>)",
    re.I | re.S,
)
# The department-level "Contact Us" form link (NOT a person). The first
# /forms.aspx?fid=N in a listing is the department block's Contact Us link.
_CONTACT_FORM_RE = re.compile(r"/forms\.aspx\?fid=\d+", re.I)


@dataclass
class Official:
    """One parsed person row (the parse output; status/timestamp added at write)."""

    eid: int
    name: str
    title: str
    department: str
    did: int
    department_phone: str | None
    department_address: str | None
    contact_form_url: str | None
    source_url: str


@dataclass
class Department:
    """A successfully-fetched department: its contact block + its roster."""

    did: int
    name: str
    phone: str | None
    address: str | None
    contact_form_url: str | None
    officials: list[Official]


@dataclass
class DirectoryFetchResult:
    """Pure (Azure-free) fetch+parse output for one sweep."""

    departments: list[Department] = field(default_factory=list)
    officials: list[Official] = field(default_factory=list)
    failed_dids: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class DirectorySweepResult:
    """Counters returned by :func:`ingest_officials` for reporting."""

    departments: int = 0
    parsed: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failures: int = 0


# ---------------------------------------------------------------------------
# Parsing (PURE — no network, no Azure)
# ---------------------------------------------------------------------------
def _clean_text(value: str) -> str:
    """Unescape HTML entities and collapse runs of whitespace to single spaces."""
    return " ".join(html.unescape(value).split())


def _clean_address(raw: str) -> str | None:
    """Turn the address sub-HTML into a single comma-joined line.

    The address is a run of text nodes separated by ``<br/>`` (e.g.
    ``<br/>City Hall Annex<br/>3 Pond Road<br/> Gloucester, MA 01930 <br/>``).
    Split on the breaks, strip any stray tags, drop empty pieces, join with ", ".
    """
    parts = re.split(r"<br\s*/?>", raw, flags=re.I)
    cleaned = []
    for part in parts:
        part = re.sub(r"<[^>]+>", " ", part)  # drop any residual tags
        part = _clean_text(part)
        if part:
            cleaned.append(part)
    return ", ".join(cleaned) or None


class _RosterParser(HTMLParser):
    """Collect ``(eid, name, title)`` for every person link in a listing.

    A person is an ``<a href=".../SingleStaff?EID=N">`` wrapping a
    ``<div class="staffName"><strong>Name</strong>…<em>Title</em></div>``. We
    capture the EID from the href and the ``<strong>``/``<em>`` text inside the
    anchor. The department block's own ``<strong>Phone:</strong>`` labels live
    OUTSIDE any SingleStaff anchor, so they are ignored (``_eid is None``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._eid: int | None = None
        self._in_strong = False
        self._in_em = False
        self._name_parts: list[str] = []
        self._title_parts: list[str] = []
        self.people: list[tuple[int, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href") or ""
            match = _SINGLE_STAFF_RE.search(href)
            if match:
                self._eid = int(match.group(1))
                self._name_parts = []
                self._title_parts = []
        elif tag == "strong" and self._eid is not None:
            self._in_strong = True
        elif tag == "em" and self._eid is not None:
            self._in_em = True

    def handle_data(self, data: str) -> None:
        if self._in_strong:
            self._name_parts.append(data)
        elif self._in_em:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "strong":
            self._in_strong = False
        elif tag == "em":
            self._in_em = False
        elif tag == "a" and self._eid is not None:
            name = _clean_text("".join(self._name_parts))
            title = _clean_text("".join(self._title_parts))
            if name:  # skip a stray SingleStaff link with no name text
                self.people.append((self._eid, name, title))
            self._eid = None
            self._name_parts = []
            self._title_parts = []


def parse_department_index(page_html: str) -> list[tuple[int, str]]:
    """Return ``(did, name)`` for every department link on ``/Directory.aspx``.

    Deduplicated by DID (the index can link a department more than once), keeping
    the first non-empty link text as the name.
    """
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for match in re.finditer(
        r'<a[^>]+href="[^"]*Directory\.aspx\?did=(\d+)"[^>]*>(.*?)</a>',
        page_html,
        re.I | re.S,
    ):
        did = int(match.group(1))
        if did in seen:
            continue
        name = _clean_text(re.sub(r"<[^>]+>", " ", match.group(2)))
        seen.add(did)
        out.append((did, name))
    return out


def parse_department_listing(page_html: str, did: int) -> Department:
    """Parse one ``DepartmentListing`` page into a :class:`Department` (PURE).

    The department name is taken from the page's ``<h2 id="pageHeader">`` so the
    PartitionKey comes from a single canonical source. Phone / address /
    contact-form are read from the department block; the roster comes from
    :class:`_RosterParser`.
    """
    header = _PAGE_HEADER_RE.search(page_html)
    name = _clean_text(header.group(1)) if header else ""
    if not name:
        name = f"DID {did}"

    phone_match = _PHONE_RE.search(page_html)
    phone = _clean_text(phone_match.group(1)) if phone_match else None
    phone = phone or None

    address_match = _ADDRESS_RE.search(page_html)
    address = _clean_address(address_match.group(1)) if address_match else None

    contact_match = _CONTACT_FORM_RE.search(page_html)
    contact_form_url = (
        DIRECTORY_BASE_URL + contact_match.group(0) if contact_match else None
    )

    parser = _RosterParser()
    parser.feed(page_html)

    # _RosterParser stays a faithful 1:1 mirror of the markup. Here — where raw
    # rows are finalized into the person records that get upserted — we drop
    # titleless rows: some listings carry a non-person link shaped like a person
    # (e.g. a "Contact Us / SeeClickFix" SingleStaff link) that has a name but no
    # title. Every genuine person has a title, so an empty title is the reliable
    # discriminator. Log each drop (dept, eid, name) so a future titleless-but-real
    # person surfaces in the logs instead of vanishing silently.
    officials: list[Official] = []
    for eid, person_name, title in parser.people:
        if not title:
            logger.warning(
                "Directory: dropping titleless row in %r: eid=%s name=%r",
                name, eid, person_name,
            )
            continue
        officials.append(
            Official(
                eid=eid,
                name=person_name,
                title=title,
                department=name,
                did=did,
                department_phone=phone,
                department_address=address,
                contact_form_url=contact_form_url,
                source_url=SINGLE_STAFF_TEMPLATE.format(eid=eid),
            )
        )
    return Department(
        did=did,
        name=name,
        phone=phone,
        address=address,
        contact_form_url=contact_form_url,
        officials=officials,
    )


# ---------------------------------------------------------------------------
# Fetch + parse (network, but PURE of Azure — verifiable without Table access)
# ---------------------------------------------------------------------------
def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": scraper.BROWSER_USER_AGENT})
    return session


def _fetch(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def fetch_department_dids(session: requests.Session) -> list[tuple[int, str]]:
    """Enumerate ``(did, name)`` from the department index page."""
    return parse_department_index(_fetch(session, DEPARTMENT_INDEX_URL))


def fetch_directory(
    dids: Iterable[int] | None = None,
    session: requests.Session | None = None,
) -> DirectoryFetchResult:
    """Fetch + parse the staff directory. NETWORK ONLY — never touches Azure.

    Args:
        dids: Department ids to fetch. Defaults to every DID enumerated from the
            department index page.
        session: Optional shared :class:`requests.Session`.

    Returns:
        A :class:`DirectoryFetchResult`. A department whose fetch/parse fails is
        recorded in ``failed_dids`` and excluded from ``departments`` (so the
        caller never reconciles — i.e. never "removes" — a department it couldn't
        read this run).
    """
    session = session or _new_session()

    if dids is None:
        try:
            did_names = fetch_department_dids(session)
        except Exception as exc:  # noqa: BLE001 - index failure yields no work
            logger.warning("Directory: failed to fetch department index: %s", exc)
            return DirectoryFetchResult()
        did_list = [did for did, _ in did_names]
    else:
        did_list = list(dids)

    result = DirectoryFetchResult()
    for did in did_list:
        try:
            page_html = _fetch(session, DEPARTMENT_LISTING_TEMPLATE.format(did=did))
            department = parse_department_listing(page_html, did)
        except Exception as exc:  # noqa: BLE001 - isolate a bad department
            logger.warning("Directory: failed to fetch/parse DID %s: %s", did, exc)
            result.failed_dids.append((did, ""))
            continue
        result.departments.append(department)
        result.officials.extend(department.officials)

    return result


# ---------------------------------------------------------------------------
# Table Storage
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _table_client():
    """TableClient for the officials table (AAD auth, created if missing).

    ``TableServiceClient`` authenticates with ``DefaultAzureCredential`` against
    ``https://{account}.table.core.windows.net`` — managed identity in Azure,
    developer credentials locally, never an account key.
    ``create_table_if_not_exists`` makes the pipeline runnable against a fresh
    account even though Bicep also declares the table.
    """
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    endpoint = f"https://{account}.table.core.windows.net"
    service = TableServiceClient(endpoint=endpoint, credential=DefaultAzureCredential())
    return service.create_table_if_not_exists(OFFICIALS_TABLE)


# Fields compared to decide whether a stored row actually changed (everything
# except the always-moving ingested_at_utc), so a re-sweep of unchanged data is a
# true no-op rather than a churn of identical writes.
_COMPARE_KEYS = (
    "eid",
    "name",
    "title",
    "department",
    "did",
    "department_phone",
    "department_address",
    "contact_form_url",
    "source_url",
    "status",
)


def _entity(official: Official, ingested_at_utc: datetime, status: str = STATUS_ACTIVE) -> dict:
    """Project an Official onto a Table entity. None values are omitted (= absent)."""
    entity: dict = {
        "PartitionKey": official.department,
        "RowKey": str(official.eid),
        "eid": official.eid,
        "name": official.name,
        "title": official.title,
        "department": official.department,
        "did": official.did,
        "source_url": official.source_url,
        "status": status,
        "ingested_at_utc": ingested_at_utc,
    }
    if official.department_phone:
        entity["department_phone"] = official.department_phone
    if official.department_address:
        entity["department_address"] = official.department_address
    if official.contact_form_url:
        entity["contact_form_url"] = official.contact_form_url
    return entity


def _normalized(entity: dict) -> dict:
    """Comparable view of an entity over the change-detection keys."""
    return {key: entity.get(key) for key in _COMPARE_KEYS}


def _upsert(table, official: Official, now_utc: datetime) -> str:
    """Idempotently upsert one official keyed on (department, eid).

    Returns "inserted", "updated", or "unchanged". Unchanged rows are left
    untouched (not even ingested_at_utc) so a second sweep is a verifiable no-op.
    """
    entity = _entity(official, now_utc)
    try:
        existing = table.get_entity(official.department, str(official.eid))
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


def _reconcile(table, department: str, present_eids: set[int], now_utc: datetime) -> int:
    """Flag officials that vanished from a department's roster as ``removed``.

    For ``department``, every stored row not already ``removed`` whose eid is
    absent from the freshly-fetched roster has left — mark it ``removed`` (never
    hard-delete). Returns the number flagged. Only called for departments fetched
    SUCCESSFULLY this run.
    """
    removed = 0
    query = "PartitionKey eq @pk and status ne @removed"
    parameters = {"pk": department, "removed": STATUS_REMOVED}
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
def ingest_officials(dids: Iterable[int] | None = None) -> DirectorySweepResult:
    """Sweep the staff directory and upsert officials into Table Storage.

    Fetch + parse is delegated to :func:`fetch_directory` (Azure-free). Each
    parsed official is idempotently upserted; then each SUCCESSFULLY-fetched
    department is reconciled so people missing from its current roster are flagged
    ``removed``. Departments that failed to fetch are never reconciled.

    Args:
        dids: Department ids to sweep. Defaults to every DID on the index page.

    Returns:
        A :class:`DirectorySweepResult` with per-outcome counters.
    """
    now_utc = datetime.now(timezone.utc)
    fetched = fetch_directory(dids)
    table = _table_client()

    result = DirectorySweepResult()
    result.failures += len(fetched.failed_dids)

    # Build the present-eid set per successfully-fetched department (including
    # departments that fetched but now have zero people — they must still be
    # reconciled, which will mark any lingering rows removed).
    present_by_department: dict[str, set[int]] = {
        dept.name: set() for dept in fetched.departments
    }

    for official in fetched.officials:
        result.parsed += 1
        present_by_department.setdefault(official.department, set()).add(official.eid)
        try:
            outcome = _upsert(table, official, now_utc)
        except Exception as exc:  # noqa: BLE001 - isolate a bad write
            logger.warning("Directory: failed to upsert EID %s: %s", official.eid, exc)
            result.failures += 1
            continue
        setattr(result, outcome, getattr(result, outcome) + 1)

    result.departments = len(fetched.departments)

    for department, present in present_by_department.items():
        try:
            result.removed += _reconcile(table, department, present, now_utc)
        except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
            logger.warning("Directory: reconciliation failed for %s: %s", department, exc)
            result.failures += 1

    logger.info(
        "Directory sweep: %d department(s), %d parsed, %d inserted, %d updated, "
        "%d unchanged, %d removed, %d failure(s)",
        result.departments, result.parsed, result.inserted, result.updated,
        result.unchanged, result.removed, result.failures,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Sweep the Gloucester staff directory into the officials table."
    )
    parser.add_argument(
        "--did",
        type=int,
        action="append",
        dest="dids",
        help="Department id to sweep (repeatable). Defaults to every department.",
    )
    args = parser.parse_args()
    print(ingest_officials(dids=args.dids))
