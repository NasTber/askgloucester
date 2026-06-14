"""Read-only Azure Table Storage access for the Gloucester staff directory.

API-local and deliberately self-contained: this module talks to the
``officials`` table DIRECTLY (``TableServiceClient`` over
``DefaultAzureCredential``) and does NOT import anything from ``ingestion/`` —
the deployed API image ships only ``api/`` (see the Dockerfile); importing
``ingestion`` would crash-loop the container. The ingestion side
(``ingestion/directory_source.py``) owns WRITES to the table; this module owns
the read path used by the ``directory_lookup`` agent tool, plus the presentation
helper that turns rows into resident-facing prose.

Auth is ``DefaultAzureCredential`` (managed identity in Azure, developer creds
locally) — no account keys. The API managed identity needs the **Storage Table
Data Reader** role on the account (see ``infra/modules/storage.bicep``); that
assignment is account-scoped, so it already covers this table alongside
``events``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential

# Dedicated table holding the normalized staff directory (written by ingestion).
OFFICIALS_TABLE = os.environ.get("OFFICIALS_TABLE_NAME", "officials")

# Status value for a person who left the directory (mirror ingestion's enum). We
# exclude these from every read so departed staff never surface in an answer.
STATUS_REMOVED = "removed"

# At ~118 rows the whole table is tiny, so the search path pulls every active
# row and filters in Python — no per-query Table filters needed.
#
# The cap is 25, not the size of the largest real department (Public Works, 12),
# so no genuine department roster ever truncates; it only bounds a pathological
# fuzzy match (e.g. a single common letter). When a result set is capped, the
# rendering says so and links the full directory, so a partial list is never
# presented as complete.
MAX_RESULTS = 25

# The full city staff directory, linked when a result set is capped.
FULL_DIRECTORY_URL = "https://www.gloucester-ma.gov/Directory.aspx"


@dataclass
class Official:
    """One staff-directory person as read back from the table (the tool's shape)."""

    eid: int
    name: str
    title: str
    department: str
    did: int | None
    department_phone: str | None
    contact_form_url: str | None
    source_url: str


@lru_cache(maxsize=1)
def _table_client():
    """TableClient for the officials table (AAD auth, read-only use).

    ``TableServiceClient`` authenticates with ``DefaultAzureCredential`` against
    ``https://{account}.table.core.windows.net``. We only ever read here, so the
    table is assumed to already exist (created by ingestion); no create call.
    """
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    endpoint = f"https://{account}.table.core.windows.net"
    service = TableServiceClient(endpoint=endpoint, credential=DefaultAzureCredential())
    return service.get_table_client(OFFICIALS_TABLE)


def _to_official(entity) -> Official:
    """Rebuild an :class:`Official` from a stored table entity."""
    did = entity.get("did")
    return Official(
        eid=int(entity["eid"]),
        name=entity.get("name", ""),
        title=entity.get("title", ""),
        department=entity.get("PartitionKey", entity.get("department", "")),
        did=int(did) if did is not None else None,
        department_phone=entity.get("department_phone"),
        contact_form_url=entity.get("contact_form_url"),
        source_url=entity.get("source_url", ""),
    )


def _load_active() -> list[Official]:
    """Read every active official (status != 'removed') from the table.

    The table is small enough to pull whole and filter in Python, so there is no
    OData query to maintain; departed staff are dropped here so no read path can
    surface them.
    """
    table = _table_client()
    officials: list[Official] = []
    for entity in table.list_entities():
        if entity.get("status") == STATUS_REMOVED:
            continue
        officials.append(_to_official(entity))
    return officials


def search_officials(query: str) -> list[Official]:
    """Case-insensitive substring search across name + title + department.

    Returns ALL matches, sorted by department then name for a stable rendering.
    Capping is applied at render time (so the renderer knows the true total and
    can flag a truncated list). An empty/blank query returns no matches (the tool
    declines).
    """
    needle = (query or "").strip().lower()
    if not needle:
        return []

    matches = [
        o
        for o in _load_active()
        if needle in f"{o.name} {o.title} {o.department}".lower()
    ]
    matches.sort(key=lambda o: (o.department.lower(), o.name.lower()))
    return matches


# --- Presentation -----------------------------------------------------------
def render_officials(query: str, matches: list[Official], limit: int = MAX_RESULTS) -> str:
    """Render directory matches as plain prose for the agent.

    Each person's directory page (``source_url``), the department phone, and the
    department contact-form link (when present) are rendered INLINE — there is no
    ``[n]`` citation channel for the directory (that belongs to doc_search only).
    On no match, return a clean decline. When more than ``limit`` people match,
    show the first ``limit`` and append a note with the true total plus the full
    directory link, so a partial list is never presented as complete.
    """
    if not matches:
        return f'I don\'t have a staff-directory entry for "{query}".'

    total = len(matches)
    shown = matches[:limit]
    lines = [f'Staff-directory matches for "{query}":', ""]
    for o in shown:
        title_part = f" — {o.title}" if o.title else ""
        lines.append(f"- {o.name}{title_part} ({o.department})")
        if o.department_phone:
            lines.append(f"  Department phone: {o.department_phone}")
        if o.contact_form_url:
            lines.append(f"  Contact form: {o.contact_form_url}")
        lines.append(f"  Directory page: {o.source_url}")
    if total > limit:
        lines.append("")
        lines.append(
            f"Showing {limit} of {total} matches; full staff directory: "
            f"{FULL_DIRECTORY_URL}"
        )
    return "\n".join(lines)
