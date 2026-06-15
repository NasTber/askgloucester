"""Scrape Gloucester MA meeting documents and upload them to Azure Blob Storage.

Gloucester publishes agendas and minutes through a CivicPlus "Archive Center"
(``Archive.aspx``) rather than the AgendaCenter that civic-scraper understands,
so this module talks to that system directly:

    1. fetch ``Archive.aspx?AMID={amid}`` (one AMID per document collection),
    2. parse the listing of archived documents — each is an ``<a>`` link whose
       href is ``Archive.aspx?ADID={ADID}`` and whose visible text begins with
       the document's date (e.g. "June 3, 2025 School Committee Meeting"),
    3. filter the documents to the requested ``[start_date, end_date]`` window,
    4. download each PDF straight from
       ``ArchiveCenter/ViewFile/Item/{ADID}``, and
    5. stream it into the ``raw-documents`` blob container, tagged with
       ``meeting_body``, ``document_date`` and ``document_type`` metadata so
       downstream stages can carry provenance through to the search index.

Each AMID maps to a single committee + document type (see ``ARCHIVE_SOURCES``).
By default we ingest School Committee agendas (AMID 113) and minutes (AMID 114);
pass ``amid_list`` to add other committees as their AMIDs are discovered.

Authentication to Azure uses ``DefaultAzureCredential`` — no account keys.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Iterable

import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

from utils import classify_meeting_category

load_dotenv()

logger = logging.getLogger(__name__)

# Base host for Gloucester's CivicPlus site. Both the archive listing pages and
# the file download endpoint live here.
ARCHIVE_BASE_URL = os.environ.get("ARCHIVE_BASE_URL", "https://gloucester-ma.gov")
RAW_DOCUMENTS_CONTAINER = os.environ.get("RAW_DOCUMENTS_CONTAINER", "raw-documents")

# Map each Archive Center AMID to the committee it belongs to and the kind of
# document it holds. Add new committees here (and to DEFAULT_AMID_LIST below, or
# pass them via the ``amid_list`` argument) as their AMIDs are discovered.
ARCHIVE_SOURCES: dict[int, tuple[str, str]] = {
    113: ("School Committee", "agenda"),
    114: ("School Committee", "minutes"),
    35: ("City Council", "agenda"),
    36: ("City Council", "minutes"),
    # NB: these archive AMIDs collide numerically with unrelated calendar CIDs
    # (e.g. CID 47 = Community Preservation, CID 48 = Conservation Commission) —
    # the AMID and CID number spaces are independent; do not cross-wire them.
    57: ("Planning Board", "agenda"),
    58: ("Planning Board", "minutes"),
    47: ("Conservation Commission", "agenda"),
    48: ("Conservation Commission", "minutes"),
    41: ("Zoning Board of Appeals", "agenda"),
    146: ("Zoning Board of Appeals", "minutes"),
    42: ("Zoning Board of Appeals", "meeting results"),
}

# Out of the box, ingest School Committee agendas and minutes only, to keep the
# initial test set small.
DEFAULT_AMID_LIST: tuple[int, ...] = (113, 114, 35, 36, 57, 58, 47, 48, 41, 146, 42)

# Some CivicPlus deployments reject requests without a browser-like User-Agent,
# returning an interstitial or 403 instead of the archive listing. Present one.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Each archived document is an anchor whose href is "Archive.aspx?ADID={ADID}".
# Capture the ADID.
_ARCHIVE_LINK_RE = re.compile(r"Archive\.aspx\?ADID=(\d+)", re.I)

# Date formats seen at the start of the link text, e.g. "June 3, 2025" or
# "06/03/2025". The comma may be followed by no space ("August 29,2022"), so the
# space after it is optional.
_TEXTUAL_DATE_RE = re.compile(r"([A-Za-z]{3,9}\.?\s+\d{1,2},?\s*\d{4})")
_NUMERIC_DATE_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")


@dataclass
class UploadedDocument:
    """A document that was scraped and stored in blob storage."""

    blob_name: str
    source_url: str
    meeting_body: str
    document_date: str  # ISO date, YYYY-MM-DD
    document_type: str  # "agenda" or "minutes"
    meeting_category: str  # "full_committee", "subcommittee" or "negotiations"


@dataclass
class ArchiveEntry:
    """One archived document parsed out of an Archive.aspx listing."""

    adid: int
    document_date: date
    title: str


class _ArchiveLinkParser(HTMLParser):
    """Collect ``(adid, text)`` pairs for every archived-document link.

    Each document in an Archive.aspx listing is an ``<a>`` whose href is
    ``Archive.aspx?ADID={ADID}`` and whose visible text begins with the
    document's date. We gather the ADID and the anchor text for every such link.
    """

    def __init__(self) -> None:
        super().__init__()
        self._adid: int | None = None
        self._text_parts: list[str] = []
        self.links: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href") or ""
            match = _ARCHIVE_LINK_RE.search(href)
            if match:
                self._adid = int(match.group(1))
                self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._adid is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._adid is not None:
            # Collapse runs of whitespace so date parsing sees a clean string.
            text = " ".join("".join(self._text_parts).split())
            self.links.append((self._adid, text))
            self._adid = None


def _blob_service_client() -> BlobServiceClient:
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    account_url = f"https://{account}.blob.core.windows.net"
    # DefaultAzureCredential resolves managed identity in Azure and developer
    # credentials locally — never an account key.
    return BlobServiceClient(account_url, credential=DefaultAzureCredential())


def _sanitize(value: str) -> str:
    """Make a string safe to use as part of a blob path."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in value).strip("-")


def _parse_document_date(text: str) -> date | None:
    """Extract a calendar date from a document link's visible text.

    Handles both textual ("June 3, 2025") and numeric ("06/03/2025") forms and
    a couple of common abbreviation styles. Returns ``None`` when no date can be
    found so the caller can skip the entry.
    """
    match = _TEXTUAL_DATE_RE.search(text)
    if match:
        # Normalize separators ("." / ",") to spaces and collapse whitespace so
        # both "June 3, 2025" and "August 29,2022" reduce to "Month D YYYY".
        candidate = " ".join(re.sub(r"[.,]", " ", match.group(1)).split())
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue

    match = _NUMERIC_DATE_RE.search(text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%m/%d/%Y").date()
        except ValueError:
            pass

    return None


def _fetch_archive_entries(
    session: requests.Session, amid: int
) -> list[ArchiveEntry]:
    """Fetch and parse one Archive.aspx page into a list of archive entries."""
    url = f"{ARCHIVE_BASE_URL}/Archive.aspx?AMID={amid}"
    response = session.get(url, timeout=60)
    response.raise_for_status()

    parser = _ArchiveLinkParser()
    parser.feed(response.text)

    entries: list[ArchiveEntry] = []
    seen_adids: set[int] = set()
    for adid, text in parser.links:
        if adid in seen_adids:
            continue  # the same document can be linked more than once
        seen_adids.add(adid)

        document_date = _parse_document_date(text)
        if document_date is None:
            # Some entries have no usable date (e.g. "Superintendent Search").
            logger.debug("Skipping AMID %s link with no parseable date: %r", amid, text)
            continue

        entries.append(
            ArchiveEntry(adid=adid, document_date=document_date, title=text)
        )

    logger.info("AMID %s: parsed %d archived document(s)", amid, len(entries))
    return entries


def scrape_and_upload(
    start_date: str,
    end_date: str,
    amid_list: Iterable[int] = DEFAULT_AMID_LIST,
    meeting_body: str | None = None,
    skip_source_urls: set[str] | None = None,
) -> list[UploadedDocument]:
    """Scrape Archive.aspx collections and upload matching PDFs to blob storage.

    Args:
        start_date: Inclusive start date, ``YYYY-MM-DD``.
        end_date: Inclusive end date, ``YYYY-MM-DD``.
        amid_list: Archive Center AMIDs to scrape. Defaults to School Committee
            agendas (113) and minutes (114). Each AMID must be present in
            :data:`ARCHIVE_SOURCES`.
        meeting_body: Optional committee filter (case-insensitive). When given,
            only AMIDs whose committee matches are scraped — handy for narrowing
            a broader ``amid_list`` down to a single body.
        skip_source_urls: Source URLs already present in the search index. Any
            in-window document whose ``source_url`` is in this set is skipped
            entirely — no download, no upload — since it is already indexed.
            ``None``/empty disables skipping (a deliberate full re-ingest).

    Returns:
        A list of :class:`UploadedDocument` records, one per uploaded blob.
    """
    window_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    window_end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Resolve the AMIDs to scrape, validating each and applying the optional
    # meeting_body filter.
    amids: list[int] = []
    for amid in amid_list:
        if amid not in ARCHIVE_SOURCES:
            logger.warning("Skipping unknown AMID %s (not in ARCHIVE_SOURCES)", amid)
            continue
        body, _ = ARCHIVE_SOURCES[amid]
        if meeting_body and body.lower() != meeting_body.lower():
            continue
        amids.append(amid)

    logger.info(
        "Scraping AMIDs %s from %s to %s",
        amids,
        start_date,
        end_date,
    )

    container = _blob_service_client().get_container_client(RAW_DOCUMENTS_CONTAINER)
    # The container is provisioned by Bicep, but create it if missing so the
    # pipeline is runnable against a fresh account.
    try:
        container.create_container()
    except Exception:  # noqa: BLE001 - "already exists" is the common case
        pass

    # Reuse one session (and its browser User-Agent) for every request.
    session = requests.Session()
    session.headers.update({"User-Agent": BROWSER_USER_AGENT})

    uploaded: list[UploadedDocument] = []

    for amid in amids:
        body, document_type = ARCHIVE_SOURCES[amid]

        try:
            entries = _fetch_archive_entries(session, amid)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch Archive.aspx for AMID %s: %s", amid, exc)
            continue

        for entry in entries:
            # Filter to the requested date window (inclusive).
            if not (window_start <= entry.document_date <= window_end):
                continue

            document_date = entry.document_date.strftime("%Y-%m-%d")
            # Derive the meeting category from the listing title — the AMID-level
            # meeting_body can't tell a full committee meeting from a subcommittee
            # or a negotiations session, but the title can.
            meeting_category = classify_meeting_category(entry.title)
            # The PDF is downloaded directly from the ViewFile endpoint by ADID.
            source_url = f"{ARCHIVE_BASE_URL}/ArchiveCenter/ViewFile/Item/{entry.adid}"

            # Existence-only skip: if this document is already indexed, skip it
            # before any download or upload. The ADID (and thus source_url) is
            # known from the listing, so we never touch the network for it.
            if skip_source_urls and source_url in skip_source_urls:
                logger.info("skipping already-indexed: %s", source_url)
                continue

            blob_name = (
                f"{_sanitize(body)}/{document_date}/{entry.adid}_{document_type}.pdf"
            )

            try:
                response = session.get(source_url, allow_redirects=True, timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Failed to download %s: %s", source_url, exc)
                continue

            metadata = {
                "meeting_body": body,
                "document_date": document_date,
                "document_type": document_type,
                "meeting_category": meeting_category,
                "source_url": source_url,
                # The full listing title (e.g. "March 12, 2025 School Committee
                # Meeting") names the specific meeting, which the AMID-level
                # meeting_body cannot. Azure blob metadata must be ASCII, so drop
                # any stray non-ASCII characters (e.g. curly quotes).
                "title": entry.title.encode("ascii", "ignore").decode("ascii"),
            }

            container.upload_blob(
                name=blob_name,
                data=response.content,
                overwrite=True,
                metadata=metadata,
                content_settings=ContentSettings(content_type="application/pdf"),
            )
            logger.info("Uploaded %s (%d bytes)", blob_name, len(response.content))

            uploaded.append(
                UploadedDocument(
                    blob_name=blob_name,
                    source_url=source_url,
                    meeting_body=body,
                    document_date=document_date,
                    document_type=document_type,
                    meeting_category=meeting_category,
                )
            )

    logger.info("Uploaded %d documents", len(uploaded))
    return uploaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Scrape Archive.aspx PDFs to blob storage.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--amid",
        type=int,
        action="append",
        dest="amids",
        help="Archive Center AMID to scrape (repeatable). "
        "Defaults to 113 (agendas) and 114 (minutes).",
    )
    args = parser.parse_args()

    amid_list = args.amids if args.amids else DEFAULT_AMID_LIST
    for doc in scrape_and_upload(args.start_date, args.end_date, amid_list=amid_list):
        print(doc.blob_name)
