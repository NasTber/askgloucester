"""Scrape Gloucester MA meeting documents and upload them to Azure Blob Storage.

Pulls agendas and minutes from the city's CivicPlus AgendaCenter using the
civic-scraper library, then streams each PDF into the ``raw-documents`` blob
container. Every blob is tagged with ``meeting_body``, ``document_date`` and
``document_type`` metadata so downstream stages can carry provenance through
to the search index.

Authentication to Azure uses ``DefaultAzureCredential`` — no account keys.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Iterable

import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from civic_scraper.platforms import CivicPlusSite
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default CivicPlus AgendaCenter for Gloucester, MA.
AGENDA_CENTER_URL = os.environ.get(
    "AGENDA_CENTER_URL", "https://www.gloucester-ma.gov/AgendaCenter"
)
RAW_DOCUMENTS_CONTAINER = os.environ.get("RAW_DOCUMENTS_CONTAINER", "raw-documents")

# We only ingest agendas and minutes; AgendaCenter also exposes other asset
# types (e.g. captioned media) that are not useful for text extraction.
SUPPORTED_ASSET_TYPES = ("agenda", "minutes")


@dataclass
class UploadedDocument:
    """A document that was scraped and stored in blob storage."""

    blob_name: str
    source_url: str
    meeting_body: str
    document_date: str  # ISO date, YYYY-MM-DD
    document_type: str  # "agenda" or "minutes"


def _blob_service_client() -> BlobServiceClient:
    account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    account_url = f"https://{account}.blob.core.windows.net"
    return BlobServiceClient(account_url, credential=DefaultAzureCredential())


def _sanitize(value: str) -> str:
    """Make a string safe to use as part of a blob path."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in value).strip("-")


def _matches(asset, meeting_body: str | None, asset_types: Iterable[str]) -> bool:
    """Return True if an asset is a PDF we want to ingest."""
    if asset.asset_type not in set(asset_types):
        return False
    # Restrict to a single committee for the initial small test set.
    if meeting_body and (asset.committee_name or "").strip().lower() != meeting_body.lower():
        return False
    # AgendaCenter assets are PDFs; guard against anything else.
    content_type = (asset.content_type or "").lower()
    url = (asset.url or "").lower()
    return "pdf" in content_type or url.endswith(".pdf")


def scrape_and_upload(
    start_date: str,
    end_date: str,
    meeting_body: str | None = "School Committee",
    asset_types: Iterable[str] = SUPPORTED_ASSET_TYPES,
    agenda_center_url: str = AGENDA_CENTER_URL,
) -> list[UploadedDocument]:
    """Scrape AgendaCenter and upload matching PDFs to blob storage.

    Args:
        start_date: Inclusive start date, ``YYYY-MM-DD``.
        end_date: Inclusive end date, ``YYYY-MM-DD``.
        meeting_body: Committee name to filter on (case-insensitive). Pass
            ``None`` to ingest every committee. Defaults to "School Committee"
            to keep the initial test set small.
        asset_types: Which asset types to keep (default: agendas + minutes).
        agenda_center_url: Base AgendaCenter URL to scrape.

    Returns:
        A list of :class:`UploadedDocument` records, one per uploaded blob.
    """
    logger.info(
        "Scraping %s for %s documents from %s to %s",
        agenda_center_url,
        meeting_body or "all bodies",
        start_date,
        end_date,
    )

    site = CivicPlusSite(agenda_center_url)
    assets = site.scrape(
        start_date=start_date,
        end_date=end_date,
        asset_list=list(asset_types),
    )

    container = _blob_service_client().get_container_client(RAW_DOCUMENTS_CONTAINER)
    # The container is provisioned by Bicep, but create it if missing so the
    # pipeline is runnable against a fresh account.
    try:
        container.create_container()
    except Exception:  # noqa: BLE001 - "already exists" is the common case
        pass

    session = requests.Session()
    uploaded: list[UploadedDocument] = []

    for asset in assets:
        if not _matches(asset, meeting_body, asset_types):
            continue

        document_date = (
            asset.meeting_date.strftime("%Y-%m-%d") if asset.meeting_date else "unknown"
        )
        body = (asset.committee_name or meeting_body or "unknown").strip()

        blob_name = (
            f"{_sanitize(body)}/{document_date}/"
            f"{asset.meeting_id or _sanitize(asset.asset_name or 'doc')}_{asset.asset_type}.pdf"
        )

        try:
            response = session.get(asset.url, allow_redirects=True, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to download %s: %s", asset.url, exc)
            continue

        metadata = {
            "meeting_body": body,
            "document_date": document_date,
            "document_type": asset.asset_type,
            "source_url": asset.url,
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
                source_url=asset.url,
                meeting_body=body,
                document_date=document_date,
                document_type=asset.asset_type,
            )
        )

    logger.info("Uploaded %d documents", len(uploaded))
    return uploaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Scrape AgendaCenter PDFs to blob storage.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--meeting-body", default="School Committee")
    args = parser.parse_args()

    for doc in scrape_and_upload(args.start_date, args.end_date, args.meeting_body):
        print(doc.blob_name)
