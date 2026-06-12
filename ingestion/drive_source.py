"""Ingest Gloucester School Committee *minutes* from a public Google Drive folder.

The city's Archive.aspx site (see :mod:`scraper`) carries agendas reliably but
is missing most meeting minutes. The minutes are instead posted to a world-
readable Google Drive folder, organized into per-school-year subfolders plus a
flat set of the current year's files at the root.

This module mirrors :mod:`scraper`'s contract exactly so the rest of the
pipeline is unchanged: it downloads each in-window PDF, uploads it to the same
``raw-documents`` blob container with the **same metadata keys/format**, and
returns a ``list[UploadedDocument]`` — the very type ``run_pipeline`` already
iterates. Discovery is keyless via ``gdown`` (the folder is public); blob/Azure
auth stays ``DefaultAzureCredential`` end to end.

Flow:
    1. enumerate the folder recursively with ``gdown`` (no download yet),
    2. parse a meeting date out of each *filename* (never the parent folder),
    3. keep only PDFs whose date falls in ``[start_date, end_date]``,
    4. download each survivor with ``gdown`` and stream it into blob storage,
       tagged with ``meeting_body`` / ``document_date`` / ``document_type`` /
       ``source_url`` / ``title`` just like the Archive.aspx scraper.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
from datetime import date, datetime

import gdown
from azure.storage.blob import ContentSettings
from dotenv import load_dotenv

# Reuse the scraper's blob plumbing and return type so the two sources are
# byte-for-byte identical in how they land documents for the pipeline.
import scraper
from scraper import UploadedDocument, _blob_service_client, _sanitize
from utils import classify_meeting_category

load_dotenv()

logger = logging.getLogger(__name__)

# The public School Committee minutes folder. Override via env for testing.
DRIVE_FOLDER_URL = os.environ.get(
    "DRIVE_MINUTES_FOLDER_URL",
    "https://drive.google.com/drive/folders/1rffVeE1ZrGukLvfcWObBAkxmC2PIPz9Q",
)

# Same container the scraper writes to; downstream reads only by blob name.
RAW_DOCUMENTS_CONTAINER = scraper.RAW_DOCUMENTS_CONTAINER

# Every file here is a School Committee meeting-minutes PDF.
MEETING_BODY = "School Committee"
DOCUMENT_TYPE = "minutes"

# A meeting date embedded in the filename: "M_D_YY" with "_" or ":" separators,
# e.g. "1_14_26", "10:13:21", "04_22_20". Month/day are 1-2 digits, the year is
# the 2-digit form used throughout the folder. The lookbehind/ahead stop us from
# slicing digits out of a longer run (so a stray "2024" can't masquerade as a
# date, and a "(1)" copy suffix is ignored).
_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[_:](\d{1,2})[_:](\d{2})(?!\d)")


def _parse_meeting_date(filename: str) -> tuple[date, str] | None:
    """Parse the meeting date and descriptive prefix out of a Drive filename.

    Examples (basename only; the parent folder is intentionally ignored)::

        "SC Minutes 1_14_26.pdf"            -> (2026-01-14, "SC")
        "Special SC Minutes 5_18_21.pdf"    -> (2021-05-18, "Special SC")
        "ES SC Minutes 10_28_20.pdf"        -> (2020-10-28, "ES SC")
        "Joint CC & SC Minutes 9_15_20.pdf" -> (2020-09-15, "Joint CC & SC")
        "SC Minutes 10:13:21.pdf"           -> (2021-10-13, "SC")
        "SC Minutes 3_12_25 (1).pdf"        -> (2025-03-12, "SC")  # "(1)" ignored

    The 2-digit year is pivoted as ``2000 + YY``. We take the *last* date-looking
    match so a number elsewhere in the name can't win. The prefix (the label
    before the word "Minutes", e.g. Special / Amended / ES) is captured so the
    meeting sub-type is preserved in the blob ``title``; it never blocks parsing.

    Returns ``None`` when no date can be found, so the caller can skip the file.
    """
    matches = list(_DATE_RE.finditer(filename))
    if not matches:
        return None

    # The last match is the meeting date; earlier numbers are noise.
    m = matches[-1]
    month, day, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
    year = 2000 + yy  # 2-digit-year pivot, e.g. "26" -> 2026.

    try:
        meeting_date = date(year, month, day)
    except ValueError:
        # e.g. month 13 / day 32 — a number that matched the shape but isn't a
        # real date. Treat as unparseable so the caller logs + skips it.
        return None

    # Descriptive prefix: text before "Minutes" if present, else everything
    # before the date. Strip separators/spaces so "Special SC " -> "Special SC".
    head = filename[: m.start()]
    label_match = re.search(r"(.*?)\bMinutes\b", head, re.IGNORECASE)
    prefix = (label_match.group(1) if label_match else head).strip(" -_:")

    return meeting_date, prefix


def _enumerate_pdfs() -> list[tuple[str, str]]:
    """List the folder recursively without downloading. Returns (file_id, name).

    ``gdown.download_folder(skip_download=True)`` walks the root and every
    subfolder and hands back one entry per file with its Drive ``id`` and a
    ``path`` (``"<subfolder>/<filename>"``). We keep only ``.pdf`` files and use
    the *basename* — the per-year subfolder is deliberately discarded since the
    date lives in the filename and some files are mis-filed under the wrong year.
    """
    entries = gdown.download_folder(
        DRIVE_FOLDER_URL,
        skip_download=True,  # enumerate metadata only; no bytes transferred
        quiet=True,
        use_cookies=False,  # public folder — no auth, no stored cookies
    )

    pdfs: list[tuple[str, str]] = []
    for entry in entries or []:
        name = os.path.basename(entry.path)
        if name.lower().endswith(".pdf"):
            pdfs.append((entry.id, name))
    return pdfs


def fetch_and_upload(start_date: str, end_date: str) -> list[UploadedDocument]:
    """Download in-window minutes PDFs from Drive and upload them to blob.

    Args:
        start_date: Inclusive start date, ``YYYY-MM-DD``.
        end_date: Inclusive end date, ``YYYY-MM-DD``.

    Returns:
        A list of :class:`scraper.UploadedDocument` — identical in shape to
        :func:`scraper.scrape_and_upload`, so ``run_pipeline`` processes minutes
        and agendas through the exact same code path.
    """
    window_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    window_end = datetime.strptime(end_date, "%Y-%m-%d").date()

    files = _enumerate_pdfs()
    logger.info("Drive: found %d PDF file(s) in folder", len(files))

    # Parse dates, skipping anything unparseable, then narrow to the window.
    in_window: list[tuple[str, str, date, str]] = []  # (id, name, date, prefix)
    skipped = 0
    for file_id, name in files:
        parsed = _parse_meeting_date(name)
        if parsed is None:
            logger.warning("Drive: no parseable date in %r; skipping", name)
            skipped += 1
            continue
        meeting_date, prefix = parsed
        if window_start <= meeting_date <= window_end:
            in_window.append((file_id, name, meeting_date, prefix))

    logger.info(
        "Drive: %d file(s) in window %s..%s (%d skipped, no date)",
        len(in_window),
        start_date,
        end_date,
        skipped,
    )

    container = _blob_service_client().get_container_client(RAW_DOCUMENTS_CONTAINER)
    # The container is provisioned by Bicep; create-if-missing keeps the pipeline
    # runnable against a fresh account (matches the scraper's behavior).
    try:
        container.create_container()
    except Exception:  # noqa: BLE001 - "already exists" is the common case
        pass

    uploaded: list[UploadedDocument] = []

    # Download each in-window PDF to a temp dir, then stream it into blob.
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_id, name, meeting_date, prefix in in_window:
            document_date = meeting_date.strftime("%Y-%m-%d")
            source_url = f"https://drive.google.com/file/d/{file_id}/view"
            # The filename names the specific meeting, so it (not the constant
            # MEETING_BODY) is what distinguishes a full committee meeting from a
            # subcommittee or negotiations session.
            meeting_category = classify_meeting_category(name)

            local_path = os.path.join(tmpdir, f"{file_id}.pdf")
            try:
                # Keyless download by file id from the public folder.
                result = gdown.download(
                    id=file_id, output=local_path, quiet=True, use_cookies=False
                )
                if not result:
                    raise RuntimeError("gdown returned no path (download failed)")
                with open(local_path, "rb") as fh:
                    data = fh.read()
            except Exception as exc:  # noqa: BLE001 - keep going on a bad file
                logger.warning("Drive: failed to download %r (%s): %s", name, file_id, exc)
                continue

            # Same blob naming scheme as the scraper: body/date/<id>_<type>.pdf.
            # The Drive file id stands in for the Archive ADID and keeps the path
            # unique even when two files share a date (e.g. duplicate uploads).
            blob_name = (
                f"{_sanitize(MEETING_BODY)}/{document_date}/{file_id}_{DOCUMENT_TYPE}.pdf"
            )

            metadata = {
                "meeting_body": MEETING_BODY,
                "document_date": document_date,
                "document_type": DOCUMENT_TYPE,
                "meeting_category": meeting_category,
                "source_url": source_url,
                # Preserve the full filename (incl. the Special/Amended/ES/Joint
                # prefix) for provenance. Azure blob metadata must be ASCII, so
                # drop stray non-ASCII (e.g. the "&" stays, curly quotes go).
                "title": name.encode("ascii", "ignore").decode("ascii"),
            }

            container.upload_blob(
                name=blob_name,
                data=data,
                overwrite=True,
                metadata=metadata,
                content_settings=ContentSettings(content_type="application/pdf"),
            )
            logger.info("Drive: uploaded %s (%d bytes)", blob_name, len(data))

            uploaded.append(
                UploadedDocument(
                    blob_name=blob_name,
                    source_url=source_url,
                    meeting_body=MEETING_BODY,
                    document_date=document_date,
                    document_type=DOCUMENT_TYPE,
                    meeting_category=meeting_category,
                )
            )

    logger.info("Drive: uploaded %d minutes document(s)", len(uploaded))
    return uploaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Download public-Drive School Committee minutes to blob storage."
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive end, YYYY-MM-DD.")
    args = parser.parse_args()

    for doc in fetch_and_upload(args.start_date, args.end_date):
        print(doc.blob_name)
