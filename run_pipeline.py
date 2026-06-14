"""AskGloucester ingestion pipeline.

Ties the four ingestion stages together end to end:

    1. scrape AgendaCenter PDFs and upload them to blob storage
    2. extract text from each PDF with Document Intelligence
    3. chunk the text into ~500-token segments with overlap
    4. embed each chunk with Azure OpenAI (text-embedding-3-small)
    5. index the chunks into Azure AI Search

Run it over a date window, e.g.::

    python run_pipeline.py --start-date 2025-01-01 --end-date 2025-06-30

To keep the initial test set small, it defaults to School Committee documents.
All Azure auth uses ``DefaultAzureCredential`` and all endpoints come from a
``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

# Make the ingestion modules importable when run from the project root.
sys.path.insert(0, "ingestion")

import calendar_source  # noqa: E402
import chunker  # noqa: E402
import directory_source  # noqa: E402
import drive_source  # noqa: E402
import embedder  # noqa: E402
import indexer  # noqa: E402
import processor  # noqa: E402
import scraper  # noqa: E402

load_dotenv()

logger = logging.getLogger("askgloucester.pipeline")


def _run_calendar_step() -> None:
    """Independent structured source: upsert calendar events into Table Storage.

    Wrapped in its own try/except so a calendar failure can NEVER break the
    document pipeline. Writes ONLY to Azure Table Storage — it does not touch blob
    storage, Document Intelligence, chunking, embedding, or the AI Search index.
    """
    try:
        result = calendar_source.ingest_calendar_events()
        logger.info("Calendar step complete: %s", result)
    except Exception as exc:  # noqa: BLE001 - never let calendar break documents
        logger.exception("Calendar ingestion failed (continuing with documents): %s", exc)


def _run_directory_step() -> None:
    """Independent structured source: upsert the city staff directory.

    A sibling of the calendar step. Wrapped in its own try/except so a directory
    failure can NEVER break the document pipeline. Writes ONLY to the
    ``officials`` Azure Table — it does not touch blob storage, Document
    Intelligence, chunking, embedding, or the AI Search index.
    """
    try:
        result = directory_source.ingest_officials()
        logger.info("Directory step complete: %s", result)
    except Exception as exc:  # noqa: BLE001 - never let directory break documents
        logger.exception("Directory ingestion failed (continuing with documents): %s", exc)


def run(
    start_date: str,
    end_date: str,
    meeting_body: str | None,
    skip_indexed: bool = True,
) -> int:
    """Run the full ingestion pipeline. Returns the number of chunks indexed."""
    # 0. Independent structured sources (calendar + staff directory). Only on a
    # full ('all') sweep, and run first so they execute regardless of the document
    # flow's outcome (the documents path can early-return when its window is
    # empty). Each has its own try/except so a structured-source failure can never
    # break the document pipeline.
    if meeting_body is None:
        _run_calendar_step()
        _run_directory_step()

    # Existence-only skip: pull the set of source_urls already in the index once,
    # up front, and hand it to both document sources so an already-indexed doc is
    # skipped before download/Document Intelligence/chunk/embed. New & revised
    # docs always arrive under a new source_url, so they aren't in the set and
    # process normally. An empty/missing index yields an empty set (full ingest).
    # --no-skip (skip_indexed=False) forces a full re-ingest. The calendar source
    # is untouched — it already does idempotent Table upsert + reconciliation.
    skip_source_urls: set[str] = set()
    if skip_indexed:
        skip_source_urls = indexer.get_indexed_source_urls()
        logger.info("Existence-only skip ON: %d indexed source_url(s)", len(skip_source_urls))
    else:
        logger.info("Existence-only skip OFF (--no-skip): re-ingesting all in-window docs")

    # 1a. Scrape + upload Archive.aspx PDFs (agendas).
    documents = scraper.scrape_and_upload(
        start_date=start_date,
        end_date=end_date,
        meeting_body=meeting_body,
        skip_source_urls=skip_source_urls,
    )

    # 1b. Pull School Committee minutes from the public Google Drive folder.
    # The Drive source is School-Committee-minutes only, so only run it when the
    # window isn't filtered to a different body. Its UploadedDocument records are
    # identical in shape, so they join the same list and flow through unchanged.
    if meeting_body is None or meeting_body.lower() == "school committee":
        documents += drive_source.fetch_and_upload(
            start_date, end_date, skip_source_urls=skip_source_urls
        )

    if not documents:
        logger.warning("No documents scraped for the given window; nothing to do.")
        return 0

    # 2. Make sure the search index exists before we start producing chunks.
    indexer.ensure_index()

    # 3. Process + chunk each document.
    all_chunks: list[chunker.Chunk] = []
    for doc in documents:
        try:
            pages = processor.extract_text(doc.blob_name)
        except Exception as exc:  # noqa: BLE001 - keep going on a bad document
            logger.exception("Failed to extract %s: %s", doc.blob_name, exc)
            continue

        if not pages:
            logger.warning("No text extracted from %s; skipping.", doc.blob_name)
            continue

        chunks = chunker.chunk_pages(
            pages,
            source_url=doc.source_url,
            document_date=doc.document_date,
            meeting_body=doc.meeting_body,
            document_type=doc.document_type,
            meeting_category=doc.meeting_category,
            base_id=doc.blob_name,
        )
        logger.info("%s -> %d chunk(s)", doc.blob_name, len(chunks))
        all_chunks.extend(chunks)

    # 4. Embed each chunk's text so it can be vector-searched.
    embedder.embed_chunks(all_chunks)

    # 5. Index everything in batches.
    indexed = indexer.upload_chunks(all_chunks)
    logger.info(
        "Pipeline complete: %d document(s), %d chunk(s), %d indexed.",
        len(documents),
        len(all_chunks),
        indexed,
    )
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AskGloucester ingestion pipeline.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument(
        "--meeting-body",
        default="School Committee",
        help="Committee to ingest. Pass 'all' to ingest every body. "
        "Defaults to School Committee to keep the test set small.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Disable the existence-only skip and re-ingest every in-window "
        "document, even if already indexed (deliberate full re-ingest). "
        "Skipping is ON by default.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    meeting_body = None if args.meeting_body.lower() == "all" else args.meeting_body
    run(args.start_date, args.end_date, meeting_body, skip_indexed=not args.no_skip)


if __name__ == "__main__":
    main()
