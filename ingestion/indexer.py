"""Create/update the Azure AI Search index and upload document chunks.

Defines the ``meeting-documents`` index schema and provides a batched upload
path for chunks produced by :mod:`chunker`. The schema makes every text field
keyword-searchable and makes the key/metadata fields filterable so the API
layer can scope queries by meeting body, document type or date.

Authentication uses ``DefaultAzureCredential`` — local API keys are disabled
on the service.
"""

from __future__ import annotations

import logging
import os

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "meeting-documents")

# Azure AI Search caps a single indexing batch at 1000 documents.
DEFAULT_BATCH_SIZE = 1000


def _search_endpoint() -> str:
    return os.environ["AZURE_SEARCH_ENDPOINT"]


def _credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()


def _build_index(index_name: str) -> SearchIndex:
    """Build the index schema.

    All string fields are keyword-searchable; the key and metadata fields are
    filterable (and faceted/sortable where useful) so results can be scoped.
    """
    fields = [
        # Key field — filterable per the schema requirement.
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        # The chunk text — searchable but not filterable.
        SearchableField(name="content", type=SearchFieldDataType.String),
        # Provenance / metadata fields: searchable AND filterable.
        SearchableField(
            name="source_url",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchableField(
            name="document_date",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SearchableField(
            name="meeting_body",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="document_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SearchableField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
    ]
    return SearchIndex(name=index_name, fields=fields)


def ensure_index(index_name: str = SEARCH_INDEX_NAME) -> None:
    """Create the index if it doesn't exist, or update it to match the schema."""
    client = SearchIndexClient(_search_endpoint(), credential=_credential())
    index = _build_index(index_name)
    try:
        client.get_index(index_name)
        logger.info("Updating existing index '%s'", index_name)
    except ResourceNotFoundError:
        logger.info("Creating index '%s'", index_name)
    client.create_or_update_index(index)


def upload_chunks(
    chunks,
    index_name: str = SEARCH_INDEX_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Upload chunks to the search index in batches.

    Args:
        chunks: Iterable of :class:`chunker.Chunk` (or dicts) to index.
        index_name: Target index.
        batch_size: Documents per upload request (max 1000).

    Returns:
        The number of documents successfully indexed.
    """
    client = SearchClient(_search_endpoint(), index_name, credential=_credential())

    documents = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in chunks]
    if not documents:
        logger.info("No chunks to upload")
        return 0

    succeeded = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        results = client.upload_documents(documents=batch)
        batch_ok = sum(1 for r in results if r.succeeded)
        succeeded += batch_ok
        logger.info(
            "Indexed batch %d-%d: %d/%d succeeded",
            start,
            start + len(batch),
            batch_ok,
            len(batch),
        )
        for r in results:
            if not r.succeeded:
                logger.warning("Failed to index %s: %s", r.key, r.error_message)

    logger.info("Indexed %d/%d documents into '%s'", succeeded, len(documents), index_name)
    return succeeded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_index()
    print(f"Index '{SEARCH_INDEX_NAME}' is ready.")
