"""Create/update the Azure AI Search index and upload document chunks.

Defines the ``gloucester-documents`` index schema and provides a batched upload
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
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "gloucester-documents")

# Azure AI Search caps a single indexing batch at 1000 documents.
DEFAULT_BATCH_SIZE = 1000

# Names tying the vector field to its algorithm/profile and the semantic
# ranking config together. text-embedding-3-small produces 1536-dim vectors.
VECTOR_DIMENSIONS = 1536
HNSW_CONFIG_NAME = "hnsw-config"
VECTOR_PROFILE_NAME = "vector-profile"
SEMANTIC_CONFIG_NAME = "semantic-config"


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
        # Vector field holding the chunk's embedding. It's searchable (via the
        # vector profile) but not retrievable to keep result payloads small.
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=False,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    # Vector search: an HNSW algorithm config referenced by a profile, which
    # the content_vector field points at.
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=HNSW_CONFIG_NAME)],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=HNSW_CONFIG_NAME,
            )
        ],
    )

    # Semantic ranking over the chunk text, for hybrid keyword+vector queries.
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def ensure_index(index_name: str = SEARCH_INDEX_NAME) -> None:
    """(Re)create the index so it matches the current schema.

    Adding the ``content_vector`` field, vector search and semantic configs is
    not an in-place-updatable schema change, so we delete any existing index
    and recreate it from scratch. This drops previously indexed documents — the
    pipeline re-uploads them after this runs.
    """
    client = SearchIndexClient(_search_endpoint(), credential=_credential())
    index = _build_index(index_name)
    try:
        client.get_index(index_name)
        logger.info("Deleting existing index '%s' to apply schema change", index_name)
        client.delete_index(index_name)
    except ResourceNotFoundError:
        logger.info("No existing index '%s'", index_name)
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
