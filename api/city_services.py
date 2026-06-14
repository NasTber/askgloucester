"""Read-only Azure AI Search access for the Gloucester city-services index.

API-local and deliberately self-contained: this module builds its OWN
``SearchClient`` over ``DefaultAzureCredential`` and does NOT import anything from
``ingestion/`` — the deployed API image ships only ``api/`` (see the Dockerfile);
importing ``ingestion`` would crash-loop the container. The ingestion side
(``ingestion/city_services_source.py``) owns WRITES (a wipe-and-rebuild) to the
``gloucester-city-services`` index; this module owns the read path used by the
``city_services_search`` agent tool.

The query embedding REUSES :func:`api.query.embed` — the exact same Azure OpenAI
embedding path ``doc_search`` uses — so there is one embedding mechanism, not two.
Hybrid search (BM25 keyword + vector) mirrors :func:`api.query.retrieve`, minus
the OData filters (city services has no body/date/category scoping need yet).

Auth is ``DefaultAzureCredential`` (managed identity in Azure, developer creds
locally) — no account keys. No new RBAC is required: the API managed identity's
Search **Index Data Reader** role is service-scoped, so it already covers this
index alongside ``gloucester-documents``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# Reuse the documents retrieval module's config + the SINGLE embed path. Importing
# embed here (not a second embedder) is the point: the query is vectorized exactly
# as doc_search vectorizes its queries, against the same 1536-dim model.
from .query import (
    AZURE_SEARCH_ENDPOINT,
    _credential,
    _required,
    embed,
)

# Its own index on the SAME search service as the documents index. Env-overridable
# to match ingestion's CITY_SERVICES_INDEX_NAME.
CITY_SERVICES_INDEX_NAME = os.environ.get(
    "CITY_SERVICES_INDEX_NAME", "gloucester-city-services"
)

# Default number of chunks to return. The city-services corpus is small (a handful
# of service pages), so a modest k is plenty; a question rarely spans more than a
# couple of pages.
DEFAULT_K = 6

# Fields safe to return. content_vector is intentionally absent — it is not
# retrievable in the index schema, so requesting it would error. (No meeting_body/
# document_date/meeting_category exist on this index.)
SELECT_FIELDS = ["content", "source_url", "title", "service_category"]


@lru_cache(maxsize=1)
def _search_client() -> SearchClient:
    """Azure AI Search client for the city-services index (AAD auth).

    Built api-locally (not imported from ingestion) and cached. Read-only use; the
    index is assumed to already exist (created by the ingestion wipe-and-rebuild).
    """
    return SearchClient(
        endpoint=_required("AZURE_SEARCH_ENDPOINT", AZURE_SEARCH_ENDPOINT),
        index_name=CITY_SERVICES_INDEX_NAME,
        credential=_credential(),
    )


def search_city_services(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Hybrid (keyword + vector) search over the city-services index.

    Embeds ``query`` via :func:`api.query.embed` (the same path doc_search uses),
    then issues ONE hybrid request: ``search_text`` drives the BM25 keyword half
    and ``vector_queries`` the approximate-nearest-neighbour vector half, fused by
    Azure AI Search. No OData filters are applied — city services has no
    body/date/category scoping need yet. ``select`` restricts the returned fields
    to retrievable ones (never content_vector).

    Args:
        query: The resident's natural-language city-services question.
        k: Maximum chunks to return (and the vector k-nearest-neighbours).

    Returns:
        Up to ``k`` chunk dicts, each with ``content``, ``source_url``, ``title``
        and ``service_category``. An empty/blank query returns ``[]`` (no search).
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return []

    vector = embed(cleaned)
    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=k,
        fields="content_vector",
    )
    results = _search_client().search(
        search_text=cleaned,             # keyword (BM25) half of the hybrid query
        vector_queries=[vector_query],   # vector (ANN) half of the hybrid query
        select=SELECT_FIELDS,
        top=k,
    )
    # Materialise the lazy paged iterator into a plain list of dicts.
    return [dict(r) for r in results]
