"""Read-only Azure AI Search access for the Gloucester FAQ index.

API-local and deliberately self-contained: this module builds its OWN
``SearchClient`` over ``DefaultAzureCredential`` and does NOT import anything from
``ingestion/`` — the deployed API image ships only ``api/`` (see the Dockerfile);
importing ``ingestion`` would crash-loop the container. The ingestion side
(``ingestion/faq_source.py``) owns WRITES (a wipe-and-rebuild) to the
``gloucester-faq`` index; this module owns the read path used by the ``faq_search``
agent tool.

The query embedding REUSES :func:`api.query.embed` — the exact same Azure OpenAI
embedding path ``doc_search`` uses — so there is one embedding mechanism, not two.
Hybrid search (BM25 keyword + vector) mirrors :mod:`api.city_services`.

Auth is ``DefaultAzureCredential`` — no account keys. No new RBAC: the API managed
identity's Search **Index Data Reader** role is service-scoped, so it already
covers this index alongside ``gloucester-documents`` and ``gloucester-city-services``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# Reuse the documents retrieval module's config + the SINGLE embed path, exactly
# as api.city_services does — one embedding mechanism, against the same 1536-dim
# model, for every index.
from .query import (
    AZURE_SEARCH_ENDPOINT,
    _credential,
    _required,
    embed,
)

# Its own index on the SAME search service. Env-overridable to match ingestion's
# FAQ_INDEX_NAME.
FAQ_INDEX_NAME = os.environ.get("FAQ_INDEX_NAME", "gloucester-faq")

# Default number of chunks to return. The FAQ has ~174 short Q&As; a handful of
# the closest answers is plenty for a "quick published answer" tool.
DEFAULT_K = 6

# Fields safe to return. content_vector is intentionally absent — it is not
# retrievable in the index schema, so requesting it would error.
SELECT_FIELDS = ["content", "source_url", "title", "category"]


@lru_cache(maxsize=1)
def _search_client() -> SearchClient:
    """Azure AI Search client for the FAQ index (AAD auth).

    Built api-locally (not imported from ingestion) and cached. Read-only use; the
    index is assumed to already exist (created by the ingestion wipe-and-rebuild).
    """
    return SearchClient(
        endpoint=_required("AZURE_SEARCH_ENDPOINT", AZURE_SEARCH_ENDPOINT),
        index_name=FAQ_INDEX_NAME,
        credential=_credential(),
    )


def search_faq(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Hybrid (keyword + vector) search over the FAQ index.

    Embeds ``query`` via :func:`api.query.embed` (the same path doc_search uses),
    then issues ONE hybrid request: ``search_text`` drives the BM25 keyword half
    and ``vector_queries`` the approximate-nearest-neighbour vector half, fused by
    Azure AI Search. No OData filters. ``select`` restricts returned fields to
    retrievable ones (never content_vector).

    Args:
        query: The resident's natural-language question.
        k: Maximum chunks to return (and the vector k-nearest-neighbours).

    Returns:
        Up to ``k`` chunk dicts, each with ``content``, ``source_url``, ``title``
        (the FAQ question) and ``category``. An empty/blank query returns ``[]``.
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
    return [dict(r) for r in results]
