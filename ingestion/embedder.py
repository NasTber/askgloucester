"""Generate vector embeddings for document chunks via Azure OpenAI.

Takes the :class:`chunker.Chunk` objects produced by the chunking stage and
populates each one's ``content_vector`` field with a 1536-dimension embedding
of its ``content``, using the ``text-embedding-3-small`` model.

Authentication uses ``DefaultAzureCredential`` (no API keys) — the same
pattern as the rest of the pipeline. The Azure OpenAI endpoint and the
embedding *deployment* name come from the environment / ``.env``:

    AZURE_OPENAI_ENDPOINT            e.g. https://<resource>.openai.azure.com
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT   the deployment name for the model

Chunks are sent in batches of 100 to stay within the embeddings API's
per-request input limits.
"""

from __future__ import annotations

import logging
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# Model + dimensions are fixed by the index schema (see indexer.py).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# The embeddings API accepts many inputs per call; 100 keeps us comfortably
# within request size/token limits while minimizing round trips.
DEFAULT_BATCH_SIZE = 100

# Scope used to request AAD tokens for Azure OpenAI ("Cognitive Services").
_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


def _client() -> AzureOpenAI:
    """Build an AzureOpenAI client authenticated with DefaultAzureCredential.

    ``get_bearer_token_provider`` wraps the credential in a callable that the
    SDK invokes to fetch (and refresh) AAD tokens — so no API key is needed.
    """
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), _TOKEN_SCOPE
    )
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        # API version that supports the v3 embedding models.
        api_version="2024-02-01",
    )


def embed_chunks(chunks, batch_size: int = DEFAULT_BATCH_SIZE):
    """Populate ``content_vector`` on each chunk in place.

    Args:
        chunks: Iterable of :class:`chunker.Chunk` objects to embed.
        batch_size: Number of chunk texts per embeddings request (max useful
            size is bounded by the API's input limit; defaults to 100).

    Returns:
        The same list of chunks, each with ``content_vector`` set.
    """
    chunks = list(chunks)
    if not chunks:
        logger.info("No chunks to embed")
        return chunks

    client = _client()
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        # The embeddings API takes a list of strings and returns one vector
        # per input, in the same order.
        response = client.embeddings.create(
            model=deployment,
            input=[c.content for c in batch],
            dimensions=EMBEDDING_DIMENSIONS,
        )
        for chunk, item in zip(batch, response.data):
            chunk.content_vector = item.embedding
        logger.info(
            "Embedded batch %d-%d (%d chunk(s))",
            start,
            start + len(batch),
            len(batch),
        )

    logger.info("Embedded %d chunk(s)", len(chunks))
    return chunks
