"""Generate vector embeddings for document chunks via Azure OpenAI.

Takes the :class:`chunker.Chunk` objects produced by the chunking stage and
populates each one's ``content_vector`` field with a 1536-dimension embedding
of its ``content``, using the ``text-embedding-3-small`` model.

Authentication uses ``DefaultAzureCredential`` (no API keys) — the same
pattern as the rest of the pipeline. The Azure OpenAI endpoint and the
embedding *deployment* name come from the environment / ``.env``:

    AZURE_OPENAI_ENDPOINT            e.g. https://<resource>.openai.azure.com
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT   the deployment name for the model

Chunks are sent in batches (one request per ~100 chunks) so a full reindex is a
handful of round trips instead of one call per chunk. Each batch is bounded by
BOTH a max item count and a token budget, to stay under the embeddings API's
per-request limits (see below); vectors are mapped back to chunks by the
response's ``index`` so order can never drift.
"""

from __future__ import annotations

import logging
import os

import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# Model + dimensions are fixed by the index schema (see indexer.py).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Azure OpenAI embeddings per-request limits (text-embedding-3 family):
#   * up to 2048 inputs per request,
#   * up to 300,000 tokens summed across all inputs (HTTP 400 if exceeded),
#   * up to 8192 tokens per single input (our ~500-token chunks never near this).
# We cap each request at ~100 items AND keep the summed tokens under a safety
# margin below the 300k hard cap, so a batch of unusually large chunks splits
# earlier instead of erroring.
DEFAULT_BATCH_SIZE = 100
MAX_REQUEST_TOKENS = 280_000

# text-embedding-3 tokenizes with cl100k_base; use it to budget each request.
_ENCODING = tiktoken.get_encoding("cl100k_base")

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


def _batched(chunks, max_items: int, max_tokens: int):
    """Yield lists of chunks, each within ``max_items`` AND ``max_tokens``.

    Greedily packs chunks in their original order; flushes before a chunk would
    push the batch past either cap. A single chunk always forms at least a
    one-item batch (our chunks are ~500 tokens, far under the per-input limit),
    so this never loops forever on an oversized input.
    """
    batch: list = []
    batch_tokens = 0
    for chunk in chunks:
        tokens = len(_ENCODING.encode(chunk.content))
        if batch and (len(batch) >= max_items or batch_tokens + tokens > max_tokens):
            yield batch
            batch, batch_tokens = [], 0
        batch.append(chunk)
        batch_tokens += tokens
    if batch:
        yield batch


def embed_chunks(chunks, batch_size: int = DEFAULT_BATCH_SIZE):
    """Populate ``content_vector`` on each chunk in place.

    Chunks are embedded in batches — at most ``batch_size`` items and
    ``MAX_REQUEST_TOKENS`` tokens per request — so a full reindex is a few dozen
    calls rather than thousands. Each returned embedding is written back to its
    chunk by the response item's ``index`` (not positional zip), so a chunk can
    never receive another chunk's vector even if the API reorders ``data``.

    Args:
        chunks: Iterable of :class:`chunker.Chunk` objects to embed.
        batch_size: Max chunk texts per embeddings request (default 100). The
            per-request token budget (:data:`MAX_REQUEST_TOKENS`) may flush a
            batch earlier when chunks are unusually large.

    Returns:
        The same list of chunks, each with ``content_vector`` set.
    """
    chunks = list(chunks)
    if not chunks:
        logger.info("No chunks to embed")
        return chunks

    client = _client()
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    done = 0
    for batch in _batched(chunks, max_items=batch_size, max_tokens=MAX_REQUEST_TOKENS):
        # The embeddings API takes a list of strings and returns one vector per
        # input; we map each result back by its `index` to preserve alignment.
        response = client.embeddings.create(
            model=deployment,
            input=[c.content for c in batch],
            dimensions=EMBEDDING_DIMENSIONS,
        )
        for item in response.data:
            batch[item.index].content_vector = item.embedding
        done += len(batch)
        logger.info("Embedded %d/%d chunk(s)", done, len(chunks))

    logger.info("Embedded %d chunk(s)", len(chunks))
    return chunks
