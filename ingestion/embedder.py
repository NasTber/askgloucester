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
import threading
import time

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

# Tokens-per-minute pacing.
#
# The embedding deployment's quota is 350K TPM. Without pacing the pipeline fires
# batches back-to-back, briefly exceeds the minute budget, and the service
# answers 429 with a ~54s Retry-After — the SDK then sleeps the whole minute,
# producing a sawtooth. We instead pace ourselves to ~90% of the limit so the
# cumulative tokens/min stay safely under quota and the 429 path rarely fires.
#
# The MAX_REQUEST_TOKENS per-batch cap (280K) is unchanged: it bounds a single
# request's size; the bucket below governs the *spacing between* requests. The
# budget (315K) exceeds that cap, so any single batch can always be admitted.
EMBEDDING_TPM_LIMIT = 350_000
EMBEDDING_TPM_BUDGET = int(EMBEDDING_TPM_LIMIT * 0.9)  # ~315K tokens/min


class _TokenBucket:
    """A continuously-refilling token bucket for tokens-per-minute pacing.

    Capacity and refill are both ``rate_per_min``: the bucket holds at most one
    minute's budget and refills at ``rate_per_min / 60`` tokens per second. That
    permits an initial burst up to a full minute's budget, after which steady-
    state throughput converges to ``rate_per_min`` tokens/min — keeping any
    rolling minute under quota. ``acquire`` blocks until the request fits.

    Thread-safe via a lock; the pipeline is single-threaded today but the lock
    keeps the bucket correct (and cheap) regardless.
    """

    def __init__(self, rate_per_min: int) -> None:
        self._rate_per_sec = rate_per_min / 60.0
        self._capacity = float(rate_per_min)
        self._tokens = float(rate_per_min)  # start full: first batch isn't delayed
        self._timestamp = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, amount: int) -> float:
        """Block until ``amount`` tokens are available, then deduct them.

        Returns the seconds spent waiting (0.0 when the budget was immediately
        available) so the caller can log pacing if it wants. ``amount`` larger
        than capacity is clamped so it can still be admitted after a full refill
        rather than blocking forever.
        """
        amount = min(amount, int(self._capacity))
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                # Refill for the elapsed interval, capped at capacity.
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._timestamp) * self._rate_per_sec,
                )
                self._timestamp = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return waited
                # Sleep just long enough for the deficit to refill, then re-check.
                deficit = amount - self._tokens
                wait = deficit / self._rate_per_sec
            waited += wait
            time.sleep(wait)


# Module-level bucket: shared across embed_chunks calls in a process so pacing
# carries over (e.g. if the pipeline embeds in several passes).
_rate_limiter = _TokenBucket(EMBEDDING_TPM_BUDGET)


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
    """Yield ``(batch, batch_tokens)``, each within ``max_items`` AND ``max_tokens``.

    Greedily packs chunks in their original order; flushes before a chunk would
    push the batch past either cap. A single chunk always forms at least a
    one-item batch (our chunks are ~500 tokens, far under the per-input limit),
    so this never loops forever on an oversized input. The summed tiktoken count
    is yielded alongside the batch so the caller can pace on it without
    re-tokenizing.
    """
    batch: list = []
    batch_tokens = 0
    for chunk in chunks:
        tokens = len(_ENCODING.encode(chunk.content))
        if batch and (len(batch) >= max_items or batch_tokens + tokens > max_tokens):
            yield batch, batch_tokens
            batch, batch_tokens = [], 0
        batch.append(chunk)
        batch_tokens += tokens
    if batch:
        yield batch, batch_tokens


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
    for batch, batch_tokens in _batched(chunks, max_items=batch_size, max_tokens=MAX_REQUEST_TOKENS):
        # Pace to stay under the embedding TPM quota: block until this batch's
        # token cost fits the bucket. This keeps cumulative tokens/min under
        # budget so the 429 -> ~54s Retry-After sleep rarely fires. The SDK's
        # built-in 429 retry/backoff remains as a fallback for any overshoot.
        waited = _rate_limiter.acquire(batch_tokens)
        if waited:
            logger.debug("Paced batch of %d tokens: waited %.2fs", batch_tokens, waited)

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
