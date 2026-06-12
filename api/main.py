"""FastAPI app for AskGloucester.

A thin HTTP layer over the RAG loop in ``query.ask``. The endpoint owns no
retrieval or generation logic of its own — it validates input, calls ``ask``,
and shapes the result into JSON. Keeping ``ask`` as the single source of truth
means the API and the CLI can never answer the same question differently
(including the deterministic "no documents" decline, which ``ask`` returns with
an empty source list).

``GET /`` serves a single self-contained HTML page (inline CSS + JS, no static
file directory) that calls ``POST /ask`` on the same origin — a minimal UI for
residents, kept in this file so the app stays a single deployable unit.

Run locally:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Import the module (not the bare name) so the shared RAG entry point and its
# Azure clients are initialised exactly once, lazily, on first request.
from . import query

app = FastAPI(
    title="AskGloucester",
    description="Civic AI assistant answering questions about Gloucester, MA municipal documents.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    """Body for POST /ask."""

    # min_length=1 rejects empty/whitespace-only questions at the edge so we
    # never spend an embedding call on nothing.
    question: str = Field(..., min_length=1, description="The question to ask.")


class Source(BaseModel):
    """One retrieved chunk, surfaced so the client can show provenance.

    Field names and order mirror the citation numbering produced by
    ``build_context`` — the Nth item here is what the answer cites as [N].
    """

    # Citation number matching the [n] markers in the answer text; 1-based.
    n: int
    meeting_body: str | None = None
    document_type: str | None = None
    document_date: str | None = None
    page_number: int | None = None
    source_url: str | None = None
    chunk_id: str | None = None


class AskResponse(BaseModel):
    """Response for POST /ask."""

    answer: str
    # Empty whenever the answer is a decline (no matching documents) — the same
    # signal ``ask`` gives its callers.
    sources: list[Source]


def _to_source(chunk: dict, n: int) -> Source:
    """Project a raw search chunk down to the public Source shape.

    Only whitelisted fields are exposed; ``content`` and any vector data stay
    server-side. Going through the Pydantic model drops anything unexpected.

    ``n`` is the 1-based citation number matching the [n] markers in the answer.
    """
    return Source(
        n=n,
        meeting_body=chunk.get("meeting_body"),
        document_type=chunk.get("document_type"),
        document_date=chunk.get("document_date"),
        page_number=chunk.get("page_number"),
        source_url=chunk.get("source_url"),
        chunk_id=chunk.get("chunk_id"),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page resident UI (inline HTML/CSS/JS, no static dir)."""
    return INDEX_HTML


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Container Apps — no Azure calls, always cheap."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """Answer a question about Gloucester civic documents via RAG.

    Delegates the whole retrieval-and-generation pass to ``query.ask`` and only
    reshapes its (answer, chunks) tuple into JSON.
    """
    answer_text, chunks = query.ask(request.question)
    return AskResponse(
        answer=answer_text,
        sources=[_to_source(c, n) for n, c in chunks],
    )


# The whole resident UI as one inline document: markup, plain CSS and vanilla JS
# with zero external requests, so it loads fast and works fully offline. The JS
# talks to POST /ask on the same origin and builds the DOM with textContent /
# createElement (never innerHTML on model output), so an answer or source field
# can never inject markup into the page.
INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AskGloucester</title>
  <style>
    :root {
      --ink: #1a2733;
      --muted: #5c6b7a;
      --accent: #154d7a;
      --accent-press: #0f3a5c;
      --line: #d8e0e6;
      --bg: #f4f6f8;
      --error: #9b2226;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }
    main {
      max-width: 640px;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 4rem;
    }
    header { border-bottom: 2px solid var(--accent); padding-bottom: 1rem; margin-bottom: 1.75rem; }
    h1 { margin: 0; font-size: 1.9rem; letter-spacing: -0.01em; color: var(--accent); }
    .subtitle { margin: 0.35rem 0 0; color: var(--muted); font-size: 1rem; }
    form { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    #question {
      flex: 1 1 14rem;
      min-width: 0;
      padding: 0.7rem 0.85rem;
      font-size: 1rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
    }
    #question:focus { outline: 2px solid var(--accent); outline-offset: 1px; border-color: var(--accent); }
    button {
      flex: 0 0 auto;
      padding: 0.7rem 1.25rem;
      font-size: 1rem;
      font-weight: 600;
      color: #fff;
      background: var(--accent);
      border: 0;
      border-radius: 8px;
      cursor: pointer;
    }
    button:hover { background: var(--accent-press); }
    button:disabled { opacity: 0.6; cursor: progress; }
    #status { margin-top: 1.5rem; color: var(--muted); display: none; }
    #status.show { display: flex; align-items: center; gap: 0.6rem; }
    .spinner {
      width: 1.1rem; height: 1.1rem;
      border: 2px solid var(--line);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #error { margin-top: 1.5rem; color: var(--error); display: none; }
    #error.show { display: block; }
    #result { margin-top: 1.75rem; display: none; }
    #result.show { display: block; }
    #answer { white-space: pre-wrap; }
    h2 { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin: 1.75rem 0 0.5rem; }
    ol { padding-left: 1.4rem; margin: 0; }
    li { margin-bottom: 0.6rem; }
    li .meta { color: var(--ink); }
    li a { color: var(--accent); word-break: break-word; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>AskGloucester</h1>
      <p class="subtitle">Ask questions about Gloucester, MA public meetings</p>
    </header>

    <form id="ask-form">
      <input id="question" name="question" type="text" autocomplete="off"
             placeholder="e.g. What did the School Committee discuss last meeting?" required>
      <button id="submit" type="submit">Ask</button>
    </form>

    <div id="status"><span class="spinner"></span><span>Searching the records…</span></div>
    <p id="error"></p>

    <section id="result">
      <p id="answer"></p>
      <h2 id="sources-heading" style="display:none">Sources</h2>
      <ol id="sources"></ol>
    </section>
  </main>

  <script>
    const form = document.getElementById("ask-form");
    const input = document.getElementById("question");
    const button = document.getElementById("submit");
    const statusEl = document.getElementById("status");
    const errorEl = document.getElementById("error");
    const result = document.getElementById("result");
    const answerEl = document.getElementById("answer");
    const sourcesHeading = document.getElementById("sources-heading");
    const sourcesEl = document.getElementById("sources");

    function showError(message) {
      errorEl.textContent = message;
      errorEl.classList.add("show");
    }

    // Build "Body — type — date, p.N" from whichever fields are present.
    function sourceLabel(s) {
      const parts = [s.meeting_body, s.document_type, s.document_date].filter(Boolean);
      let label = parts.join(" — ") || "Source";
      if (s.page_number != null) label += ", p." + s.page_number;
      return label;
    }

    function renderSources(sources) {
      sourcesEl.replaceChildren();
      if (!sources || sources.length === 0) {
        sourcesHeading.style.display = "none";
        return;
      }
      sourcesHeading.style.display = "";
      for (const s of sources) {
        const li = document.createElement("li");
        if (s.source_url) {
          const a = document.createElement("a");
          a.href = s.source_url;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.textContent = sourceLabel(s);
          li.appendChild(a);
        } else {
          const span = document.createElement("span");
          span.className = "meta";
          span.textContent = sourceLabel(s);
          li.appendChild(span);
        }
        sourcesEl.appendChild(li);
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;

      // Reset to a clean loading state before each request.
      button.disabled = true;
      statusEl.classList.add("show");
      errorEl.classList.remove("show");
      result.classList.remove("show");

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });
        if (!response.ok) {
          throw new Error("The server returned an error (" + response.status + "). Please try again.");
        }
        const data = await response.json();
        answerEl.textContent = data.answer || "";
        renderSources(data.sources);
        result.classList.add("show");
      } catch (err) {
        // Covers both non-OK responses and network failures (fetch rejecting).
        showError(err instanceof TypeError
          ? "Could not reach the server. Check your connection and try again."
          : err.message);
      } finally {
        button.disabled = false;
        statusEl.classList.remove("show");
      }
    });
  </script>
</body>
</html>
"""
