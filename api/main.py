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

from typing import Literal

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


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    """Body for POST /ask."""

    # min_length=1 rejects empty/whitespace-only questions at the edge so we
    # never spend an embedding call on nothing.
    question: str = Field(..., min_length=1, description="The question to ask.")
    history: list[Message] = Field(default_factory=list,
        description="Prior turns, alternating user/assistant.")


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
    history = [{"role": m.role, "content": m.content} for m in request.history]
    answer_text, chunks = query.ask(request.question, history=history or None)
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
      --ink: #1a2733; --muted: #5c6b7a; --accent: #154d7a;
      --accent-press: #0f3a5c; --line: #d8e0e6; --bg: #f4f6f8;
      --error: #9b2226;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; }
    body {
      display: flex; flex-direction: column;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   Helvetica, Arial, sans-serif;
      color: var(--ink); background: var(--bg); line-height: 1.55;
    }
    header {
      flex-shrink: 0; padding: 0.9rem 1.25rem;
      background: #fff; border-bottom: 2px solid var(--accent);
    }
    header h1 { font-size: 1.4rem; color: var(--accent); }
    header p  { font-size: 0.88rem; color: var(--muted); margin-top: 0.1rem; }
    #thread {
      flex: 1 1 0; overflow-y: auto; padding: 1.25rem;
      display: flex; flex-direction: column; gap: 1rem;
      max-width: 700px; width: 100%; margin: 0 auto;
    }
    #welcome {
      margin: auto; text-align: center;
      color: var(--muted); padding: 2rem; font-size: 0.97rem;
    }
    .msg { display: flex; flex-direction: column; max-width: 85%; }
    .msg.user      { align-self: flex-end;   align-items: flex-end; }
    .msg.assistant { align-self: flex-start; align-items: flex-start; }
    .bubble {
      padding: 0.65rem 0.95rem; border-radius: 14px; font-size: 0.96rem;
    }
    .msg.user .bubble {
      background: var(--accent); color: #fff; border-bottom-right-radius: 3px;
    }
    .msg.assistant .bubble {
      background: #fff; border: 1px solid var(--line);
      border-bottom-left-radius: 3px; white-space: pre-wrap;
    }
    .sources { margin-top: 0.45rem; }
    .src-toggle {
      background: none; border: none; cursor: pointer;
      color: var(--accent); font-size: 0.82rem; padding: 0;
    }
    .src-toggle:hover { text-decoration: underline; }
    .src-list {
      display: none; margin-top: 0.3rem; padding-left: 1.1rem;
      font-size: 0.82rem; color: var(--muted);
    }
    .src-list.open { display: block; }
    .src-list li { margin-bottom: 0.3rem; }
    .src-list a { color: var(--accent); word-break: break-word; }
    .typing {
      align-self: flex-start; display: flex; gap: 5px;
      padding: 0.75rem 1rem; background: #fff;
      border: 1px solid var(--line); border-radius: 14px;
      border-bottom-left-radius: 3px;
    }
    .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--muted); animation: bounce 1.2s infinite;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%,60%,100% { transform: translateY(0); }
      30%          { transform: translateY(-5px); }
    }
    .err-bubble {
      background: #fff5f5; border: 1px solid #f5c6c6;
      color: var(--error); padding: 0.65rem 0.95rem;
      border-radius: 14px; font-size: 0.95rem;
    }
    #input-bar {
      flex-shrink: 0; padding: 0.8rem 1.25rem;
      background: #fff; border-top: 1px solid var(--line);
    }
    #input-inner {
      display: flex; gap: 0.5rem;
      max-width: 700px; margin: 0 auto;
    }
    #question {
      flex: 1; padding: 0.65rem 0.85rem; font-size: 0.97rem;
      border: 1px solid var(--line); border-radius: 8px;
      background: var(--bg); color: var(--ink);
    }
    #question:focus {
      outline: 2px solid var(--accent); outline-offset: 1px;
      border-color: var(--accent);
    }
    #submit {
      padding: 0.65rem 1.1rem; font-size: 0.97rem; font-weight: 600;
      color: #fff; background: var(--accent); border: 0;
      border-radius: 8px; cursor: pointer; white-space: nowrap;
    }
    #submit:hover    { background: var(--accent-press); }
    #submit:disabled { opacity: 0.6; cursor: progress; }
  </style>
</head>
<body>
  <header>
    <h1>AskGloucester</h1>
    <p>Ask questions about Gloucester, MA public meetings</p>
  </header>

  <div id="thread">
    <div id="welcome">
      Ask anything about City Council, School Committee, or other
      Gloucester municipal meetings.
    </div>
  </div>

  <div id="input-bar">
    <div id="input-inner">
      <input id="question" type="text" autocomplete="off"
             placeholder="What was discussed at the last City Council meeting?">
      <button id="submit">Ask</button>
    </div>
  </div>

  <script>
    const thread   = document.getElementById("thread");
    const welcome  = document.getElementById("welcome");
    const input    = document.getElementById("question");
    const submit   = document.getElementById("submit");

    let history = [];

    function srcLabel(s) {
      const p = [s.meeting_body, s.document_type, s.document_date].filter(Boolean);
      let t = p.join(" \u2014 ") || "Source";
      if (s.page_number != null) t += ", p." + s.page_number;
      return t;
    }

    function addUser(text) {
      welcome.style.display = "none";
      const d = document.createElement("div");
      d.className = "msg user";
      const b = document.createElement("div");
      b.className = "bubble";
      b.textContent = text;
      d.appendChild(b);
      thread.appendChild(d);
      thread.scrollTop = thread.scrollHeight;
    }

    function addTyping() {
      const d = document.createElement("div");
      d.className = "typing";
      for (let i = 0; i < 3; i++) {
        const dot = document.createElement("div");
        dot.className = "dot";
        d.appendChild(dot);
      }
      thread.appendChild(d);
      thread.scrollTop = thread.scrollHeight;
      return d;
    }

    function addAssistant(text, sources) {
      const d = document.createElement("div");
      d.className = "msg assistant";

      const b = document.createElement("div");
      b.className = "bubble";
      b.textContent = text;
      d.appendChild(b);

      if (sources && sources.length) {
        const wrap   = document.createElement("div");
        wrap.className = "sources";
        const toggle = document.createElement("button");
        toggle.className = "src-toggle";
        toggle.textContent = "\u25b8 " + sources.length +
          " source" + (sources.length === 1 ? "" : "s");
        const list = document.createElement("ol");
        list.className = "src-list";
        sources.forEach(s => {
          const li = document.createElement("li");
          if (s.source_url) {
            const a = document.createElement("a");
            a.href = s.source_url; a.target = "_blank";
            a.rel = "noopener noreferrer";
            a.textContent = srcLabel(s);
            li.appendChild(a);
          } else {
            li.textContent = srcLabel(s);
          }
          list.appendChild(li);
        });
        toggle.addEventListener("click", () => {
          const open = list.classList.toggle("open");
          toggle.textContent = (open ? "\u25be " : "\u25b8 ") +
            sources.length + " source" + (sources.length === 1 ? "" : "s");
        });
        wrap.appendChild(toggle); wrap.appendChild(list);
        d.appendChild(wrap);
      }

      thread.appendChild(d);
      thread.scrollTop = thread.scrollHeight;
    }

    function addError(msg) {
      const d = document.createElement("div");
      d.className = "msg assistant";
      const b = document.createElement("div");
      b.className = "err-bubble";
      b.textContent = msg;
      d.appendChild(b);
      thread.appendChild(d);
      thread.scrollTop = thread.scrollHeight;
    }

    async function send() {
      const q = input.value.trim();
      if (!q) return;
      input.value = "";
      submit.disabled = true;
      addUser(q);
      const typing = addTyping();
      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ question: q, history }),
        });
        typing.remove();
        if (!res.ok) throw new Error("Server error (" + res.status + "). Please try again.");
        const data = await res.json();
        addAssistant(data.answer, data.sources);
        history.push({role: "user",      content: q});
        history.push({role: "assistant", content: data.answer});
      } catch(e) {
        typing.remove();
        addError(e instanceof TypeError
          ? "Could not reach the server. Check your connection and try again."
          : e.message);
      } finally {
        submit.disabled = false;
        input.focus();
      }
    }

    submit.addEventListener("click", send);
    input.addEventListener("keydown", e => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });
    input.focus();
  </script>
</body>
</html>
"""
