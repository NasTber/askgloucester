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

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

# Import the module (not the bare name) so the shared RAG entry point and its
# Azure clients are initialised exactly once, lazily, on first request.
from . import query

app = FastAPI(
    title="AskGloucester",
    description="Civic AI assistant answering questions about Gloucester, MA municipal documents.",
    version="0.1.0",
)


# --- Rate limiting for POST /ask ------------------------------------------
# Applies to /ask only — GET / and GET /health are left unlimited (rate-limiting
# the liveness probe would restart-loop the container). Two independent limits,
# named here so they're easy to tune. Storage is slowapi's default in-memory
# backend, which is authoritative because Container Apps runs a single replica
# (max-replicas=1) — no external store needed.
PER_IP_RATE_LIMIT = "10/minute"   # per real client IP
GLOBAL_RATE_LIMIT = "30/minute"   # all IPs combined
# Constant key that buckets every /ask request together for the global cap.
_GLOBAL_BUCKET_KEY = "__all__"
# Both limits use a 60s window, so a flat Retry-After is accurate enough.
_RETRY_AFTER_SECONDS = "60"
_RATE_LIMIT_MESSAGE = (
    "You're sending requests too quickly — please wait a moment and try again."
)


def _client_ip(request: Request) -> str:
    """Per-IP rate-limit key: the *real* client IP.

    Container Apps' ingress is a reverse proxy, so ``request.client.host`` is the
    ingress IP — keying on it would throttle every user as one bucket. Read the
    left-most entry of ``X-Forwarded-For`` (the originating client) and fall back
    to the socket peer only when the header is absent.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Default key_func is the per-IP function above; the global limit overrides it
# with a constant key on its own decorator. Default storage is memory://.
limiter = Limiter(key_func=_client_ip)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Friendly 429 for either limit: small JSON body + Retry-After header."""
    return JSONResponse(
        status_code=429,
        content={"detail": _RATE_LIMIT_MESSAGE},
        headers={"Retry-After": _RETRY_AFTER_SECONDS},
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


# Two stacked limits: the inner decorator caps each IP (default key_func), the
# outer one caps all traffic combined via the constant bucket key. Exceeding
# either raises RateLimitExceeded -> _rate_limit_handler (429). slowapi requires
# the `request: Request` parameter; the JSON body is `body: AskRequest`.
@app.post("/ask", response_model=AskResponse)
@limiter.limit(GLOBAL_RATE_LIMIT, key_func=lambda request: _GLOBAL_BUCKET_KEY)
@limiter.limit(PER_IP_RATE_LIMIT)
def ask_endpoint(request: Request, body: AskRequest) -> AskResponse:
    """Answer a question about Gloucester civic documents via RAG.

    Delegates the whole retrieval-and-generation pass to ``query.ask`` and only
    reshapes its (answer, chunks) tuple into JSON.
    """
    history = [{"role": m.role, "content": m.content} for m in body.history]
    answer_text, chunks = query.ask(body.question, history=history or None)
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
    .beta {
      display: inline-block;
      margin-top: 0.4rem;
      font-size: 0.78rem;
      color: #fff;
      background: var(--accent);
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      letter-spacing: 0.03em;
    }
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
    /* Inline citation chips: [n] in the answer is rendered as a small
       superscript link to its source. */
    .cite {
      font-size: 0.68em; vertical-align: super; line-height: 0;
      font-weight: 600; color: var(--accent); text-decoration: none;
      background: #e7eef4; border-radius: 4px;
      padding: 0.05em 0.32em; margin: 0 0.08em;
    }
    a.cite:hover { text-decoration: underline; }
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
    <p class="beta">Beta · Document Q&amp;A: School Committee &amp; City Council ·
  Meeting schedules: city-wide · Answers may be incomplete</p>
  </header>

  <div id="thread">
    <div id="welcome">
      Ask what was discussed or decided at City Council and School Committee
      meetings, or when any Gloucester public body meets.
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

    // Render the answer into `bubble`, turning numeric [n] markers into small
    // superscript citation chips that link to source n. Pure display transform:
    // the [n] stay in the model output / API response; we only change how they
    // look. Built with text nodes + createElement (never innerHTML on model
    // output), so an answer can't inject markup. Non-numeric bracketed tokens
    // (e.g. a stray [toolname]) are left untouched as plain text.
    function renderAnswer(bubble, text, sources) {
      const byN = {};
      (sources || []).forEach(s => { byN[s.n] = s; });
      // Capturing split: even indices are literal text, odd indices are the
      // digits captured from a [<digits>] token.
      const parts = String(text).split(/\\[(\\d+)\\]/);
      parts.forEach((part, i) => {
        if (i % 2 === 1) {
          const n = parseInt(part, 10);
          const s = byN[n];
          const chip = (s && s.source_url)
            ? document.createElement("a")
            : document.createElement("span");
          chip.className = "cite";
          chip.textContent = n;                 // shows "1", not "[1]"
          chip.title = s ? srcLabel(s) : ("Source " + n);
          if (s && s.source_url) {
            chip.href = s.source_url;
            chip.target = "_blank";
            chip.rel = "noopener noreferrer";
          }
          bubble.appendChild(chip);
        } else if (part) {
          bubble.appendChild(document.createTextNode(part));
        }
      });
    }

    function addAssistant(text, sources) {
      const d = document.createElement("div");
      d.className = "msg assistant";

      const b = document.createElement("div");
      b.className = "bubble";
      renderAnswer(b, text, sources);
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
        // Rate limited (per-IP or global cap on /ask): show the server's
        // friendly message in the thread rather than the generic error path.
        if (res.status === 429) {
          addError("You're sending requests too quickly — please wait a moment and try again.");
          return;
        }
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
