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
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AskGloucester</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root{
      --navy:#1E3E80; --navy-deep:#162F61; --navy-soft:#E9EEF7;
      --ink:#1C2A3F; --ink-soft:#43536A; --page:#F2F5FA; --surface:#FFFFFF;
      --accent:#2C66AB; --maroon:#581824; --maroon-deep:#43121B;
      --maroon-soft:#F4E8EB; --maroon-line:#E4CDD2; --muted:#5C6776;
      --hairline:#DBE2EE;
    }
    *{box-sizing:border-box;}
    html,body{margin:0;padding:0;}
    body{
      font-family:'Public Sans',system-ui,sans-serif;
      color:var(--ink); background:var(--page); line-height:1.6;
      -webkit-font-smoothing:antialiased;
      min-height:100vh; display:flex; flex-direction:column;
    }

    header{ background:var(--navy); color:#F6F8FC; border-bottom:3px solid var(--navy-deep); }
    .bar{ max-width:860px; margin:0 auto; padding:16px 24px; display:flex; align-items:center; gap:14px; }
    .mark{ width:30px; height:30px; flex:0 0 auto; color:#9FB6DD; }
    .wordmark{ font-family:'Fraunces',serif; font-weight:600; font-size:24px; letter-spacing:0.2px; line-height:1; color:#F6F8FC; }
    .tag{ font-size:11px; font-weight:700; letter-spacing:0.6px; text-transform:uppercase; color:var(--navy); background:#fff; border:none; padding:3px 9px; border-radius:999px; white-space:nowrap; }

    main{ flex:1; width:100%; max-width:860px; margin:0 auto; padding:28px 24px 8px; }
    .welcome{ color:var(--muted); font-size:15px; text-align:center; padding:40px 12px; max-width:560px; margin:6px auto 0; }
    .turn{ margin-bottom:22px; }
    .who{ font-size:12px; letter-spacing:0.5px; text-transform:uppercase; color:var(--muted); margin:0 0 7px 2px; font-weight:600; }
    .user .bubble{ background:var(--navy-soft); border:1px solid #D4DEF0; border-left:3px solid var(--navy); color:var(--ink); border-radius:12px; padding:13px 16px; font-size:16px; max-width:80%; }
    .bot .card{ background:var(--surface); border:1px solid var(--hairline); border-radius:14px; padding:18px 20px; font-size:16px; box-shadow:0 1px 0 rgba(28,42,63,0.03); white-space:pre-wrap; }
    .bot .card.error{ border-color:var(--maroon-line); background:var(--maroon-soft); color:var(--maroon-deep); }
    .bot p{ margin:0 0 12px; }
    .bot ul{ margin:0 0 12px; padding-left:18px; }
    .bot li{ margin:0 0 7px; }
    .bot p:last-child,.bot ul:last-child{ margin-bottom:0; }
    .bot .card a.link{color:var(--accent);text-decoration:underline;text-underline-offset:2px;word-break:break-word;}
    .bot .card a.link:hover{color:var(--navy-deep);}

    .cite{
      display:inline-flex; align-items:center; justify-content:center;
      min-width:18px; height:18px; padding:0 5px; margin:0 1px; vertical-align:1px;
      font-size:11px; font-weight:600; color:var(--maroon);
      background:var(--maroon-soft); border:1px solid var(--maroon-line);
      border-radius:5px; text-decoration:none; cursor:pointer;
      transition:background .12s,color .12s,border-color .12s;
    }
    .cite:hover{ background:var(--maroon); color:#fff; border-color:var(--maroon); }

    details.sources{ margin-top:14px; border-top:1px solid var(--hairline); padding-top:11px; }
    details.sources summary{ cursor:pointer; list-style:none; font-size:13px; font-weight:600; color:var(--accent); display:flex; align-items:center; gap:6px; }
    details.sources summary::-webkit-details-marker{ display:none; }
    details.sources summary .chev{ transition:transform .15s; font-size:13px; }
    details.sources[open] summary .chev{ transform:rotate(90deg); }
    .src{ display:flex; gap:10px; align-items:baseline; font-size:13.5px; color:var(--ink-soft); padding:8px 0 0; }
    .src .n{ flex:0 0 auto; font-weight:600; color:var(--maroon); font-size:12px; min-width:16px; }
    .src a{ color:var(--accent); text-decoration:none; word-break:break-word; }
    .src a:hover{ text-decoration:underline; }

    .typing{display:inline-flex;gap:5px;align-items:center;padding:2px 0;}
    .typing span{width:7px;height:7px;border-radius:50%;background:var(--maroon);animation:blink 1.2s infinite both;}
    .typing span:nth-child(2){animation-delay:.18s;}
    .typing span:nth-child(3){animation-delay:.36s;}
    @keyframes blink{0%,80%,100%{opacity:.25;}40%{opacity:.95;}}

    .composer-wrap{ position:sticky; bottom:0; background:linear-gradient(to bottom, rgba(242,245,250,0) 0%, var(--page) 38%); padding:14px 0 18px; }
    .composer{ max-width:860px; margin:0 auto; padding:0 24px; }
    .field{ display:flex; align-items:flex-end; gap:10px; background:var(--surface); border:1px solid var(--hairline); border-radius:16px; padding:8px 8px 8px 16px; box-shadow:0 2px 10px rgba(28,42,63,0.05); }
    .field:focus-within{ border-color:var(--accent); box-shadow:0 0 0 3px rgba(44,102,171,0.16); }
    .field textarea{ flex:1; border:0; outline:0; resize:none; background:transparent; font-family:inherit; font-size:16px; color:var(--ink); padding:9px 0; max-height:120px; line-height:1.45; }
    .field textarea::placeholder{ color:#8893A4; }
    .send{ flex:0 0 auto; border:0; border-radius:11px; background:var(--maroon); color:#fff; height:40px; padding:0 16px; font-family:inherit; font-size:15px; font-weight:600; display:inline-flex; align-items:center; gap:7px; cursor:pointer; transition:background .14s,transform .06s; }
    .send:hover{ background:var(--maroon-deep); }
    .send:active{ transform:scale(0.98); }
    .send:disabled{ opacity:0.6; cursor:progress; }
    .send svg{ width:16px; height:16px; }

    footer{ max-width:860px; margin:0 auto; padding:6px 24px 22px; font-size:12.5px; color:var(--muted); line-height:1.5; }
    footer .anchor{ color:var(--accent); text-decoration:none; }

    @media (max-width:560px){
      .wordmark{ font-size:21px; }
      .user .bubble{ max-width:92%; }
      main{ padding-top:20px; }
    }
    @media (prefers-reduced-motion:reduce){ *{ transition:none!important; } }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <svg class="mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="10.2" stroke="currentColor" stroke-width="1.3"/>
        <path d="M12 2.5 L13.7 10.3 L21.5 12 L13.7 13.7 L12 21.5 L10.3 13.7 L2.5 12 L10.3 10.3 Z" fill="currentColor"/>
      </svg>
      <span class="wordmark">AskGloucester</span>
      <span class="tag">Beta</span>
    </div>
  </header>

  <main id="thread">
    <div id="welcome" class="welcome">
      Ask what was discussed or decided at City Council and School Committee
      meetings, or when any Gloucester public body meets.
    </div>
  </main>

  <div class="composer-wrap">
    <div class="composer">
      <div class="field">
        <textarea id="question" rows="1" autocomplete="off"
                  placeholder="Ask about a meeting, schedule, or city service…"
                  aria-label="Ask a question"></textarea>
        <button id="submit" class="send" type="button">
          Ask
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12h13M13 6l6 6-6 6" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
      </div>
    </div>
    <footer>
      AskGloucester is an AI and can make mistakes — please double-check important information on the <a class="anchor" href="https://www.gloucester-ma.gov" target="_blank" rel="noopener noreferrer">city website</a>.
    </footer>
  </div>

  <script>
    const thread   = document.getElementById("thread");
    const welcome  = document.getElementById("welcome");
    const input    = document.getElementById("question");
    const submit   = document.getElementById("submit");

    let history = [];

    // The whole page scrolls (sticky composer pins to the bottom), so keep the
    // newest turn in view by scrolling the document, not an inner container.
    function scrollDown() {
      window.scrollTo(0, document.body.scrollHeight);
    }

    function srcLabel(s) {
      const p = [s.meeting_body, s.document_type, s.document_date].filter(Boolean);
      let t = p.join(" \u2014 ") || "Source";
      if (s.page_number != null) t += ", p." + s.page_number;
      return t;
    }

    // div.turn.user > p.who("You") + div.bubble
    function addUser(text) {
      welcome.style.display = "none";
      const turn = document.createElement("div");
      turn.className = "turn user";
      const who = document.createElement("p");
      who.className = "who";
      who.textContent = "You";
      const b = document.createElement("div");
      b.className = "bubble";
      b.textContent = text;
      turn.appendChild(who);
      turn.appendChild(b);
      thread.appendChild(turn);
      scrollDown();
    }

    // Loading state: div.turn.bot > p.who + div.card > div.typing(3 spans).
    // Returns the whole turn so the caller can remove it on response.
    function addTyping() {
      welcome.style.display = "none";
      const turn = document.createElement("div");
      turn.className = "turn bot";
      const who = document.createElement("p");
      who.className = "who";
      who.textContent = "AskGloucester";
      const card = document.createElement("div");
      card.className = "card";
      const typing = document.createElement("div");
      typing.className = "typing";
      for (let i = 0; i < 3; i++) typing.appendChild(document.createElement("span"));
      card.appendChild(typing);
      turn.appendChild(who);
      turn.appendChild(card);
      thread.appendChild(turn);
      scrollDown();
      return turn;
    }

    // Linkify bare http(s) URLs inside a single plain-text answer segment.
    // Runs ONLY on answer prose (the calendar/schedule channel emits its
    // source_url inline as plain text); never on .cite chips or .src rows.
    // XSS-safe: the URL goes in via textContent + .href, never innerHTML.
    function appendLinkified(container, text) {
      const re = /https?:\\/\\/[^\\s]+/g;
      let last = 0, m;
      while ((m = re.exec(text)) !== null) {
        let url = m[0];
        // Trailing punctuation belongs to the prose, not the URL — strip it
        // off the match and re-emit it as plain text after the link.
        const trail = (url.match(/[.,;:!?)\\]}'"]+$/) || [""])[0];
        if (trail) url = url.slice(0, url.length - trail.length);
        // Text before the URL.
        if (m.index > last) {
          container.appendChild(document.createTextNode(text.slice(last, m.index)));
        }
        // Guard: only real http(s) URLs become anchors; else leave as text.
        if (url.indexOf("http://") === 0 || url.indexOf("https://") === 0) {
          const a = document.createElement("a");
          a.textContent = url;
          a.href = url;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.className = "link";
          container.appendChild(a);
        } else if (url) {
          container.appendChild(document.createTextNode(url));
        }
        if (trail) container.appendChild(document.createTextNode(trail));
        last = m.index + m[0].length;
      }
      // Remaining prose after the last URL (or the whole string if no match).
      if (last < text.length) {
        container.appendChild(document.createTextNode(text.slice(last)));
      }
    }

    // Render the answer into `card`, turning numeric [n] markers into small
    // citation chips that link to source n. Pure display transform: the [n]
    // stay in the model output / API response; we only change how they look.
    // Built with text nodes + createElement (never innerHTML on model output),
    // so an answer can't inject markup. Non-numeric bracketed tokens (e.g. a
    // stray [toolname]) are left untouched as plain text.
    function renderAnswer(card, text, sources) {
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
          card.appendChild(chip);
        } else if (part) {
          appendLinkified(card, part);
        }
      });
    }

    // div.turn.bot > p.who("AskGloucester") + div.card (answer + sources)
    function addAssistant(text, sources) {
      const turn = document.createElement("div");
      turn.className = "turn bot";
      const who = document.createElement("p");
      who.className = "who";
      who.textContent = "AskGloucester";
      const card = document.createElement("div");
      card.className = "card";
      renderAnswer(card, text, sources);

      if (sources && sources.length) {
        // details.sources > summary(span.chev + "Sources (N)") + div.src*
        const det = document.createElement("details");
        det.className = "sources";
        const sum = document.createElement("summary");
        const chev = document.createElement("span");
        chev.className = "chev";
        chev.textContent = "\u203a";            // chevron, rotates via CSS when open
        sum.appendChild(chev);
        sum.appendChild(document.createTextNode(
          " Sources (" + sources.length + ")"));
        det.appendChild(sum);
        sources.forEach(s => {
          const row = document.createElement("div");
          row.className = "src";
          const num = document.createElement("span");
          num.className = "n";
          num.textContent = s.n;
          const txt = document.createElement("span");
          txt.textContent = srcLabel(s);
          row.appendChild(num);
          row.appendChild(txt);
          if (s.source_url) {
            txt.appendChild(document.createTextNode(" \u00b7 "));
            const a = document.createElement("a");
            a.href = s.source_url;
            a.target = "_blank";
            a.rel = "noopener noreferrer";
            let host = s.source_url;
            try { host = new URL(s.source_url).hostname; } catch (e) {}
            a.textContent = host;
            txt.appendChild(a);
          }
          det.appendChild(row);
        });
        card.appendChild(det);
      }

      turn.appendChild(who);
      turn.appendChild(card);
      thread.appendChild(turn);
      scrollDown();
    }

    // Errors reuse the bot turn shape with an error-tinted card.
    function addError(msg) {
      const turn = document.createElement("div");
      turn.className = "turn bot";
      const who = document.createElement("p");
      who.className = "who";
      who.textContent = "AskGloucester";
      const card = document.createElement("div");
      card.className = "card error";
      card.textContent = msg;
      turn.appendChild(who);
      turn.appendChild(card);
      thread.appendChild(turn);
      scrollDown();
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
