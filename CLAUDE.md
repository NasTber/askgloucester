# CLAUDE.md — AskGloucester

Context primer for Claude Code. Keep this current; it is the single source of truth for how this project is built and how we work on it.

## What this is

AskGloucester is a civic AI assistant for Gloucester, MA. It answers questions about municipal **meeting documents** (agendas and minutes for School Committee, City Council, Planning Board, Conservation Commission, and Zoning Board of Appeals), **meeting schedules** (a city-wide calendar), and other published city info — the **staff directory**, **appointed board/commission rosters**, **city-service pages** (trash, permits, beaches, harbor, clerk), and the city **FAQ** — using a tool-using RAG agent on Azure (six tools; see RAG / agent below). It runs on a **personal** Azure subscription (not a work tenant) and doubles as hands-on AZ-305 preparation (AZ-900 and AI-900 already done). Live at **https://www.askgloucester.com**.

## Working agreement (read this first)

- **Design / architecture / diagnosis** happens in Claude.ai chat. **File edits, git, and verification harnesses** happen in Claude Code. **Nas runs all Azure CLI commands himself.**
- **Verify before building.** Investigate and spot-check against the real repo/index before changing anything. Read-only trace harnesses live in `/tmp` and must never touch the repo or the search index. This has caught bad assumptions repeatedly — do it.
- **Report and stop before committing.** A push to `main` triggers CI/CD (`deploy-app.yml`) and deploys to prod. Never commit without Nas's go-ahead; he reviews diffs and runs browser smoke-tests first.
- **Scope changes tightly.** Keep presentation-only changes separate from logic/behavior changes so a regression's blame is unambiguous.
- **Overengineering check.** Fix a real *current* wrong, not hypotheticals. Prefer removing an over-eager rule/filter over stacking new ones. Don't hand-code per-case branches. When unsure, say so.
- zsh: never put inline `#` comments in terminal commands. Always `source .venv/bin/activate` before running anything (conda `(base)` lacks deps like `dotenv`).
- Add clear comments explaining Azure SDK calls.

## Architecture

**Ingestion** (`run_pipeline.py` orchestrates; calendar step runs first, isolated, on `--meeting-body all`):
- Documents: custom **Archive.aspx** scraper (CivicPlus; Gloucester does NOT use AgendaCenter). PDFs at `…/ArchiveCenter/ViewFile/Item/{ADID}`.
- SC minutes: public Google Drive folder via `gdown` (keyless).
- Calendar: CivicPlus per-CID **iCalendar** feed → Azure Table Storage `events`.
- Pipeline: download → Blob (`raw-documents`) → Document Intelligence (`prebuilt-read`) → 500-token chunks / 50 overlap (`cl100k_base`) with a human-readable prefix `"{body} {type} — {Month D, YYYY} ({ISO})\n\n{chunk}"` (makes date/body keyword-searchable in BM25) → `text-embedding-3-small` (1536-dim) → AI Search hybrid index.
- Auth: `DefaultAzureCredential` throughout (Azure CLI cred locally, UAMI in the container).

**RAG / agent** (`api/agent.py`, LangChain `create_agent` / LangGraph, `temperature=0`):
- `TOOLS = [doc_search, schedule_lookup, directory_lookup, city_services_search, board_lookup, faq_search]` — the **router seam**; each new data source is a new tool, not a re-migration. `TOOL_GUIDANCE` (appended to the system prompt) routes between them by intent/scope. Each api-local tool module reads its own backing store DIRECTLY over `DefaultAzureCredential` and must NOT import `ingestion/` (the image ships `api/` only — importing ingestion crash-loops the container); the matching `ingestion/*_source.py` owns writes.
- `doc_search`: hybrid (BM25 + vector, RRF) over the `gloucester-documents` AI Search index with OData filters (`meeting_body`, `meeting_category`, date). Body **allowlist** — five bodies with indexed documents: School Committee, City Council, Planning Board, Conservation Commission, Zoning Board of Appeals (single source of truth: `BODY_KEYWORDS` in `api/query.py`, reused by `api/agent.py`). Off-allowlist bodies get a structural decline inside the tool. (Planning Board is allowlisted but may be empty in-window pending its calendar CID.)
- `schedule_lookup` (`api/calendar.py`): api-local, read-only `events` Table reader (roster of bodies). Returns prose with inline calendar links — NO `[n]` citation channel.
- `directory_lookup` (`api/directory.py`): read-only `officials` Table — current city STAFF identity/contact. Prose, NO `[n]`.
- `board_lookup` (`api/boards.py`): read-only `boards` Table — APPOINTED board/commission members + term dates (PERSON rows + `__board__` summary rows). Prose, NO `[n]`.
- `city_services_search` (`api/city_services.py`): hybrid search over the `gloucester-city-services` index — published service pages (trash/recycling, permits, beaches/parking, harbor/moorings, clerk services). Returns `[n]` sources.
- `faq_search` (`api/faq.py`): hybrid search over the `gloucester-faq` index — concise published FAQ answers across any department. Returns `[n]` sources.
- Citations: `doc_search`, `city_services_search`, and `faq_search` share ONE per-request `_CITATION_STATE` ContextVar + running `next_n` counter, so `[n]` numbers never collide across tools in a turn. `ask()` keeps only chunks whose `[n]` appears in the final answer, preserving original numbers. The Table-backed tools (schedule/directory/board) are prose-only — no `[n]` channel.
- Memory: stateless / client-carried. `POST /ask {question, history[]}` reconstructs messages each call.

**API** (`api/main.py`, FastAPI): `GET /` (inline chat UI), `POST /ask {question, history[]} → {answer, sources[]}`, `GET /health` (lazy, never touches Azure).

**Frontend**: a single inline HTML/CSS/JS string `INDEX_HTML` in `api/main.py` (no static dir). Gloucester civic theme — **navy `#1E3E80`** for structure (header band, wordmark, user bubble, Sources/links), **maroon `#581824`** for actions/citations (Ask button, citation chips, source numbers), Fraunces + Public Sans from Google Fonts. **XSS-safe DOM** (`textContent`/`createElement`, never `innerHTML` with model output). Maroon citation chips + collapsible Sources expander; URLs in answers are linkified to clickable links; AI-disclaimer footer linking the city website.

## Azure resources (`rg-askgloucester-dev`, eastus)

- Storage `stakgloucesterdev` — `raw-documents` blob container + `events` Table.
- Document Intelligence `docintel-askgloucester-dev` (prebuilt-read).
- AI Search `srch-askgloucester-dev` — **Basic SKU** (migrated from Free; semantic ranker available but unused). Index `gloucester-documents`: ~3,700 chunks across ~95 docs, **2026 window only**.
- Key Vault `kv-askgloucester-dev`.
- UAMI `id-askgloucester-dev` — clientId `2927ca7b-530b-483d-8fd1-85c97c8a41bb` (the `AZURE_CLIENT_ID` in the container), object/principalId `76a3753b-9bf6-4184-af0e-1dfd4f277266` (RBAC targets this).
- Azure OpenAI `aoai-askgloucester-dev-7c8ac` — `text-embedding-3-small` (350K TPM) + `gpt-4.1-mini` (100K TPM).
- ACR `acraskgloucesterdev`, Log Analytics `log-askgloucester-dev`, Container Apps Env `cae-askgloucester-dev`, Container App `ca-askgloucester-dev` (scale 0→1, startup+liveness probes on `/health`, custom domain `www.askgloucester.com`, TLS SniEnabled).

Index fields: `id, content, source_url, document_date, meeting_body, document_type, page_number, chunk_id, meeting_category, content_vector`. `title` is NOT indexed (blob metadata only). `document_date` is String ISO `YYYY-MM-DD` (sortable + filterable, not facetable).

Data sources (AMIDs): 35 City Council agendas ✅, 36 City Council minutes ✅, 113 School Committee agendas ✅, 114 School Committee minutes ❌ DEAD (1 doc, 2019); plus the expanded bodies — 57/58 Planning Board agendas/minutes, 47/48 Conservation Commission agendas/minutes, 41/146 Zoning Board of Appeals agendas/minutes, 42 ZBA "meeting results". NB: 114 is still in the scraper's `DEFAULT_AMID_LIST = (113, 114, 35, 36, 57, 58, 47, 48, 41, 146, 42)` — it is requested every run but yields nothing in-window, so real SC minutes come from the public Drive folder instead. (The AMID→body/type map lives in `ARCHIVE_SOURCES` in `scraper.py`; AMID and calendar CID number spaces are independent — do not cross-wire them.) `meeting_category` (`full_committee` / `subcommittee` / `negotiations`) is derived from the title at ingest (`classify_meeting_category`; "negotiat" outranks "subcommittee", else `full_committee`).

## Identity (don't mix these up)

- UAMI: clientId `2927ca7b…` (container env var), object id `76a3753b…` (RBAC target). `az role assignment create --assignee <clientId>` auto-resolves to the object id.
- GitHub Actions SP `askgloucester-github-actions`: appId `ad5ced71-357f-43a9-8f8d-bc890f1647f7`, SP object id `dd7b6255-d761-4b86-a3c7-69c92f442fd1`. Federated cred `repo:NasTber/askgloucester:ref:refs/heads/main`. Tenant `250ce8f4-9d0a-4a44-9993-ea84c9082ca9`.

## Build & deploy

```bash
# Image (--platform linux/amd64 MANDATORY on Apple Silicon, else exec format error)
docker build --platform linux/amd64 -t acraskgloucesterdev.azurecr.io/askgloucester-api:latest .
docker push acraskgloucesterdev.azurecr.io/askgloucester-api:latest
```

```bash
# Bicep — ⚠️ WILL CLOBBER the www hostname binding + manual RBAC until those are in IaC (see Known issues)
az deployment group create \
  --resource-group rg-askgloucester-dev \
  --template-file infra/main.bicep \
  --parameters location=eastus \
  --parameters containerImage=acraskgloucesterdev.azurecr.io/askgloucester-api:latest \
  --parameters githubActionsSpObjectId=dd7b6255-d761-4b86-a3c7-69c92f442fd1
```

```bash
# Pipeline
python run_pipeline.py --start-date 2026-01-01 --end-date 2026-12-31 --meeting-body all
# --no-skip forces reprocessing of already-indexed docs (default skips them)
```

## CI/CD (GitHub Actions, OIDC — no stored secrets)

- `deploy-app.yml`: push to `main` on paths `api/**`, `Dockerfile`, `.dockerignore` → docker build `--platform linux/amd64` → push ACR → `az containerapp update`. (`ingestion/**` is intentionally NOT a trigger — the image ships `api/` only, so ingestion changes can't alter the deployed image; `ingest.yml` checks out fresh each run so ingestion code is never stale regardless.)
- `ingest.yml`: cron Mondays 09:00 UTC + `workflow_dispatch` → `run_pipeline.py … --meeting-body all`.
- Actions pinned to Node 24 runtime (`checkout@v6`, `setup-python@v6`, `azure/login@v3`).
- `CLAUDE.md` and other root files do NOT trigger a deploy (not in any path filter).

## Security posture

- **Headers middleware** (`_security_headers`, on all responses incl. 429/413/422): CSP, in code order — `default-src 'self'`, `script-src 'self' 'nonce-…'` (**per-request script nonce**, no `unsafe-inline` for scripts), `style-src 'self' 'unsafe-inline' https://fonts.googleapis.com`, `font-src 'self' https://fonts.gstatic.com`, `img-src 'self' data:`, `connect-src 'self'`, `base-uri 'self'`, `form-action 'self'`, `frame-ancestors 'none'`, `object-src 'none'`; plus `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, HSTS (`max-age=31536000; includeSubDomains`, no preload), `Permissions-Policy: geolocation=(), camera=(), microphone=()`. The nonce is injected into the single `<script>` via a targeted `INDEX_HTML.replace("<script>", …, 1)` — **never `.format`** (INDEX_HTML is full of literal `{ }`).
- **Input caps** (Pydantic — rejected with 422 *before* the rate bucket and *before* any LLM/embedding call): `question` max 2000, `Message.content` max 12000, `history` max 20 items. **Body-size 413 guard** at 512 KB (Content-Length check in the middleware, before the body is read). Client trims `history.slice(-20)` and the textarea has `maxlength="2000"`.
- **Rate limit** (slowapi): 10/min per-IP + 30/min global, `memory://` storage (valid only because max-replicas = 1), 429 with `Retry-After: 60`. Wired via `app.state.limiter` + per-route decorators + a `RateLimitExceeded` handler (no `SlowAPIMiddleware`).

## Gotchas & lessons

- **`recency` is for "last/latest/next meeting" questions only — never topical/subject questions.** It pins retrieval to a single most-recent full-committee *minutes* date (`date_eq`), so applying it to "what's the budget" hid the June agendas that held the FY27 figures. The agent's `doc_search` recency guidance enforces this; do not loosen it. (The pin mechanism itself is correct for genuine recency queries — don't re-architect it.)
- **The agent MUST call `doc_search` before answering OR declining a civic factual question.** The grounding rules were reconciled to the agent paradigm: "NOT IN SOURCES" declines fire only *after* a search returns nothing — never assert the docs don't cover something without searching. (Earlier the prompt still said "sources provided by the user," a leftover from the pre-agent injected-context loop, which let the model decline without retrieving.)
- **Budget/figure precision is a known data limit — disclaim, don't engineer.** City Council packets carry the School Department budget with *both* FY26-proposed and FY27-proposed framings, and the numeric tables are OCR-garbled. The footer hedge covers this; do not chase figure fidelity with prompt rules.
- **Ingest skip is existence-only and that's sufficient.** Gloucester revisions arrive under a NEW `source_url` (new ArchiveCenter Item id), so "already-indexed source_url → skip" is correct; no content hash needed. (Pruning superseded revisions is a separate, deferred concern that needs a meeting-identity key, not a hash.)
- **Embedder paces with a token-bucket** (~315K/min ≈ 90% of the 350K TPM) to avoid the 429 sawtooth on backfill; 280K per-batch cap + 429 retry remain as fallback.
- `az acr build` is BLOCKED on this subscription (`TasksOperationsNotAllowed`) — always local/CI `docker build` + `docker push`.
- `exec format error` = ARM64 image on AMD64 host → always `docker build --platform linux/amd64`.
- `DefaultAzureCredential` + UAMI requires `AZURE_CLIENT_ID` in the container. RBAC propagation is 2–5 min after assignment (caused early 403s).
- `ensure_index()` is "create if not exists" (not delete+recreate) so a per-body run doesn't wipe other bodies. Schema changes require deleting the index first.
- AI Search SKU is immutable → a tier change = delete + recreate. Vectors make storage bind before doc count (3,700 docs blew the 50 MB Free cap). `documentCount` is the live signal; `storageSize`/`vectorIndexSize` read 0 transiently after a reindex.
- Bicep RBAC can show "Succeeded" while assignments are absent if a deploy was canceled — always verify with `az role assignment list`. `customSubDomainName` on Azure OpenAI is immutable → use an `existing` reference.
- Container Apps: `az containerapp update --image` with the same tag doesn't make a new revision — use `--revision-suffix` to force one when a revision is stuck Unhealthy.
- 422 (Pydantic) and body validation run BEFORE slowapi and BEFORE the endpoint — bad/oversized bodies are rejected cheaply, no LLM call.

## Known issues / parked

- **IaC reconciliation (landmine).** The `www` hostname binding and several RBAC assignments live OUTSIDE Bicep, so any `main.bicep` apply wipes them and breaks the live site. Put them in Bicep (idempotent) before the next infra deploy. App deploys (`deploy-app.yml`) do NOT trip this.
- **Citation dedup-by-source.** Citations are numbered per *chunk*, so one PDF (chunked into dozens) surfaces as dozens of `[n]` chips. Fix = number by distinct `source_url` so chunks of a doc share one citation + one Sources row. Parked as cosmetic.
- **Working-tree WIP.** Keep `CLAUDE.md` and `scripts/` committed or discarded — main↔local skew has caused a deploy failure before.
- **Historical data = 2026 only.** Backfill (SC Drive 2020–2025, City Council back to 2009) is gated on the Microsoft Founders Hub application (Nas-owned) to avoid embedding cost at scale.
- Calendar launch gates: confirm the exact `CANCELLED` token against a real cancelled event; find the Planning Board CID.
- Optional/later: index `title`; `document_date` facetable; upsert + index aliasing; semantic ranker / agentic retrieval (now unblocked on Basic).

## Repo structure

```
askgloucester/
├── infra/
│   ├── main.bicep
│   ├── main.bicepparam
│   └── modules/  # kebab-case files: identity, storage, search (basic), document-intelligence,
│                 # openai, keyvault, containerapp.  (Also container-apps.bicep +
│                 # container-registry.bicep — appear superseded by containerapp.bicep; verify
│                 # which main.bicep references before the next infra change.)
├── ingestion/
│   ├── utils.py            # classify_meeting_category
│   ├── scraper.py          # Archive.aspx; DEFAULT_AMID_LIST = (113, 114, 35, 36, 57, 58, 47, 48, 41, 146, 42); existence-only skip
│   ├── processor.py
│   ├── chunker.py          # 500-token/50-overlap chunks; metadata prefix on every chunk
│   ├── embedder.py         # token-bucket pacing
│   ├── indexer.py          # ensure_index = create-if-not-exists; index field schema lives here
│   ├── drive_source.py     # SC minutes via gdown; existence-only skip
│   ├── calendar_source.py  # per-CID iCal → events Table; CANCEL/RESCHEDUL status derivation
│   ├── directory_source.py # writes the officials Table (staff directory)
│   ├── boards_source.py    # writes the boards Table (appointments: PERSON + __board__ rows)
│   ├── city_services_source.py  # wipe-and-rebuild of the gloucester-city-services index
│   ├── faq_source.py       # wipe-and-rebuild of the gloucester-faq index
│   └── requirements.txt
├── api/
│   ├── __init__.py
│   ├── agent.py            # create_agent; 6 tools + TOOL_GUIDANCE; recency + grounding rules
│   ├── calendar.py         # api-local read-only events Table reader (schedule_lookup)
│   ├── directory.py        # api-local read-only officials Table reader (directory_lookup)
│   ├── boards.py           # api-local read-only boards Table reader (board_lookup)
│   ├── city_services.py    # api-local hybrid search over gloucester-city-services (city_services_search)
│   ├── faq.py              # api-local hybrid search over gloucester-faq (faq_search)
│   ├── query.py            # SYSTEM_PROMPT + thin delegator to agent.ask
│   ├── main.py             # FastAPI; chat UI; security headers + nonce CSP; rate limit; input caps
│   ├── routes/.gitkeep     # empty placeholder dir
│   ├── Dockerfile          # legacy/local; CI builds the ROOT Dockerfile, not this one
│   └── requirements.txt
├── scripts/                # data-inspection / trace helpers (read-only, untracked WIP)
├── .github/workflows/      # deploy-app.yml, ingest.yml
├── Dockerfile              # python:3.12-slim, api/ only, linux/amd64 — the one CI uses
├── run_pipeline.py
├── README.md
├── CLAUDE.md
├── .env                    # local config (gitignored; no secrets — all DefaultAzureCredential)
└── .env.example            # committed template
```
