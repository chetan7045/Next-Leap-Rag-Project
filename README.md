# Niva

> Your factual guide to HDFC mutual funds.

A facts-only assistant for HDFC Mutual Fund schemes. It answers factual questions
(expense ratio, exit load, minimum SIP, lock-in, benchmark, NAV, fund manager) from
an indexed corpus of public scheme pages, and refuses everything else.

- **Backend** — Python 3.12+, FastAPI, sentence-transformers, Chroma, Gemini
- **Frontend** — Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4
- **No investment advice, no forecasts, no personal data**

The customer-facing experience is **Niva**: one question box, one status chip, and a
sourced answer. RAG internals (embeddings, chunks, retrieval scores, model names,
API versions) stay behind the API and never reach the UI.

## Safety model

The backend decides safety; the frontend only renders what the API returns.

1. **Deterministic classifier first.** Requests for advice, performance promises,
   forecasts, personal account data, or non-scheme topics are answered without ever
   calling the LLM. The refusal reason is returned to the client.
2. **Confidence gate.** If the best retrieved chunk scores below
   `similarity_threshold`, the assistant returns `NO_CONTEXT` instead of guessing.
3. **Grounded generation.** The model only sees retrieved chunks and must cite them.
4. **Post-generation guardrails.** The answer is re-checked; advice-shaped or
   return-guaranteeing output is replaced with a refusal.
5. **Honest provenance.** Every citation carries the source type. The configured
   sources are third-party pages, so they are labelled `REFERENCE` — never
   `AMC_OFFICIAL`. `last_updated` comes only from a date stated in the page, never
   from the retrieval time.

`answer_type` is one of `ANSWER`, `REFUSAL`, `CLARIFICATION`, `NO_CONTEXT`, `ERROR`.
`refusal_reason` explains a `REFUSAL`. The legacy booleans `refusal` and
`clarification` are kept in sync for compatibility. The `retrieval` diagnostic block
is only populated when `DEBUG_RAG=true` outside production.

## Requirements

- Python 3.12 or newer
- Node.js 20 or newer
- A Gemini API key (only needed for real answers; tests and evaluation are offline)

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then set GEMINI_API_KEY
```

Ingest the sources (downloads each page, embeds it, writes the Chroma index):

```bash
HF_HOME=$PWD/.cache/huggingface .venv/bin/python scripts/ingest.py
```

Ingestion is idempotent: an unchanged source reports `unchanged` and adds no
duplicates.

Run the API:

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

Interactive API docs are at <http://127.0.0.1:8000/docs>.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local    # NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

Open <http://localhost:3000>. The API allows CORS from the configured origins
(`ALLOWED_ORIGINS`, comma-separated) — add your deployed frontend origin there.

### Frontend design system

`src/app/globals.css` holds the design tokens (accent ramp, ink, muted text, radius,
elevation, easing, keyframes) as Tailwind v4 `@theme` variables. Components consume
those tokens and never hard-code hex values.

| Component | Responsibility |
| --- | --- |
| `NivaHeader`, `NivaWordmark`, `NivaMark` | Brand lockup and the single status chip |
| `LiveStatus` | Polls `GET /health`; renders only `Live` / `Offline` |
| `WelcomeHero`, `PromptSuggestion` | Empty state and example questions |
| `UserMessage`, `AssistantMessage`, `ErrorMessage` | Transcript rendering |
| `SourceCitation` | Source name, authority, external link, last-updated date |
| `TypingIndicator` | Conversational loading state |
| `ChatComposer` | Enter to send, Shift+Enter for a newline, auto-growing input |
| `DisclaimerLine`, `AboutDialog` | Facts-only notice and secondary product info |

`src/lib/format.ts` maps the API response onto customer-facing language: internal
source enums become `HDFC Mutual Fund` / `AMFI` / `SEBI` / `Public source`, dates
become `25 Sep 2026`, and long page titles are shortened. Relevance scores, chunk
ids and document types are deliberately not rendered.

## Tests and evaluation

```bash
cd backend
HF_HOME=$PWD/.cache/huggingface .venv/bin/python -m pytest tests/ -q
HF_HOME=$PWD/.cache/huggingface .venv/bin/python scripts/evaluate.py
```

The test suite is hermetic: it uses ephemeral Chroma collections and the offline
fake provider, and never touches the network or the Gemini API.

`scripts/evaluate.py` runs 24 cases from `evaluation/questions.json` covering
expense/fees, lock-in, fund profile, NAV, ambiguity, advice, performance, forecast,
PII, and out-of-scope questions. It reports a safety pass rate and checks that
grounded answers carry citations. Add `--json report.json` for a structured report.

```bash
.venv/bin/python scripts/test_retrieval.py --top-k 5 "lock-in period of HDFC ELSS"
```

## Deployment

`render.yaml` in the repo root is a complete Render Blueprint. Both services run on
the **free** plan (0.1 CPU, 512 MB RAM).

### You only need to provide one secret

Everything except the Gemini key is already committed — either as a plain value in
`render.yaml` or as a default in `backend/app/core/config.py`.

| What | Where it comes from | Manual? |
| --- | --- | --- |
| `GEMINI_API_KEY` | You, in the Render dashboard | **Yes — the only one** |
| `NEXT_PUBLIC_API_BASE_URL` | `render.yaml` (`https://niva-api.onrender.com`) | No |
| `ALLOWED_ORIGIN_SUFFIXES` | `render.yaml` (`.onrender.com`) | No |
| Model, chunking, retrieval, CORS, cache paths, threads | `config.py` defaults + `render.yaml` | No |
| Python / Node versions | `render.yaml` (`PYTHON_VERSION`, `NODE_VERSION`) | No |

### Steps

1. Push this repository to GitHub.
2. In Render: **New → Blueprint**, select the repository, and apply. Render creates
   both services and every environment variable except the key.
3. Open the `niva-api` service → **Environment**, and add:
   - Key: `GEMINI_API_KEY`
   - Value: your key from <https://aistudio.google.com/apikey>
4. Save. Render redeploys the API.
5. Open `https://niva-web.onrender.com`.

The first API boot downloads the embedding model (~87 MB) and ingests the corpus by
fetching all five source pages, which takes roughly a minute on a normal machine and
noticeably longer on the 0.1 CPU free tier. Because the free instance spins down after
15 minutes idle, the next request pays that cost again while it wakes.

### Service settings

Both are defined in `render.yaml`; the values are listed here for reference.

| | `niva-api` | `niva-web` |
| --- | --- | --- |
| Runtime | Python 3.12.7 | Node 24 |
| Root directory | `backend` | `frontend` |
| Plan | `free` | `free` |
| Build command | `pip install --no-cache-dir -r requirements.txt` | `npm ci && npm run build` |
| Start command | `python scripts/ingest.py && uvicorn app.main:app --host 0.0.0.0 --port $PORT` | `npm run start` |
| Health check path | `/health` | `/` |

If you would rather not use the Blueprint, those are the only values to enter when
creating each service by hand.

The `&&` in the API start command is deliberate. `scripts/ingest.py` exits non-zero
when the corpus cannot be built. Chaining means a failed index is a failed boot,
rather than a service that reports `status: "ok"` on `/health` while every chat
reply says the index is empty.

### Notes and limits of the free plan

- **Memory is the tight constraint.** Two things dominate it, and both were measured
  rather than guessed.
  - *PyTorch is not installed at all.* Importing it costs roughly 750 MB RSS on its
    own — more than the entire budget. The embedder runs on ONNX Runtime instead.
  - *ONNX Runtime's CPU memory arena is the second-largest term.* It grows to the
    largest activation it has ever seen and never returns that memory to the OS, so
    peak RSS is set by the **biggest embedding batch**, not by the size of the corpus.
    The corpus itself is trivial: 77 chunks x 384 dims is under 1 MB of vectors.
    Measured peak RSS for a full ingest of the real corpus, 1 thread:

    | arena | batch 32 | batch 8 | batch 4 | batch 1 |
    | --- | --- | --- | --- | --- |
    | on | 451 MB | 301 MB | — | 255 MB |
    | **off** | 353 MB | 275 MB | **257 MB** | 254 MB |

    So `EMBEDDING_ENABLE_CPU_MEM_ARENA=false` and `EMBEDDING_BATCH_SIZE=4` are the
    defaults. Disabling the arena is also *faster* at this corpus size (9.2s vs 14.2s
    for the full embed), because the arena's grow-and-copy bookkeeping costs more than
    plain `malloc`/`free` when there are only 77 chunks.

  End-to-end on the real `ingest.py` run: **417 MB peak before, 347 MB after.**
  The serving process settles at **302 MB steady / 313 MB peak**, and does not grow
  across queries (+3 MB over 30 sequential requests, none of it reclaimable, so it is
  allocator high-water rather than a leak).

  These are local measurements taken with `/proc/self/status` `VmHWM` on Linux
  x86-64, Python 3.14. Render runs Python 3.12.7 and its accounting includes page
  cache, so treat them as indicative rather than a guarantee. The headroom is real
  but it is not infinite — see *Remaining risks* below.

- **The filesystem is ephemeral and there is no persistent disk.** The Chroma index
  is written to `/tmp/chroma` and rebuilt by the start command on every cold start.
  It must stay in `persistent` mode even so, because `ingest.py` and Uvicorn are
  separate processes that hand the index over through the filesystem. "Persistent"
  here describes how Chroma stores data, not durability across a restart.
- The corpus is fetched from the network at boot, so a failed upstream fetch fails
  the boot rather than serving an empty index.
- If you rename a service, update the other service's URL in `render.yaml`; the
  hostname is derived from the service name.
- `NEXT_PUBLIC_API_BASE_URL` is inlined into the client bundle at build time, so
  changing it needs a rebuild, not just a restart.

## Verifying a deployment

```bash
# API is up and the index built
curl -s https://niva-api.onrender.com/health

# retrieval + grounded generation through Gemini, with a real citation
curl -s -X POST https://niva-api.onrender.com/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is the exit load of HDFC ELSS Tax Saver Fund?"}'

# guardrails: each of these returns a refusal, not an answer
# "Should I invest in HDFC Large Cap Fund?"      -> ADVICE_REQUEST
# "What will the NAV be 10 years from now?"       -> PERFORMANCE_PROMISE
# "My PAN is ABCDE1234F"                          -> PII_DETECTED
# "What is the weather in Mumbai?"                -> OUT_OF_SCOPE
# "What is the exit load?"                        -> CLARIFICATION
```

`/health` returning `200` with `index_ready: true` and `llm.configured: true` is the
signal that the whole boot chain — dependencies, embedding model, corpus fetch,
index build, credential — succeeded.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status, version, whether the index is ready. Drives the UI status chip. |
| `GET` | `/api/v1/schemes` | Known schemes, aliases, and whether each is indexed |
| `GET` | `/api/v1/schemes/indexed` | Only the schemes that actually have chunks in the index |
| `GET` | `/api/v1/sources` | Source provenance, dates, and indexed chunk counts |
| `POST` | `/api/v1/chat` | Ask a question; returns a typed, cited answer |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is the exit load of HDFC ELSS Tax Saver Fund?"}'
```

## Configuration

All settings are environment variables with safe defaults (see
`backend/app/core/config.py` and `backend/.env.example`).

| Variable | Default | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Server-side only; never exposed to the browser |
| `LLM_PROVIDER` | `gemini` | `fake` is rejected in production |
| `LLM_MODEL` | `gemini-3.8-flash` | |
| `TOP_K` | `5` | Chunks passed to the model |
| `SIMILARITY_THRESHOLD` | `0.45` | Below this the answer is `NO_CONTEXT` |
| `EMBEDDING_ENABLE_CPU_MEM_ARENA` | `false` | ONNX Runtime's caching arena. It never returns memory to the OS, so leaving it on makes peak RSS depend on the largest batch ever run rather than on one forward pass. Turning it off also runs faster at this corpus size. |
| `EMBEDDING_BATCH_SIZE` | `4` | Chunks per forward pass during ingestion. Directly sets peak activation memory — 32 measured 451 MB, 4 measured 257 MB. |
| `FASTEMBED_CACHE_PATH` | _(empty)_ | Where the downloaded ONNX weights are cached. Set it to a scratch dir on Render. |
| `CHROMA_MODE` | `persistent` | `ephemeral` for tests |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | Relative paths resolve against `backend/`. On Render use `/tmp/chroma`; the default would write into the deployed source tree. |
| `WEB_CONCURRENCY` | `1` | Uvicorn workers. Each extra worker loads its own ~87 MB of ONNX weights and its own Chroma client. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `ALLOWED_ORIGIN_SUFFIXES` | _(empty)_ | Comma-separated https-only hostname suffixes, e.g. `.onrender.com`. Set on Render so the deployment accepts its own frontend without a manual value. Not a credential. |
| `AUTO_INGEST_ON_STARTUP` | `false` | Render ingests in the start command instead |
| `DEBUG_RAG` | `false` | Populates the `retrieval` diagnostic field |

`GEMINI_API_KEY` is the only value that must be supplied per environment. Every
other setting has a working default, so a local checkout runs with nothing but that
key set.

## Project layout

```
backend/
  app/
    api/routes/        chat, health, schemes, sources
    core/              config, security, logging, errors
    models/            chat and retrieval contracts
    services/
      ingestion/       loader, extractor, cleaner, chunker, metadata, pipeline
      embeddings/      local MiniLM encoder (ONNX Runtime, no PyTorch)
      rag/             chroma, retriever, answer service
      safety/          classifier, guardrails, scheme resolver, pii
      llm/             gemini, fake, factory
  scripts/             ingest, evaluate, test_retrieval
  tests/               hermetic unit and API tests
evaluation/            questions.json
frontend/              Next.js app (Niva UI, src/components/niva)
render.yaml            Render Blueprint for both services
sources.json           source registry (the only place source URLs are defined)
sources.schema.json    JSON Schema for sources.json, referenced by its $schema key
```

## Limitations

- Answers reflect the ingested pages as of their stated date. Scheme facts change;
  re-run ingestion to refresh.
- The indexed sources are third-party aggregator pages, not official AMC
  documents. Verify anything important against the AMC, AMFI, or SEBI.
- The assistant deliberately cannot rank funds, predict returns, or give
  recommendations.
- **The memory budget has real but finite headroom.** Serving sits near 302 MB and
  ingestion peaks near 347 MB against a 512 MB ceiling. The measurements above were
  taken on Python 3.14 locally; Render runs 3.12.7 and counts page cache toward the
  limit, so the two numbers are not directly comparable. Raising
  `EMBEDDING_BATCH_SIZE`, re-enabling the ONNX arena, adding a Uvicorn worker, or
  adding a dependency with a heavy transitive import will each push it back over.
- **Ingestion is the fragile part of a cold start.** It runs before Uvicorn binds a
  port, so it must fetch all five source pages successfully or the boot fails. That
  is deliberate — a loud failure beats a service that reports healthy and answers
  nothing — but it does mean a Groww outage or a schema change takes the API down
  until ingestion succeeds. The corpus is small enough to pre-build an index and
  serve it read-only if that trade-off ever becomes worth making.
- On 0.1 CPU the boot-time ingest is the slowest part of the deploy; the first
  request after a spin-down waits for it.
