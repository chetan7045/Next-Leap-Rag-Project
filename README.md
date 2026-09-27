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

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status, version, whether the index is ready |
| `GET` | `/api/v1/schemes` | Known schemes, aliases, and whether each is indexed |
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
| `CHROMA_MODE` | `persistent` | `ephemeral` for tests |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `ALLOWED_ORIGIN_SUFFIXES` | _(empty)_ | Comma-separated https-only hostname suffixes, e.g. `.onrender.com`. Set on Render so the deployment accepts its own frontend without a manual value. Not a credential. |
| `EMBEDDING_THREADS` | `1` | ONNX threads; keep at 1 on 0.1-CPU hosts |
| `AUTO_INGEST_ON_STARTUP` | `false` | Render ingests in the start command instead |
| `DEBUG_RAG` | `false` | Populates the `retrieval` diagnostic field |

`GEMINI_API_KEY` is the only value that must be supplied per environment. Every
other setting has a working default, so a local checkout runs with nothing but that
key set.

## Deployment (Render)

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

### Steps

1. Push this repository to GitHub.
2. In Render: **New → Blueprint**, select the repository, and apply. Render creates
   both services and every environment variable except the key.
3. Open the `niva-api` service → **Environment**, and add:
   - Key: `GEMINI_API_KEY`
   - Value: your key from <https://aistudio.google.com/apikey>
4. Save. Render redeploys the API, which triggers a frontend rebuild through the
   Blueprint.
5. Open `https://niva-web.onrender.com`.

The first API boot downloads the embedding model and ingests the corpus, which takes
roughly a minute. Because the free instance spins down after 15 minutes idle, the
next request takes a few seconds while it wakes.

### Service settings

Both are defined in `render.yaml`; the values are listed here for reference.

| | `niva-api` | `niva-web` |
| --- | --- | --- |
| Runtime | Python | Node |
| Root directory | `backend` | `frontend` |
| Plan | `free` | `free` |
| Build command | `pip install --no-cache-dir -r requirements.txt` | `npm ci && npm run build` |
| Start command | `python scripts/ingest.py && uvicorn app.main:app --host 0.0.0.0 --port $PORT` | `npm run start` |
| Health check path | `/health` | `/` |

The `&&` in the API start command is deliberate. `scripts/ingest.py` exits non-zero
when the corpus cannot be built. Chaining means a failed index is a failed boot,
rather than a service that reports `status: "ok"` on `/health` while every chat
reply says the index is empty.

### Notes and limits of the free plan

- **Memory is the tight constraint.** The embedder runs on ONNX Runtime rather than
  PyTorch: importing torch alone costs about 750 MB RSS, more than the whole 512 MB
  budget. The full stack peaks near 300 MB, which is what makes `free` viable. Note
  that Render's `starter` plan is also 512 MB — it buys CPU, not RAM.
- **The filesystem is ephemeral and there is no persistent disk.** The Chroma index
  is written to `/tmp/chroma` and rebuilt by the start command on every cold start.
  It must stay in `persistent` mode even so, because `ingest.py` and Uvicorn are
  separate processes that hand the index over through the filesystem.
- The corpus is fetched from the network at boot, so a failed upstream fetch fails
  the boot rather than serving an empty index.
- If you rename a service, update the other service's URL in `render.yaml`; the
  hostname is derived from the service name.
- `NEXT_PUBLIC_API_BASE_URL` is inlined into the client bundle at build time, so
  changing it needs a rebuild, not just a restart.

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
sources.json           source registry
```

## Limitations

- Answers reflect the ingested pages as of their stated date. Scheme facts change;
  re-run ingestion to refresh.
- The indexed sources are third-party aggregator pages, not official AMC
  documents. Verify anything important against the AMC, AMFI, or SEBI.
- The assistant deliberately cannot rank funds, predict returns, or give
  recommendations.
