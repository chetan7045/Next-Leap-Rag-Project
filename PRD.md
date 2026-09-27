# PRD — HDFC Fund Facts

> **Product name:** HDFC Fund Facts
> **Tagline:** Facts-only answers from public mutual fund sources.
> **Type:** Full-stack RAG (Retrieval-Augmented Generation) prototype — Python/FastAPI backend, Next.js frontend.

---

## 1. Product Overview

HDFC Fund Facts is a small, deliberately narrow information assistant. It answers **factual questions
about specific HDFC Mutual Fund schemes** using content indexed from public, publicly-linked fund
documents, and it always shows the user where each fact came from.

The product answers **"What do the indexed public sources state?"**
It does **not** answer **"What should I invest in?"**

## 2. Problem Statement

Retail investors asking basic scheme questions (minimum SIP, exit load, expense ratio, benchmark,
riskometer, ELSS lock-in) must currently either:

- read dense PDFs (factsheets, KIMs, SIDs),
- navigate several regulator/aggregator sites, or
- rely on informal advice that may be inaccurate or promotional.

Generic AI chatbots are a poor substitute: they answer from model memory, invent facts, and
routinely drift into recommendations. The failure mode is not "the model is wrong" — it is
**"the model is confident and unsourced."**

HDFC Fund Facts fixes exactly that: retrieval is mandatory, every factual answer carries a citation
and a source date, and refusal to answer is a first-class, expected outcome.

## 3. Target Users

| User | Need |
| --- | --- |
| First-time retail investor | Plain-language scheme facts without reading PDFs |
| Existing investor | Quick confirmation of fees/lock-ins before an action |
| CX / relationship manager | Quick factual reference during client conversations |
| Compliance/QA reviewer | Evidence that the system refuses advice and cites sources |

## 4. Product Goals

1. Answer simple factual scheme questions **concisely** (≤ 3 sentences).
2. Make provenance **unmissable**: source name, source type, clickable URL, source date.
3. **Refuse** advice, performance prediction, and ranking — reliably, without a lecture.
4. **Never** fabricate. Low confidence → say "not found in the available sources."
5. **Never** store or transmit user PII.
6. Be deployable: local dev first, then Render, with configuration parity.

## 5. Non-Goals (explicitly out of scope)

- Investment / portfolio / asset-allocation advice
- Buy, sell, hold, or switch recommendations
- Fund rankings, "best fund", scores, leaderboards
- Return predictions, CAGR/SIP/XIRR calculators, projections
- Fund comparison on future performance
- Personalization of any kind (no age, income, risk-tolerance, holdings collection)
- Portfolio management, transaction execution, account servicing
- Multi-AMC breadth, scheduled crawling, admin dashboards (roadmap only)

## 6. Scope

### 6.1 AMC

HDFC Mutual Fund. Structured so another AMC is a **configuration** change, not a rewrite.

### 6.2 Schemes (initial five)

| scheme_id | Scheme |
| --- | --- |
| `HDFC_LARGE_CAP` | HDFC Large Cap Fund — Direct Growth |
| `HDFC_FLEXI_CAP` | HDFC Flexi Cap Fund (formerly HDFC Equity Fund) — Direct Growth |
| `HDFC_ELSS` | HDFC ELSS Tax Saver Fund — Direct Growth |
| `HDFC_SMALL_CAP` | HDFC Small Cap Fund — Direct Growth |
| `HDFC_BALANCED_ADVANTAGE` | HDFC Balanced Advantage Fund — Direct Growth |

### 6.3 Supported question types

Minimum investment · minimum SIP · additional investment · expense ratio · exit load · scheme
category · benchmark · riskometer · ELSS lock-in period · scheme/document information.

### 6.4 Unsupported (refused or "not found")

Recommendations · best/worst fund · expected returns · future NAV/performance · suitability ·
tax planning advice · jokes/weather/general knowledge.

## 7. Source Strategy

Preference order, reflected in `source_type` + `authority_level`:

| source_type | authority_level | Meaning |
| --- | --- | --- |
| `AMC_OFFICIAL` | 1 | Published by HDFC Mutual Fund itself |
| `AMFI` | 2 | Association of Mutual Funds of India |
| `SEBI` | 2 | Securities and Exchange Board of India |
| `REFERENCE` | 3 | Third-party public aggregator (e.g. Groww) |

The five seed sources are Groww scheme pages → `REFERENCE`. **A `REFERENCE` page is never labelled
"official HDFC Mutual Fund".** Authority level is used internally to resolve conflicts and is not
shown as a numeric score to users.

## 8. User Journeys

### 8.1 Factual (happy path)
Open site → read disclaimer → pick a scheme or ask generally → ask "exit load for HDFC ELSS" →
classification `FACTUAL` → scheme resolved → embedding → Chroma top-K → confidence OK → Gemini
answers using only context → backend attaches citation → UI shows answer + source + date.

### 8.2 Ambiguous
"What is the exit load?" → multiple schemes plausible → system asks
"Which HDFC Mutual Fund scheme would you like the exit load for?" and lists options. **No guessing.**

### 8.3 Advice refusal
"Should I invest in HDFC ELSS?" → `ADVICE` → short refusal + redirect to factual topics.

### 8.4 Performance refusal
"Which will give the highest return?" → `PERFORMANCE` → no prediction, no ranking; points to
official documents for published historical figures.

### 8.5 PII
"My PAN is ABCDE1234F" → `PII_RISK` → safe notice, nothing stored, nothing forwarded to the LLM.

### 8.6 Unknown / not in corpus
"What's the NAV in 2036?" → `OUT_OF_SCOPE`/low confidence → "I couldn't find that information in
the available sources."

## 9. Functional Requirements

| ID | Requirement |
| --- | --- |
| FR-01 | `GET /health` — cheap liveness, no LLM, no embedding work |
| FR-02 | `GET /api/v1/schemes` — configured schemes; frontend must not hard-code them |
| FR-03 | `GET /api/v1/sources` — indexed sources with type, doc type, URL, last-updated |
| FR-04 | `POST /api/v1/chat` — validated request → grounded answer + citations |
| FR-05 | Question length 1–1000 chars; empty rejected; unknown `scheme_id` rejected |
| FR-06 | Scheme selection optionally constrains retrieval |
| FR-07 | Ambiguity triggers clarification, not a guess |
| FR-08 | Advice/performance requests get canned refusals, no LLM call |
| FR-09 | PII-shaped input gets a safe notice and is not persisted |
| FR-10 | Confidence below threshold → "not found" fallback, **LLM is not called** |
| FR-11 | Every factual answer carries ≥1 backend-generated citation |
| FR-12 | `last_updated` comes from document metadata or is honestly "Date not available" |
| FR-13 | Ingestion is CLI-only — no public ingestion endpoint |
| FR-14 | No chat history persistence |

## 10. RAG Requirements

- **Ingestion:** configured URL allow-list only (SSRF-safe) → fetch with timeout/retry/size cap and
  identifiable UA → HTML extraction → cleaning → metadata → structure-aware chunking → local
  embedding → Chroma.
- **Chunking:** heading-anchored, 500–800 tokens target with 50–100 token overlap, **smaller chunks
  allowed for compact factual fields**. A heading never detaches from the fact it labels.
  Tables keep header/row relationships.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, loaded **once** as a process singleton,
  normalized. No external embedding API.
- **Retrieval:** configurable `TOP_K` (default 5) + optional `scheme_id` metadata filter +
  `SIMILARITY_THRESHOLD` (default 0.45) gate. Whole documents are never dumped into the prompt.
- **Idempotency:** deterministic `document_id`/`chunk_id` from content hash; unchanged content is not
  re-embedded; changed content replaces its chunks (no duplicates on re-run).

## 11. Citation Requirements

- Citations are **constructed by the backend from retrieved chunk metadata**, never by the LLM.
- Only URLs that exist in the indexed/configured source set may be returned.
- `source_type` is passed through verbatim so the UI can distinguish official vs reference.
- If no valid source survives validation → do not present the answer as verified.

## 12. Safety Requirements

- Server-side facts-only system prompt; retrieved content and user text are **data**, never instructions.
- Response validation: non-empty, cited, no recommendation phrasing, no performance prediction, length cap.
- Advice/performance/PII paths are deterministic and do not reach the LLM.

## 13. PII Requirements

- Detect PAN, Aadhaar, bank account, card, OTP, phone, email, folio, password/token patterns.
- Do not persist raw messages; no chat history; no PII in logs.
- PII-shaped input is answered with a notice; the text is not forwarded to Gemini.

## 14. UX Requirements

- States: Welcome → Retrieving → Answer (+ clarification, refusal, error, empty).
- Header with product name, tagline, and a **real** "N schemes indexed" badge from the API.
- Three tappable example questions on the welcome screen + "Facts-only. No investment advice."
- Assistant answers: answer first, source immediately below, not every message a giant card.
- Source card: title, human-readable authority label (HDFC Mutual Fund / AMFI / SEBI / Reference
  source), `View source ↗` with `target="_blank" rel="noopener noreferrer"`, last-updated line.
- Loading copy is human ("Searching fund documents…"), never exposes embeddings/IDs/prompts/scores.
- Distinct error copy for network, generation, no-result, and validation failures.
- Responsive from 375px up; long URLs must not break layout.
- Accessible: semantic HTML, labelled form, keyboard operable, visible focus, sensible ARIA,
  screen-reader announcements for new messages.
- Permanent disclaimer: facts-only, not advice, verify with official AMC/AMFI/SEBI docs, no PII.

## 15. API Requirements

Backend **Python + FastAPI only** (non-negotiable). Frontend is Next.js/React/TypeScript and talks
HTTP/JSON. Typed Pydantic request/response models. CORS restricted via `ALLOWED_ORIGINS` (never
`*` in production). Rate-limit-ready architecture (client-IP + request-ID middleware, no external
store yet). No stack traces to users. Swagger documented as a developer surface.

## 16. Deployment Requirements

- Runs locally first: `uvicorn` + `npm run dev`.
- Render: backend Web Service (`uvicorn app.main:app --host 0.0.0.0 --port $PORT`), frontend Static
  Site with `NEXT_PUBLIC_API_BASE_URL`.
- `GEMINI_API_KEY` supplied via environment only — local `.env`, Render dashboard/Blueprint `sync: false`.
- **No secret in Git, no `NEXT_PUBLIC_*` secret, no secrets in `render.yaml`.**
- Same env var names locally and on Render (values differ, e.g. `./data/chroma` vs `/var/data/chroma`).
- Chroma persistence mode configurable; ephemeral default on Render is documented as **not durable**.

## 17. Success Metrics

| Metric | Target |
| --- | --- |
| Factual answer grounded in a cited source | ≥ 90% on the evaluation set |
| Advice refusal correctness | 100% |
| Performance refusal correctness | 100% |
| PII-shaped input handled without forwarding | 100% |
| Fabricated citations | 0 |
| Answer length | ≤ 3 sentences |
| p95 end-to-end latency (warm) | < 8s |

## 18. Evaluation Strategy

`evaluation/questions.json` (factual, ambiguous, advice, performance, PII, unknown) with expected
behaviours, executed by `scripts/evaluate.py`, producing a report across retrieval, generation,
citation, and safety dimensions.

## 19. Acceptance Criteria

**Ingestion** — 5 sources downloadable; cleaning, metadata, chunking, embeddings, Chroma storage all
work; re-ingestion creates no uncontrolled duplicates; every chunk maps to scheme + source.

**Retrieval** — query embedding works; top-K configurable; scheme filter works; threshold exists;
unknown questions never produce fabricated answers; ambiguous queries ask for clarification.

**LLM** — Gemini called only from FastAPI; key from environment only; key never reaches frontend or
Git; provider abstraction exists; facts-only prompt enforced; LLM sees only retrieved context; no
returns, no recommendations.

**Citations** — every factual answer cited; citations backend-generated; URLs not LLM-generated;
clickable; source type labelled correctly; last-updated metadata honest.

**Security** — `.env`/`.env.local`/`*.secret` ignored; no hard-coded secrets; no API keys in logs; no
PII persistence; no public ingestion endpoint; CORS restricted; input validated; no stack traces leaked.

**Backend** — Python + FastAPI + Pydantic; all four endpoints; `/docs` works locally; backend runs
standalone.

**Frontend** — Next.js + React + TypeScript; responsive; welcome state; examples; scheme selector;
chat, loading, error, source cards, disclaimer, mobile usable.

**Render** — config exists; `$PORT` + `0.0.0.0`; configurable API URL; secret via Render config; no
secrets in `render.yaml`; persistent-disk path configurable; documented.

## 20. Known Limitations

- Single AMC, five schemes, English only.
- Seed corpus is `REFERENCE` (aggregator) grade; official HDFC/AMFI documents would raise authority.
- Manual ingestion only — no scheduled refresh.
- Dense retrieval only; no hybrid lexical, no reranking.
- No per-user memory by design.
- Ephemeral Render deployments must re-ingest after restart/deploy.
- Gemini project billing/credit state is an operational dependency outside the codebase.
- Aggregator pages are JS-heavy; extraction quality depends on the page's server-rendered HTML.

## 21. Future Roadmap (not implemented)

More AMCs/schemes · official HDFC + AMFI + SEBI document ingestion · hybrid lexical + vector
retrieval · reranking · scheduled source refresh · document version diffing · multilingual
questions · Hindi/Hinglish UI · authenticated admin ingestion & source management · analytics and
evaluation dashboards · production rate limiting (Redis) · managed vector database.
