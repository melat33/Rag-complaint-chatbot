# CrediTrust Complaint Intelligence — RAG-Powered Chatbot

**🔗 [Live App](https://rag-complaint-chatbot.vercel.app/) · [API](https://rag-complaint-chatbot.onrender.com) · [API Docs](https://rag-complaint-chatbot.onrender.com/docs)**

> Note: the backend runs on Render's free tier, which spins down after 15 minutes of inactivity. The first question after idling may take 30-60 seconds to wake up — that's expected, not a bug.

A full-stack Retrieval-Augmented Generation (RAG) system that turns 456K+ unstructured
customer complaint narratives into evidence-backed answers for Product, Support, and
Compliance teams — no data analyst required.

Built for CrediTrust Financial, a digital finance platform serving 500,000+ users across
Credit Cards, Personal Loans, Savings Accounts, and Money Transfers in East Africa.

---

## Contents

- [Why this exists](#why-this-exists)
- [Results](#results)
- [Architecture](#architecture)
- [Tech stack & key decisions](#tech-stack--key-decisions)
- [Project structure](#project-structure)
- [Setup](#setup)
- [API reference](#api-reference)
- [Evaluation](#evaluation)
- [Challenges & what I learned](#challenges--what-i-learned)
- [Roadmap](#roadmap--future-improvements)

---

## Why this exists

CrediTrust's Product Managers were spending days manually reading complaints to spot
trends. This system collapses that to seconds: ask a plain-English question, get an
answer grounded in real complaint excerpts, with every claim traceable back to its
source.

## Results

Built and tested against the full historical CFPB complaint export, not a sample:

| Metric | Value |
|---|---|
| Raw complaints scanned | 9,609,797 |
| Complaints with a narrative | 2,980,756 (31%) |
| Filtered to 4 target products | 456,104 |
| Chunks indexed (evaluation build) | 35,517 |
| Retrieval + generation latency | sub-second (Groq inference) |

## Architecture
This deployment saga is genuinely great material for your README's "Challenges" section — diagnosing a memory ceiling and redesigning around it is real engineering work. Here's the complete final README with live links up top and a clickable table of contents:
markdown# CrediTrust Complaint Intelligence — RAG-Powered Chatbot

**🔗 [Live App](https://rag-complaint-chatbot.vercel.app/) · [API](https://rag-complaint-chatbot.onrender.com) · [API Docs](https://rag-complaint-chatbot.onrender.com/docs)**

> Note: the backend runs on Render's free tier, which spins down after 15 minutes of inactivity. The first question after idling may take 30-60 seconds to wake up — that's expected, not a bug.

A full-stack Retrieval-Augmented Generation (RAG) system that turns 456K+ unstructured
customer complaint narratives into evidence-backed answers for Product, Support, and
Compliance teams — no data analyst required.

Built for CrediTrust Financial, a digital finance platform serving 500,000+ users across
Credit Cards, Personal Loans, Savings Accounts, and Money Transfers in East Africa.

---

## Contents

- [Why this exists](#why-this-exists)
- [Results](#results)
- [Architecture](#architecture)
- [Tech stack & key decisions](#tech-stack--key-decisions)
- [Project structure](#project-structure)
- [Setup](#setup)
- [API reference](#api-reference)
- [Evaluation](#evaluation)
- [Challenges & what I learned](#challenges--what-i-learned)
- [Roadmap](#roadmap--future-improvements)

---

## Why this exists

CrediTrust's Product Managers were spending days manually reading complaints to spot
trends. This system collapses that to seconds: ask a plain-English question, get an
answer grounded in real complaint excerpts, with every claim traceable back to its
source.

## Results

Built and tested against the full historical CFPB complaint export, not a sample:

| Metric | Value |
|---|---|
| Raw complaints scanned | 9,609,797 |
| Complaints with a narrative | 2,980,756 (31%) |
| Filtered to 4 target products | 456,104 |
| Chunks indexed (evaluation build) | 35,517 |
| Retrieval + generation latency | sub-second (Groq inference) |

## Architecture
┌─────────────┐     HTTP/SSE     ┌──────────────┐     similarity search    ┌─────────────┐
│   React UI   │ ───────────────▶ │   FastAPI     │ ─────────────────────▶ │  ChromaDB    │
│  (frontend/) │ ◀─────────────── │  (backend/)   │ ◀───────────────────── │(vector_store)│
└─────────────┘   answer+sources  └──────┬───────┘      top-k chunks       └─────────────┘
│
┌───────────┴────────────┐
▼                         ▼
┌─────────────┐          ┌──────────────┐
│  Groq LLM    │          │ HF Inference  │
│ Llama 3.3 70B│          │ API (query    │
│ (generation) │          │ embedding)    │
└─────────────┘          └──────────────┘
1. **Retriever** embeds the question (via Hugging Face's Inference API in production —
   see [Challenges](#challenges--what-i-learned) for why) and pulls the top-k most
   similar complaint chunks from ChromaDB, optionally filtered by product.
2. **Generator** feeds those chunks into a strict analyst prompt sent to Groq's
   Llama 3.3 70B, which must answer *only* from the retrieved context.
3. **Backend** (FastAPI) exposes `/ask` and `/ask/stream` (Server-Sent Events for
   token-by-token streaming) and validates every request/response with Pydantic.
4. **Frontend** (React) streams the answer live and shows every source excerpt it was
   grounded in, so a PM can verify the AI isn't fabricating a trend.

## Tech stack & key decisions

| Layer | Choice | Why |
|---|---|---|
| Embeddings (offline indexing) | `sentence-transformers/all-MiniLM-L6-v2`, run locally | 384-dim, strong semantic similarity for its size, fast to embed hundreds of thousands of chunks on CPU |
| Embeddings (live queries) | Hugging Face Inference API | Deployed backend embeds one short question at a time — an API call is lighter than loading torch for that |
| Vector store | ChromaDB | Persists to disk out of the box; stores metadata alongside vectors natively |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, 500 chars / 50 overlap | Validated against alternatives (250/25 and 1000/100) on real narratives |
| Product matching | Keyword-based, not exact-string | CFPB renames product categories over time; exact-match filtering silently dropped 3 of 4 target categories on the real dataset |
| LLM | Groq — Llama 3.3 70B Versatile | Sub-second inference (LPU hardware); strong instruction-following for "stay grounded, admit uncertainty" |
| Backend | FastAPI, deployed via Docker on Render | Async, auto-generated OpenAPI docs, Pydantic validation |
| Frontend | React + Vite, deployed on Vercel | Real product UI — streaming chat, evidence panel, product filtering |

## Project structure
rag-complaint-chatbot/
├── src/                      # RAG library
│   ├── config.py                # single source of truth for paths/hyperparameters
│   ├── data_processing.py       # Task 1: streaming EDA + cleaning (memory-safe for multi-GB files)
│   ├── chunking.py              # Task 2: stratified sampling + text splitting
│   ├── embedding.py             # Task 2: OFFLINE bulk embedding + indexing (local, torch-based)
│   ├── query_embedding.py       # Task 3: LIVE query embedding via Hugging Face API (deployed backend)
│   ├── vector_store.py          # ChromaDB access, no heavy ML dependencies
│   ├── retriever.py             # Task 3: similarity search
│   ├── generator.py             # Task 3: Groq LLM calls (sync + streaming)
│   ├── prompt_templates.py      # the analyst prompt
│   ├── rag_pipeline.py          # orchestrates retriever + generator
│   ├── build_index.py           # CLI: run Task 2 end-to-end
│   └── run_rag_demo.py          # CLI: run Task 3 evaluation end-to-end
├── backend/                   # FastAPI web layer
│   ├── main.py                   # routes: /health, /ask, /ask/stream
│   ├── schemas.py                # request/response contracts
│   └── services/rag_service.py
├── frontend/                   # React chat UI, deployed on Vercel
│   └── src/{App.jsx, components/, api/}
├── tests/                      # pytest, mocked LLM/vector store — no network calls in CI
├── notebooks/                   # EDA, chunking experiments, RAG demo + evaluation
├── evaluation/eval_questions.md      # Task 3 qualitative evaluation table
├── Dockerfile                  # deployed backend image (leaner than local dev)
├── requirements.txt             # full local dev deps (includes torch, sentence-transformers)
├── requirements-backend.txt      # deployed backend deps (no torch — see Challenges)
└── .github/workflows/unittests.yml
## Setup

### 1. Backend (local development)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env         # then add your GROQ_API_KEY and HF_TOKEN
```

Get a free Groq key at [console.groq.com](https://console.groq.com) and a free
Hugging Face token at [huggingface.co](https://huggingface.co) (Settings → Access Tokens).

**Run the pipeline:**

```bash
python -m src.data_processing   # Task 1 — produces data/processed/filtered_complaints.csv
python -m src.build_index       # Task 2 — samples, chunks, embeds, indexes into ChromaDB
python -m src.run_rag_demo      # Task 3 — runs 10-question evaluation, saves to evaluation/
```

or step through `notebooks/01` → `02` → `03` for the visual, annotated version.

**Start the API:**

```bash
uvicorn backend.main:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`.

### 3. Tests

```bash
pytest tests/ -v
```

### 4. Deployment

Backend is deployed via Docker on [Render](https://render.com) (`Dockerfile` +
`requirements-backend.txt`); frontend on [Vercel](https://vercel.com) (root directory
`frontend/`, env var `VITE_API_BASE_URL` pointing at the Render URL). See
[Challenges](#challenges--what-i-learned) for the memory-constraint redesign this required.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + vector store chunk count |
| `/ask` | POST | `{question, product_filter?}` → `{answer, sources[]}` |
| `/ask/stream` | POST | Same input, Server-Sent Events: `sources` → `token`* → `done` |

## Evaluation

See [`evaluation/eval_questions.md`](evaluation/eval_questions.md) for the 10-question
qualitative evaluation table. Notable result: on "Are there recurring complaints about
a specific company?", the system correctly declined to over-claim a company name it
wasn't confident about, rather than fabricating one — the groundedness constraint in
the prompt working as designed.

## Challenges & what I learned

- **CFPB's product taxonomy isn't stable.** Exact-string product filtering worked on a
  small test sample but silently returned zero rows for 3 of 4 target categories
  against the real 9.6M-row dataset, since CFPB had renamed categories. Fixed with
  keyword-based substring matching instead of exact matches.

- **Loading a multi-GB CSV in memory doesn't scale.** Switched to chunked streaming
  reads so the pipeline runs on a standard laptop regardless of file size, and to avoid
  the sampling bias from truncating to the first N rows of a date-sorted file.

- **Streaming and non-streaming API responses can silently drift out of sync.** The
  `/ask/stream` endpoint returned differently-shaped source metadata than `/ask`,
  causing complaint IDs to render blank in the UI. Fixed by extracting one shared
  flattening function both endpoints call.

- **Free-tier deployment memory limits forced a real architecture decision.** The
  initial deploy loaded `torch` + `sentence-transformers` + a 384MB ChromaDB index in
  the same process as the API server — comfortably exceeding Render's free-tier 512MB
  RAM limit and crash-looping on every request. Rather than just paying for more RAM,
  I split the pipeline: bulk indexing (`src/embedding.py`) stays local and
  torch-based, since it only ever runs once, offline, on my own machine; the deployed
  backend embeds each live query via a hosted Hugging Face Inference API call instead
  (`src/query_embedding.py`) — using the *same* underlying model, so retrieval quality
  is unaffected, but the deployed process never loads torch at all. This also meant
  discovering that a plain `pip install torch` pulls in unused CUDA/GPU libraries even
  on a CPU-only host — switching to the explicit CPU-only wheel cut the deploy image
  down significantly.

## Roadmap / future improvements

- Tune `top_k` per question type — broad questions likely need more retrieved chunks
  than narrow ones
- Swap ChromaDB for a managed vector DB (Pinecone/Weaviate) for multi-instance deployment
- Add conversation memory (multi-turn follow-up questions)
- Add a `/compare` endpoint for cross-product trend comparison
- Cache frequent queries to cut LLM cost
- Add reranking (cross-encoder) on top of the initial vector search for higher precision

---

**Author:** Melat — AI/ML Engineer 