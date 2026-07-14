# CrediTrust Complaint Intelligence — RAG-Powered Chatbot

A full-stack Retrieval-Augmented Generation (RAG) system that turns 456K+ unstructured
customer complaint narratives into evidence-backed answers for Product, Support, and
Compliance teams — no data analyst required.

Built for CrediTrust Financial, a digital finance platform serving 500,000+ users across
Credit Cards, Personal Loans, Savings Accounts, and Money Transfers in East Africa.

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
| Chunks indexed (12K stratified sample) | 35,517 |
| Retrieval + generation latency | sub-second (Groq inference) |

## Architecture
┌─────────────┐     HTTP/SSE     ┌──────────────┐     similarity search    ┌─────────────┐
│   React UI   │ ───────────────▶ │   FastAPI     │ ─────────────────────▶ │  ChromaDB    │
│  (frontend/) │ ◀─────────────── │  (backend/)   │ ◀───────────────────── │(vector_store)│
└─────────────┘   answer+sources  └──────┬───────┘      top-k chunks       └─────────────┘
│
▼
┌─────────────┐
│  Groq LLM    │
│ Llama 3.3 70B│
└─────────────┘
1. **Retriever** embeds the question with `all-MiniLM-L6-v2` and pulls the top-k most
   similar complaint chunks from ChromaDB.
2. **Generator** feeds those chunks into a strict analyst prompt sent to Groq's
   Llama 3.3 70B, which must answer *only* from the retrieved context.
3. **Backend** (FastAPI) exposes `/ask` and `/ask/stream` (Server-Sent Events for
   token-by-token streaming) and validates every request/response with Pydantic.
4. **Frontend** (React) streams the answer live and shows every source excerpt it was
   grounded in, so a PM can verify the AI isn't fabricating a trend.

## Tech stack & key decisions

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, ~80MB, fast enough to embed 1M+ chunks on CPU, strong semantic similarity for its size |
| Vector store | ChromaDB | Persists to disk out of the box; stores metadata alongside vectors natively |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, 500 chars / 50 overlap | Validated against alternatives (250/25 and 1000/100) on real narratives; 500/50 balances topical focus against fragmentation |
| Product matching | Keyword-based, not exact-string | CFPB renames product categories over time; exact-match filtering silently dropped 3 of 4 target categories on the real dataset — see Challenges below |
| LLM | Groq — Llama 3.3 70B Versatile | Sub-second inference (LPU hardware) so the chat UI feels real-time; generous free tier for a public portfolio deploy; strong instruction-following for "stay grounded, admit uncertainty" |
| Backend | FastAPI | Async, auto-generated OpenAPI docs, Pydantic validation |
| Frontend | React + Vite | Real product UI (not a notebook demo) — streaming chat, evidence panel, product filtering |

## Project structure
rag-complaint-chatbot/
├── src/                 # RAG library — no web framework dependency, fully unit-testable
│   ├── config.py           # single source of truth for all paths/hyperparameters
│   ├── data_processing.py  # Task 1: streaming EDA + cleaning (memory-safe for multi-GB files)
│   ├── chunking.py         # Task 2: stratified sampling + text splitting
│   ├── embedding.py        # Task 2: embedding + ChromaDB indexing
│   ├── retriever.py        # Task 3: similarity search
│   ├── generator.py        # Task 3: Groq LLM calls (sync + streaming)
│   ├── prompt_templates.py # the analyst prompt
│   ├── rag_pipeline.py     # orchestrates retriever + generator
│   ├── build_index.py      # CLI: run Task 2 end-to-end
│   └── run_rag_demo.py     # CLI: run Task 3 evaluation end-to-end
├── backend/              # FastAPI web layer
│   ├── main.py              # routes: /health, /ask, /ask/stream
│   ├── schemas.py           # request/response contracts
│   └── services/rag_service.py
├── frontend/              # React chat UI
│   └── src/{App.jsx, components/, api/}
├── tests/                 # pytest, mocked LLM/vector store — no network calls in CI
├── notebooks/              # thin wrappers around src/ for Tasks 1-3 exploration + eval
├── evaluation/eval_questions.md   # Task 3 qualitative evaluation table
└── .github/workflows/unittests.yml
## Setup

### 1. Backend

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env         # then add your GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

**Run the pipeline:**

Place the raw CFPB export at `data/raw/complaints.csv`, then either run the CLI scripts
or walk through the notebooks (recommended — they include EDA plots and chunking
experiments referenced in the report):

```bash
python -m src.data_processing   # Task 1 — produces data/processed/filtered_complaints.csv
python -m src.build_index       # Task 2 — samples, chunks, embeds, indexes into ChromaDB
python -m src.run_rag_demo      # Task 3 — runs 10-question evaluation, saves to evaluation/
```

or open and run top-to-bottom:
- `notebooks/01_eda_preprocessing.ipynb` — product distribution, narrative length
  histograms, narrative-coverage breakdown, then filters/cleans and saves the
  processed CSV.
- `notebooks/02_chunking_embedding_experiments.ipynb` — stratified sampling with
  before/after distribution plots, chunk-size sensitivity comparison, embedding speed
  benchmark, then runs the full chunk → embed → index pipeline and sanity-checks it
  with a test query against `vector_store/chroma_db/`.
- `notebooks/03_rag_pipeline_demo.ipynb` — retriever demo (with/without product
  filter), the exact prompt sent to the LLM, a full `RAGPipeline.ask()` trace, the
  10-question qualitative evaluation (auto-exported to `evaluation/eval_questions.md`),
  and a failure-case check confirming the system admits uncertainty instead of
  hallucinating on out-of-scope questions.

**Start the API:**
```bash
uvicorn backend.main:app --reload --port 8000
```
Docs available at `http://localhost:8000/docs`.

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
  against the real 9.6M-row dataset, since CFPB had renamed categories (e.g.
  "Personal loan" → "Payday loan, title loan, personal loan, or advance loan"). Fixed
  with keyword-based substring matching instead of exact matches.
- **Loading a multi-GB CSV in memory doesn't scale.** Switched to chunked streaming
  reads so the pipeline runs on a standard laptop regardless of file size, and to avoid
  the sampling bias that comes from truncating to the first N rows of a date-sorted file.
- **Streaming and non-streaming API responses can silently drift out of sync.** The
  `/ask/stream` endpoint returned differently-shaped source metadata than `/ask`,
  causing complaint IDs to render blank in the UI. Fixed by extracting one shared
  flattening function both endpoints call.

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