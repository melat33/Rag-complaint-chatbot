# Notebooks

Run these in order. Each is a thin, visual wrapper around `src/` — the tested,
reusable logic lives there; the notebooks exist for exploration, plots, and the
report deliverables.

1. **`01_eda_preprocessing.ipynb`** — Task 1. Loads `data/raw/complaints.csv`, plots
   product distribution and narrative word-count histograms, breaks down
   narrative-coverage, filters to the five target products, cleans the text, and
   saves `data/processed/filtered_complaints.csv`.

2. **`02_chunking_embedding_experiments.ipynb`** — Task 2. Draws a proportional
   stratified sample (10-15K complaints), compares chunk-size options visually,
   benchmarks embedding speed, then runs the full chunk → embed → index pipeline
   and sanity-checks it with a live query against `vector_store/chroma_db/`.

3. **`03_rag_pipeline_demo.ipynb`** — Task 3. Loads the vector store, demonstrates
   the retriever in isolation (with and without a product filter), shows the exact
   prompt sent to the LLM, runs the full `RAGPipeline.ask()` call, then runs 10
   representative questions through the pipeline and exports the qualitative
   evaluation table to `evaluation/eval_questions.md`. Also includes a failure-case
   check (out-of-scope question) to confirm the system admits uncertainty instead
   of hallucinating.

**Prerequisites:** `pip install -r requirements.txt`, `GROQ_API_KEY` set in `.env`
(needed from notebook 3 onward), and `data/raw/complaints.csv` present (needed from
notebook 1 onward).
