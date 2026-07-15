## Engineering decisions worth knowing about

A few choices in this codebase came from debugging real failures against real
production-scale data, not from following a tutorial:

- **Product filtering by keyword, not exact string.** CFPB renames its product
  categories over time — "Personal loan" became "Payday loan, title loan, personal
  loan, or advance loan" at some point. Exact-match filtering worked fine on a small
  test sample and then silently returned **zero rows** for 3 of 4 target categories
  against the real 9.6M-row dataset. Fixed with substring keyword matching that
  survives future renames.

- **Streaming CSV reads, not `pd.read_csv()`.** The full CFPB export is several GB.
  Loading it whole exhausts memory on a normal laptop, and `nrows=N` truncation
  biases the sample toward whatever the file happens to be sorted by (usually
  most-recent-first, which skews heavily toward low-narrative-consent complaint
  types). Chunked streaming reads solve both problems at once.

- **A shared metadata-flattening function, because two endpoints drifted apart.**
  `/ask/stream` returned differently-shaped source metadata than `/ask`, so complaint
  IDs rendered as a blank `#` in the UI even though retrieval was working correctly.
  The fix wasn't patching the symptom — it was extracting one function both endpoints
  call, so they structurally can't diverge again.

- **Query embedding moved off the deployed server entirely.** The first deploy loaded
  `torch` + `sentence-transformers` + a 384MB ChromaDB index in the same process as
  the API — comfortably over Render's free-tier 512MB limit, crash-looping on every
  request. Rather than just paying for more RAM, bulk indexing stays local (it only
  ever runs once, offline), while the deployed backend calls a hosted embedding API
  for the one short query it needs per question — same underlying model, so retrieval
  quality is unaffected, but the deployed process never loads torch at all. Along the
  way: a plain `pip install torch` pulls in unused CUDA libraries even on a CPU-only
  host — the explicit CPU-only wheel cut real weight from the deploy image too.

## Evaluation

10 representative questions run through the full pipeline, scored for groundedness
and usefulness — full table in [`evaluation/eval_questions.md`](evaluation/eval_questions.md).
Notable result: asked whether complaints recur against a specific company, the system
explicitly declined to name one it wasn't confident about rather than guessing — the
prompt's groundedness constraint working as intended, not a gap in the answer.

## Roadmap

- Tune `top_k` per question type — broad questions likely need more retrieved chunks
  than narrow ones
- Add conversation memory for multi-turn follow-ups
- Cross-encoder reranking on top of initial vector search
- Swap ChromaDB for a managed vector DB for multi-instance deployment

---

**Author:** Melat — AI/ML Engineer 

</div>