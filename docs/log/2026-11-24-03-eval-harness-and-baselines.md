# Evaluation harness, cascade skeleton and trivial baselines

- Date: 2026-11-24
- Type: feature

## What changed
- **Metrics** (`eval/metrics.py`).
  - Rank of the first correct result, Recall@k (hit rate) and MRR.
  - A query whose correct items fall outside the returned ranking counts as a miss.
- **Timing** (`timing.py`).
  - `StageTimer` collects wall-clock time per stage. It takes an optional sync hook, so GPU timings will be honest.
  - `LatencyStats` reports median, p95, mean and max.
- **Cascade skeleton** (`cascade.py`).
  - A stage-1 `Retriever`, then optional `RerankStage`s, each re-ranking the top-k of the previous stage.
  - The ranking is kept after every stage. The untouched tail is appended, so Recall@k stays comparable across stages.
  - One object serves both interactive `run(query)` and batch evaluation.
- **Harness** (`eval/harness.py`).
  - `evaluate(cascade, split)` gives metrics per stage (stage 1 vs re-ranked for free), latency per stage plus total, a per-query table for failure inspection, and the offline gallery-indexing time.
  - `comparison_table(reports)` builds ablation tables.
- **Retrieval base** (`retrieval/`).
  - `Retriever` and `EmbeddingRetriever` interfaces, plus an exact brute-force cosine index. FAISS isn't needed at the planned gallery sizes.
  - Two baselines: `RandomRetriever` (chance level) and `ColorHistogramRetriever` (HSV histogram; the query is cropped to its prompt box).
- **Notebook.** `notebooks/cascade.ipynb` runs both baselines on the synthetic set and the real `data/` folder.
- **Dependencies.**
  - pandas was added explicitly; it was already installed through mlcvtools.
  - jupyterlab, ipykernel, nbformat and nbconvert were added as dev dependencies.

## Why
TASK.md asks for the evaluation harness to be the first thing that works end to end, so every later method is compared on the same footing. The baselines prove the harness distinguishes a weak signal from chance.

## Impact
- Behavior/results (M2 dev machine, CPU):

| Dataset | Pipeline | R@1 | R@5 | MRR | median ms/query |
|---|---|---|---|---|---|
| synthetic (10 q, 40 gallery, SMOKE TEST) | random | 0.0 | 0.1 | 0.10 | 0.04 |
| synthetic | color_hist | 0.3 | 0.6 | 0.44 | 0.14 |
| real `data/` (1 q, 2 gallery) | color_hist | 1.0 | 1.0 | 1.00 | 14 (includes reading the PNG) |

  - The real set is too small to mean anything yet: `bank_01` has 2 images, `billy_01` has only 1 (dropped), and there is 1 distractor.
- Code/interfaces: stage 1 (DINOv2) implements `EmbeddingRetriever`. Matchers and dense matching will be `RerankStage`s.
- Follow-ups:
  - Step 3, DINOv2-small retrieval with a disk cache.
  - The per-query timing currently includes image loading from disk. Once caching is in place, the interactive path will reuse loaded images.

## Evidence (optional)
`uv run pytest` (54 passed). `notebooks/cascade.ipynb` executed headless with nbconvert, seed 0.
