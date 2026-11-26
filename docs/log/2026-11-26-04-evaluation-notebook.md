# Evaluation notebook and the notebook split (step 6d)

- Date: 2026-11-26
- Type: feature, decision

## What changed
- **The notebooks are reorganized into three, one per job.** `cascade.ipynb` is removed; its parts moved into the other two.
  - `labeling.ipynb`: unchanged.
  - `search.ipynb`: free play, with *Search the test set* (✓/✗) and *Search a new gallery* (grounding, top-n).
  - `evaluation.ipynb`: all measurements.
  - Every code cell has an explanation cell above it.
- **Judges** (`eval/judges.py`) define when a result counts as correct:
  - `ItemJudge`: same label, the exact item.
  - `GroupJudge`: same `group` (model); items without a group count as their own group.
  - `GroundedJudge`: a found region overlaps an annotated box of the right label (or group) with IoU ≥ a threshold; a whole-image result counts when the photo contains such a box.
- **Harness:**
  - `run_evaluation` evaluates several judges in one run. The primary judge gets plain columns (`R@1`, `R@5`, `MRR`); the others get prefixed columns (`model R@1`, …).
  - Everything from the query's own photo is excluded, so an object must be found in other photos.
  - New protocol `evaluate_leave_one_out`: every labelled box is a query once.
  - `evaluate` (one query per label) still works and now also excludes the query's own photo.
- **Test-set helpers:** `AnnotationSet.boxes_by_image()` and `image_records()` (the test set as an unlabelled photo collection, for the grounding evaluation).
- **`eval/inspect.py`:** a per-label table, worst queries, failure grids, localization statistics, a latency table, and CSV export to `results/<date-time>_<run>/` (gitignored).
- **`evaluation.ipynb` sections:**
  1. Setup.
  2. Test set.
  3. Stage 1 comparison.
  4. Stage 2 comparison.
  5. Grounding (annotated boxes vs DINOv2 grounding vs whole images, plus localization IoU).
  6. Per-label breakdown.
  7. Failure gallery.
  8. Latency.
  9. Save results.
- **Small fix:** taller rows in the result grid, so titles don't overlap the row above.

## Why
The owner wants to compare approaches on their own test set, with scores per box, and to separate free play from evaluation. Their choices for this step:
- three notebooks;
- leave-one-out, because it uses every label of a small test set, gives more queries and steadier numbers;
- grounding is correct at IoU ≥ 0.3, which is lenient because regions sit on a 32×32 grid; the threshold is a parameter.

The model level covers the owner's blurry line between "the exact item" and "the same model".

## Impact
- **Results** on a generated test set (SMOKE TEST: 8 synthetic objects × 3 views, 2 look-alike pairs sharing a group, 6 distractor photos; 24 queries):

| Pipeline | R@1 | MRR | model R@1 | model MRR |
|---|---|---|---|---|
| color histogram | 0.42 | 0.53 | 0.46 | 0.58 |
| DINOv2 `crop_cls` | 0.63 | 0.75 | 0.83 | 0.90 |
| + SIFT H MAGSAC++ | **1.00** | 1.00 | 1.00 | 1.00 |
| + ALIKED+LightGlue (top 10) | 0.92 | 0.94 | 0.96 | 0.96 |

  - **The model level shows what it should:** DINOv2 alone often returns the look-alike twin (model R@1 0.83 vs item R@1 0.63), and SIFT verification separates the twins. The failure gallery shows this case directly.
- **Grounding (same generated set):**
  - Stage-1 R@1 with annotated boxes is 0.63, vs 0.29 with DINOv2 grounding or whole images.
  - After SIFT: 1.00 (annotated boxes), 0.96 (DINOv2 grounding), 1.00 (whole images).
  - Localization: median IoU 0.63; 75 % of regions ≥ 0.3, 63 % ≥ 0.5.
  - These are synthetic numbers; real photos are needed before drawing conclusions.
- **Latency columns in leave-one-out are search cost only.** Query boxes are gallery boxes, so their features exist already. This is documented in the notebook and the guide; interactive latency belongs to `search.ipynb`.
- **Tests:** 131 pass.
  - The judges (item, group fallback, grounded IoU, model level).
  - Leave-one-out with two levels: the query's own photo is never returned, and unanswerable queries are skipped.
  - The grounding evaluation and localization statistics.
  - The per-label table, worst queries, failure grid, latency table and CSV export.
- **Verification:** both notebooks execute headless on the generated test set, and `evaluation.ipynb` also runs on its synthetic fallback (section 5 skipped). Not checked in a real browser.
- **Follow-ups:**
  - Run it on the owner's real test set.
  - Step 7 (a scoring combiner; per-label results show where inlier counts fail).
  - Possibly a configurable list of pipelines in one place, as the comparison grows.

## Evidence (optional)
`notebooks/evaluation.ipynb` with `VS_TESTSET` pointing to the generated set (seed 3). `uv run pytest tests/test_evaluation.py`.
