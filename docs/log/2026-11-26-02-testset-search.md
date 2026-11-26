# Search and evaluate on the labelled test set (step 6c, part 2)

- Date: 2026-11-26
- Type: feature, fix

## What changed
- **Cascade notebook, section 6, "Search the test set":** an interactive search whose gallery is every annotated box of `annotations.json`, plus the reviewed images without boxes as distractors.
  - Queries are either a labelled box from the dropdown, or any image with a box you draw.
  - An **Expected** label field (autocompleted from the test-set labels) decides the ✓/✗ marks. It's pre-filled for a test-set box, typed for a new image, and empty means no marks.
  - Results show a crop around each gallery box, with the box drawn.
- **Self-exclusion:** `Cascade.run(query, exclude=...)` leaves given gallery ids out of the results. The app uses it so a query picked from the test set never finds its own box.
- **Datasets cell:** the labelled test set is now the "real" dataset for the stage-1 and stage-2 tables. Metrics are per query box (one box per label is the query). The folder-per-object layout is used only when there's no test set.
- **Notebook structure:** numbered sections (1–7), a markdown explanation before every code cell, and a placeholder section 7 for "Search a new gallery" (part 3).
- **`DinoRetriever.crop_descriptors`:** `crop_cls` descriptors of boxed items are now **batched and cached on disk** per image + box (`cache/dinov2-crop/`). A test set with many boxes is embedded once, and repeated query crops are free. File fingerprints are also memoized.
- **`viz.gallery.box_thumbnail`:** a crop with context and the box drawn, for the result grid.
- **Fix:** `ColorHistogramRetriever` ignored gallery boxes and used the whole image for every box of a photo. It now uses each item's box, like the DINOv2 retriever.

## Why
TASK.md and the owner's request: compare a query box against the labelled test set and see correctness per box, separately from the upcoming new-gallery search. The labelled set is where per-box metrics come from.

## Impact
- **Behavior:** box-to-box comparison on the test set in both stages. Stage 1 crops gallery boxes, and stage 2 matches inside gallery boxes; both were already supported for prompted gallery items.
- **Results (synthetic, smoke test):**
  - The color-histogram baseline went from R@1 0.13 to **0.40**, because synthetic gallery items have boxes and the baseline now uses them. Earlier baseline numbers in this log are not comparable.
  - DINOv2 results are unchanged.
  - On a temporary five-object synthetic test set, `crop_cls` reaches R@1 1.0. That only proves the plumbing works.
- **Tests:** 115 pass, including per-box records, the crop-descriptor cache, self-exclusion, per-box evaluation with SIFT (R@1 1.0), and the app on a test set (pre-filled expected label, self-exclusion, the expected label for a new image).
- **Verification:** the notebook executes headless against a temporary labelled test set. It was not checked in a real browser.
- **Follow-ups:**
  - Part 3: DINOv2 gallery grounding and section 7.
  - The evaluation notebook: per-group (model-level) metrics, and grounding evaluated against annotated boxes.

## Evidence (optional)
`uv run pytest tests/test_testset_search.py`. `notebooks/cascade.ipynb` with `VS_TESTSET` pointing to a five-object synthetic test set.
