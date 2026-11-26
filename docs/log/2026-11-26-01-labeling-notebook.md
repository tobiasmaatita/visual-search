# Per-object annotations and a labeling notebook (step 6c, part 1)

- Date: 2026-11-26
- Type: feature, decision

## What changed
- **`data/annotations.py`:** a new test-set format, `annotations.json`, stored next to the images.
  - Each image has a list of boxes, each with `label` (the exact physical item) and an optional `group` (a coarser identity such as the product model).
  - An image listed with no boxes is *reviewed* and becomes a whole-image distractor. Images not listed are *not labelled yet* and stay out of the dataset.
  - `AnnotationSet` handles loading, saving (atomic, with an `annotations.json.bak` backup), adding, relabelling and deleting boxes, plus the reviewed state.
  - Label statistics: boxes and images per label, with warnings for single-box labels and look-alike labels.
  - `to_records()` turns the set into dataset items: one item per box (id `image#n`, the box as prompt, `image_id` and `group` as extra keys). It feeds straight into the existing split and evaluation.
- **Near-duplicate label check** (`similar_labels`):
  - Labels are compared ignoring case, spaces, dashes and underscores, with spelling tolerance for typos.
  - Labels whose numbers differ are never flagged, because `bank_01` vs `bank_02` is deliberate. Numbers compare by value, so `bank_1` and `bank_01` count as the same.
- **`ui/labeling.py`: `LabelingApp`.**
  - **Navigation:** previous/next, *next unlabelled*, and a dropdown with a ✓ and box count per image.
  - **Adding:** draw a box (reusing the prompt canvas), then type a label and group with autocomplete over the labels already in use. A look-alike new label needs an explicit *Add anyway*.
  - **Fixing:** a box list with relabel and delete, plus a *reviewed* checkbox for images without objects.
  - **Overview:** a live label table, and every change is saved immediately.
- **`PromptWidget.show_boxes`:** draws labelled boxes on the canvas. The labeler uses it now; the result views will use it for regions.
- **`notebooks/labeling.ipynb`:** a separate labeling notebook. Every code cell is preceded by a markdown cell explaining what it does.
  - The default folder is `data/testset/` (gitignored); override it with `VS_TESTSET`.
  - It includes a label check table and a preview of the split the evaluation will use.

## Why
The owner wants to build their own per-object test set, with metrics scored per box. Labels must stay consistent across the whole set: the same item keeps exactly the same label, and different spellings must not create false "different objects". The `group` field covers the blurry *same model* level next to the *exact item* level.

mlcvtools' anywidget labeller (`label_dataset_items`) was considered and not used. It works with a fixed class list, while these labels are an open, growing vocabulary of item ids that needs autocomplete and duplicate checks.

## Impact
- **Behavior:** the owner can label now. Evaluation code already accepts the records, since each box is an item with a box prompt. The test-set search section (part 2) and gallery grounding (part 3) build on this.
- **Tests:** 109 pass, including the format roundtrip and backup, reviewed semantics, records and split, the label rules, and the labeling app driven by simulated mouse drags (add, duplicate confirmation, relabel, delete, navigation, reviewed toggle).
- **Verification:** the notebook executes headless on a temporary three-image set. It was not checked in a real browser session.
- **Follow-ups:**
  - Part 2: test-set mode in the cascade notebook, with gallery = annotated boxes and ✓/✗ per box, plus per-box evaluation.
  - Part 3: DINOv2 grounding and a new-gallery mode (directory or glob, top-n).
  - Later: possibly split the cascade notebook into "free play" and "evaluation".

## Evidence (optional)
`uv run pytest tests/test_annotations.py`. `notebooks/labeling.ipynb` executed with nbconvert (`VS_TESTSET` pointing to a temporary set).
