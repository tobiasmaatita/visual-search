# Repo skeleton, dataset schema, split and synthetic smoke data

- Date: 2026-11-24
- Type: feature

## What changed
- **Project setup.**
  - Set up as a uv project (`pyproject.toml` + `uv.lock`) with a `src/visual_search` package.
  - Base dependencies: numpy, pillow, pyyaml, opencv-python-headless (<5) and mlcvtools as an editable path dependency.
  - Dev tools: pytest and ruff.
- **Dataset schema** (`data/schema.py`).
  - An `Item` has an image, a label (`None` = distractor) and an optional `Prompt` (box, labelled points, or mask).
  - Any other key in a record, such as a future `group`, is kept untouched in `Item.extra`.
  - A pair is a correct match only when both labels are equal and not `None`.
- **Validation and split** (`data/split.py`).
  - Each instance needs at least 2 images.
  - The split into queries and gallery is reproducible from a seed and doesn't depend on input order.
  - All distractors go to the gallery, and at least one image of each instance always stays in the gallery.
  - In strict mode a broken dataset raises an error; in lenient mode the broken instances are dropped.
- **Folder loader** (`data/folder.py`).
  - The dataset is one folder per instance, with `singletons/`, `_distractors/` or `distractors/` for `label=None`.
  - Optional `prompts.json`.
  - A separate distractor folder loader (e.g. for COCO).
  - A read-only importer for the `own-annotations.json` written by mlcvtools' visual-search notebook.
- **Synthetic smoke-test set** (`data/synthetic.py`, clearly marked SMOKE TEST ONLY).
  - Procedural textured patches pasted into cluttered scenes under random perspective and photometric changes.
  - Each image carries a ground-truth mask, a box prompt and the homography that created it.
  - Optional near-duplicate instances that share a `group` (hard negatives).

## Why
TASK.md asks for the dataset loader, validator and a smoke-test fallback first, so the evaluation harness (next step) has something to run on.

The folder convention means a small real dataset needs no annotation tool. The mlcvtools importer keeps boxes already drawn in that notebook usable, without writing to mlcvtools.

## Impact
- Behavior/results: 40 tests pass, covering schema, split, folder loader and synthetic data.
- Current state of the owner's real data:
  - `data/billy_01` has 1 image, so the validator flags it.
  - `data/bank_01` is empty.
  - The 3 images in `data/singletons` are treated as distractors.
  - So there are no queries yet. Two of the singletons are byte-identical to mlcvtools' `bank1.png` / `bank2.png`, which belong in `bank_01/` if they show the same physical sofa.
- Code/interfaces: later stages consume `Item` and `Split`. Masking, matching and scoring tests can use the synthetic `gt_mask` and `gt_homography`.
- Dependency note: mlcvtools requires `opencv-python-headless<5`, so we pin the same range.
- Follow-ups:
  - Next is the evaluation harness (Recall@1/5, MRR, latency) with a trivial baseline.
  - The `distractors_dir` config option arrives together with `config.py` in a later step.

## Evidence (optional)
`uv run pytest` (40 passed), `uv run ruff check src tests` (clean). mlcvtools `git status` was identical before and after adding the dependency.
