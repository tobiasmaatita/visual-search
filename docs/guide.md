# Repository guide

This guide explains what the code in this repository does and how to use it. It starts from the notebooks ([`notebooks/`](../notebooks/)) and works down into the `visual_search` package. It describes the state after step 6d: the labeling notebook, search on your labelled test set and on unlabelled photo collections (gallery grounding), and the evaluation notebook. The step-by-step history, with measurements, is in the [change log](log/README.md).

**Contents**

1. [What this repository is for](#1-what-this-repository-is-for)
2. [Getting started](#2-getting-started)
3. [Your data: what to provide and what to label](#3-your-data-what-to-provide-and-what-to-label)
4. [How a search works: the pipeline](#4-how-a-search-works-the-pipeline)
5. [The notebooks, cell by cell](#5-the-notebooks-cell-by-cell)
6. [Reading the scores and tables](#6-reading-the-scores-and-tables)
7. [What is available: models, retrievers, maskers, matchers](#7-what-is-available-models-retrievers-maskers-matchers)
8. [Extending the code](#8-extending-the-code)
9. [A map of the package](#9-a-map-of-the-package)
10. [Known limitations and what comes next](#10-known-limitations-and-what-comes-next)

---

## 1. What this repository is for

You point at an object in a photo, for example by drawing a box around a sofa. The system then looks for **other photos of that same physical object** in a collection (the *gallery*). "Same" means the same instance, not just the same kind of object: *this* red cap, not any red cap.

Doing that well and fast takes several steps, each more precise and more expensive than the one before. That is the **cascade**:

1. A cheap **semantic search** narrows thousands of gallery images down to about a hundred that *look* similar.
2. A **geometric check** with local keypoints looks at the best of those and asks whether the same physical details line up between the two images.
3. Later: a **dense matching** step for the final few (not built yet).

The notebook drives this cascade. All the logic lives in the Python package `src/visual_search/`, so the notebook stays readable and the same code runs in interactive and batch mode.

---

## 2. Getting started

### Install

You need [uv](https://docs.astral.sh/uv/). [mlcvtools](../../mlcvtools) must sit next to this repository, because it is installed from there as an editable dependency (it is read-only for this project).

```bash
uv sync --extra onnx --extra sparse
```

- **Base install:** data handling, DINOv2 (PyTorch), SIFT (OpenCV), the notebook UI and JupyterLab.
- **`--extra onnx`** adds onnxruntime, which the SAM3 adapter needs.
- **`--extra sparse`** adds kornia, which the learned matchers (ALIKED, XFeat) need.

A plain `uv sync` without the extras *removes* onnxruntime and kornia again, so always pass the extras you use.

Model weights are downloaded on first use:
- **DINOv2:** from Hugging Face into `~/.cache/huggingface`.
- **Learned matchers:** about 67 MB from GitHub into `~/.cache/torch/hub/checkpoints`.

The SAM3 and RF-DETR ONNX files are not downloaded; they live in `models/` (gitignored).

### Run the notebook

```bash
uv run jupyter lab
```

Open one of the three notebooks in `notebooks/` and choose *Run → Run All Cells*:
- `labeling.ipynb` builds your test set;
- `search.ipynb` is free play (search with your own boxes);
- `evaluation.ipynb` measures and compares approaches.

They are described in [section 5](#5-the-notebooks-cell-by-cell).

The first run computes and caches features in `cache/`. Later runs reuse them and only pay for the queries.

### Use the code without the notebook

Everything the notebook does is a few lines of Python:

```python
from visual_search.data import AnnotationSet
from visual_search.data.schema import parse_dataset
from visual_search.cascade import Cascade
from visual_search.models.dinov2 import Dinov2Backbone
from visual_search.retrieval import DinoRetriever
from visual_search.stages import SparseVerificationStage
from visual_search.eval import default_judges, evaluate_leave_one_out

items = parse_dataset(AnnotationSet.load("data/testset").to_records())   # one item per labelled box

cascade = Cascade(
    DinoRetriever(Dinov2Backbone("facebook/dinov2-small"), pooling="crop_cls", cache_dir="cache"),
    retrieval_k=100,                                    # stage 1 keeps the top-100
    stages=[SparseVerificationStage("sift", top_k=20, cache_dir="cache")],  # stage 2 re-ranks the top-20
)

report = evaluate_leave_one_out(cascade, items, judges=default_judges())  # every box is a query once
print(report)                              # R@k / MRR per stage (item + model level), latency per stage

query = next(it for it in items if it.label)
result = cascade.run(query, exclude={query.item_id})   # one query
[(c.item_id, c.score) for c in result.final[:5]]
```

### Run the tests

```bash
uv run pytest
```

Tests that need the learned-matcher weights are skipped if those weights aren't downloaded.

---

## 3. Your data: what to provide and what to label

### Two ways to provide data

- **A labelled test set (recommended for evaluation).** Put the images in one folder (default `data/testset/`) and label them in [`notebooks/labeling.ipynb`](../notebooks/labeling.ipynb).
  - You draw a box around each object and give it a label, so a photo can contain several labelled objects.
  - The labels are saved to `annotations.json` next to the images, and metrics are computed per box. See [the labeling section](#the-labelled-test-set-annotationsjson) below.
- **Folder per object (quick start).** Sort whole photos into one folder per object, as described next. This is fine for photos with one object each.

### The short answer (folder per object)

- **Put images in folders; that's the labelling.** One folder per physical object, and the folder name is the label. No annotation tool or label file is needed.
- **Every object you want to evaluate on needs at least two photos.** One becomes the query and the rest stay in the gallery to be found.
- **Unrelated photos go in one distractor folder.** They need no label at all.
- **For interactive use only, nothing needs labels.** Labels are only used to *score* results (the ✓/✗ marks and the Recall numbers).

### The folder layout

```
data/                     # gitignored; never committed
  bank_01/                # one folder per physical object; folder name = label
    img-001.png
    img-002.png
  billy_01/
    photo_a.jpg
    photo_b.jpg
  singletons/             # distractors: label None (also accepted: _distractors/, distractors/)
    room.png
  prompts.json            # optional: boxes/clicks per image (see below)
```

Rules the loader applies ([`data/folder.py`](../src/visual_search/data/folder.py)):
- **Formats:** JPG, PNG, BMP, WebP and TIFF. Keep the originals; any size works. HEIC files are skipped with a warning, so convert them to JPG first.
- **Rotation:** phone photos are rotated upright from their EXIF orientation.
- **Skipped or flagged:** empty folders are skipped. Images directly in `data/` (not in a subfolder) become distractors, with a warning.
- **Ids:** each image's id is its path relative to `data/` (e.g. `bank_01/img-001.png`). Ids are stable, so they also serve as cache and split keys.

### Do I need to label queries, the gallery, or both?

That depends on what you want to do:

| Goal | What needs a label |
|---|---|
| **Interactive exploration** (load any image, draw a box, browse results) | Nothing. Load a query from any path; the gallery can be unlabelled. You just won't get ✓/✗ marks. |
| **Evaluation** (Recall@1/5, MRR) | Every image of every object you want measured: it must be in its object's folder (or boxed and labelled in the test set). Distractors stay unlabelled. |
| **A "hard negative"** (same model, different physical object) | Give it its *own* label (e.g. `bank_02`), not the same label. The optional `group` field marks such look-alike groups (see below). |

You don't label queries and gallery separately. The evaluation picks queries from your labelled data itself:
- **Leave-one-out** (the evaluation notebook's default, `evaluate_leave_one_out`): every labelled box is a query once. It is searched against all other boxes and the distractors, with everything from its own photo left out, so it must be found in *other* photos. This uses all your labels.
- **One query per label** (`split_dataset` + `evaluate`, [`data/split.py`](../src/visual_search/data/split.py)): for every object one box is drawn at random (reproducibly from a seed) as the query; the rest stay in the gallery.

A result is correct when the query and the gallery item have the same label, and the label isn't `None`. Distractors are never correct.

### The labelled test set (`annotations.json`)

The labeling notebook writes this file into the image folder:

```json
{"version": 1, "images": {
  "living_room_3.jpg": [
    {"box": [63, 335, 535, 568], "label": "bank_01", "group": "sofa_model_x"},
    {"box": [600, 120, 780, 500], "label": "billy_01"}
  ],
  "kitchen.jpg": []
}}
```

- **`label`** = the exact item; **`group`** (optional) = a coarser identity such as the model.
- **Every box becomes one test item** with id `image#n` and the box as its prompt, so metrics are per box.
  - For each label, one box is the query and the other boxes of that label must be found among all boxes and distractors.
- **`"kitchen.jpg": []`** means reviewed, with nothing to label: a whole-image distractor. Images that aren't in the file at all are *not labelled yet* and are left out.
- **Consistent labels:** the label field autocompletes from labels already used, and a new label that looks like an existing one (`Bank_01` vs `bank_01`, a typo) needs an explicit *Add anyway*. Different numbers (`bank_01` vs `bank_02`) are treated as deliberately different items.
- **A label with a single box can't be evaluated** (there's nothing to find), and the notebook flags it.
- **In code:** `AnnotationSet.load(folder)` reads the file, and `.to_records()` gives the items for `split_dataset` and `evaluate`.

### Exact item or same model?

Your use case has two levels of "the same", both stricter than "the same kind of thing":
- **Exact instance:** this physical sofa, photographed in other rooms or from other angles.
- **Same model:** any sofa of this product line, such as the two different sofas in `data/bank_01/`.

The scoring rule above has a single level: whatever shares a label counts as "the same". Today, `bank_01` is therefore evaluated as a *model*-level match.

The planned way to support both levels: the label stays the **instance**, the optional `group` names the **model**, and the evaluation reports Recall for both levels side by side. The labeling notebook already records the `group`; reporting metrics per group comes with the evaluation notebook. Until then, choose per label what you want to measure.

Note also that geometric verification (stage 2) is an instance-level tool: two physical copies of a model share their shape but not their exact creases and wear.

Before splitting, the validator checks the dataset and reports problems. The most common is an object with only one image or box (currently `billy_01`). In the notebook's lenient mode such objects are dropped from evaluation with a warning instead of stopping the run.

### Prompts: telling the system *which* object you mean

A photo often shows several objects, so a query can carry a **prompt**, meaning a hint where the object is. One format serves both the dataset and the notebook UI:

```python
{"box": [x0, y0, x1, y1]}                                   # pixel box
{"points": [[x, y], ...], "point_labels": [1, 0, ...]}      # clicks; 1 = on the object, 0 = exclude
{"mask": HxW boolean array}                                 # a full mask
```

- **In the notebook** you draw prompts with the mouse.
- **For a folder-per-object dataset**, put them in `data/prompts.json`, keyed by relative image path:
  ```json
  {"bank_01/img-002.png": {"box": [63, 335, 535, 568]}}
  ```
- **Without a prompt**, the whole image counts as the object. That's fine for product-style photos with one object, but poor for room scenes.

Gallery images may also have prompts; the boxes of your labelled test set are exactly that. Then matching on the gallery side is restricted to that box. Unlabelled photo collections have no boxes; there, grounding finds the object in each photo (see [section 4](#4-how-a-search-works-the-pipeline)).

### The underlying format and other sources

Under the folder convention, a dataset is simply a **list of dicts**, the format from the task description ([`data/schema.py`](../src/visual_search/data/schema.py)):

```python
{"input_image": <path | PIL image | numpy array>, "label": "bank_01" or None, "prompt": {...}, ...}
```

Any other key (for example `"group"`) is kept untouched in `Item.extra`, so the schema can grow without code changes. Besides the folder loader and the test set there are:
- **`load_gallery(source)`:** an unlabelled gallery from a folder, a glob pattern or a list of files (used by "Search a new gallery").
- **`load_image_folder(dir)`:** a folder of distractors from anywhere, e.g. a COCO image directory.
- **`load_mlcvtools_annotations(json, image_root)`:** reads the boxes you drew in mlcvtools' `visual-search.ipynb` (`own-annotations.json`).
- **`make_synthetic_dataset()`:** generated test data. **For smoke tests only:** random textured patches pasted into cluttered scenes, with known ground-truth masks and warps. It checks that the code runs, not which method is better.

---

## 4. How a search works: the pipeline

### The flow

```
 query image ──► prompt (box / clicks / none)
                    │
                    ▼
            ┌──────────────┐   box mask today; SAM3 object mask once it passes its check
            │  MASK         │   (masking/)
            └──────┬───────┘
                   │ the mask becomes the query's prompt
                   ▼
            ┌──────────────┐   DINOv2-small descriptors, cosine similarity
 STAGE 1    │  RETRIEVAL    │   query: the prompted region  ↔  gallery: its boxes (test set),
            └──────┬───────┘   found regions (grounding) or whole images; keeps top-100 (retrieval_k)
                   │
                   ▼
            ┌──────────────┐   keypoints (SIFT / ALIKED / XFeat) + matching
 STAGE 2    │  SPARSE       │   query: keypoints INSIDE the mask  ↔  gallery: inside its box / region (else whole image)
            │  VERIFICATION │   robust model fit (homography / affine / fundamental; MAGSAC++ or RANSAC)
            └──────┬───────┘   re-ranks the top-20 (top_k) by verified inliers; the rest keeps its order
                   │
                   ▼
            final ranking  (+ per-stage rankings, score features, timings)
```

### Semantic search first, then keypoints

Stage 1 is a **semantic search**. [DINOv2](https://github.com/facebookresearch/dinov2) is a self-supervised vision transformer that turns an image into a 384-number description of what it looks like. Images of similar things get similar descriptions, which are compared with cosine similarity. This is fast (about 55 ms for the query, the gallery is precomputed) but *semantic*: it finds "a dark grey sofa", not necessarily *your* sofa.

Stage 2 is the **instance check**. It finds distinctive local points (corners, texture details) in both images and pairs them up. It then asks whether a single geometric transformation explains many of those pairs at once. If yes, the two images very likely show the same physical surface. The count of pairs that agree (*inliers*) becomes the new score.

DINOv2 is **not** an object detector here. It produces a description vector (and patch features), not boxes; grounding (below) turns the patch features into regions.

### What exactly is compared: boxes or whole images?

It depends on what the gallery items carry and which stage-1 retriever is used:

| | Query side | Gallery item **with a box** (labelled test set) | Photo **without a box**, grounded (`GroundedDinoRetriever`, `search.ipynb` section 4) | Photo **without a box**, not grounded (`DinoRetriever`) |
|---|---|---|---|---|
| **Stage 1 (DINOv2)** | Only the prompted region. `crop_cls` crops to the prompt box (plus 10 % padding); grounding averages the patch features under it. | Only the box. | **The regions where the object was found:** up to 3 per photo, each its own result. | The whole image. |
| **Stage 2 (keypoints)** | Only keypoints **inside the mask** (the box, or later the SAM3 mask). | Only keypoints inside the box. | Only keypoints inside the found region. | All keypoints of the whole image. |

**Your labelled test set compares object to object.** Each annotated box is its own gallery entry.

**An unlabelled photo collection compares object to object too, with grounding.** Without it, an object is compared against whole scenes, which works for product shots but not for cluttered rooms. Both early real-data failures came from that (see the [step 6a log](log/2026-11-24-07-sparse-verification-sift.md) and the [step 6b log](log/2026-11-24-08-learned-sparse-matchers.md)).

### Gallery grounding: finding the object in each photo

`GroundedDinoRetriever` ([`retrieval/grounding.py`](../src/visual_search/retrieval/grounding.py)) needs no extra model and no labels:

1. **Query descriptor.** The query's DINOv2 patch features under your box are averaged into one vector.
2. **Similarity map.** That vector is compared with every patch of every gallery photo. Each photo has a 32×32 patch grid, cached from stage 1, so this is one matrix product.
3. **Regions.**
   - In each photo, the patches within `delta` (0.1) of the photo's best similarity are switched on, and connected blobs become regions (boxes, plus one patch of context).
   - Up to `max_regions` (3) blobs are kept if their peak is close to the best one (`region_delta`, 0.1).
4. **Scoring.** Each region is scored by the cosine between the query vector and the region's average feature, so a photo can appear more than once (one result per region).
5. **Stage 2** matches keypoints only inside the region.

Measurements so far:
- **Speed:** about 4 ms per query for a handful of photos and about 26 ms for 70 photos on an M2, warm.
- **Memory:** all patch grids are kept in RAM, about 0.75 MB per photo.
- **Localization:** on the synthetic set, the best correct region overlaps the true object box with a median IoU of 0.66 (73 % above 0.5).

Grounding is *semantic*: it finds regions that *look like* your object (e.g. a dark-grey armchair for a dark-grey sofa). Confirming that a region is the same physical item is stage 2's job.

### What happens under the hood, step by step

When you press **Search** in the notebook, or call `cascade.run(query)`:

1. **Mask:** the `Masker` turns your prompt into a pixel mask. Today that's `BoxMasker`, which uses the box itself as the mask. The mask replaces the prompt on a copy of the query item (`with_mask`), so every later step reads it the same way.
2. **Stage 1** (default `DinoRetriever`, `crop_cls`):
   - The query is cropped to the mask's box, resized to 448×448 and run through DINOv2. Its CLS token is the query vector.
   - Gallery vectors were computed once when the gallery was indexed (`cascade.prepare`) and are cached on disk.
   - One matrix product scores all gallery items; the 100 best are kept.
3. **Stage 2,** for each of the top 20:
   - Load the gallery image's keypoints from the cache (computed once at indexing).
   - Keep only the query keypoints inside the mask, and the gallery keypoints inside its box or found region.
   - Match descriptors: SIFT uses Lowe's ratio test plus a mutual-best check; ALIKED and XFeat use the LightGlue network.
   - Fit a homography with MAGSAC++ and count the inliers. Fewer than 8 means no match (0).
   - The new score is `inliers + stage-1 cosine`, so the cosine only breaks ties.
   - The 20 are re-sorted; positions 21–100 keep their stage-1 order.
4. **Result:** a `QueryResult` with the full ranking after each stage, the score components of every candidate, the matches and fitted model (for the detail view), and the time per stage.

The same `Cascade` object runs a single interactive query and a full evaluation, so interactive and batch results are directly comparable.

### Caching: why the second run is fast

Expensive work is stored in `cache/` (gitignored), keyed by the **image content** (a hash) plus the **model configuration**:
- `cache/dinov2/<config>/…`: the DINOv2 CLS vector, the 32×32 grid of patch features and the image size, per image. Pooling modes, prompts and grounding reuse these without re-running the model.
- `cache/dinov2-crop/<config>/…`: the DINOv2 description of each *box crop* (test-set boxes and query crops), per image + box.
- `cache/kp-<matcher>/<config>/…`: keypoints and descriptors per image, per matcher.

Change the model, resolution or keypoint settings and a new cache folder is used automatically; stale results are never mixed in. Delete `cache/` at any time to start fresh. At 448 px the DINOv2 cache takes about 0.75 MB per image.

The SAM3 image encoding (about 120 MB per image) is kept only **in memory** for the last two images, so refining a prompt on the same image skips the slow encoder.

---

## 5. The notebooks, cell by cell

There are three notebooks, one per job. In all of them, every code cell is preceded by a markdown cell explaining what it does.

| Notebook | Job | Needs |
|---|---|---|
| `labeling.ipynb` | Build your test set: boxes + labels → `annotations.json` | A folder of photos |
| `search.ipynb` | Free play: draw a box and see what is found | DINOv2 (and a test set for section 3) |
| `evaluation.ipynb` | Measure and compare approaches on the test set | A labelled test set (falls back to synthetic data) |

### `labeling.ipynb`

Described in [section 3](#the-labelled-test-set-annotationsjson). Setup, the labeling app, a label check table, and a preview of the test items.

### `search.ipynb` (free play)

**1. Setup.** `%matplotlib widget`, the main knobs, and loading DINOv2 once:
- `TESTSET_DIR` (default `data/testset/`, or the `VS_TESTSET` environment variable);
- `DINO_MODEL`, `DINO_RESOLUTION` (448), `RETRIEVAL_K` (100), `SPARSE_K` (20);
- `MASKER` (box mask; SAM3 once validated).

**2. Your test set.** Loads the boxes from `annotations.json`, falling back to synthetic data.

**3. Search the test set.** An interactive app (`SearchApp`) whose gallery is **every box of your test set**, plus the reviewed images without boxes as distractors. The pipeline is DINOv2 `crop_cls` → SIFT verification.
1. **Pick a query:** a labelled box from the dropdown (`image#n [label]`), or any image via a path and *Load path*.
2. **Prompt it.**
   - *Box* mode: drag a rectangle.
   - *Points* mode: left click = on the object; right click or shift+click = exclude.
   - *Clear* resets the prompt; the blue overlay previews the mask; zoom and pan via the canvas toolbar.
3. **Expected label.** This decides the ✓/✗ marks.
   - It's pre-filled for a test-set box.
   - For a new image, type or pick the label of the object you drew (autocompleted from the test set). Leave it empty for no marks.
4. **Search.** Shows one row per stage: the query, then the top 5, each as a crop around its box (yellow) with score, label and a green ✓ / red ✗ frame.
   - A query picked from the test set never finds its own box.
   - The line above the gallery gives the time per step and the total.
5. **Detail.** Pick a result to see why it scored as it did:
   - Left: both images with the match lines (green = inliers, faint red = rejected matches).
   - Right: the gallery image warped onto the query with the fitted homography. If the object is really the same, the warped outline (orange) sits on the object.
   - Matches the stage rejected (too few inliers) are labelled *REJECTED* and get no warp panel.

After a search, `testset_app.last_result` holds all rankings and score components, and `testset_app.timings` the step timings.

**4. Search a new gallery.** Search any collection of photos, with no labels needed.
1. **Choose the gallery.** Set `GALLERY_SOURCE` to a folder (all images, including subfolders), a glob pattern (`"/path/**/*.jpg"`) or a list of file paths, and `TOP_N` to the number of results.
2. **Build the app.** The cascade is `GroundedDinoRetriever` → SIFT verification of the top `max(SPARSE_K, TOP_N)` regions.
3. **Query.** Load a query by path (or pick a test-set box), draw a box and press **Search**.
4. **Read the results.** Each result is a crop around the **found region** (yellow box). There is no *Expected* field and no ✓/✗: you judge the results. **Detail** shows the keypoint matches inside the region.

After a search, `gallery_app.last_result` holds every candidate with its `region`.

### `evaluation.ipynb`

All numbers come from **leave-one-out** on your test set: every labelled box is a query once, searched against all other boxes and the distractors, with everything from its own photo left out. Every table has two levels of correctness side by side:
- **item:** the exact object, the same label;
- **model:** the same `group`, where boxes without a group count as their own group.

**1. Setup.** The knobs of the run:
- `TESTSET_DIR`, `DINO_MODEL`, `DINO_RESOLUTION`, `RETRIEVAL_K`, `SPARSE_K`;
- `IOU_THRESHOLD` (0.3, for section 5);
- `RUN_NAME` and `RESULTS_DIR` (for section 9).

**2. Test set.** Loads `annotations.json`, prints the validator report (labels with a single box can't be queries) and the number of queries, distractors and groups. Without a test set it uses synthetic data, and section 5 is skipped.

**3. Stage 1.** Random, color histogram, and DINOv2 `cls` / `crop_cls` / `masked_mean`, without re-ranking.

**4. Stage 2.** DINOv2 `crop_cls` plus a verification variant:
- the estimator (MAGSAC++ vs RANSAC);
- the geometric model (`H`, `A`, `F`);
- no masks, and plain SIFT;
- the learned matchers (top 10).

**5. Grounding: what if the photos had no boxes?** The boxes are hidden from the gallery, which becomes your test-set photos; the query boxes stay. Three variants:
- **annotated boxes:** the upper bound;
- **DINOv2 grounding:** a found region is correct if it overlaps a box of the right label with IoU ≥ `IOU_THRESHOLD`;
- **whole images:** correct if the photo contains the object, anywhere.

Each is shown at stage 1 and after SIFT, plus a localization table: the median IoU of the found regions with your boxes, and the share above 0.3 and 0.5.

**6. Per-label breakdown.** For the default pipeline (DINOv2 `crop_cls` → SIFT H MAGSAC++): one row per label, hardest first, with queries, R@1, R@5, MRR, median rank and how many queries found nothing.

**7. Failure gallery.** The worst queries drawn as result grids: the query box, then the top 5 per stage, with ✓/✗ at item level. `N_FAILURES` sets how many.

**8. Latency.** Median and p95 per stage and in total, for every pipeline. **This is search cost, not interactive latency.** Every query box is also a gallery box, so its features were computed during indexing. A fresh query adds its own feature extraction: about 55 ms for DINOv2 at 448 px on an M2, plus the matcher's keypoints.

**9. Save the results.** All tables as CSV in `results/<date-time>_<RUN_NAME>/` (gitignored), so runs can be compared over time.

---

## 6. Reading the scores and tables

### Evaluation tables (`evaluation.ipynb`)

Each row is one pipeline, evaluated over all queries (leave-one-out: every labelled box once).

| Column | Meaning |
|---|---|
| **R@1** | Recall@1: the share of queries whose **first** result is correct at *item* level (the exact object). The headline number. |
| **R@5** | The share of queries with at least one correct result in the **top 5**. |
| **MRR** | Mean reciprocal rank: for each query, 1 / (rank of the first correct result), averaged. 1.0 = always first, 0.5 = typically second. Queries whose correct result falls outside the returned list count as 0. |
| **model R@1**, **model R@5**, **model MRR** | The same at *model* level: a result also counts when it has the query's `group` (e.g. the same sofa model). Never lower than the item-level numbers. |
| **n_queries** | Number of queries evaluated. Queries whose object can't be found anywhere else (e.g. its only other box is in the same photo) are skipped. |
| **stage** | Which stage's ranking the metrics refer to (the last stage of that pipeline). |
| **total_median_ms**, **total_p95_ms** | Search time per query (all stages, excluding indexing and, in leave-one-out, the query's own feature extraction): median and 95th percentile. |
| **prepare_s** | Seconds spent indexing the gallery (offline, once; near zero when cached). |

**Grounding table** (section 5): the same columns, for three galleries (annotated boxes, DINOv2 grounding, whole images), at stage 1 and after SIFT. Here a grounded result is correct when its region overlaps an annotated box of the query's label (model level: of its group) with IoU ≥ `IOU_THRESHOLD`. A whole-image result is correct when the photo contains such a box.

**Localization table:**
- `median_iou`: per query, take the first found region in another photo that contains the object, and its IoU with that object's box; the median over all queries;
- `share_iou_0.3` / `share_iou_0.5`: the share of those regions with IoU at or above 0.3 / 0.5.

**Per-label table:** one row per label, hardest first.
- `queries`: how many of its boxes were queries.
- `R@k`, `MRR`: as above, for that label only.
- `median rank`: of the first correct result.
- `not found`: queries without any correct result in the list.

In code, every evaluation returns:
- `report.metrics`: one row per stage, so "stage 1 only" vs "after re-ranking" comes from the same run;
- `report.latency`: median, p95, mean and max per stage;
- `report.per_query`: rank per stage and level, top-1 and time per query.

Three caveats when reading the numbers:
- **Synthetic numbers are a smoke test.** The synthetic objects are flat and heavily textured, so stage 2 reaches R@1 = 1.0 for almost every variant. That shows the code works, not which variant is best.
- **Small test sets give jumpy numbers.** With 20 queries, one query is 5 percentage points of R@1.
- **The first query after loading can be slow.** On a Mac, PyTorch on the GPU compiles kernels for each new image size; later queries are fast.

### Scores in the interactive gallery

Every thumbnail title reads, for example, `#1 41.623 ✓`, then the label, then up to two score components:

- **`#1`:** the rank in that stage.
- **The score:** its meaning depends on the stage.
  - Retrieval row: the **cosine similarity** between the query and gallery descriptions, from −1 to 1. In practice about 0.3 (unrelated) to above 0.8 (very similar). With grounding (`search.ipynb` section 4) it compares the query with a found **region**, and the crop shows that region.
  - Sparse-stage row: **`inliers + cosine`**. `41.623` means 41 verified inliers plus a cosine of 0.623. A score below 1 means the geometric check failed (0 inliers) and only the cosine is left.
- **✓ / ✗** (green / red frame): whether the result has the query's label. Only shown when the query is labelled.
- **Score components** (stored in `candidate.features`):

| Component | Set by | Meaning |
|---|---|---|
| `cosine` | stage 1 | DINOv2 cosine similarity (with grounding: query vs found region). |
| `peak` | grounding | The best single-patch similarity inside the region. |
| `region_patches` | grounding | How many patches (of the 32×32 grid) the region's blob covers. |
| `inliers` | stage 2 | Matches consistent with the fitted geometry. 0 if fewer than `min_inliers` (8). |
| `inlier_ratio` | stage 2 | inliers / matches. High = the matches are mostly consistent. |
| `matches` | stage 2 | Tentative matches before the geometric check. |
| `residual_px` | stage 2 | Mean distance in pixels between where an inlier lands and where the model predicts it. Lower = tighter fit. |
| `stage1` | stage 2 | The candidate's score from the previous stage. |

Inlier counts depend strongly on texture: a patterned cushion gives many, a plain upholstered sofa few. That is why step 7 adds more components (spatial coverage of the mask, mask overlap after warping, post-warp appearance difference, warp distortion) and a combiner, instead of relying on inliers alone.

---

## 7. What is available: models, retrievers, maskers, matchers

The code separates **models** (neural networks that produce features or masks) from the **components** that use them in the pipeline. Each component type has a common interface, so components of the same type are interchangeable.

### Models

| Model | Role | Runtime | License | Status |
|---|---|---|---|---|
| **DINOv2-small** (`facebook/dinov2-small`) | Image descriptions for stage 1 | PyTorch (CUDA / MPS / CPU) | Apache-2.0 | **In use.** About 55 ms per image at 448 px on an M2. Larger variants and "with registers" variants are a config change. |
| **SAM3** (`models/sam3/`) | Object mask from a box or clicks | ONNX (onnxruntime) | Meta SAM License (custom, commercial use with restrictions) | **Code ready, not validated.** The q8 export is rejected (11 s per image on CPU, fails on CoreML). The fp16 adapter needs your manual test with `scripts/try_sam3.py`. |
| **RF-DETR small / seg-small** (`models/rf-detr/`) | Object detection, COCO classes | ONNX | Apache-2.0 | **Present, not used yet.** Inspected (0.27 s per image on CPU); candidate for gallery grounding. Note that the seg export's output names are shuffled, so outputs must be decoded by shape. |
| **ALIKED-N16, XFeat** | Learned keypoints + descriptors | PyTorch via kornia | BSD-3 / Apache-2.0 | In use inside the learned matchers below. |
| **LightGlue, LighterGlue** | Learned keypoint matching | PyTorch via kornia | Apache-2.0 | In use inside the learned matchers below. |

### Stage-1 retrievers (`retrieval/`)

| Name | What it does | Use |
|---|---|---|
| `RandomRetriever` | Random order | Chance level for evaluation. |
| `ColorHistogramRetriever` | HSV color histogram of the query region vs each gallery item's box (or whole image) | Sanity baseline. |
| `DinoRetriever(pooling="crop_cls")` | DINOv2 CLS token of the query **cropped to its prompt** | **Default**, the best stage-1 mode so far. One extra model run per query. |
| `DinoRetriever(pooling="masked_mean")` | Average of DINOv2 patch features **under the prompt**, from the full-image pass | Instant when the prompt changes (no extra model run). Weaker than `crop_cls` on synthetic data. Benefits most from real SAM3 masks. |
| `DinoRetriever(pooling="cls")` | CLS token of the whole query image | Ablation: ignores the prompt. |
| `GroundedDinoRetriever` | Finds the query object inside each gallery photo (DINOv2 patch similarity) and ranks the **regions** (up to 3 per photo) | For unlabelled photo collections (`search.ipynb` section 4). Results carry a `region` box. |

### Maskers (`masking/`)

| Name | Output |
|---|---|
| `NoMasker` | The whole image: the "no mask" ablation. |
| `BoxMasker` | The prompt box as a rectangle; clicks give the box around the positive clicks. **The current default.** |
| `Sam3Masker` | A real object mask from SAM3. Boxes pick the best-overlapping SAM3 candidate. Clicks pick the candidate masks that contain positive clicks and no negative ones; if nothing fits, it falls back to the box. *Untested on the real model.* |

### Matchers (`matching/`), for stage 2

| Registry name | Keypoints / matching | Speed (M2, per pair) | Notes |
|---|---|---|---|
| `"sift"` | SIFT with RootSIFT descriptors, ratio test + mutual check (OpenCV, CPU) | about 4 ms | **The default.** Rotation and scale invariant, sub-pixel precise. Weak on low-texture surfaces. |
| `"aliked-lightglue"` | ALIKED-N16 + LightGlue | about 47 ms | Not rotation invariant (failed a 35°-rotated synthetic pair). Gave a false verification on a real distractor. |
| `"xfeat-lighterglue"` | XFeat + LighterGlue | about 36–50 ms | Very fast keypoints; coarser localization (2–3 px). The replacement for OmniGlue. |

Create a matcher with `create_matcher(name, **options)`; `available_matchers()` lists the names.

**Geometric verification** (`geometry/verify.py`) fits one of three models:
- **Homography `H`:** flat objects, or small viewpoint changes. The default; it also allows warping.
- **Affine `A`:** a simpler, more stable homography.
- **Fundamental `F`:** general 3D objects; looser (accepts more false matches) and gives no warp.

Each can be fitted with `"magsac"` (OpenCV MAGSAC++, the default) or classic `"ransac"`.

### Not adopted, and why

Some models from the task were rejected up front, on latency or license. They're documented as "re-open later" options in the [latency entry](log/2026-11-24-01-latency-budget-and-rejections.md):
- **OmniGlue:** restrictive SuperPoint weights, and it needs TensorFlow.
- **MASt3R, DUSt3R, VGGT:** heavy, and non-commercial licenses.
- **SAM3 q8:** too slow.
- **RoMa:** conditional, pending its benchmark in step 8.

---

## 8. Extending the code

The package is built around small interfaces, so new methods are new classes, not edits to the pipeline.

### Adding a matcher

A matcher is one class with three things: an `identity` (the settings that change the features, used as cache key), `extract(image)` and `match_features(fq, fc)`. Masking, caching, verification and the UI work automatically.

1. **Create `src/visual_search/matching/my_matcher.py`:**

```python
import numpy as np
from visual_search.matching.base import Correspondences, Features, Matcher
from visual_search.matching.registry import register_matcher


@register_matcher("my-matcher")
class MyMatcher(Matcher):
    """One-line description."""

    def __init__(self, max_keypoints: int = 2048) -> None:
        """Load the model."""
        self.max_keypoints = max_keypoints

    @property
    def identity(self) -> dict:
        """Settings that change the extracted features (used as cache key)."""
        return {"matcher": "my-matcher", "max_keypoints": self.max_keypoints, "version": 1}

    def extract(self, image: np.ndarray) -> Features:
        """Keypoints in ORIGINAL-image pixels (x, y) + descriptors for an RGB uint8 image."""
        ...
        return Features(keypoints, descriptors, scores, image.shape[:2])

    def match_features(self, fq: Features, fc: Features) -> Correspondences:
        """Return matched point pairs (query, candidate) with a confidence per pair."""
        ...
        return Correspondences(fq.keypoints[iq], fc.keypoints[ic], confidence)
```

   Keypoints must be in **original-image pixels**, even if the model runs on a resized image. That's the most common mistake, and the contract test catches it.
2. **Register the module for lazy import:** add `"my-matcher": "visual_search.matching.my_matcher"` to `_LAZY_MODULES` in [`matching/registry.py`](../src/visual_search/matching/registry.py). Heavy dependencies are then only imported when the matcher is requested.
3. **Add the name to `MATCHERS` in [`tests/test_matching.py`](../tests/test_matching.py).** The shared tests then check that it:
   - returns well-formed features;
   - reports points in the right pixel frame (by matching an image against a 3× upscaled copy);
   - respects masks;
   - is deterministic;
   - recovers a known homography on synthetic data;
   - gives clearly fewer inliers for different objects.
4. **Put new dependencies in an optional extra:** `uv add --optional <extra> <package>`, after checking the license (see CLAUDE.md).
5. **Use it:**
   ```python
   SparseVerificationStage("my-matcher", top_k=20, cache_dir="cache")
   ```
   Or add it to `stage2_configs()` in the notebook to see it in the comparison table.

Dense matchers (RoMa, step 8) will implement the same interface. They will also put their dense warp into `Correspondences.extra`.

### Adding other components

The other component types follow the same pattern:

| To add… | Subclass | Implement | Then |
|---|---|---|---|
| a stage-1 method | `retrieval.base.EmbeddingRetriever` (one vector per image) or `Retriever` (anything else) | `embed_query`, `embed_gallery` (or `fit`, `search`) | `Cascade(MyRetriever(), ...)` |
| a re-ranking stage | `cascade.RerankStage` | `rerank(query, candidates, gallery)` and optionally `prepare(gallery)` | `Cascade(..., stages=[stage2, MyStage(top_k=5)])` |
| a mask source | `masking.base.Masker` | `_predict(image, prompt, image_key)` | `SearchApp(..., masker=MyMasker())` |

Use `timing.StageTimer` and `cache.FeatureCache` for consistent timings and caching. Evaluation, per-stage tables and the UI pick up new components without changes.

---

## 9. A map of the package

```
src/visual_search/
  data/         schema.py       Item, Prompt, is_match: the dataset format
                annotations.py  annotations.json: per-object boxes + labels, label checks
                folder.py       folder-per-object loader, load_gallery (folder / glob / list), mlcvtools importer
                split.py        validator + seeded query/gallery split
                synthetic.py    SMOKE-TEST-ONLY generated data
  models/       dinov2.py       DINOv2 backbone (CLS + patch features)
  retrieval/    base.py         Retriever / EmbeddingRetriever / Candidate (with optional region)
                index.py        exact cosine search
                dino.py         DinoRetriever (cls / crop_cls / masked_mean)
                grounding.py    GroundedDinoRetriever, localize (find the object in each photo)
                baselines.py    random, color histogram
  masking/      base.py         Masker, NoMasker, BoxMasker, with_mask
                sam3.py         SAM3 ONNX adapter (untested on the real model)
  matching/     base.py         Matcher, Features, Correspondences
                registry.py     register_matcher / create_matcher
                sift.py, lightglue.py   the matchers
  geometry/     verify.py       homography / affine / fundamental, MAGSAC++ / RANSAC
  stages/       sparse.py       SparseVerificationStage (stage 2)
  cascade.py                    Cascade, RerankStage, QueryResult
  eval/         metrics.py      Recall@k, MRR
                judges.py       when a result is correct: item / model (group) / grounded (IoU)
                harness.py      run_evaluation, evaluate, evaluate_leave_one_out, comparison_table
                inspect.py      per-label table, failure grids, localization stats, latency, CSV export
  cache.py                      on-disk feature cache
  device.py, timing.py, imaging.py   helpers (device choice, stage timers, prompt→box/mask)
  ui/           prompt_widget.py  draw box / click points (ipympl)
                labeling.py       the labeling app
                search_app.py     the interactive search app
  viz/          gallery.py, pairs.py, overlay.py   result grid, match/warp view, mask overlays

notebooks/labeling.ipynb   build your own test set (boxes + labels)
notebooks/search.ipynb     free play: test-set search, new-gallery search
notebooks/evaluation.ipynb measurements and comparisons on the test set
results/                   saved evaluation tables (gitignored)
scripts/try_sam3.py       manual SAM3 go/no-go check (timings + overlay)
tests/                    pytest suite, one file per area
docs/log/                 change log: what changed, why, with numbers
models/  data/  cache/    local only (gitignored)
```

From mlcvtools (read-only), the code reuses the ONNX model loader and provider selection, its preprocessing (for SAM3), image utilities and the COCO class list. The prompt widget was adapted from mlcvtools' visual-search notebook.

---

## 10. Known limitations and what comes next

**Current limitations.**
- **Grounding is coarse and semantic.** Regions sit on a 32×32 patch grid, and they find what *looks like* the object (a similar armchair for a sofa). Stage 2 still has to confirm the exact item. The thresholds (`delta`, `region_delta`) were tuned on synthetic data only.
- **Masks are boxes.** SAM3 masks are wired in but unvalidated.
- **Stage 2 scores by inlier count only,** which gave one false verification on real data (ALIKED on a room photo) and favors textured objects.
- **There is no real test set yet,** so no real numbers. Also, `bank_01` holds two *different* sofas of the same model, so it measures model-level rather than instance-level matching (see [Exact item or same model?](#exact-item-or-same-model)).
- **Latency was measured on an 8 GB M2 development machine,** not on the target 16 GB Mac or the GPU server.

**Next steps.**
1. **Step 7, scoring:** spatial coverage of the query mask, mask overlap after warping, post-warp appearance difference, reprojection error and warp distortion, combined first with hand-set weights and later with weights learned from labelled data.
2. **Step 8, dense matching:** RoMa (or EfficientLoFTR if RoMa is too slow) for the final top-5, plus a displacement map in the detail view.
3. **Step 9, full ablations** on real data and on target hardware, using the evaluation notebook.

**What you can do now.**
- Label a test set in `notebooks/labeling.ipynb`, with several photos per object, and run `notebooks/evaluation.ipynb` on it for the first real numbers.
- Try "Search a new gallery" (`search.ipynb`, section 4) on one of your own photo folders.
- Test SAM3 fp16 on your 16 GB Mac and on the GPU server with `scripts/try_sam3.py`.
- Decide how to label "same model": the same label (model level), or separate labels that share a `group` (item level, with the model recorded).
