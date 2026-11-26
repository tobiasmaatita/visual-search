# Gallery grounding with DINOv2 and new-gallery search (step 6c, part 3)

- Date: 2026-11-26
- Type: feature, decision

## What changed
- **`retrieval/grounding.py`:**
  - **`localize()`** finds where the query object is in an image. It compares the query descriptor (mean of the query's DINOv2 patch features under its box) with each of the image's patch features, thresholds the similarity map at `peak - delta`, and turns connected blobs into regions (boxes plus one patch of context). It keeps up to `max_regions` blobs whose peak is within `region_delta` of the best.
  - **`GroundedDinoRetriever`** is a stage-1 retriever that does this for every gallery photo and ranks **regions**: one candidate per region, so a photo can appear several times. A region is scored by the cosine between the query and the region's mean feature.
  - Gallery items that already have a box are not localized; their box is their region.
  - It reuses the cached DINOv2 patch grids of stage 1, so there is no extra model run.
- **`Candidate.region`:** a new field. `SparseVerificationStage` matches gallery keypoints only inside the candidate's region, if one is set. The result grid, thumbnails and detail view show the region.
- **`load_gallery(source)`:** builds an unlabelled gallery from a folder (recursive), a glob pattern or a list of files.
- **Result grid:** callbacks now receive the candidate instead of only its id, so per-region thumbnails work. Long result lists wrap onto extra rows (`per_row`, 8).
- **Cascade notebook section 7, "Search a new gallery":** `GALLERY_SOURCE` (folder, glob or list), `TOP_N`, `MAX_REGIONS`. The cascade is `GroundedDinoRetriever` → SIFT verification. There are no ✓/✗ marks; the owner judges. Explanatory markdown precedes each code cell.

## Why
The owner wants to search arbitrary photo collections with a box query and get the top-n places where the object appears. The decision (2026-11-25) was DINOv2 localization only (no RF-DETR, no OWLv2), used in both stages, with one result per object region.

## Impact
- **Localization quality** (synthetic, smoke test; gallery boxes removed, best correct region vs the true box):

| `delta` | median IoU | IoU > 0.5 | R@1 | MRR |
|---|---|---|---|---|
| 0.04 | 0.13 | 7 % | 0.40 | 0.51 |
| 0.07 | 0.44 | 40 % | 0.20 | 0.43 |
| **0.10 (default)** | **0.66** | **73 %** | 0.33 | 0.49 |
| 0.15 | 0.40 | 40 % | 0.20 | 0.45 |
| 0.20 | 0.27 | 20 % | 0.13 | 0.45 |
| *whole image, `crop_cls` (no grounding)* | – | – | 0.33 | 0.47 |

  - `delta = 0.10` gives the best regions, which matter most for stage 2. The R@1 differences rest on 15 synthetic queries, so they're too few to trust.
- **Latency:** about 26 ms per query for 70 gallery photos on an M2 (one matrix product plus a connected-component pass per photo), and about 4 ms for 3 photos.
  - The first query in a fresh process paid about 0.6 s of GPU warm-up.
  - Memory: about 0.75 MB of patch features per gallery photo.
- **Real photos** (query box on the sofa in `bank_01/img-002`, gallery = the other `data/` photos):
  - Grounding found the other sofa of the same model, with its region around the sofa, as #2.
  - #1 was a dark-grey armchair, which is semantically similar.
  - Stage 2 found no geometric confirmation, which is correct for a different physical sofa.
  - In short, grounding is semantic and stage 2 is the instance check.
- **Tests:** 124 pass.
  - `localize` on constructed similarity maps: box geometry, two objects, weak-blob rejection, padding.
  - The grounded retriever: regions per object, boxed items not localized.
  - Stage 2 restricted to the region, and `load_gallery` for all three source kinds.
  - The app on a new gallery (region thumbnails, no expected label).
  - Real-DINOv2 localization on synthetic data (median IoU > 0.3).
- **Verification:** the notebook executes headless. It hasn't been checked in a real browser.
- **Follow-ups:**
  - The evaluation notebook: grounding judged against the annotated test-set boxes (region IoU), and metrics per `group`.
  - Tuning `delta` and `region_delta` on real data.
  - An optional image-level prefilter for very large galleries (memory).

## Evidence (optional)
`uv run pytest tests/test_grounding.py`. The `delta` sweep used DINOv2-small at 448 px, seed 0, 12 synthetic instances × 3 views, and 40 distractors.
