# Stage 2: sparse matching (SIFT/RootSIFT) + geometric verification, pair detail view

- Date: 2026-11-24
- Type: feature

## What changed
- **`matching/`:** a `Matcher` interface that splits extraction from matching.
  - `extract(image) -> Features` and `match_features(fq, fc) -> Correspondences`; all points are in original-image pixels.
  - `match_extracted(...)` restricts keypoints to the query and candidate masks.
  - A name registry (`register_matcher`, `create_matcher`, with lazy imports for heavy matchers).
  - A shared **contract test** that every new matcher must pass: shapes, pixel frame across resolutions, mask restriction, determinism.
- **`matching/sift.py`:** OpenCV SIFT with RootSIFT descriptors (default), detection at a longer side of 1024 px or less, Lowe's ratio test (0.8) and a mutual nearest-neighbor check.
- **`geometry/verify.py`:** `verify(corr, model_type, method, threshold)`.
  - Models: homography `H`, affine `A`, fundamental `F`.
  - Estimators: OpenCV `USAC_MAGSAC` (MAGSAC++) or classic `RANSAC`.
  - Returns inliers and per-match residuals: reprojection error, or Sampson distance for `F`.
  - Convention: the model maps candidate points to query points, so the candidate can be warped onto the query.
- **`stages/sparse.py`: `SparseVerificationStage`** (a `RerankStage`).
  - Matches the top-k candidates inside the query mask and verifies them; fewer than `min_inliers` (8) counts as no match.
  - Re-scores by `inliers + stage-1 score`. The stage-1 score only breaks ties; step 7 replaces this with a combiner.
  - Records `inliers`, `inlier_ratio`, `matches`, `residual_px` and `stage1` as score features, and keeps correspondences and verification as artifacts.
  - Gallery keypoints are extracted once and cached on disk (`cache/kp-sift/...`).
- **Pair detail view** (`viz/pairs.py`, plus a **Detail** dropdown in `SearchApp`):
  - Shows query | candidate with match lines (inliers green, outliers faint red).
  - Shows the candidate warped onto the query, with the warped candidate mask or image border.
  - Rejected verifications are labelled "REJECTED" and get no warp panel.
- **`Candidate.artifacts`:** a new field for non-numeric by-products. `imaging.prompt_to_mask` is a new helper.
- **Notebook:** a stage-2 ablation table, and the interactive app now runs DINOv2 → SIFT.

## Why
TASK.md step 6 starts with a SIFT baseline, MAGSAC vs RANSAC and H/F/A model choice. Splitting extraction from matching lets gallery features be precomputed, which keeps interactive queries fast. The registry and contract test are there so new matchers drop in without cascade changes, as the owner asked.

## Impact
- **Results** (DINOv2-small `crop_cls` top-100, then top-20 re-ranked; M2, MPS + CPU):

| Synthetic, SMOKE TEST (15 q, 70 gallery) | R@1 | R@5 | MRR | median ms/query |
|---|---|---|---|---|
| stage 1 only | 0.67 | 0.80 | 0.73 | 51 |
| + SIFT, H, MAGSAC++ | **1.00** | 1.00 | 1.00 | 65 |
| + SIFT, H, RANSAC | 1.00 | 1.00 | 1.00 | 67 (p95 86) |
| + SIFT, A, MAGSAC++ | 0.93 | 1.00 | 0.97 | 64 |
| + SIFT, F, MAGSAC++ | 1.00 | 1.00 | 1.00 | 68 |
| + SIFT, H, no mask | 1.00 | 1.00 | 1.00 | 62 |
| + plain SIFT (not Root), H | 0.93 | 1.00 | 0.97 | 63 |

  - The synthetic set is planar and textured by construction, so it **saturates**; it shows the stage works, not which variant is better.
- **Latency:**
  - Stage 2 for 20 pairs takes about 22 ms median with warm gallery features.
  - Cold SIFT extraction takes 18 ms on a 600 px image and 40 ms at 1240×1540.
  - Both are far inside the 1 s stage budget.
- **Real data: an important early warning.** The `bank_01` pair shows the same sofa *model* in two different rooms.
  - With the whole image as prompt, SIFT finds 21 tentative matches and 5 "inliers", which are geometrically wrong (lamp to bookshelf). The stage rejects the pair (below 8 inliers).
  - Upholstered furniture is low-texture and the backgrounds differ, so SIFT alone will not verify real furniture instances.
  - This is what learned matchers (6b), a tight object mask (SAM3) and a score combiner that doesn't rely on inlier count alone (step 7) are for.
  - It also raises a dataset question for the owner: are those two photos really the same physical sofa?
- **Follow-ups:**
  - **6b:** ALIKED + LightGlue (kornia) and XFeat + LighterGlue, after a license check. Both need weight downloads, so the owner is asked first.
  - Multi-scale or rotation attempts for large scale changes are not done yet.
  - Cycle consistency is covered by the mutual nearest-neighbor check for sparse matching; an explicit A→B→A check comes with dense matching.

## Evidence (optional)
`notebooks/cascade.ipynb` (stage-2 table, seed 0). `uv run pytest`: 94 passed, including homography recovery against the synthetic ground-truth warp (corner error under 6 px, MAGSAC and RANSAC), F-matrix residual convention, and the matcher contract.
