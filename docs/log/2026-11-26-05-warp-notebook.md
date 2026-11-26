# Warp notebook: animate a search result warping onto the query (step 6e)

- Date: 2026-11-26
- Type: feature, dependency

## What changed
- **`warp/models.py`:** warp models fitted from keypoint matches, all usable *backwards* (query pixel → candidate pixel) and *partially* for animation.
  - `HomographyWarp` (H) and `AffineWarp` (A) come from MAGSAC++.
  - **`TPSWarp`** is a thin-plate spline (numpy, no new dependency) with a stiffness parameter. Its control points are the inliers of a robust fundamental-matrix fit (rigid 3D, not necessarily flat), falling back to homography inliers.
  - `fit_warps` fits all three and notes why a model failed.
  - Partial warps interpolate sampling positions between a base placement (candidate box resized into the query box, t = 0) and the full warp (t = 1).
- **`viz/animation.py`:**
  - frames on a canvas around the query box, in four views: **morph** (with travelling inlier dots), **grid**, **flicker** and **residual**;
  - `compare_frames` (H | A | TPS side by side), `side_by_side` (the pair preview) and `warp_stats` (points, reprojection error, strain, residual per model);
  - **GIF export** (`save_gif`, `gif_bytes`).
- **`ui/warp_app.py`: `WarpApp`.** The familiar search panel on top (an embedded `SearchApp`), then:
  - pick a candidate from the top-n (parameter `top_n`) and see the pair side by side;
  - choose model and view (or all models side by side), press **Warp ▶** to play, scrub with the frame slider, set fps, and **Save GIF**;
  - stage-2 matches are reused, otherwise matches are computed on the spot;
  - fits with fewer than 8 agreeing matches get a warning that the warp is unreliable.
- **`notebooks/warp.ipynb`:** a new notebook built around this app, with explanation cells before every code cell and an optional "export every view" cell. The other notebooks are unchanged.
- **`SearchApp`:** two read-only properties (`query_image`, `searched_query`), so the warp app can read the searched query. Behavior is unchanged.
- **Dependency:** `imageio` (BSD-2-Clause) is now an explicit dependency; it was already installed through mlcvtools.

## Why
The owner finds the warp the most interesting part and wanted it central and animated ("almost a GIF"):
- run the cascade;
- pick a file from the parameterized top-n;
- see the pair side by side;
- start the morph with a button.

TPS was approved now as the non-rigid model (TASK.md lists thin-plate splines). GIF export only; MP4 is not needed for now.

## Impact
- **Rendering:** about 15 ms per frame at a 480 px canvas on the M2; a 32-frame GIF is about 1.6 MB.
- **A bug found on the way and fixed before release:** upsampling the coarse sampling map with `cv2.resize` shifted positions by up to about half a grid cell (around 3 px at the canvas edges). Exact node-aligned interpolation now matches the exact map to within 0.2 px; a test covers it.
- **Real photos:** the `bank_01` pair (same model, different sofas) gets only 3–5 agreeing matches with SIFT or ALIKED, and every model smears the candidate diagonally (homography strain 3.0, vs ≈ 0 for a rigid fit). That is the visual "not the same item" verdict, and the reason for the unreliable-warp warning.
- **Synthetic pair (same object):** H/A/TPS align it, and the residual inside the box drops clearly over the animation (a test checks it falls below 80 % of the start).
- **Tests:** 146 pass with all extras.
  - Base placement; homography sampling vs OpenCV's `warpPerspective`; partial-warp endpoints.
  - TPS exact through its points and smooth with stiffness.
  - `fit_warps` against ground truth.
  - All views, compare frames, stats, GIF export.
  - The app flow: search, pick, warp, scrub, compare, save; computed matches; reset on a new search.
- **Verification:** `warp.ipynb` executes headless on a generated test set. Interactive playback was not checked in a real browser.
- **Follow-ups:**
  - RoMa's dense warp as a fourth model (step 8).
  - The owner's question about regions (pending).

## Evidence (optional)
`uv run --all-extras pytest tests/test_warp.py`. `notebooks/warp.ipynb` with `VS_TESTSET` pointing to the generated set.
