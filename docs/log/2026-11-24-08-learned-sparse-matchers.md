# Learned sparse matchers: ALIKED + LightGlue, XFeat + LighterGlue

- Date: 2026-11-24
- Type: feature, dependency

## What changed
- **`matching/lightglue.py`:** `aliked-lightglue` and `xfeat-lighterglue` via kornia.
  - One class (`LearnedSparseMatcher`): a learned keypoint/descriptor network followed by LightGlue-style attention matching.
  - Both registered in the matcher registry; both pass the shared matcher contract test.
- **Defaults:** 1024 keypoints, extraction at a longer side of 640 px or less. LightGlue cost grows quadratically with keypoints, and these settings were the fastest measured.
- **Dependency:** a new optional extra, `sparse` (`kornia>=0.8.2,<0.9`, Apache-2.0, the same range as mlcvtools). Install with `uv sync --extra sparse`.
- **Weights** (downloaded with the owner's OK into `~/.cache/torch/hub/checkpoints`):

| File | Source | License | Size |
|---|---|---|---|
| `aliked-n16.pth` | github.com/Shiaoming/ALIKED | BSD-3-Clause | 2.7 MB |
| `aliked_lightglue.pth` | github.com/cvg/LightGlue v0.1_arxiv | Apache-2.0 (code + weights) | 47.6 MB |
| `xfeat.pt`, `xfeat-lighterglue.pt` | github.com/verlab/accelerated_features | Apache-2.0 | 6.2 + 10.8 MB |

  The licenses were checked against each repository's LICENSE file. XFeat + LighterGlue replaces OmniGlue (see entry 01).
- **Tests:** the contract test now checks for *no systematic offset* (under 1.5 px) plus a looser median error, instead of SIFT-level precision. The homography test has per-matcher tolerances.
- **Notebook:** the stage-2 table includes both learned matchers, re-ranking the top 10.

## Why
TASK.md step 6 asks for ALIKED+LightGlue, and the owner approved XFeat+LighterGlue as the fast, recent OmniGlue replacement.

## Impact
- **Results** (DINOv2-small `crop_cls` top-100, then H + MAGSAC++; M2, MPS):

| Synthetic, SMOKE TEST (15 q, 70 gallery) | R@1 | R@5 | MRR | median ms/query |
|---|---|---|---|---|
| stage 1 only | 0.67 | 0.80 | 0.73 | 51 |
| SIFT, top-20 | **1.00** | 1.00 | 1.00 | 63 |
| ALIKED+LightGlue, top-10 | 0.73 | 0.87 | 0.81 | 454 |
| XFeat+LighterGlue, top-10 | 0.87 | 0.87 | 0.88 | 427 |

  - The learned matchers are *worse* than SIFT on the synthetic set. That set has in-plane rotations up to 35°. ALIKED-n16 isn't rotation invariant (it missed one pair completely), and XFeat localizes on a stride-8 grid (2–3 px). Rotated planar textures favor SIFT, so this says little about real furniture.
- **Latency per image / per pair** (M2, steady state):

| Matcher | Extract at 600 px | Extract, 1240×1540 input | Match one pair |
|---|---|---|---|
| SIFT (CPU) | 18 ms | 49 ms | 4 ms |
| ALIKED+LightGlue (MPS, 640 px, 1024 kp) | 115 ms | 105 ms | 47 ms |
| XFeat+LighterGlue (MPS, 640 px, 1024 kp) | 13 ms | 17 ms | 50 ms |
| XFeat+LighterGlue (CPU, 640 px, 1024 kp) | 21 ms | 28 ms | 36 ms |

  - With 2048 keypoints at 1024 px, matching rises to 56–114 ms per pair, so 20 pairs take over 1 s.
  - The first call with a new input shape on MPS can take seconds (kernel compilation). This shows up as a slow first interactive query.
- **Real data warning** (no prompt, whole images):
  - ALIKED+LightGlue gives the *room* distractor 10 homography inliers. That's above the 8-inlier acceptance threshold, so it's a **false verification**.
  - On the same-model sofa pair, all matchers find at most 5 homography inliers (XFeat finds only 2 matches).
  - Inlier count alone does not separate instances on real furniture. This motivates step 7 (coverage, warped-mask IoU and residual-based scoring) and gallery grounding (box or mask per gallery object instead of the whole image).
- **Follow-ups:**
  - The `aliked-n16rot` variant for rotation robustness.
  - Batching LightGlue over the candidates, if per-pair latency matters on the target GPU.
  - Re-evaluating on real data and on the MIG box before choosing the default stage-2 matcher. SIFT stays the default for now.

## Evidence (optional)
`notebooks/cascade.ipynb` (stage-2 table). `uv run pytest tests/test_matching.py` (21 passed, including the learned matchers' contract and homography tests).
