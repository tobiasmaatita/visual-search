# Stage 1: DINOv2-small retrieval with prompt-focused pooling and a feature cache

- Date: 2026-11-24
- Type: feature

## What changed
- **`models/dinov2.py`: `Dinov2Backbone`.**
  - Loads any `facebook/dinov2-*` (or `-with-registers-*`) checkpoint with Hugging Face transformers on CUDA, MPS or CPU.
  - Returns the CLS token and the full patch-token grid.
  - Images are resized to a square (default 448 px, so a 32×32 grid) without cropping, so every patch maps linearly back to the original image.
- **`retrieval/dino.py`: `DinoRetriever`**, with three pooling modes:
  - `cls`: global CLS token of the query; ignores the prompt.
  - `crop_cls`: CLS of the query cropped to its padded prompt box; one extra forward pass. **New default.**
  - `masked_mean`: mean of the patch tokens under the prompt; no extra forward pass, so prompt changes are instant.
  - Gallery images are described as whole images, or by their own prompt if they have one.
- **`cache.py`: on-disk feature cache.**
  - Keyed by image content (file bytes or pixel bytes) plus backbone identity (model id, revision, resolution, cache version).
  - Writes are atomic, and a different config gets a different directory.
  - Stores the CLS token, the float16 patch grid and the original image size, so any pooling can be recomputed without re-running the model.
- **`device.py`:** device selection (CUDA > MPS > CPU) and a sync hook, so GPU latency includes queued work.
- **`imaging.prompt_patch_mask`:** maps a box, points or mask prompt onto the patch grid. It's never empty, even for a single click.
- **Notebook:** compares the baselines with all three DINOv2 modes.
- **Dependencies:** `torch>=2.9` and `transformers>=4.40,<5` are base dependencies; transformers is kept below 5 to stay compatible with mlcvtools. Linux wheels from PyPI include CUDA for the MIG box.

## Why
Stage 1 has to be fast (budget 0.1 s per query) and focused on the prompted object rather than the whole scene.
- `crop_cls` is the simplest way to focus the descriptor.
- `masked_mean` serves the interactive use case, where the prompt changes often and should not cost a forward pass each time.
- Torch is a base dependency, not an optional group: DINOv2 is the core of stage 1, so without it the package doesn't do its main job.
- FAISS isn't used; brute-force cosine is enough at these gallery sizes.

## Impact
- **Results** on the synthetic set (SMOKE TEST ONLY: planar textures, 15 queries, 70 gallery images of which 40 distractors; not representative of real objects):

| Pipeline | R@1 | R@5 | MRR | median ms/query |
|---|---|---|---|---|
| random | 0.00 | 0.00 | 0.06 | 0.04 |
| color histogram | 0.13 | 0.33 | 0.23 | 0.2 |
| DINOv2-small `cls` @448 | 0.20 | 0.60 | 0.37 | 57 |
| DINOv2-small `crop_cls` @448 | **0.67** | **0.80** | **0.73** | 56 |
| DINOv2-small `masked_mean` @448 | 0.33 | 0.73 | 0.50 | 0.7* |
| DINOv2-small `crop_cls` @224 | 0.33 | 0.80 | 0.53 | 16 |

  \* Measured with the query's full-image features already cached, which is the interactive situation after the first prompt. A cold query costs one forward pass, about 55 ms at 448 px.
- **Latency (M2, MPS, fp32):** one DINOv2-small forward pass takes about 54 ms at 448 px and about 16 ms at 224 px, so stage 1 fits its 0.1 s budget. Indexing the 70-image gallery from scratch took about 4 s, and cached re-runs take about 0.05 s.
- **Real `data/`:** still 1 query against 2 gallery images, so everything scores 1.0 and no conclusions can be drawn.
- **Cache size:** about 0.75 MB per image at 448 px (float16 patch grid). 1,000 gallery images take about 750 MB; 224 px is 4× smaller.
- **Repo hygiene:** the `.gitignore` entry `models/` also matched `src/visual_search/models/`. It's now anchored as `/models/`, like `/data/`. `cache/` is ignored everywhere.
- **Follow-ups:**
  - A gallery-side "patch max-sim" option (score each query patch against all gallery patches) for small objects in cluttered gallery images.
  - The registers checkpoint and dinov2-base as ablations.
  - Once SAM3 masks are available, `masked_mean` gets a real mask instead of the box.

## Evidence (optional)
`notebooks/cascade.ipynb`, executed headless (seed 0, synthetic: 12 instances × 3 views, 3 near-duplicate pairs, 40 distractors). `uv run pytest`: 61 passed, including an integration test with the real dinov2-small (224 px) that beats random on the synthetic set.
