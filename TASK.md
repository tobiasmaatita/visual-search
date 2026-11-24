# Task: instance-level re-ranking pipeline for visual prompting (from scratch)

Standing rules (language, docstrings, licenses, working method) are in `CLAUDE.md`. Follow them.

## Background
I'm researching how to go beyond *semantic* matching in visual prompting and find the **best visual match of the same instance**.

"Exact" means **same instance**, e.g. the same red cap or the same car seen earlier, regardless of pose or viewpoint. Objects are diverse: cars, caps, wallpaper patterns, etc. Rigid, non-rigid and textured/planar objects all occur.

The core idea is a **cascade**:
1. **Retrieval**: fast semantic candidate selection with DINOv2 embeddings (top-K, e.g. 100).
2. **Fast geometric verification**: sparse keypoint matching + robust estimation on a smaller set (e.g. top-20).
3. **Dense matching and scoring**: heavier dense matching/warping on the final shortlist (e.g. top-5).

There is **no existing codebase**. This is a greenfield repo.

## Environment
- Python + PyTorch, **a single GPU** (details: 20GB nvidia mig profile, or CoreML 8GB - code should run on both).
- ONNX models, used via `onnxruntime` (GPU provider if available):
  - SAM3: `models/sam3/`
  - RF-DETR: `models/rf-detr/`
- First inspect each model's inputs/outputs (e.g. does the DINOv2 export give CLS only, or also patch tokens? What input resolution? What prompt format does SAM3 expect?) and tell me what you find, since it affects the design.

## Dataset
I will create the real dataset. You build the loader, validator, and a smoke-test fallback.

The dataset is a **list of dicts**:
```python
{
  "input_image": <PIL.Image | np.ndarray | path>,
  "label": <str instance id, or None>,   # same id => same instance
  # optional, ignored by core code unless used:
  "prompt": {"box": [x0, y0, x1, y1]} | {"points": [...]} | {"mask": ...},
}
```

- Images of the same object share a `label` (e.g. `"cap_red_01"`).
- **Distractors come from COCO** and have `label = None` (easy negatives). A candidate with `label=None` is never a correct match.
- A prediction is correct iff `label_query == label_candidate and label_query is not None`.
- Hard negatives will be added later, possibly via an extra key such as `"group"`. Design the schema so this needs no refactor.
- Loader/validator: split into **queries** and **gallery**.
  - Each instance needs >= 2 images: one or more as queries, the rest in the gallery.
  - Distractors go in the gallery.
  - The split must be reproducible via a seed.
- Since the real dataset does not exist yet, provide a small synthetic set for end-to-end testing: random geometric/photometric augmentations of a few images, mixed with a few COCO images or placeholders. Mark clearly in the code that it is for smoke testing only.

## Interactive notebook (core requirement)
The notebook must be **interactive**: I want to define the visual prompt myself on a query image, then see what the cascade returns.

### Prompting UI
- I can **draw a box** on the query image, or **click one or more points** on an item (positive clicks, ideally also negative clicks to exclude regions).
- The prompt is converted to the `"prompt"` format from the dataset schema (`{"box": ...}` / `{"points": ...}`), so interactive and dataset-driven runs share exactly the same code path.
- After each prompt, show a **live SAM3 mask preview** (overlay on the query) so I can see what the object is before running the cascade. I can refine the prompt and re-run.
- Query image selection: pick from the query set (e.g. dropdown or index slider), or load any image from a path.
- Environment: JupyterLab. Pick a widget approach that works there and tell me the trade-offs in the plan. Candidates: `ipympl` (matplotlib `RectangleSelector` + click events), `anywidget` (custom canvas), `ipycanvas`, `jupyter-bbox-widget`. Prefer the simplest robust option with minimal dependencies.
- Keep the widget code in the package (e.g. `ui/`), not inline in the notebook, so the notebook stays readable.

### Results view
- After running the cascade, show the **top-k results** per stage as a simple gallery: stage-1 ranking vs final ranking, with the score (and the main score features) under each image, and a clear marker for GT-correct/incorrect when labels exist.
- Selecting a result (click or dropdown) shows a **detail view** for that query/candidate pair: the matches/inliers drawn on both images, the warped candidate mask on the query and, once available, the displacement/morph. Keep styling plain; polish comes later.
- Support running a single interactive query as well as the batch evaluation, both through the same pipeline object.
- Interactive use should feel responsive: reuse cached gallery embeddings/keypoints, and report per-stage timing for each interactive run.

## Methods to implement and compare
All pluggable behind common interfaces.

### Stage 1: retrieval
- DINOv2 embeddings (ONNX).
- Support global (CLS) descriptors and masked/prompted pooling of patch tokens (if available), so the visual prompt focuses the descriptor.
- Cosine similarity; brute-force or FAISS (decide in the plan).

### Masking
- SAM3 (ONNX): turn the visual prompt into an object mask so keypoints and warps outside the object are ignored. A bounding box is too coarse.
- Also support running without masks as an ablation.
- RF-DETR (ONNX) is available. Propose in the plan whether and how it is useful, e.g. object proposals/crops on gallery images so retrieval can be object-level instead of image-level. Weigh it up; don't just assume.

### Matchers (stage 2/3)
- Sparse: SIFT/RootSIFT (OpenCV) baseline; ALIKED + LightGlue; OmniGlue.
- Dense / detector-free: RoMa (dense warp + certainty map); optionally EfficientLoFTR.
- Optional, clearly flagged: MASt3R / DUSt3R / VGGT.
- If you know newer matchers that fit (post OmniGlue), propose them in the plan with license info. Do not add them without my approval.

### Geometric verification
- MAGSAC++ / USAC via OpenCV (`cv2.USAC_MAGSAC`) or PoseLib; compare with plain RANSAC.
- Model choice: homography (planar), fundamental matrix (3D), affine (simplified).
- Non-rigid options for deformable objects: thin-plate spline, moving least squares, dense flow from RoMa.
- Extra filters: mutual nearest neighbours, cycle-consistency (A->B->A).

### Scoring
Do not rely on inlier count alone, since it depends on texture. Implement separate, composable features:
- inlier count and inlier ratio
- **spatial coverage**: fraction of the query mask covered by inliers, grid-based (not convex hull)
- **mask IoU after warping** the candidate mask onto the query
- **post-warp residual** (e.g. DINO feature distance, SSIM or LPIPS inside the mask)
- reprojection error of inliers
- warp strain/distortion magnitude as a false-match indicator
- a combiner: first a simple hand-weighted score, with a hook to learn weights later (logistic regression / gradient boosting) once labels exist

### Displacement map / morphing
Nice-to-have diagnostic. Compute a displacement map (from the dense matcher or from keypoint correspondences) and warp candidate -> query. Keep it simple; no polished visualization yet.

## Evaluation
- Metrics: **Recall@1, Recall@5, MRR**, plus **latency per cascade stage** (median and p95).
- Report **before vs after re-ranking** (stage 1 only vs full cascade) and per-matcher ablations.
- Make the evaluation harness the first thing that works end to end, so every later method is compared on the same footing.
- Pitfalls to handle or at least log:
  - repetitive patterns and symmetric objects (false inliers)
  - large scale/rotation differences (consider multi-scale or multi-rotation attempts, or a dense matcher)
  - near-duplicates of the same model (different instance)

## Working method for this task
1. **Plan first. Do not write code yet.** Explore the environment (GPU, installed packages, ONNX files), inspect the ONNX models, then present a plan covering: repo layout, interfaces, dependency list with licenses, order of implementation, how each stage is tested, risks, and open questions for me.
2. **Wait for my approval of the plan.**
3. Then build step by step in this order, stopping after each step:
   1. Repo skeleton, dataset schema/loader/split, synthetic smoke-test data
   2. Evaluation harness (metrics, latency timing) with a trivial baseline
   3. Stage 1: DINOv2 (ONNX) retrieval + caching
   4. SAM3 (ONNX) masking from visual prompts
   5. Interactive prompting UI (box + click) with live SAM3 mask preview, and a basic stage-1 results gallery. From here on, I can try things myself in the notebook.
   6. Sparse matching + geometric verification (SIFT baseline first, then ALIKED+LightGlue, OmniGlue), with the pair detail view
   7. Scoring features + combiner
   8. Dense matching (RoMa) + warping/displacement map in the detail view
   9. Full cascade + ablation tables in the notebook