# Masking interface and SAM3 adapter (code only, not run on the real model)

- Date: 2026-11-24
- Type: feature

## What changed
- **`masking/base.py`:**
  - A `Masker` interface returns a `MaskResult`: a full-resolution bool mask, a score, the source, alternative masks and timings.
  - Two fallbacks: `NoMasker` (whole image, the "no mask" ablation) and `BoxMasker` (the prompt box; clicks use the box around the positive points).
  - A mask prompt is always passed through unchanged.
  - `with_mask(item, result)` replaces an item's prompt with `Prompt(mask=...)` and keeps the original box or clicks in `extra["source_prompt"]`. Retrieval and matching therefore see masks through the same prompt schema; `DinoRetriever`'s `masked_mean` already pools under a mask prompt.
- **`masking/sam3.py`: `Sam3Masker`** for the ONNX export (vision encoder, geometry encoder, decoder). The fp16 variant is the default.
  - **Box prompt:** the box is encoded as a positive exemplar. Among the up-to-200 candidates, the one overlapping the box best is chosen (ties broken by score). Runner-ups are returned as alternatives.
  - **Click prompt** (the export has no point input):
    - A pseudo-box is built around the positive clicks.
    - Candidates that contain a positive click and no negative click are kept and unioned greedily by score.
    - If no candidate qualifies, the masker falls back to a box mask (`source="box(sam3-fallback)"`).
  - Vision-encoder features (about 120 MB per image) are kept in a small in-memory LRU keyed by image fingerprint, so prompt refinements on the same image skip the encoder. A disk cache isn't worth it at that size.
  - Inputs are cast to the dtypes each model declares, and fp16 outputs are converted back to float32, so fp16 exports with float16 I/O also work.
- **`scripts/try_sam3.py`:** a manual go/no-go check. It loads the parts, runs one cold and two warm prompts, prints per-step timings and writes an overlay PNG.
- **`viz/overlay.py`:** plain mask and prompt overlays, also for the upcoming UI.
- **Dependencies:** a new optional extra, `onnx`: onnx, plus onnxruntime on macOS or onnxruntime-gpu elsewhere (1.32.0 is locked with a Linux x86_64 CUDA wheel). Install it with `uv sync --extra onnx`. A plain `uv sync` removes it again.

## Why
The owner asked for SAM3 code without running it on the 8 GB dev machine; they will test the fp16 export on the target hardware.

**Deviation from the plan:** the plan said to wrap mlcvtools' `experimental.sam_onnx.Sam3InferenceHelper`. Instead we reuse its tensor contract, `mt.model_load` and mlcvtools' `preprocess` (the exact same preprocessing call), but not the helper class, for two reasons:
- mlcvtools validates input dtypes strictly and never casts. The helper passes float32 everywhere, which would fail if the fp16 export has float16 inputs.
- Its geometry API goes through `BBoxLabel.class_index` for the box label.

No change to mlcvtools is needed for this.

## Impact
- **Behavior/results:** none measured. SAM3 is only covered by unit tests with fake ONNX models: tensor names and dtypes, box conversion, candidate selection, encoder caching and fallback. 75 tests pass.
- **Open questions for the manual test** (most likely failure points first):
  1. **Geometry-only prompting.** We feed only the geometry features to the decoder. The original SAM3 pairs exemplar boxes with a text prompt (default "visual"). If masks look poor, the next thing to try is concatenating `text_encode("visual")` features. That needs the CLIP BPE vocab file `bpe_simple_vocab_16e6.txt.gz`, which isn't in `models/sam3/`.
  2. **fp16 tensor names and dtypes.** These are assumed to match the q8 export. A mismatch raises a clear error from mlcvtools' validation.
  3. **Scores.** `presence_logits` is ignored for scoring, as in mlcvtools. `score_threshold` (0.3) and `mask_threshold` (0.5) may need tuning.
  4. **Latency** against the 1 s budget, both cold (encoder) and warm (decoder only). On the M2 CPU the q8 decoder alone took 1.35 s, so a warm prompt on the Mac may also be borderline.
- **Code/interfaces:** the UI (step 5) uses `Masker.mask(...)` for the live preview. Until SAM3 passes, the default is `BoxMasker`.
- **Follow-ups:** the owner runs `scripts/try_sam3.py` on the 16 GB Mac and on the MIG box and records the go/no-go in a new log entry.

## Evidence (optional)
`uv run pytest tests/test_masking.py`, using fake models only.
