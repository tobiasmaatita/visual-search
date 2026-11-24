"""SAM3 (ONNX) masker: box and click prompts -> object mask.

!!! UNTESTED WITH THE REAL MODEL. !!!
This adapter is written against the tensor contract of the SAM3 ONNX export in
``models/sam3/`` (inspected on the q8 variant) and mlcvtools'
``experimental/sam_onnx.py``. It has only been tested with fake models. The owner
tests the fp16 export manually (see ``scripts/try_sam3.py``); the q8 export is
rejected on latency (``docs/log/2026-11-24-01-latency-budget-and-rejections.md``).

Tensor contract (q8 export, assumed identical for fp16 apart from dtypes):

- vision encoder: ``images`` (1, 3, 1008, 1008), RGB, plain resize, (x/255 - 0.5)/0.5
  -> ``fpn_feat_0`` (1,256,288,288), ``fpn_feat_1`` (1,256,144,144),
  ``fpn_feat_2`` (1,256,72,72), ``fpn_pos_2`` (1,256,72,72).
- geometry encoder: ``input_boxes`` (1, N, 4) normalized cx, cy, w, h;
  ``input_boxes_labels`` (1, N) int64, 1 = positive exemplar, 0 = negative;
  ``fpn_feat_2``, ``fpn_pos_2`` -> ``geometry_features`` (1, N+1, 256), ``geometry_mask``.
- decoder: fpn feats + ``prompt_features``, ``prompt_mask`` -> ``pred_masks``
  (1, 200, 288, 288) logits, ``pred_boxes`` (1, 200, 4) normalized x0, y0, x1, y1,
  ``pred_logits`` (1, 200), ``presence_logits`` (1, 1).

SAM3 box prompts are *exemplars* ("segment everything like this"), and the export
has no point input. So:

- **box**: encode the box and pick the candidate that overlaps the box best.
- **points**: encode a pseudo-box around the positive clicks, then keep the
  candidates that contain positive clicks and no negative click, and union them.
  If nothing qualifies, fall back to a box mask.

Because the image is resized without cropping, all normalized outputs map
linearly back to the original image.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from visual_search.data.schema import Prompt
from visual_search.masking.base import BoxMasker, Masker, MaskResult

SAM3_INPUT_SIZE = 1008
MODEL_PARTS = ("vision-encoder", "geometry-encoder", "decoder")


class OnnxModel(Protocol):
    """The subset of mlcvtools' tensor model wrapper used here (``mt.model_load``)."""

    input_names: list[str]
    input_dtypes: list[np.dtype | None]
    output_names: list[str]

    def predict(self, x: dict[str, NDArray]) -> list[NDArray] | NDArray:
        """Run the model on named inputs; returns outputs in ``output_names`` order."""
        ...


@dataclass
class Sam3Config:
    """Configuration of :class:`Sam3Masker`.

    Attributes:
        model_dir: Directory with ``vision-encoder<suffix>.onnx``, ``geometry-encoder<suffix>.onnx``
            and ``decoder<suffix>.onnx``.
        variant: File suffix without dash: ``"fp16"`` (default), ``"q8"`` (rejected on
            latency) or ``""`` (fp32).
        providers: ONNX Runtime providers, e.g. ``["CUDAExecutionProvider"]`` or
            ``["CPUExecutionProvider"]``; ``None`` lets mlcvtools pick the best available.
        score_threshold: Minimum candidate score (sigmoid of ``pred_logits``) to be
            considered; if no candidate passes, all candidates are considered.
        mask_threshold: Probability threshold that binarizes a mask.
        point_box_pad: Relative padding of the pseudo-box around positive clicks.
        point_box_min_frac: Minimum pseudo-box side as a fraction of the shorter image side.
        n_alternatives: How many runner-up masks to return for display.
        cache_size: Images whose encoder features are kept in memory (each ~120 MB
            in float32), so prompt refinements on the same image skip the encoder.
        runtime_options: Extra mlcvtools ``runtime_options`` (e.g. ``{"session_options": so}``);
            ``providers`` above is merged in.
    """

    model_dir: str | Path = "models/sam3"
    variant: str = "fp16"
    providers: list[str] | None = None
    score_threshold: float = 0.3
    mask_threshold: float = 0.5
    point_box_pad: float = 0.25
    point_box_min_frac: float = 0.1
    n_alternatives: int = 3
    cache_size: int = 2
    runtime_options: dict[str, Any] = field(default_factory=dict)

    def model_path(self, part: str) -> Path:
        """Return the ONNX file path of one model part (e.g. ``"decoder"``)."""
        suffix = f"-{self.variant}" if self.variant else ""
        return Path(self.model_dir) / f"{part}{suffix}.onnx"


# --- pure helpers (unit-tested without the model) ---------------------------------


def sigmoid(x: NDArray) -> NDArray[np.float32]:
    """Numerically safe logistic function."""
    return (1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float32), -64, 64)))).astype(np.float32)


def box_to_cxcywh_norm(box: tuple[float, float, float, float], image_hw: tuple[int, int]) -> NDArray[np.float32]:
    """Convert a pixel box ``(x0, y0, x1, y1)`` to normalized ``(cx, cy, w, h)``.

    Args:
        box: Box in original-image pixels.
        image_hw: ``(H, W)``.

    Returns:
        ``(4,)`` float32 in [0, 1].
    """
    h, w = image_hw
    x0, y0, x1, y1 = box
    out = np.array([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h], np.float32)
    return np.clip(out, 0.0, 1.0)


def points_pseudo_box(
    points: NDArray[np.float32],
    labels: NDArray[np.int64],
    image_hw: tuple[int, int],
    pad: float = 0.25,
    min_frac: float = 0.1,
) -> tuple[float, float, float, float]:
    """Build an exemplar box around the positive clicks.

    Args:
        points: ``(N, 2)`` click positions ``(x, y)`` in pixels.
        labels: ``(N,)`` 1 = positive, 0 = negative.
        image_hw: ``(H, W)``.
        pad: Relative padding per side.
        min_frac: Minimum side length as a fraction of the shorter image side.

    Returns:
        ``(x0, y0, x1, y1)`` in pixels, clipped to the image.
    """
    h, w = image_hw
    pos = points[labels == 1]
    x0, y0 = pos.min(axis=0)
    x1, y1 = pos.max(axis=0)
    min_side = min_frac * min(h, w)
    bw = max((x1 - x0) * (1 + 2 * pad), min_side)
    bh = max((y1 - y0) * (1 + 2 * pad), min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (
        float(max(cx - bw / 2, 0)),
        float(max(cy - bh / 2, 0)),
        float(min(cx + bw / 2, w)),
        float(min(cy + bh / 2, h)),
    )


def box_iou(boxes: NDArray[np.floating], box: NDArray[np.floating]) -> NDArray[np.float32]:
    """IoU between each row of ``boxes`` (N, 4) and one ``box`` (4,), all ``x0, y0, x1, y1``."""
    ix0 = np.maximum(boxes[:, 0], box[0])
    iy0 = np.maximum(boxes[:, 1], box[1])
    ix1 = np.minimum(boxes[:, 2], box[2])
    iy1 = np.minimum(boxes[:, 3], box[3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    return (inter / np.maximum(area + area_b - inter, 1e-9)).astype(np.float32)


def rank_candidates_for_box(
    boxes_norm: NDArray[np.floating],
    scores: NDArray[np.floating],
    prompt_box_norm: NDArray[np.floating],
    score_threshold: float,
) -> list[int]:
    """Order candidates for a box prompt: best overlap with the prompt box first.

    Candidates below ``score_threshold`` are dropped unless none pass. Ties in IoU
    are broken by score.

    Args:
        boxes_norm: ``(N, 4)`` candidate boxes, normalized ``x0, y0, x1, y1``.
        scores: ``(N,)`` candidate scores in [0, 1].
        prompt_box_norm: ``(4,)`` prompt box, normalized ``x0, y0, x1, y1``.
        score_threshold: Minimum score.

    Returns:
        Candidate indices, best first.
    """
    keep = np.flatnonzero(scores >= score_threshold)
    if keep.size == 0:
        keep = np.arange(len(scores))
    iou = box_iou(boxes_norm[keep], prompt_box_norm)
    order = np.lexsort((-scores[keep], -iou))  # last key is primary
    return keep[order].tolist()


def select_candidates_for_points(
    masks_prob: NDArray[np.floating],
    scores: NDArray[np.floating],
    points_norm: NDArray[np.floating],
    labels: NDArray[np.int64],
    score_threshold: float,
    mask_threshold: float = 0.5,
) -> list[int]:
    """Pick the candidate masks that explain the clicks.

    A candidate qualifies if it contains at least one positive click and no
    negative click. Qualifying candidates are taken greedily by score until every
    coverable positive click is covered; their union is the object.

    Args:
        masks_prob: ``(N, h, w)`` mask probabilities at decoder resolution.
        scores: ``(N,)`` candidate scores.
        points_norm: ``(P, 2)`` clicks, normalized ``(x, y)`` in [0, 1].
        labels: ``(P,)`` 1 = positive, 0 = negative.
        score_threshold: Candidates below this score are ignored.
        mask_threshold: Probability above which a pixel is inside the mask.

    Returns:
        Indices of the selected candidates (empty if none qualifies), best first.
    """
    h, w = masks_prob.shape[1:]
    cols = np.clip((points_norm[:, 0] * w).astype(int), 0, w - 1)
    rows = np.clip((points_norm[:, 1] * h).astype(int), 0, h - 1)
    inside = masks_prob[:, rows, cols] >= mask_threshold  # (N, P)
    pos, neg = labels == 1, labels == 0
    ok = (scores >= score_threshold) & inside[:, pos].any(axis=1) & ~inside[:, neg].any(axis=1)
    chosen: list[int] = []
    covered = np.zeros(int(pos.sum()), bool)
    for i in np.flatnonzero(ok)[np.argsort(-scores[ok], kind="stable")]:
        new = inside[i, pos] & ~covered
        if new.any():
            chosen.append(int(i))
            covered |= new
        if covered.all():
            break
    return chosen


def upsample_mask(mask_prob: NDArray[np.floating], image_hw: tuple[int, int], threshold: float) -> NDArray[np.bool_]:
    """Resize a decoder-resolution probability map to the image and binarize it."""
    h, w = image_hw
    return cv2.resize(mask_prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= threshold


# --- the masker -------------------------------------------------------------------


class Sam3Masker(Masker):
    """Object masks from box or click prompts with the SAM3 ONNX export.

    Encoder features are kept in a small in-memory LRU keyed by ``image_key``, so
    refining the prompt on the same image only re-runs the (cheaper) geometry
    encoder and decoder.

    Example:
        >>> masker = Sam3Masker(Sam3Config(model_dir="models/sam3", variant="fp16"))
        >>> result = masker.mask(rgb, Prompt(box=(100, 80, 300, 260)), image_key="img1")
        >>> result.mask.shape, result.score, result.timings
    """

    name = "sam3"

    def __init__(self, config: Sam3Config | None = None, models: dict[str, OnnxModel] | None = None) -> None:
        """Load the three ONNX parts (or use injected models).

        Args:
            config: Paths, variant, providers and thresholds.
            models: Optional ``{"vision-encoder", "geometry-encoder", "decoder"}`` ->
                model objects; bypasses loading (used by tests).

        Raises:
            FileNotFoundError: If a model file is missing.
        """
        self.config = config or Sam3Config()
        self.models = models if models is not None else self._load_models()
        self._features: OrderedDict[str, dict[str, NDArray]] = OrderedDict()
        self._fallback = BoxMasker(min_size=32)

    def _load_models(self) -> dict[str, OnnxModel]:
        """Load the ONNX parts with mlcvtools' loader."""
        import mlcvtools as mt  # deferred: pulls in onnxruntime

        options: dict[str, Any] = dict(self.config.runtime_options)
        if self.config.providers is not None:
            options["providers"] = self.config.providers
        models = {}
        for part in MODEL_PARTS:
            path = self.config.model_path(part)
            if not path.is_file():
                raise FileNotFoundError(f"SAM3 model part not found: {path}")
            models[part] = mt.model_load(path, runtime_options=options or None)
        return models

    @staticmethod
    def _run(model: OnnxModel, feeds: dict[str, NDArray]) -> dict[str, NDArray]:
        """Run a model, casting inputs to its declared dtypes and outputs to float32."""
        cast = {}
        for name, dtype in zip(model.input_names, model.input_dtypes):
            value = feeds[name]
            cast[name] = value.astype(dtype) if dtype is not None and value.dtype != dtype else value
        outputs = model.predict(cast)
        outputs = outputs if isinstance(outputs, list) else [outputs]
        return {
            name: (out.astype(np.float32) if out.dtype == np.float16 else out)
            for name, out in zip(model.output_names, outputs)
        }

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Resize to 1008 x 1008 and normalize to [-1, 1]; returns ``(1, 3, 1008, 1008)``.

        Args:
            image: ``(H, W, 3)`` uint8 RGB.

        Returns:
            NCHW float32 tensor.
        """
        from mlcvtools.pipeline.preprocess import preprocess  # same call as mlcvtools' Sam3InferenceHelper

        x = preprocess(
            image[:, :, :3],
            mean_subtract=(0.5, 0.5, 0.5),
            std_divide=(0.5, 0.5, 0.5),
            normalize=False,
            img_shape=[SAM3_INPUT_SIZE, SAM3_INPUT_SIZE],
        )
        return np.transpose(x, (2, 0, 1)).astype(np.float32)[np.newaxis]

    def encode(self, image: NDArray[np.uint8], image_key: str | None = None) -> dict[str, NDArray]:
        """Return the vision-encoder features of an image (cached by ``image_key``).

        Args:
            image: ``(H, W, 3)`` uint8 RGB.
            image_key: Stable id of the image content; ``None`` disables caching.

        Returns:
            Output name -> array (``fpn_feat_0..2``, ``fpn_pos_2``, possibly more).
        """
        if image_key is not None and image_key in self._features:
            self._features.move_to_end(image_key)
            return self._features[image_key]
        model = self.models["vision-encoder"]
        feats = self._run(model, {model.input_names[0]: self.preprocess(image)})
        if image_key is not None and self.config.cache_size > 0:
            self._features[image_key] = feats
            while len(self._features) > self.config.cache_size:
                self._features.popitem(last=False)
        return feats

    def decode_box(
        self, feats: dict[str, NDArray], box: tuple[float, float, float, float], image_hw: tuple[int, int]
    ) -> dict[str, NDArray]:
        """Run geometry encoder + decoder for one positive exemplar box.

        Args:
            feats: Output of :meth:`encode`.
            box: Exemplar box ``(x0, y0, x1, y1)`` in pixels.
            image_hw: Original ``(H, W)``.

        Returns:
            ``{"masks": (200, 288, 288) probabilities, "boxes": (200, 4) normalized
            x0, y0, x1, y1, "scores": (200,), "presence": float}``.
        """
        geo = self.models["geometry-encoder"]
        boxes = box_to_cxcywh_norm(box, image_hw)[None, None]
        labels = np.ones((1, 1), np.int64)
        # Inputs in export order, as in mlcvtools' Sam3InferenceHelper.geo_encode.
        values = [boxes, labels, feats["fpn_feat_2"], feats["fpn_pos_2"]]
        prompt = self._run(geo, dict(zip(geo.input_names, values)))
        prompt_features, prompt_mask = (prompt[n] for n in geo.output_names[:2])

        dec = self.models["decoder"]
        feeds = {
            "fpn_feat_0": feats["fpn_feat_0"],
            "fpn_feat_1": feats["fpn_feat_1"],
            "fpn_feat_2": feats["fpn_feat_2"],
            "fpn_pos_2": feats["fpn_pos_2"],
            "prompt_features": prompt_features,
            "prompt_mask": prompt_mask.astype(bool),
        }
        out = self._run(dec, feeds)
        return {
            "masks": sigmoid(out["pred_masks"][0]),
            "boxes": out["pred_boxes"][0],
            "scores": sigmoid(out["pred_logits"][0]),
            "presence": float(sigmoid(out["presence_logits"]).ravel()[0]),
        }

    def _predict(self, image: NDArray[np.uint8], prompt: Prompt | None, image_key: str | None) -> MaskResult:
        """Encode (or reuse) the image, decode the prompt and select the object mask."""
        cfg = self.config
        hw = (image.shape[0], image.shape[1])
        if prompt is None:
            return MaskResult(np.ones(hw, bool), None, "none")
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        feats = self.encode(image, image_key)
        timings["encode"] = time.perf_counter() - t0

        if prompt.box is not None:
            box = prompt.box
        else:
            assert prompt.points is not None and prompt.point_labels is not None
            box = points_pseudo_box(prompt.points, prompt.point_labels, hw, cfg.point_box_pad, cfg.point_box_min_frac)
        t0 = time.perf_counter()
        pred = self.decode_box(feats, box, hw)
        timings["decode"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if prompt.box is not None:
            h, w = hw
            box_norm = np.array([box[0] / w, box[1] / h, box[2] / w, box[3] / h], np.float32)
            ranked = rank_candidates_for_box(pred["boxes"], pred["scores"], box_norm, cfg.score_threshold)
            chosen = ranked[:1]
            others = ranked[1 : 1 + cfg.n_alternatives]
        else:
            points_norm = prompt.points / np.array([hw[1], hw[0]], np.float32)
            chosen = select_candidates_for_points(
                pred["masks"], pred["scores"], points_norm, prompt.point_labels, cfg.score_threshold, cfg.mask_threshold
            )
            others = [i for i in np.argsort(-pred["scores"]).tolist() if i not in chosen][: cfg.n_alternatives]

        mask = np.zeros(hw, bool)
        for i in chosen:
            mask |= upsample_mask(pred["masks"][i], hw, cfg.mask_threshold)
        alternatives = [(upsample_mask(pred["masks"][i], hw, cfg.mask_threshold), float(pred["scores"][i]))
                        for i in others]
        timings["select"] = time.perf_counter() - t0

        if not mask.any():
            fallback = self._fallback.mask(image, prompt)
            fallback.source = "box(sam3-fallback)"
            fallback.alternatives = alternatives
            fallback.timings = timings
            return fallback
        score = float(max(pred["scores"][i] for i in chosen))
        return MaskResult(mask, score, self.name, alternatives, timings)
