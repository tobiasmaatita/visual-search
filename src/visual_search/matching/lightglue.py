"""Learned sparse matchers via kornia: ALIKED + LightGlue and XFeat + LighterGlue.

Requires the optional ``sparse`` extra (``uv sync --extra sparse``). Weights are
downloaded once by ``torch.hub`` into ``~/.cache/torch/hub/checkpoints``:

- ``aliked-n16.pth`` (github.com/Shiaoming/ALIKED, BSD-3-Clause, 2.7 MB)
- ``aliked_lightglue.pth`` (github.com/cvg/LightGlue v0.1_arxiv, Apache-2.0, 47.6 MB)
- ``xfeat.pt`` and ``xfeat-lighterglue.pt`` (github.com/verlab/accelerated_features,
  Apache-2.0, 6.2 MB + 10.8 MB)

Both pipelines share one class: a keypoint/descriptor network + a LightGlue-style
attention matcher; they differ only in the extractor.

Latency note (M2): the first call with a new input shape on MPS can take seconds
(kernel compilation); steady state is ~15 ms (XFeat) / ~110 ms (ALIKED) per image
at 640 px and ~35-50 ms per pair for matching. Re-rank only a short list (e.g. 10).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from visual_search.device import resolve_device
from visual_search.matching.base import Correspondences, Features, Matcher
from visual_search.matching.registry import register_matcher

logger = logging.getLogger(__name__)

Extractor = Literal["aliked", "xfeat"]


class LearnedSparseMatcher(Matcher):
    """Keypoints + descriptors from a learned extractor, matched with LightGlue.

    Images are downscaled so the longer side is at most ``max_side`` before
    extraction; keypoints are reported in original-image pixels.
    """

    extractor_name: Extractor = "aliked"

    def __init__(
        self,
        max_keypoints: int = 1024,
        max_side: int = 640,
        match_threshold: float = 0.1,
        device: str = "auto",
    ) -> None:
        """Load the extractor and the matcher (downloads weights on first use).

        Args:
            max_keypoints: Keep at most this many keypoints per image. LightGlue cost grows
                quadratically with it; 1024 keeps a pair at ~35-50 ms on an M2 (MPS/CPU).
            max_side: Longer image side used for extraction, in pixels.
            match_threshold: LightGlue matchability threshold (lower = more matches).
            device: ``"auto"`` or an explicit torch device.
        """
        import kornia.feature as KF  # optional dependency ("sparse" extra)

        self.max_keypoints = max_keypoints
        self.max_side = max_side
        self.match_threshold = match_threshold
        self.device = resolve_device(device)
        if self.extractor_name == "aliked":
            self._extractor = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=max_keypoints,
                                                        device=self.device)
        else:
            self._extractor = KF.XFeat.from_pretrained(top_k=max_keypoints).to(self.device)
        self._extractor.eval()
        self._lightglue = KF.LightGlue(self.extractor_name, filter_threshold=match_threshold).eval().to(self.device)

    @property
    def identity(self) -> dict[str, Any]:
        """Return the extraction settings (the matcher does not affect cached features)."""
        return {"matcher": self.name, "extractor": self.extractor_name, "max_keypoints": self.max_keypoints,
                "max_side": self.max_side, "version": 1}

    @torch.inference_mode()
    def extract(self, image: NDArray[np.uint8]) -> Features:
        """Detect and describe keypoints with the learned extractor.

        Args:
            image: ``(H, W, 3)`` uint8 RGB.

        Returns:
            Features; descriptors are float16 ``(N, 128)`` for ALIKED, ``(N, 64)`` for XFeat.
        """
        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = image if scale >= 1 else cv2.resize(image, (round(w * scale), round(h * scale)),
                                                     interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.array(small)).permute(2, 0, 1)[None].float().div(255.0).to(self.device)
        if self.extractor_name == "aliked":
            out = self._extractor(x)[0]
            kpts, desc, scores = out.keypoints, out.descriptors, out.keypoint_scores
        else:
            out = self._extractor.detectAndCompute(x)[0]
            kpts, desc, scores = out["keypoints"], out["descriptors"], out["scores"]
        sy, sx = small.shape[0] / h, small.shape[1] / w
        pts = kpts.float().cpu().numpy()
        pts = ((pts + 0.5) / np.array([sx, sy], np.float32) - 0.5).astype(np.float32)
        return Features(pts, desc.half().cpu().numpy(), scores.float().cpu().numpy(), (h, w))

    @torch.inference_mode()
    def match_features(self, fq: Features, fc: Features) -> Correspondences:
        """Match two feature sets with LightGlue.

        Args:
            fq: Query features.
            fc: Candidate features.

        Returns:
            Correspondences with LightGlue's matching scores as confidence.
        """

        def side(f: Features) -> dict[str, torch.Tensor]:
            return {
                "keypoints": torch.from_numpy(f.keypoints)[None].to(self.device),
                "descriptors": torch.from_numpy(f.descriptors.astype(np.float32))[None].to(self.device),
                "image_size": torch.tensor([[f.image_hw[1], f.image_hw[0]]], dtype=torch.float32,
                                           device=self.device),
            }

        out = self._lightglue({"image0": side(fq), "image1": side(fc)})
        matches = out["matches"][0].cpu().numpy()
        conf = out["scores"][0].float().cpu().numpy().astype(np.float32)
        if len(matches) == 0:
            return Correspondences.empty()
        return Correspondences(fq.keypoints[matches[:, 0]], fc.keypoints[matches[:, 1]], conf)


@register_matcher("aliked-lightglue")
class AlikedLightGlueMatcher(LearnedSparseMatcher):
    """ALIKED-N16 keypoints + LightGlue (ICCV 2023) matcher."""

    extractor_name: Extractor = "aliked"


@register_matcher("xfeat-lighterglue")
class XFeatLighterGlueMatcher(LearnedSparseMatcher):
    """XFeat keypoints (CVPR 2024) + LighterGlue, the fast lightweight variant of LightGlue."""

    extractor_name: Extractor = "xfeat"
