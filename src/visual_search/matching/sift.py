"""SIFT / RootSIFT baseline (OpenCV): detection, description, ratio-test matching."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from visual_search.matching.base import Correspondences, Features, Matcher
from visual_search.matching.registry import register_matcher


@register_matcher("sift")
class SiftMatcher(Matcher):
    """OpenCV SIFT with optional RootSIFT descriptors, Lowe's ratio test and mutual check.

    Images are downscaled so the longer side is at most ``max_side`` before
    detection (speed); keypoints are reported in original-image pixels.

    Example:
        >>> corr = SiftMatcher(root_sift=True).match(img_q, img_c, mask_q)
    """

    def __init__(
        self,
        root_sift: bool = True,
        max_keypoints: int = 2048,
        max_side: int = 1024,
        ratio: float = 0.8,
        mutual: bool = True,
    ) -> None:
        """Create the matcher.

        Args:
            root_sift: Use RootSIFT (L1-normalize, then square root) descriptors,
                which compare better with Euclidean distance.
            max_keypoints: Keep at most this many strongest keypoints per image.
            max_side: Longer image side used for detection, in pixels.
            ratio: Lowe ratio threshold (lower = stricter).
            mutual: Also require the match to be the mutual nearest neighbor.
        """
        self.root_sift = root_sift
        self.max_keypoints = max_keypoints
        self.max_side = max_side
        self.ratio = ratio
        self.mutual = mutual
        self._sift = cv2.SIFT_create(nfeatures=max_keypoints)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)

    @property
    def identity(self) -> dict[str, Any]:
        """Return the extraction settings (matching settings don't affect features)."""
        return {"matcher": "sift", "root_sift": self.root_sift, "max_keypoints": self.max_keypoints,
                "max_side": self.max_side, "version": 1}

    def extract(self, image: NDArray[np.uint8]) -> Features:
        """Detect SIFT keypoints and compute (Root)SIFT descriptors.

        Args:
            image: ``(H, W, 3)`` uint8 RGB.

        Returns:
            Features; descriptors are ``(N, 128)`` float32.
        """
        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if scale < 1:
            gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        kps, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or not kps:
            return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), np.zeros(0, np.float32),
                            (h, w))
        pts = np.array([k.pt for k in kps], np.float32)
        # +0.5/-0.5 maps pixel centers correctly between resolutions.
        pts = (pts + 0.5) / scale - 0.5
        scores = np.array([k.response for k in kps], np.float32)
        desc = desc.astype(np.float32)
        if self.root_sift:
            desc /= np.maximum(desc.sum(axis=1, keepdims=True), 1e-7)
            desc = np.sqrt(desc)
        return Features(pts, desc, scores, (h, w))

    def match_features(self, fq: Features, fc: Features) -> Correspondences:
        """Ratio-test (and optionally mutual) nearest-neighbor matching.

        Args:
            fq: Query features.
            fc: Candidate features.

        Returns:
            Correspondences; confidence = ``1 - d1 / d2`` (0 = ambiguous, 1 = distinctive).
        """
        if len(fq) < 2 or len(fc) < 2:
            return Correspondences.empty()
        knn = self._bf.knnMatch(fq.descriptors, fc.descriptors, k=2)
        qi, ci, conf = [], [], []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                qi.append(m.queryIdx)
                ci.append(m.trainIdx)
                conf.append(1.0 - m.distance / max(n.distance, 1e-9))
        if not qi:
            return Correspondences.empty()
        qi_a, ci_a, conf_a = np.array(qi), np.array(ci), np.array(conf, np.float32)
        if self.mutual:
            back = self._bf.match(fc.descriptors[ci_a], fq.descriptors)
            back_idx = np.array([b.trainIdx for b in back])
            keep = back_idx == qi_a
            qi_a, ci_a, conf_a = qi_a[keep], ci_a[keep], conf_a[keep]
        return Correspondences(fq.keypoints[qi_a], fc.keypoints[ci_a], conf_a)
