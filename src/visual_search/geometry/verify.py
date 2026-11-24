"""Robust geometric verification of correspondences (OpenCV RANSAC / USAC MAGSAC++).

Convention: every model maps **candidate** points to **query** points
(``pts_q ~ model(pts_c)``), so the candidate image or mask can be warped into the
query frame for scoring and display.

Model choice:
- ``"H"`` homography: planar objects or small viewpoint changes (posters, wallpaper, flat sides).
- ``"A"`` affine: a simplified homography; more stable with few matches.
- ``"F"`` fundamental matrix: general 3D rigid objects under viewpoint change; only
  constrains points to epipolar lines, so it accepts more false matches. No warp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from visual_search.matching.base import Correspondences

ModelType = Literal["H", "A", "F"]
Method = Literal["magsac", "ransac"]

MIN_SAMPLES: dict[str, int] = {"H": 4, "A": 3, "F": 8}


@dataclass
class Verification:
    """Result of robust model fitting.

    Attributes:
        model_type: ``"H"``, ``"A"`` or ``"F"``.
        model: ``3x3`` matrix (H, F; affine is lifted to 3x3), or ``None`` if fitting failed.
        inliers: ``(M,)`` bool over the input correspondences.
        residuals: ``(M,)`` float32 error per correspondence in pixels: reprojection
            error for H/A, symmetric epipolar (Sampson) distance for F. ``inf`` if no model.
        method: Estimator used.
        threshold: Inlier threshold in pixels.
    """

    model_type: str
    model: NDArray[np.float64] | None
    inliers: NDArray[np.bool_]
    residuals: NDArray[np.float32]
    method: str
    threshold: float

    @property
    def n_inliers(self) -> int:
        """Return the number of inliers."""
        return int(self.inliers.sum())

    @property
    def ok(self) -> bool:
        """Return True if a model was found with at least the minimal sample size of inliers."""
        return self.model is not None and self.n_inliers >= MIN_SAMPLES[self.model_type]

    @property
    def inlier_ratio(self) -> float:
        """Return inliers / correspondences (0 if there are none)."""
        return float(self.inliers.mean()) if len(self.inliers) else 0.0

    @property
    def can_warp(self) -> bool:
        """Return True if the model maps candidate pixels to query pixels (H or A)."""
        return self.ok and self.model_type in ("H", "A")


def _residuals(model_type: str, model: NDArray[np.float64], pts_q: NDArray, pts_c: NDArray) -> NDArray[np.float32]:
    """Per-correspondence error in pixels for a fitted model."""
    if model_type in ("H", "A"):
        proj = cv2.perspectiveTransform(pts_c.reshape(-1, 1, 2).astype(np.float64), model).reshape(-1, 2)
        return np.linalg.norm(proj - pts_q, axis=1).astype(np.float32)
    # Sampson distance for x_q^T F x_c = 0.
    xq = np.hstack([pts_q, np.ones((len(pts_q), 1))])
    xc = np.hstack([pts_c, np.ones((len(pts_c), 1))])
    fxc = xc @ model.T
    ftxq = xq @ model
    num = np.sum(xq * fxc, axis=1) ** 2
    den = fxc[:, 0] ** 2 + fxc[:, 1] ** 2 + ftxq[:, 0] ** 2 + ftxq[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12)).astype(np.float32)


def verify(
    corr: Correspondences,
    model_type: ModelType = "H",
    method: Method = "magsac",
    threshold: float = 4.0,
    confidence: float = 0.999,
    max_iters: int = 5000,
) -> Verification:
    """Fit a geometric model to correspondences and classify inliers.

    Args:
        corr: Tentative correspondences.
        model_type: ``"H"`` (homography), ``"A"`` (affine) or ``"F"`` (fundamental).
        method: ``"magsac"`` (OpenCV ``USAC_MAGSAC``, MAGSAC++) or ``"ransac"`` (classic RANSAC).
        threshold: Inlier threshold in pixels (MAGSAC uses it as the maximum sigma).
        confidence: Desired probability of finding the best model.
        max_iters: Iteration cap.

    Returns:
        The verification. With fewer correspondences than the minimal sample, no
        model is fitted (``ok`` is False).

    Raises:
        ValueError: For an unknown model type or method.

    Example:
        >>> v = verify(corr, "H", "magsac", threshold=4.0)
        >>> v.ok, v.n_inliers, v.inlier_ratio
    """
    if model_type not in MIN_SAMPLES:
        raise ValueError(f"Unknown model type {model_type!r}.")
    if method not in ("magsac", "ransac"):
        raise ValueError(f"Unknown method {method!r}.")
    m = len(corr)
    fail = Verification(model_type, None, np.zeros(m, bool), np.full(m, np.inf, np.float32), method, threshold)
    if m < MIN_SAMPLES[model_type]:
        return fail
    src = corr.pts_c.astype(np.float64)
    dst = corr.pts_q.astype(np.float64)
    flag = cv2.USAC_MAGSAC if method == "magsac" else cv2.RANSAC
    try:
        if model_type == "H":
            model, mask = cv2.findHomography(src, dst, flag, threshold, maxIters=max_iters, confidence=confidence)
        elif model_type == "A":
            affine, mask = cv2.estimateAffine2D(src, dst, method=flag, ransacReprojThreshold=threshold,
                                                maxIters=max_iters, confidence=confidence)
            model = None if affine is None else np.vstack([affine, [0.0, 0.0, 1.0]])
        else:
            fm_flag = cv2.USAC_MAGSAC if method == "magsac" else cv2.FM_RANSAC
            # OpenCV's F satisfies p2^T F p1 = 0; with p1 = candidate, p2 = query we get x_q^T F x_c = 0.
            model, mask = cv2.findFundamentalMat(src, dst, fm_flag, threshold, confidence, max_iters)
            if model is not None and model.shape != (3, 3):
                model = model[:3]  # OpenCV may return several stacked solutions
    except cv2.error:
        return fail
    if model is None or mask is None:
        return fail
    inliers = mask.ravel().astype(bool)
    residuals = _residuals(model_type, model, dst, src)
    return Verification(model_type, model, inliers, residuals, method, threshold)
