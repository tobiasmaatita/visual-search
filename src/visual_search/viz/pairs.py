"""Pair detail view: matches/inliers between query and candidate, and the warp."""

from __future__ import annotations

import io

import cv2
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray

from visual_search.geometry.verify import Verification
from visual_search.matching.base import Correspondences
from visual_search.viz.overlay import overlay_mask

INLIER_COLOR = (0.1, 0.8, 0.1, 0.8)
OUTLIER_COLOR = (0.9, 0.1, 0.1, 0.25)


def side_by_side(
    left: NDArray[np.uint8], right: NDArray[np.uint8], height: int = 480
) -> tuple[NDArray[np.uint8], float, float, int]:
    """Place two images next to each other at a common height.

    Args:
        left: First RGB image.
        right: Second RGB image.
        height: Output height in pixels.

    Returns:
        ``(canvas, scale_left, scale_right, x_offset_right)``.
    """
    sl, sr = height / left.shape[0], height / right.shape[0]
    a = cv2.resize(left, (max(1, round(left.shape[1] * sl)), height), interpolation=cv2.INTER_AREA)
    b = cv2.resize(right, (max(1, round(right.shape[1] * sr)), height), interpolation=cv2.INTER_AREA)
    gap = np.full((height, 8, 3), 255, np.uint8)
    return np.hstack([a, gap, b]), sl, sr, a.shape[1] + gap.shape[1]


def warp_to_query(
    image_c: NDArray[np.uint8], model: NDArray[np.float64], query_hw: tuple[int, int]
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Warp a candidate image into the query frame with a candidate->query homography/affine.

    Args:
        image_c: Candidate RGB image.
        model: ``3x3`` matrix mapping candidate pixels to query pixels.
        query_hw: Query ``(H, W)``.

    Returns:
        ``(warped, valid)``: the warped image and the mask of query pixels covered by it.
    """
    h, w = query_hw
    warped = cv2.warpPerspective(image_c, model, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(image_c.shape[:2], 255, np.uint8), model, (w, h)) > 127
    return warped, valid


def render_pair_detail(
    image_q: NDArray[np.uint8],
    image_c: NDArray[np.uint8],
    corr: Correspondences,
    verification: Verification | None,
    mask_q: NDArray[np.bool_] | None = None,
    mask_c: NDArray[np.bool_] | None = None,
    max_lines: int = 200,
    title: str = "",
    accepted: bool = True,
) -> bytes:
    """Render matches and, if the model allows it, the candidate warped onto the query.

    Panels: (1) query | candidate with match lines (inliers green, outliers faint
    red); (2) the candidate warped into the query frame and blended with it, with
    the warped candidate mask (or warped image border) and the query mask outline.

    Args:
        image_q: Query RGB image.
        image_c: Candidate RGB image.
        corr: Tentative correspondences.
        verification: Fitted model and inliers, or ``None``.
        mask_q: Query mask (drawn in panel 2).
        mask_c: Candidate mask (warped into the query frame in panel 2).
        max_lines: Maximum number of drawn lines per class (inliers/outliers).
        title: Figure title.
        accepted: False if the re-ranking stage rejected the model (e.g. too few
            inliers); the warp panel is then omitted, since it would be meaningless.

    Returns:
        PNG bytes.
    """
    canvas, sl, sr, off = side_by_side(image_q, image_c)
    can_warp = accepted and verification is not None and verification.can_warp
    fig = Figure(figsize=(14 if can_warp else 9, 4.5))
    ratios = [2, 1] if can_warp else [1]
    axes = fig.subplots(1, len(ratios), squeeze=False, gridspec_kw={"width_ratios": ratios})
    ax = axes[0, 0]
    ax.imshow(canvas)
    ax.set_axis_off()
    inl = verification.inliers if verification is not None else np.zeros(len(corr), bool)
    rng = np.random.default_rng(0)
    for sel, color in ((~inl, OUTLIER_COLOR), (inl, INLIER_COLOR)):
        idx = np.flatnonzero(sel)
        if len(idx) > max_lines:
            idx = rng.choice(idx, max_lines, replace=False)
        if len(idx):
            segs = np.stack([corr.pts_q[idx] * sl, corr.pts_c[idx] * sr + [off, 0]], axis=1)
            ax.add_collection(LineCollection(segs, colors=[color], linewidths=0.8))
            ax.scatter(segs[:, 0, 0], segs[:, 0, 1], s=4, color=color[:3])
            ax.scatter(segs[:, 1, 0], segs[:, 1, 1], s=4, color=color[:3])
    n_inl = int(inl.sum())
    info = f"{len(corr)} matches, {n_inl} inliers"
    if verification is not None:
        info += f" ({verification.model_type}, {verification.method}, thr {verification.threshold:g}px)"
    if not accepted:
        info += " - REJECTED (too few inliers)"
    ax.set_title(f"query | candidate: {info}", fontsize=9)

    if can_warp:
        assert verification is not None and verification.model is not None
        ax = axes[0, 1]
        warped, valid = warp_to_query(image_c, verification.model, image_q.shape[:2])
        blend = image_q.copy()
        blend[valid] = (0.5 * image_q[valid] + 0.5 * warped[valid]).astype(np.uint8)
        if mask_c is not None:
            wm = cv2.warpPerspective(mask_c.astype(np.uint8), verification.model,
                                     (image_q.shape[1], image_q.shape[0]), flags=cv2.INTER_NEAREST) > 0
            blend = overlay_mask(blend, wm, color=(255, 140, 0), alpha=0.3)
        else:
            contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blend, contours, -1, (255, 140, 0), max(1, min(image_q.shape[:2]) // 200))
        if mask_q is not None:
            blend = overlay_mask(blend, mask_q, alpha=0.0)
        ax.imshow(blend)
        ax.set_axis_off()
        ax.set_title("candidate warped onto query (orange), query mask (blue)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    return buf.getvalue()
