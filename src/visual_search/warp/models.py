"""Warp models that map a candidate image onto the query, including *partial* warps for animation.

All warps are fitted on correspondences (query point ``q_i`` <-> candidate point
``c_i``, in original-image pixels) and are used *backwards*: for a pixel in the
query frame they give the candidate pixel to sample, which is what
``cv2.remap`` needs.

For the animation every warp is blended with a simple **base** placement that
puts the candidate's box onto the query's box (scale + shift only). At ``t = 0``
the candidate is just resized into the query box; at ``t = 1`` the full warp is
applied; in between, the sampling positions are interpolated linearly. So the
motion you see is exactly the part of the alignment that the warp model adds.

Models:
- :class:`HomographyWarp` (``H``): a plane seen from another viewpoint.
- :class:`AffineWarp` (``A``): rotation, scale, shear and shift; no perspective.
- :class:`TPSWarp`: a thin-plate spline through the matches, a smooth *non-rigid*
  warp that can bend (cushions, fabric, small 3D effects). ``stiffness`` trades
  exact fit through every match against smoothness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from visual_search.geometry.verify import verify
from visual_search.matching.base import Correspondences

Box = tuple[float, float, float, float]


def base_transform(query_box: Box, cand_box: Box) -> NDArray[np.float64]:
    """Return the 3x3 scale+shift that maps the candidate box onto the query box.

    Args:
        query_box: ``(x0, y0, x1, y1)`` in the query image.
        cand_box: ``(x0, y0, x1, y1)`` in the candidate image.

    Returns:
        Matrix mapping candidate pixels to query pixels.
    """
    qx0, qy0, qx1, qy1 = query_box
    cx0, cy0, cx1, cy1 = cand_box
    sx = (qx1 - qx0) / max(cx1 - cx0, 1e-6)
    sy = (qy1 - qy0) / max(cy1 - cy0, 1e-6)
    return np.array([[sx, 0, qx0 - sx * cx0], [0, sy, qy0 - sy * cy0], [0, 0, 1]], np.float64)


def _apply_h(m: NDArray[np.float64], pts: NDArray[np.floating]) -> NDArray[np.float64]:
    """Apply a 3x3 projective matrix to ``(N, 2)`` points."""
    p = np.c_[pts.astype(np.float64), np.ones(len(pts))] @ m.T
    return p[:, :2] / np.where(np.abs(p[:, 2:]) < 1e-12, 1e-12, p[:, 2:])


class Warp(ABC):
    """A fitted candidate -> query warp, evaluated backwards (query pixel -> candidate pixel)."""

    #: Short name for tables and titles.
    name: str = "warp"

    @abstractmethod
    def to_candidate(self, query_pts: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map query-frame points to candidate-image points.

        Args:
            query_pts: ``(N, 2)`` ``(x, y)`` in query pixels.

        Returns:
            ``(N, 2)`` candidate pixels.
        """

    @abstractmethod
    def to_query(self, cand_pts: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map candidate points into the query frame (forward direction).

        Args:
            cand_pts: ``(N, 2)`` candidate pixels.

        Returns:
            ``(N, 2)`` query pixels.
        """

    def sample_map(
        self,
        out_shape: tuple[int, int],
        origin: tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
        t: float = 1.0,
        base: NDArray[np.float64] | None = None,
        step: int = 1,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Build ``cv2.remap`` maps for a (partial) warp onto a query-frame canvas.

        The canvas pixel ``(u, v)`` shows query pixel ``origin + (u, v) / scale``.

        Args:
            out_shape: Canvas ``(height, width)``.
            origin: Query-pixel position of the canvas's top-left corner.
            scale: Canvas pixels per query pixel.
            t: 0 = base placement only, 1 = full warp, in between = interpolated.
            base: Candidate -> query base transform (see :func:`base_transform`);
                ``None`` uses the full warp at every ``t``.
            step: Evaluate the warp on every ``step``-th pixel and upsample (speed).

        Returns:
            ``(map_x, map_y)`` float32 arrays of shape ``out_shape``.
        """
        h, w = out_shape
        gh = h if step <= 1 else max(2, -(-(h - 1) // step) + 1)
        gw = w if step <= 1 else max(2, -(-(w - 1) // step) + 1)
        us = np.linspace(0, w - 1, gw)
        vs = np.linspace(0, h - 1, gh)
        uu, vv = np.meshgrid(us, vs)
        q = np.c_[uu.ravel() / scale + origin[0], vv.ravel() / scale + origin[1]]
        full = self.to_candidate(q)
        if base is not None and t < 1.0:
            start = _apply_h(np.linalg.inv(base), q)
            src = (1 - t) * start + t * full
        else:
            src = full
        mx = src[:, 0].reshape(gh, gw).astype(np.float32)
        my = src[:, 1].reshape(gh, gw).astype(np.float32)
        if (gh, gw) != (h, w):
            # Interpolate exactly between grid nodes (cv2.resize would shift by up to half a cell).
            cu = np.arange(w, dtype=np.float32) * (gw - 1) / max(w - 1, 1)
            cv = np.arange(h, dtype=np.float32) * (gh - 1) / max(h - 1, 1)
            gu, gv = np.meshgrid(cu, cv)
            mx = cv2.remap(mx, gu, gv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            my = cv2.remap(my, gu, gv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return mx, my


class HomographyWarp(Warp):
    """Projective warp (3x3 matrix mapping candidate -> query)."""

    name = "H"

    def __init__(self, matrix: NDArray[np.float64]) -> None:
        """Store the candidate -> query matrix and its inverse."""
        self.matrix = np.asarray(matrix, np.float64)
        self.inverse = np.linalg.inv(self.matrix)

    def to_candidate(self, query_pts: NDArray[np.floating]) -> NDArray[np.float64]:
        """Query -> candidate via the inverse matrix."""
        return _apply_h(self.inverse, query_pts)

    def to_query(self, cand_pts: NDArray[np.floating]) -> NDArray[np.float64]:
        """Candidate -> query via the matrix."""
        return _apply_h(self.matrix, cand_pts)


class AffineWarp(HomographyWarp):
    """Affine warp (a homography with last row ``[0, 0, 1]``)."""

    name = "A"


class TPSWarp(Warp):
    """Thin-plate spline through point pairs, fitted in both directions.

    Coordinates are normalized internally for numerical stability. With
    ``stiffness = 0`` the spline passes exactly through every pair; larger values
    give a smoother warp that tolerates noisy matches.
    """

    name = "TPS"

    def __init__(self, query_pts: NDArray[np.floating], cand_pts: NDArray[np.floating], stiffness: float = 0.0) -> None:
        """Fit query -> candidate (for sampling) and candidate -> query (for points).

        Args:
            query_pts: ``(N, 2)`` control points in the query (N >= 3, not collinear).
            cand_pts: ``(N, 2)`` corresponding candidate points.
            stiffness: Regularization (relative to the point spread); 0 = interpolate exactly.

        Raises:
            ValueError: With fewer than 3 control points.
        """
        if len(query_pts) < 3:
            raise ValueError("TPS needs at least 3 point pairs.")
        self.stiffness = stiffness
        self._backward = _TPS(np.asarray(query_pts, np.float64), np.asarray(cand_pts, np.float64), stiffness)
        self._forward = _TPS(np.asarray(cand_pts, np.float64), np.asarray(query_pts, np.float64), stiffness)
        self.n_points = len(query_pts)

    def to_candidate(self, query_pts: NDArray[np.floating]) -> NDArray[np.float64]:
        """Query -> candidate via the backward spline."""
        return self._backward(np.asarray(query_pts, np.float64))

    def to_query(self, cand_pts: NDArray[np.floating]) -> NDArray[np.float64]:
        """Candidate -> query via the forward spline."""
        return self._forward(np.asarray(cand_pts, np.float64))


class _TPS:
    """2-D thin-plate spline ``src -> dst`` with kernel ``r^2 log r``."""

    def __init__(self, src: NDArray[np.float64], dst: NDArray[np.float64], stiffness: float) -> None:
        """Solve the TPS system in normalized coordinates."""
        self.mean = src.mean(axis=0)
        self.std = float(src.std()) or 1.0
        self.dmean = dst.mean(axis=0)
        self.dstd = float(dst.std()) or 1.0
        s = (src - self.mean) / self.std
        d = (dst - self.dmean) / self.dstd
        n = len(s)
        k = _kernel(s, s) + stiffness * np.eye(n)
        p = np.c_[np.ones(n), s]
        a = np.zeros((n + 3, n + 3))
        a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
        b = np.zeros((n + 3, 2))
        b[:n] = d
        self.ctrl = s
        self.coef = np.linalg.lstsq(a, b, rcond=None)[0]

    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate at ``(M, 2)`` points (in chunks to bound memory)."""
        out = np.empty((len(pts), 2))
        n = len(self.ctrl)
        for start in range(0, len(pts), 20000):
            q = (pts[start : start + 20000] - self.mean) / self.std
            u = _kernel(q, self.ctrl)
            out[start : start + 20000] = u @ self.coef[:n] + np.c_[np.ones(len(q)), q] @ self.coef[n:]
        return out * self.dstd + self.dmean


def _kernel(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """TPS radial basis ``r^2 log r`` between all rows of ``a`` and ``b``."""
    r2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 0.5 * r2 * np.log(r2)
    return np.nan_to_num(k)


@dataclass
class FittedWarps:
    """All warp models fitted on one query/candidate pair.

    Attributes:
        warps: Model name (``"H"``, ``"A"``, ``"TPS"``) -> fitted warp (missing if the fit failed).
        inliers: Model name -> ``(query_pts, cand_pts)`` used for or found by that fit.
        notes: Model name -> why a fit is missing or what it used.
    """

    warps: dict[str, Warp]
    inliers: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]]
    notes: dict[str, str]


def fit_warps(
    corr: Correspondences,
    threshold: float = 4.0,
    tps_stiffness: float = 0.05,
    tps_max_points: int = 200,
    seed: int = 0,
) -> FittedWarps:
    """Fit homography, affine and TPS warps from tentative matches.

    ``H`` and ``A`` come from robust fitting (MAGSAC++). The TPS needs clean
    control points but must not be limited to a plane, so it uses the inliers of a
    robust **fundamental-matrix** fit (consistent with a rigid 3D scene, not
    necessarily planar). If that fails, it falls back to the homography inliers.
    At most ``tps_max_points`` control points are kept (spatially spread at random).

    Args:
        corr: Tentative correspondences (query <-> candidate).
        threshold: Inlier threshold in pixels.
        tps_stiffness: TPS regularization (0 = exact through all control points).
        tps_max_points: Maximum TPS control points.
        seed: Seed for subsampling control points.

    Returns:
        The fitted warps, their inliers and notes.
    """
    warps: dict[str, Warp] = {}
    inliers: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
    notes: dict[str, str] = {}
    for model_type, cls in (("H", HomographyWarp), ("A", AffineWarp)):
        v = verify(corr, model_type, "magsac", threshold)
        if v.ok and v.model is not None:
            warps[model_type] = cls(v.model)
            inliers[model_type] = (corr.pts_q[v.inliers], corr.pts_c[v.inliers])
            notes[model_type] = f"{v.n_inliers} inliers"
        else:
            notes[model_type] = f"fit failed ({len(corr)} matches)"
    v = verify(corr, "F", "magsac", threshold)
    source = "F"
    if not (v.ok and v.n_inliers >= 8):
        v = verify(corr, "H", "magsac", threshold)
        source = "H"
    if v.ok and v.n_inliers >= 4:
        q, c = corr.pts_q[v.inliers], corr.pts_c[v.inliers]
        if len(q) > tps_max_points:
            keep = np.random.default_rng(seed).choice(len(q), tps_max_points, replace=False)
            q, c = q[keep], c[keep]
        try:
            warps["TPS"] = TPSWarp(q, c, tps_stiffness)
            inliers["TPS"] = (q, c)
            notes["TPS"] = f"{len(q)} control points ({source} inliers), stiffness {tps_stiffness:g}"
        except (ValueError, np.linalg.LinAlgError) as exc:
            notes["TPS"] = f"fit failed: {exc}"
    else:
        notes["TPS"] = f"fit failed ({len(corr)} matches)"
    return FittedWarps(warps, inliers, notes)
