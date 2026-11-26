"""Animation frames that show a candidate being warped onto the query, plus GIF export.

Every frame is drawn on a canvas around the **query box** (with some context),
in query coordinates. Views:

- ``"morph"``: the candidate deforms from its base placement (its box simply
  resized into the query box) to the fully warped state, blended over the query;
  inlier matches travel from their candidate position to their query position.
- ``"grid"``: a grid drawn on the candidate bends along with the warp, showing
  where and how strongly it deforms.
- ``"flicker"``: blink between the query and the fully warped candidate; for the
  same item details stay put, for a different copy they jump.
- ``"residual"``: the difference between query and (partially) warped candidate
  inside the query box, as a heatmap that should fade as the warp completes.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from visual_search.warp.models import Box, FittedWarps, Warp, _apply_h, base_transform

VIEWS = ("morph", "grid", "flicker", "residual")
QUERY_BOX_COLOR = (255, 220, 0)
DOT_COLOR = (40, 230, 40)


@dataclass
class WarpPair:
    """A query/candidate pair and the canvas it is animated on.

    Attributes:
        query_image: ``(H, W, 3)`` uint8 RGB.
        query_box: The query object ``(x0, y0, x1, y1)``.
        cand_image: Candidate RGB image.
        cand_box: The candidate object (its box or found region; the whole image if unknown).
        canvas_side: Longer side of the animation canvas in pixels.
        pad: Context around the query box, as a fraction of its size.
    """

    query_image: NDArray[np.uint8]
    query_box: Box
    cand_image: NDArray[np.uint8]
    cand_box: Box
    canvas_side: int = 480
    pad: float = 0.3

    def __post_init__(self) -> None:
        """Compute the canvas placement (origin and scale in query pixels)."""
        h, w = self.query_image.shape[:2]
        x0, y0, x1, y1 = self.query_box
        bw, bh = x1 - x0, y1 - y0
        ox0, oy0 = max(0.0, x0 - self.pad * bw), max(0.0, y0 - self.pad * bh)
        ox1, oy1 = min(float(w), x1 + self.pad * bw), min(float(h), y1 + self.pad * bh)
        self.origin = (ox0, oy0)
        self.scale = self.canvas_side / max(ox1 - ox0, oy1 - oy0, 1.0)
        self.shape = (max(1, round((oy1 - oy0) * self.scale)), max(1, round((ox1 - ox0) * self.scale)))
        self.base = base_transform(self.query_box, self.cand_box)
        m = np.array([[self.scale, 0, -ox0 * self.scale], [0, self.scale, -oy0 * self.scale]], np.float64)
        self.query_canvas = cv2.warpAffine(self.query_image, m, (self.shape[1], self.shape[0]), flags=cv2.INTER_AREA)
        self._box_mask = np.zeros(self.shape, bool)
        bx0, by0 = self.to_canvas(np.array([[x0, y0]]))[0]
        bx1, by1 = self.to_canvas(np.array([[x1, y1]]))[0]
        self.canvas_box = (int(bx0), int(by0), int(bx1), int(by1))
        self._box_mask[max(0, int(by0)) : int(by1), max(0, int(bx0)) : int(bx1)] = True

    def to_canvas(self, query_pts: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map query pixels to canvas pixels."""
        return (np.asarray(query_pts, np.float64) - np.array(self.origin)) * self.scale

    def warped(self, warp: Warp, t: float, image: NDArray[np.uint8] | None = None) -> tuple[NDArray, NDArray]:
        """Sample the (partially) warped candidate on the canvas.

        Args:
            warp: The fitted warp.
            t: 0 = base placement, 1 = full warp.
            image: Candidate-sized image to warp instead of the candidate (e.g. with a grid drawn on it).

        Returns:
            ``(warped_rgb, valid_mask)`` on the canvas.
        """
        mx, my = warp.sample_map(self.shape, self.origin, self.scale, t, self.base, step=8)
        src = self.cand_image if image is None else image
        out = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ch, cw = self.cand_image.shape[:2]
        valid = (mx >= 0) & (my >= 0) & (mx <= cw - 1) & (my <= ch - 1)
        return out, valid


def _blend(base: NDArray[np.uint8], top: NDArray[np.uint8], mask: NDArray[np.bool_], alpha: float) -> NDArray[np.uint8]:
    """Blend ``top`` over ``base`` inside ``mask``."""
    out = base.copy()
    out[mask] = (base[mask].astype(np.float32) * (1 - alpha) + top[mask].astype(np.float32) * alpha).astype(np.uint8)
    return out


def _label(img: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    """Write a small caption in the top-left corner."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (8 + 7 * len(text), 20), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_query_box(img: NDArray[np.uint8], pair: WarpPair) -> NDArray[np.uint8]:
    """Outline the query box on a canvas image."""
    x0, y0, x1, y1 = pair.canvas_box
    cv2.rectangle(img, (x0, y0), (x1, y1), QUERY_BOX_COLOR, 2)
    return img


def _grid_image(image: NDArray[np.uint8], box: Box, cells: int = 10) -> NDArray[np.uint8]:
    """Return a copy of ``image`` with a regular grid drawn over the object box."""
    out = image.copy()
    x0, y0, x1, y1 = box
    t = max(1, round(max(x1 - x0, y1 - y0) / 250))
    for i in range(cells + 1):
        x = int(x0 + i * (x1 - x0) / cells)
        y = int(y0 + i * (y1 - y0) / cells)
        cv2.line(out, (x, int(y0)), (x, int(y1)), (255, 0, 200), t)
        cv2.line(out, (int(x0), y), (int(x1), y), (255, 0, 200), t)
    return out


def _residual_overlay(pair: WarpPair, warped: NDArray[np.uint8], valid: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Heatmap of |query - warped| (grayscale) inside the query box, over a dimmed query."""
    q = cv2.cvtColor(pair.query_canvas, cv2.COLOR_RGB2GRAY).astype(np.float32)
    w = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = np.clip(np.abs(q - w) * 2, 0, 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    dim = (pair.query_canvas * 0.35).astype(np.uint8)
    return _blend(dim, heat, valid & pair._box_mask, 0.9)


def warp_frames(
    pair: WarpPair,
    warp: Warp,
    view: str = "morph",
    n_frames: int = 24,
    hold: int = 8,
    inliers: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
    alpha: float = 0.6,
) -> list[NDArray[np.uint8]]:
    """Render the animation frames of one warp in one view.

    Args:
        pair: The pair and its canvas.
        warp: The fitted candidate -> query warp.
        view: ``"morph"``, ``"grid"``, ``"flicker"`` or ``"residual"``.
        n_frames: Frames from ``t = 0`` to ``t = 1`` (flicker: number of blinks x 2).
        hold: Extra copies of the last frame, so the end state stays visible in a loop.
        inliers: ``(query_pts, cand_pts)`` to draw as travelling dots (morph view).
        alpha: Opacity of the candidate over the query.

    Returns:
        RGB frames, all of the canvas size.

    Raises:
        ValueError: For an unknown view.
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}.")
    frames: list[NDArray[np.uint8]] = []
    if view == "flicker":
        full, valid = pair.warped(warp, 1.0)
        b = _draw_query_box(pair.query_canvas.copy(), pair)
        a = _draw_query_box(_blend(pair.query_canvas, full, valid, 1.0), pair)
        for i in range(max(2, n_frames // 3)):
            frame = _label(b, f"{warp.name} | query") if i % 2 == 0 else _label(a, f"{warp.name} | warped candidate")
            frames += [frame] * 3
        return frames
    grid_src = _grid_image(pair.cand_image, pair.cand_box) if view == "grid" else None
    dots_start = dots_end = None
    if inliers is not None and view == "morph" and len(inliers[0]):
        dots_start = pair.to_canvas(_apply_h(pair.base, inliers[1]))
        dots_end = pair.to_canvas(inliers[0])
    for t in np.linspace(0.0, 1.0, n_frames):
        if view == "grid":
            warped, valid = pair.warped(warp, float(t), grid_src)
            frame = _blend((pair.query_canvas * 0.6).astype(np.uint8), warped, valid, 0.85)
        elif view == "residual":
            warped, valid = pair.warped(warp, float(t))
            frame = _residual_overlay(pair, warped, valid)
        else:
            warped, valid = pair.warped(warp, float(t))
            frame = _blend(pair.query_canvas, warped, valid, alpha)
            if dots_start is not None:
                for x, y in (1 - t) * dots_start + t * dots_end:
                    cv2.circle(frame, (int(x), int(y)), 3, DOT_COLOR, -1, cv2.LINE_AA)
        frames.append(_label(_draw_query_box(frame, pair), f"{warp.name} | {view} | t={t:.2f}"))
    return frames + [frames[-1]] * hold


def compare_frames(
    pair: WarpPair, fitted: FittedWarps, view: str = "morph", n_frames: int = 24, hold: int = 8
) -> list[NDArray[np.uint8]]:
    """Animate every fitted warp model side by side (same view, same timing).

    Args:
        pair: The pair and its canvas.
        fitted: Fitted warps (``fit_warps`` result).
        view: Animation view.
        n_frames: Frames from ``t = 0`` to ``t = 1``.
        hold: Copies of the last frame.

    Returns:
        Frames with one panel per model, left to right in the order ``H``, ``A``, ``TPS``.
    """
    names = [n for n in ("H", "A", "TPS") if n in fitted.warps]
    if not names:
        return []
    per_model = [warp_frames(pair, fitted.warps[n], view, n_frames, hold, fitted.inliers.get(n)) for n in names]
    gap = np.full((pair.shape[0], 6, 3), 255, np.uint8)
    out = []
    for i in range(min(len(f) for f in per_model)):
        parts = []
        for f in per_model:
            parts += [f[i], gap]
        out.append(np.hstack(parts[:-1]))
    return out


def side_by_side(pair: WarpPair, height: int = 320) -> NDArray[np.uint8]:
    """Query crop (with its box) next to the candidate crop (with its box or region), for a first look.

    Args:
        pair: The pair.
        height: Output height in pixels.

    Returns:
        One RGB image.
    """
    from visual_search.viz.gallery import box_thumbnail

    q = box_thumbnail(pair.query_image, pair.query_box, pad=pair.pad, max_side=2000)
    c = box_thumbnail(pair.cand_image, pair.cand_box, pad=pair.pad, max_side=2000)
    q = cv2.resize(q, (max(1, round(q.shape[1] * height / q.shape[0])), height), interpolation=cv2.INTER_AREA)
    c = cv2.resize(c, (max(1, round(c.shape[1] * height / c.shape[0])), height), interpolation=cv2.INTER_AREA)
    gap = np.full((height, 10, 3), 255, np.uint8)
    return np.hstack([_label(q, "query"), gap, _label(c, "candidate")])


def warp_stats(pair: WarpPair, fitted: FittedWarps) -> pd.DataFrame:
    """Numbers per warp model for the pair, as early indicators of a good or bad alignment.

    Columns:
        - ``points``: inliers (``H``, ``A``) or control points (``TPS``) used.
        - ``reproj_px``: median distance between warped candidate points and their
          query points (lower = tighter fit; TPS is near 0 by construction).
        - ``strain``: spread of the local area change of the warp inside the query box
          (std of log local scale); 0 = rigid, large = strong local stretching, often a
          sign of a wrong match.
        - ``residual``: mean absolute grayscale difference (0-1) between query and fully
          warped candidate inside the query box.

    Args:
        pair: The pair.
        fitted: Fitted warps.

    Returns:
        One row per fitted model, plus a ``note`` column.
    """
    rows = {}
    for name, warp in fitted.warps.items():
        q, c = fitted.inliers[name]
        reproj = float(np.median(np.linalg.norm(warp.to_query(c) - q, axis=1))) if len(q) else float("nan")
        mx, my = warp.sample_map(pair.shape, pair.origin, pair.scale, 1.0, step=4)
        dxu, dxv = np.gradient(mx)[1], np.gradient(mx)[0]
        dyu, dyv = np.gradient(my)[1], np.gradient(my)[0]
        det = np.abs(dxu * dyv - dxv * dyu)
        inside = pair._box_mask & (det > 1e-9)
        strain = float(np.std(np.log(det[inside]))) if inside.any() else float("nan")
        warped, valid = pair.warped(warp, 1.0)
        m = valid & pair._box_mask
        qg = cv2.cvtColor(pair.query_canvas, cv2.COLOR_RGB2GRAY).astype(np.float32)
        wg = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
        residual = float(np.abs(qg - wg)[m].mean() / 255) if m.any() else float("nan")
        rows[name] = {"points": len(q), "reproj_px": reproj, "strain": strain, "residual": residual,
                      "note": fitted.notes.get(name, "")}
    for name, note in fitted.notes.items():
        if name not in rows:
            rows[name] = {"points": 0, "reproj_px": np.nan, "strain": np.nan, "residual": np.nan, "note": note}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    return df


def encode_png(frame: NDArray[np.uint8]) -> bytes:
    """PNG bytes of an RGB frame (for ``ipywidgets.Image``)."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encoding failed.")
    return buf.tobytes()


def save_gif(frames: Sequence[NDArray[np.uint8]], path: str | Path, fps: float = 12.0) -> Path:
    """Write frames as a looping GIF.

    Args:
        frames: RGB frames of equal size.
        path: Output file (parent folders are created).
        fps: Frames per second.

    Returns:
        The written path.

    Raises:
        ValueError: Without frames.
    """
    import imageio.v3 as iio

    if not frames:
        raise ValueError("No frames to save.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), extension=".gif", duration=1000.0 / fps, loop=0)
    return path


def gif_bytes(frames: Sequence[NDArray[np.uint8]], fps: float = 12.0) -> bytes:
    """The GIF of ``frames`` as bytes (for ``IPython.display.Image``)."""
    import imageio.v3 as iio

    buf = io.BytesIO()
    iio.imwrite(buf, np.stack(frames), extension=".gif", duration=1000.0 / fps, loop=0)
    return buf.getvalue()


def all_views(pair: WarpPair, warp: Warp, inliers=None, n_frames: int = 24, hold: int = 8) -> dict[str, list]:
    """Frames for every view of one warp (handy for exporting all at once)."""
    return {v: warp_frames(pair, warp, v, n_frames, hold, inliers) for v in VIEWS}

