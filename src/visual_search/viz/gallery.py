"""Static result gallery: the query plus the top-k candidates after each cascade stage."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from visual_search.retrieval.base import Candidate

CORRECT_COLOR = "#1a9e1a"
WRONG_COLOR = "#d01c1c"


def thumbnail(image: NDArray[np.uint8], max_side: int = 256) -> NDArray[np.uint8]:
    """Downscale an image so its longer side is at most ``max_side`` (never upscales).

    Args:
        image: ``(H, W, 3)`` uint8.
        max_side: Maximum side length in pixels.

    Returns:
        The (possibly) resized image.
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def box_thumbnail(
    image: NDArray[np.uint8], box: tuple[float, float, float, float], pad: float = 0.3, max_side: int = 256
) -> NDArray[np.uint8]:
    """Crop around a box (with context), draw the box, and downscale.

    Small objects in large photos stay recognizable, while the context shows
    where they are.

    Args:
        image: ``(H, W, 3)`` uint8 RGB.
        box: ``(x0, y0, x1, y1)`` in pixels.
        pad: Context added on each side, as a fraction of the box size.
        max_side: Longer side of the result in pixels.

    Returns:
        The thumbnail.
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cx0, cy0 = int(max(0, x0 - pad * bw)), int(max(0, y0 - pad * bh))
    cx1, cy1 = int(min(w, x1 + pad * bw)), int(min(h, y1 + pad * bh))
    crop = np.ascontiguousarray(image[cy0 : max(cy1, cy0 + 1), cx0 : max(cx1, cx0 + 1)])
    t = max(2, round(max(crop.shape[:2]) / 120))
    cv2.rectangle(crop, (int(x0 - cx0), int(y0 - cy0)), (int(x1 - cx0), int(y1 - cy0)), (255, 220, 0), t)
    return thumbnail(crop, max_side)


def _feature_text(c: Candidate, max_features: int) -> str:
    """Format the main score features of a candidate (numbers only)."""
    parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(c.features.items())[:max_features]
             if isinstance(v, (int, float))]
    return "\n".join(parts)


def render_ranking_grid(
    query_image: NDArray[np.uint8],
    stages: Mapping[str, list[Candidate]],
    load_thumb: Callable[[Candidate], NDArray[np.uint8]],
    is_correct: Callable[[Candidate], bool | None],
    k: int = 5,
    label_of: Callable[[Candidate], str | None] | None = None,
    max_features: int = 2,
    title: str = "",
    cell_inches: float = 2.0,
    per_row: int = 8,
) -> bytes:
    """Render one row per stage: the query, then its top-``k`` candidates.

    Each candidate shows its rank, score and main features, and gets a green
    (correct) / red (wrong) frame when ground truth is known.

    Args:
        query_image: Query image to show at the start of each row (e.g. with mask overlay).
        stages: Stage name -> ranking (best first), e.g. ``QueryResult.stages``.
        load_thumb: Candidate -> RGB thumbnail (the candidate's ``region`` may matter).
        is_correct: Candidate -> True/False, or None when ground truth is unknown.
        k: Candidates shown per stage.
        label_of: Optional candidate -> label, shown under the thumbnail.
        max_features: Number of score features printed per candidate.
        title: Figure title (e.g. timings).
        cell_inches: Size of one cell.
        per_row: Candidates per grid row; a stage with more than this wraps onto extra rows.

    Returns:
        PNG bytes (render with ``ipywidgets.Image`` or ``IPython.display.Image``).
    """
    names = list(stages) or [""]
    cols = min(k, per_row)
    rows_per_stage = max(1, -(-k // per_row))
    n_rows, n_cols = len(names) * rows_per_stage, cols + 1
    fig = Figure(figsize=(cell_inches * n_cols, (cell_inches + 0.9) * n_rows))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    query_thumb = thumbnail(query_image)
    for s_idx, name in enumerate(names):
        ranking = stages.get(name, []) if name else []
        for sub in range(rows_per_stage):
            r = s_idx * rows_per_stage + sub
            ax = axes[r, 0]
            ax.set_axis_off()
            if sub == 0:
                ax.imshow(query_thumb)
                ax.set_title(f"query\n[{name}]", fontsize=9)
            for c in range(1, n_cols):
                ax = axes[r, c]
                ax.set_axis_off()
                rank = sub * cols + c - 1
                if rank >= min(k, len(ranking)):
                    continue
                cand = ranking[rank]
                thumb = load_thumb(cand)
                ax.imshow(thumb)
                correct = is_correct(cand)
                mark = "" if correct is None else (" ✓" if correct else " ✗")
                label = label_of(cand) if label_of else None
                lines = [f"#{rank + 1} {cand.score:.3f}{mark}", label or "", _feature_text(cand, max_features)]
                ax.set_title("\n".join(x for x in lines if x), fontsize=8,
                             color="black" if correct is None else (CORRECT_COLOR if correct else WRONG_COLOR))
                if correct is not None:
                    h, w = thumb.shape[:2]
                    ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, linewidth=4,
                                           edgecolor=CORRECT_COLOR if correct else WRONG_COLOR))
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    return buf.getvalue()
