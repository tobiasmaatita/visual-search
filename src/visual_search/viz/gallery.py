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


def _feature_text(c: Candidate, max_features: int) -> str:
    """Format the main score features of a candidate (numbers only)."""
    parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(c.features.items())[:max_features]
             if isinstance(v, (int, float))]
    return "\n".join(parts)


def render_ranking_grid(
    query_image: NDArray[np.uint8],
    stages: Mapping[str, list[Candidate]],
    load_thumb: Callable[[str], NDArray[np.uint8]],
    is_correct: Callable[[str], bool | None],
    k: int = 5,
    label_of: Callable[[str], str | None] | None = None,
    max_features: int = 2,
    title: str = "",
    cell_inches: float = 2.0,
) -> bytes:
    """Render one row per stage: the query, then its top-``k`` candidates.

    Each candidate shows its rank, score and main features, and gets a green
    (correct) / red (wrong) frame when ground truth is known.

    Args:
        query_image: Query image to show at the start of each row (e.g. with mask overlay).
        stages: Stage name -> ranking (best first), e.g. ``QueryResult.stages``.
        load_thumb: Item id -> RGB thumbnail.
        is_correct: Item id -> True/False, or None when ground truth is unknown.
        k: Candidates per row.
        label_of: Optional item id -> label, shown under the thumbnail.
        max_features: Number of score features printed per candidate.
        title: Figure title (e.g. timings).
        cell_inches: Size of one cell.

    Returns:
        PNG bytes (render with ``ipywidgets.Image`` or ``IPython.display.Image``).
    """
    names = list(stages)
    n_rows, n_cols = max(len(names), 1), k + 1
    fig = Figure(figsize=(cell_inches * n_cols, (cell_inches + 0.6) * n_rows))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    query_thumb = thumbnail(query_image)
    for r, name in enumerate(names or [""]):
        ax = axes[r, 0]
        ax.imshow(query_thumb)
        ax.set_title(f"query\n[{name}]", fontsize=9)
        ax.set_axis_off()
        ranking = stages.get(name, []) if names else []
        for c in range(1, n_cols):
            ax = axes[r, c]
            ax.set_axis_off()
            if c - 1 >= len(ranking):
                continue
            cand = ranking[c - 1]
            thumb = load_thumb(cand.item_id)
            ax.imshow(thumb)
            correct = is_correct(cand.item_id)
            mark = "" if correct is None else (" ✓" if correct else " ✗")
            label = label_of(cand.item_id) if label_of else None
            lines = [f"#{c} {cand.score:.3f}{mark}", label or "", _feature_text(cand, max_features)]
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
