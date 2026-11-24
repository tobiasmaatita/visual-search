"""Small image helpers shared by retrievers and matchers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from visual_search.data.schema import Prompt


def prompt_box(
    prompt: Prompt | None, image_shape: tuple[int, ...], pad: float = 0.0, min_size: int = 16
) -> tuple[int, int, int, int]:
    """Return the integer pixel box ``(x0, y0, x1, y1)`` to crop for a prompt.

    The prompt's enclosing box is padded by ``pad`` times its size on each side,
    grown to at least ``min_size`` pixels (so single clicks give a usable region),
    and clipped to the image. Without a prompt the whole image is returned.

    Args:
        prompt: The visual prompt, or ``None``.
        image_shape: ``(H, W, ...)`` of the image.
        pad: Relative padding per side, e.g. 0.1 = 10 % of width/height.
        min_size: Minimum side length in pixels before clipping.

    Returns:
        ``(x0, y0, x1, y1)`` with ``0 <= x0 < x1 <= W`` and ``0 <= y0 < y1 <= H``.
    """
    h, w = image_shape[:2]
    if prompt is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = prompt.bbox()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw = max((x1 - x0) * (1 + 2 * pad), min_size)
    bh = max((y1 - y0) * (1 + 2 * pad), min_size)
    ix0, iy0 = int(np.floor(max(cx - bw / 2, 0))), int(np.floor(max(cy - bh / 2, 0)))
    ix1, iy1 = int(np.ceil(min(cx + bw / 2, w))), int(np.ceil(min(cy + bh / 2, h)))
    return ix0, iy0, max(ix1, ix0 + 1), max(iy1, iy0 + 1)


def crop_to_prompt(image: NDArray[np.uint8], prompt: Prompt | None, pad: float = 0.0) -> NDArray[np.uint8]:
    """Crop an image to the (padded) box around a prompt; no prompt keeps the full image.

    Args:
        image: ``(H, W, C)`` image.
        prompt: The visual prompt, or ``None``.
        pad: Relative padding per side (see :func:`prompt_box`).

    Returns:
        A view into ``image`` covering the crop.
    """
    x0, y0, x1, y1 = prompt_box(prompt, image.shape, pad)
    return image[y0:y1, x0:x1]
