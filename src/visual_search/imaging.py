"""Small image helpers shared by retrievers and matchers."""

from __future__ import annotations

import cv2
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


def prompt_patch_mask(
    prompt: Prompt | None, image_hw: tuple[int, int], grid_hw: tuple[int, int]
) -> NDArray[np.bool_]:
    """Map a prompt onto a ViT patch grid that covers the whole (resized) image.

    The image is assumed to be resized to the grid without cropping, so patch
    ``(i, j)`` covers the image region ``[j*W/gw, (j+1)*W/gw) x [i*H/gh, (i+1)*H/gh)``.

    - No prompt: all patches.
    - Box / points: patches whose center lies inside the prompt box (points use
      the box around the positive points, at least one patch wide).
    - Mask: patches whose area is at least half covered by the mask.

    If no patch qualifies (tiny prompt), the patch at the prompt center is used, so
    the result is never empty.

    Args:
        prompt: The visual prompt, or ``None``.
        image_hw: Original image ``(H, W)``.
        grid_hw: Patch grid ``(gh, gw)``.

    Returns:
        ``(gh, gw)`` bool array.
    """
    gh, gw = grid_hw
    h, w = image_hw
    if prompt is None:
        return np.ones((gh, gw), bool)
    if prompt.mask is not None:
        frac = cv2.resize(prompt.mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        sel = frac >= 0.5
        if not sel.any():
            sel = frac == frac.max()
        return sel
    min_size = int(np.ceil(max(h / gh, w / gw)))
    x0, y0, x1, y1 = prompt_box(prompt, (h, w), pad=0.0, min_size=min_size)
    cx = (np.arange(gw) + 0.5) * w / gw
    cy = (np.arange(gh) + 0.5) * h / gh
    sel = ((cy >= y0) & (cy < y1))[:, None] & ((cx >= x0) & (cx < x1))[None, :]
    if not sel.any():
        i = min(int((y0 + y1) / 2 * gh / h), gh - 1)
        j = min(int((x0 + x1) / 2 * gw / w), gw - 1)
        sel[i, j] = True
    return sel


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
