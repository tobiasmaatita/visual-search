"""Plain overlays of masks and prompts on images (no styling)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from visual_search.data.schema import Prompt


def overlay_mask(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    color: tuple[int, int, int] = (0, 160, 255),
    alpha: float = 0.45,
    contour: bool = True,
) -> NDArray[np.uint8]:
    """Blend a colored mask onto an RGB image.

    Args:
        image: ``(H, W, 3)`` uint8 RGB.
        mask: ``(H, W)`` bool.
        color: RGB fill color.
        alpha: Fill opacity in [0, 1].
        contour: Also draw the mask outline.

    Returns:
        A new ``(H, W, 3)`` uint8 image.
    """
    out = image.copy()
    fill = np.array(color, np.float32)
    out[mask] = (out[mask].astype(np.float32) * (1 - alpha) + fill * alpha).astype(np.uint8)
    if contour:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, max(1, min(image.shape[:2]) // 300))
    return out


def draw_prompt(image: NDArray[np.uint8], prompt: Prompt | None) -> NDArray[np.uint8]:
    """Draw a box prompt (yellow) or click prompts (green = positive, red = negative).

    Args:
        image: ``(H, W, 3)`` uint8 RGB.
        prompt: The prompt to draw; mask prompts and ``None`` are not drawn.

    Returns:
        A new image.
    """
    out = image.copy()
    t = max(1, min(image.shape[:2]) // 250)
    if prompt is None:
        return out
    if prompt.box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in prompt.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 220, 0), t)
    elif prompt.points is not None and prompt.point_labels is not None:
        for (x, y), lbl in zip(prompt.points, prompt.point_labels):
            color = (0, 220, 0) if lbl == 1 else (230, 0, 0)
            cv2.circle(out, (int(round(x)), int(round(y))), 3 * t + 2, color, -1)
            cv2.circle(out, (int(round(x)), int(round(y))), 3 * t + 2, (255, 255, 255), max(1, t // 2))
    return out
