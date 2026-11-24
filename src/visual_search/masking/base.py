"""Masker interface: turn a visual prompt into an object mask, plus simple fallbacks.

Downstream code (retrieval pooling, keypoint filtering, warping) consumes masks
through the ordinary prompt schema: :func:`with_mask` replaces an item's prompt by
``Prompt(mask=...)``, so masked and unmasked runs share one code path.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from visual_search.data.schema import Item, Prompt
from visual_search.imaging import prompt_box


@dataclass
class MaskResult:
    """A predicted object mask.

    Attributes:
        mask: ``(H, W)`` bool, True on the object, in original-image pixels.
        score: Confidence in [0, 1] if the masker provides one.
        source: Which masker produced it (``"sam3"``, ``"box"``, ``"none"``, ``"prompt"``).
        alternatives: Other candidate masks with scores, best first (for UIs that let
            the user pick another instance). May be empty.
        timings: Seconds per internal step, e.g. ``{"encode": 0.4, "decode": 0.1}``.
    """

    mask: NDArray[np.bool_]
    score: float | None = None
    source: str = ""
    alternatives: list[tuple[NDArray[np.bool_], float]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class Masker(ABC):
    """Predict an object mask from an image and a visual prompt."""

    #: Short identifier used in reports.
    name: str = "masker"

    def mask(self, image: NDArray[np.uint8], prompt: Prompt | None, image_key: str | None = None) -> MaskResult:
        """Return the object mask for ``prompt`` on ``image``.

        A mask prompt is returned as-is (the user already gave the mask). For other
        prompts the subclass's :meth:`_predict` is called.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.
            prompt: Box / points / mask prompt, or ``None`` (whole image).
            image_key: Optional stable id of the image content (e.g. a fingerprint)
                that maskers may use to cache per-image work across prompt changes.

        Returns:
            The mask result.
        """
        if prompt is not None and prompt.mask is not None:
            if prompt.mask.shape != image.shape[:2]:
                raise ValueError(f"Mask prompt shape {prompt.mask.shape} != image shape {image.shape[:2]}.")
            return MaskResult(prompt.mask.astype(bool), 1.0, "prompt")
        return self._predict(image, prompt, image_key)

    @abstractmethod
    def _predict(self, image: NDArray[np.uint8], prompt: Prompt | None, image_key: str | None) -> MaskResult:
        """Predict a mask for a box/points prompt (or None)."""


class NoMasker(Masker):
    """Ablation: the whole image is the object."""

    name = "none"

    def _predict(self, image: NDArray[np.uint8], prompt: Prompt | None, image_key: str | None) -> MaskResult:
        """Return an all-True mask."""
        return MaskResult(np.ones(image.shape[:2], bool), None, self.name)


class BoxMasker(Masker):
    """Fallback: the (padded) prompt box is the object; points use the box around the positive clicks."""

    name = "box"

    def __init__(self, pad: float = 0.0, min_size: int = 32) -> None:
        """Create the masker.

        Args:
            pad: Relative padding of the box per side.
            min_size: Minimum box side in pixels (so a single click gives a region).
        """
        self.pad = pad
        self.min_size = min_size

    def _predict(self, image: NDArray[np.uint8], prompt: Prompt | None, image_key: str | None) -> MaskResult:
        """Return a rectangular mask (whole image without prompt)."""
        x0, y0, x1, y1 = prompt_box(prompt, image.shape, pad=self.pad, min_size=self.min_size)
        mask = np.zeros(image.shape[:2], bool)
        mask[y0:y1, x0:x1] = True
        return MaskResult(mask, None, self.name)


def with_mask(item: Item, result: MaskResult) -> Item:
    """Return a copy of ``item`` whose prompt is the predicted mask.

    The original prompt is kept in ``extra["source_prompt"]`` (for display and for
    stages that want the user's box or clicks).

    Args:
        item: The item that was masked.
        result: The masker output for that item.

    Returns:
        A new item with ``prompt=Prompt(mask=result.mask)``.
    """
    extra = {**item.extra, "source_prompt": item.prompt, "mask_source": result.source}
    return dataclasses.replace(item, prompt=Prompt(mask=result.mask), extra=extra)
