"""Judges: when does a returned candidate count as a correct match for a query?

A judge answers two questions:
- ``correct(query, item, candidate)``: is this returned candidate a correct match?
- ``has_positive(query, gallery)``: can the query be answered at all (is anything
  correct in the gallery)? Queries without any positive are skipped in metrics.

Levels:
- :class:`ItemJudge`: same instance label (the exact physical item).
- :class:`GroupJudge`: same ``group`` (e.g. product model); items without a group
  fall back to their label, so a group is never *required*.
- :class:`GroundedJudge`: for galleries of unlabelled whole images (grounding
  evaluation). A candidate is correct when its found region overlaps an annotated
  box with the query's label (or group) by IoU >= ``iou_threshold``; a candidate
  without a region (whole image) is correct when the image contains such a box.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from visual_search.data.schema import Item
from visual_search.retrieval.base import Candidate

Box = tuple[float, float, float, float]


class Judge(Protocol):
    """Decides correctness of candidates (see module docstring)."""

    def correct(self, query: Item, item: Item, candidate: Candidate) -> bool:
        """Return True if ``candidate`` (gallery ``item``) is a correct result for ``query``."""
        ...

    def has_positive(self, query: Item, gallery: Sequence[Item]) -> bool:
        """Return True if at least one correct result exists in ``gallery``."""
        ...


def group_key(item: Item) -> str | None:
    """Return an item's group, or its label when it has no group (None for distractors)."""
    if item.label is None:
        return None
    return item.group or item.label


class ItemJudge:
    """Correct = same non-None instance label."""

    def correct(self, query: Item, item: Item, candidate: Candidate) -> bool:
        """Return True if the labels are equal and not None."""
        return query.label is not None and query.label == item.label

    def has_positive(self, query: Item, gallery: Sequence[Item]) -> bool:
        """Return True if some gallery item has the query's label."""
        return query.label is not None and any(g.label == query.label for g in gallery)


class GroupJudge:
    """Correct = same group (model); an item without a group stands for its own label."""

    def correct(self, query: Item, item: Item, candidate: Candidate) -> bool:
        """Return True if the group keys are equal and not None."""
        key = group_key(query)
        return key is not None and key == group_key(item)

    def has_positive(self, query: Item, gallery: Sequence[Item]) -> bool:
        """Return True if some gallery item has the query's group key."""
        key = group_key(query)
        return key is not None and any(group_key(g) == key for g in gallery)


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class GroundedJudge:
    """Correct = the found region overlaps an annotated box of the right label (or group).

    Args:
        boxes_by_image: Image id -> annotated objects ``(box, label, group)`` in that image.
            Gallery items are looked up by their ``extra["image_id"]`` (or their id).
        iou_threshold: Minimum IoU between the candidate's region and the box.
        level: ``"item"`` (label must match) or ``"model"`` (group key must match).
    """

    def __init__(
        self,
        boxes_by_image: Mapping[str, Sequence[tuple[Box, str | None, str | None]]],
        iou_threshold: float = 0.3,
        level: str = "item",
    ) -> None:
        """Store the annotations, the threshold and the level."""
        if level not in ("item", "model"):
            raise ValueError("level must be 'item' or 'model'.")
        self.boxes_by_image = boxes_by_image
        self.iou_threshold = iou_threshold
        self.level = level

    def _key(self, label: str | None, group: str | None) -> str | None:
        """Identity of an annotated object at this judge's level."""
        if label is None:
            return None
        return (group or label) if self.level == "model" else label

    def _query_key(self, query: Item) -> str | None:
        """Identity of the query at this judge's level."""
        return group_key(query) if self.level == "model" else query.label

    def _image_id(self, item: Item) -> str:
        """Image id of a gallery item."""
        return item.extra.get("image_id", item.item_id)

    def matching_boxes(self, query: Item, item: Item) -> list[Box]:
        """Return the annotated boxes in ``item``'s image that belong to the query's object."""
        key = self._query_key(query)
        if key is None:
            return []
        return [box for box, label, group in self.boxes_by_image.get(self._image_id(item), [])
                if self._key(label, group) == key]

    def correct(self, query: Item, item: Item, candidate: Candidate) -> bool:
        """Return True if the region overlaps a matching box enough; without a region, if the image has one."""
        boxes = self.matching_boxes(query, item)
        if not boxes:
            return False
        if candidate.region is None:
            return True
        return max(box_iou(candidate.region, b) for b in boxes) >= self.iou_threshold

    def has_positive(self, query: Item, gallery: Sequence[Item]) -> bool:
        """Return True if some gallery image contains a box of the query's object."""
        return any(self.matching_boxes(query, g) for g in gallery)
