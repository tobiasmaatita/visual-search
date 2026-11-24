"""Dataset schema: items, visual prompts and the ground-truth match rule.

The raw dataset is a list of dicts::

    {
        "input_image": PIL.Image | np.ndarray | str | Path,
        "label": str | None,              # same id => same instance; None => distractor
        "prompt": {"box": [x0, y0, x1, y1]}
                | {"points": [[x, y], ...], "point_labels": [1, 0, ...]}
                | {"mask": HxW array},     # optional
        ...                               # any other key is kept in Item.extra
    }

Unknown keys (e.g. a future ``"group"`` for hard negatives) are preserved in
``Item.extra`` so that schema extensions need no refactor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mlcvtools.image_ops import to_rgb_if_needed
from numpy.typing import NDArray
from PIL import Image, ImageOps

ImageSource = Image.Image | np.ndarray | str | Path
PromptKind = Literal["box", "points", "mask"]

KNOWN_KEYS = frozenset({"input_image", "label", "prompt", "id"})


@dataclass(frozen=True)
class Prompt:
    """A visual prompt on an image: a box, a set of labelled points, or a mask.

    Exactly one of ``box``, ``points`` or ``mask`` is set. All coordinates are in
    pixels of the original (unresized) image, ``x`` to the right and ``y`` down.

    Attributes:
        box: ``(x0, y0, x1, y1)`` with ``x0 < x1`` and ``y0 < y1``.
        points: ``(N, 2)`` float array of ``(x, y)`` click positions.
        point_labels: ``(N,)`` int array, 1 = positive (on the object),
            0 = negative (exclude this region). Set iff ``points`` is set.
        mask: ``(H, W)`` bool array, True on the object.

    Example:
        >>> Prompt.from_dict({"points": [[10, 20], [30, 40]], "point_labels": [1, 0]}).kind
        'points'
    """

    box: tuple[float, float, float, float] | None = None
    points: NDArray[np.float32] | None = None
    point_labels: NDArray[np.int64] | None = None
    mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        """Check that exactly one prompt type is given and that it is well formed."""
        n_set = sum(v is not None for v in (self.box, self.points, self.mask))
        if n_set != 1:
            raise ValueError(f"Prompt needs exactly one of box/points/mask, got {n_set}.")
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            if not (x0 < x1 and y0 < y1):
                raise ValueError(f"Box must satisfy x0 < x1 and y0 < y1, got {self.box}.")
        if self.points is not None:
            if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) == 0:
                raise ValueError(f"Points must have shape (N>0, 2), got {self.points.shape}.")
            if self.point_labels is None or self.point_labels.shape != (len(self.points),):
                raise ValueError("point_labels must have shape (N,) matching points.")
            if not np.isin(self.point_labels, (0, 1)).all():
                raise ValueError("point_labels must be 0 (negative) or 1 (positive).")
            if not (self.point_labels == 1).any():
                raise ValueError("At least one positive point (label 1) is required.")
        elif self.point_labels is not None:
            raise ValueError("point_labels given without points.")
        if self.mask is not None and (self.mask.ndim != 2 or not self.mask.any()):
            raise ValueError("Mask must be a non-empty 2-D array.")

    @property
    def kind(self) -> PromptKind:
        """Return which prompt type this is: ``"box"``, ``"points"`` or ``"mask"``."""
        if self.box is not None:
            return "box"
        if self.points is not None:
            return "points"
        return "mask"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Prompt:
        """Build a prompt from the dataset dict format.

        Args:
            d: One of ``{"box": [x0, y0, x1, y1]}``,
                ``{"points": [[x, y], ...], "point_labels": [...]}`` (labels optional,
                default all positive; ``[[x, y, label], ...]`` is also accepted), or
                ``{"mask": HxW array}``.

        Returns:
            The parsed prompt.

        Raises:
            ValueError: If the dict has no or several prompt keys, or invalid values.
        """
        keys = {"box", "points", "mask"} & set(d)
        if len(keys) != 1:
            raise ValueError(f"Prompt dict needs exactly one of box/points/mask, got keys {sorted(d)}.")
        if "box" in d:
            box = tuple(float(v) for v in d["box"])
            if len(box) != 4:
                raise ValueError(f"Box needs 4 values, got {d['box']}.")
            return cls(box=box)  # type: ignore[arg-type]
        if "points" in d:
            pts = np.asarray(d["points"], dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 3 and "point_labels" not in d:
                labels = pts[:, 2].astype(np.int64)
                pts = pts[:, :2]
            else:
                labels = np.asarray(d.get("point_labels", np.ones(len(pts))), dtype=np.int64)
            return cls(points=pts, point_labels=labels)
        return cls(mask=np.asarray(d["mask"]).astype(bool))

    def to_dict(self) -> dict[str, Any]:
        """Convert back to the dataset dict format (lists for box/points, array for mask).

        Returns:
            A dict accepted by :meth:`from_dict`.
        """
        if self.box is not None:
            return {"box": list(self.box)}
        if self.points is not None:
            assert self.point_labels is not None
            return {"points": self.points.tolist(), "point_labels": self.point_labels.tolist()}
        return {"mask": self.mask}

    def bbox(self) -> tuple[float, float, float, float]:
        """Return a bounding box ``(x0, y0, x1, y1)`` enclosing the prompt.

        For points only the positive points are used; the box can be degenerate
        (zero width/height) for a single point.

        Returns:
            The enclosing box in original-image pixels.
        """
        if self.box is not None:
            return self.box
        if self.points is not None:
            assert self.point_labels is not None
            pos = self.points[self.point_labels == 1]
            return (float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 0].max()), float(pos[:, 1].max()))
        assert self.mask is not None
        ys, xs = np.nonzero(self.mask)
        return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


@dataclass
class Item:
    """One dataset entry: an image, its instance label and an optional prompt.

    Attributes:
        item_id: Unique id within the dataset (from the ``"id"`` key or the list index).
        image: The image source; use :meth:`load_image` to get pixels.
        label: Instance id. Items with the same non-None label show the same instance.
            ``None`` marks a distractor, which never counts as a correct match.
        prompt: Optional visual prompt on the image.
        extra: All other keys of the raw dict, untouched (e.g. ``"group"``).
    """

    item_id: str
    image: ImageSource
    label: str | None = None
    prompt: Prompt | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_distractor(self) -> bool:
        """Return True if the item has no instance label."""
        return self.label is None

    @property
    def group(self) -> str | None:
        """Return the optional hard-negative group id (``extra["group"]``), if any."""
        return self.extra.get("group")

    def load_image(self) -> NDArray[np.uint8]:
        """Load the image as an RGB uint8 array.

        Paths are opened with PIL and EXIF orientation is applied, so phone photos
        come out upright. Grayscale is expanded to 3 channels and alpha is dropped.

        Returns:
            ``(H, W, 3)`` uint8 RGB array.

        Raises:
            ValueError: If an array source has an unsupported shape or dtype.
            FileNotFoundError: If a path source does not exist.
        """
        src = self.image
        if isinstance(src, (str, Path)):
            with Image.open(src) as im:
                src = ImageOps.exif_transpose(im).convert("RGB")
        if isinstance(src, Image.Image):
            return np.asarray(src.convert("RGB"), dtype=np.uint8)
        arr = np.asarray(src)
        if arr.dtype != np.uint8:
            raise ValueError(f"Array images must be uint8, got {arr.dtype}.")
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = to_rgb_if_needed(arr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported image shape {arr.shape}.")
        return np.ascontiguousarray(arr)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], index: int) -> Item:
        """Parse one raw dataset dict.

        Args:
            d: Raw entry with at least ``"input_image"``.
            index: Position in the dataset list; used for the id if ``"id"`` is absent.

        Returns:
            The parsed item.

        Raises:
            KeyError: If ``"input_image"`` is missing.
            ValueError: If the label is not a string/None or the prompt is invalid.
        """
        if "input_image" not in d:
            raise KeyError(f"Entry {index} has no 'input_image'.")
        label = d.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"Entry {index}: label must be str or None, got {type(label).__name__}.")
        prompt = d.get("prompt")
        return cls(
            item_id=str(d.get("id", f"item_{index:06d}")),
            image=d["input_image"],
            label=label,
            prompt=Prompt.from_dict(prompt) if prompt is not None else None,
            extra={k: v for k, v in d.items() if k not in KNOWN_KEYS},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert back to the raw dataset dict format.

        Returns:
            A dict accepted by :meth:`from_dict`, including all ``extra`` keys.
        """
        d: dict[str, Any] = {"id": self.item_id, "input_image": self.image, "label": self.label, **self.extra}
        if self.prompt is not None:
            d["prompt"] = self.prompt.to_dict()
        return d


def is_match(query: Item, candidate: Item) -> bool:
    """Return True iff ``candidate`` shows the same instance as ``query``.

    A prediction is correct iff both labels are equal and not None, so
    distractors (``label=None``) never match anything.

    Args:
        query: The query item.
        candidate: A gallery item.

    Returns:
        Whether the pair counts as a correct match.
    """
    return query.label is not None and query.label == candidate.label


def parse_dataset(records: Sequence[Mapping[str, Any]]) -> list[Item]:
    """Parse a raw dataset (list of dicts) into items.

    Args:
        records: The raw dataset.

    Returns:
        One :class:`Item` per record, in order.

    Raises:
        ValueError: If two records share an ``"id"``, or any record is invalid
            (see :meth:`Item.from_dict`).
    """
    items = [r if isinstance(r, Item) else Item.from_dict(r, i) for i, r in enumerate(records)]
    dupes = sorted(i for i, n in Counter(it.item_id for it in items).items() if n > 1)
    if dupes:
        raise ValueError(f"Duplicate item ids: {dupes[:10]}")
    return items
