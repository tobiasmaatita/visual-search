"""Per-object annotations (boxes with instance labels) for a self-built test set.

Layout: a directory of images (subfolders allowed) plus one ``annotations.json``::

    {
      "version": 1,
      "images": {
        "living_room_3.jpg": [
          {"box": [63, 335, 535, 568], "label": "bank_01", "group": "sofa_model_x"},
          {"box": [600, 120, 780, 500], "label": "billy_01"}
        ],
        "kitchen.jpg": []
      }
    }

- ``label`` is the **instance** (same label = same physical object).
- ``group`` (optional) is a coarser identity such as the product **model**.
- An image listed with an empty box list is *reviewed and contains no objects of
  interest*: it becomes a whole-image distractor. Images not listed at all are
  *not labelled yet* and are left out of the dataset, so half-labelled sets never
  produce wrong distractors.

Each box becomes one dataset item (id ``"<image>#<n>"``), with the box as its prompt.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from visual_search.data.folder import IMAGE_EXTENSIONS

ANNOTATIONS_FILE = "annotations.json"
FORMAT_VERSION = 1


@dataclass
class BoxAnnotation:
    """One labelled object box.

    Attributes:
        box: ``(x0, y0, x1, y1)`` in original-image pixels.
        label: Instance id (same label = same physical object); ``None`` = unlabelled object.
        group: Optional coarser identity (e.g. product model).
    """

    box: tuple[float, float, float, float]
    label: str | None
    group: str | None = None

    def __post_init__(self) -> None:
        """Normalize types and validate the box."""
        self.box = tuple(round(float(v), 1) for v in self.box)  # type: ignore[assignment]
        x0, y0, x1, y1 = self.box
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"Box must satisfy x0 < x1 and y0 < y1, got {self.box}.")
        self.label = _clean(self.label)
        self.group = _clean(self.group)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON format (``group`` only when set)."""
        d: dict[str, Any] = {"box": list(self.box), "label": self.label}
        if self.group is not None:
            d["group"] = self.group
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BoxAnnotation:
        """Parse one JSON box entry."""
        return cls(tuple(d["box"]), d.get("label"), d.get("group"))  # type: ignore[arg-type]


def _clean(value: str | None) -> str | None:
    """Strip whitespace; empty strings become None."""
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def normalize_label(label: str) -> str:
    """Return a spelling-insensitive key: lowercase, letters and digits only.

    ``"Bank_01"``, ``"bank-01"`` and ``"bank 01"`` all map to ``"bank01"``.

    Args:
        label: A label.

    Returns:
        The normalized key.
    """
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _numbers(label: str) -> tuple[int, ...]:
    """Return the digit groups of a label as integers (``"bank_01b2"`` -> ``(1, 2)``)."""
    return tuple(int(d) for d in re.findall(r"[0-9]+", label))


def similar_labels(label: str, existing: Iterable[str], cutoff: float = 0.8) -> list[str]:
    """Find existing labels that are probably meant to be the same as ``label``.

    Labels whose numbers differ are never similar (``bank_01`` and ``bank_02`` are
    deliberately different items); numbers compare by value, so ``bank_1`` and
    ``bank_01`` agree. Otherwise a label counts as similar when its letters match
    ignoring case, spaces, dashes and underscores, or are close in spelling
    (``difflib`` ratio at or above ``cutoff``: ``chiar_01`` vs ``chair_01``). The
    exact same label is not reported.

    Args:
        label: The new label being typed.
        existing: Labels already in use.
        cutoff: Similarity threshold in [0, 1] for spelling matches.

    Returns:
        Similar existing labels, most similar first.

    Example:
        >>> similar_labels("Bank_01", ["bank_01", "bank_02", "billy_01"])
        ['bank_01']
    """
    key = normalize_label(label)
    if not key:
        return []
    letters, numbers = re.sub(r"[0-9]", "", key), _numbers(key)
    scored = []
    for other in set(existing):
        okey = normalize_label(other)
        if other == label or _numbers(okey) != numbers:
            continue
        oletters = re.sub(r"[0-9]", "", okey)
        if oletters == letters:
            ratio = 1.0
        elif letters and oletters:
            ratio = difflib.SequenceMatcher(None, letters, oletters).ratio()
        else:
            ratio = 0.0
        if ratio >= cutoff:
            scored.append((ratio, other))
    return [o for _, o in sorted(scored, key=lambda t: (-t[0], t[1]))]


class AnnotationSet:
    """The annotations of one image directory, backed by ``annotations.json``.

    Example:
        >>> ann = AnnotationSet.load("data/testset")
        >>> ann.add("living_room_3.jpg", BoxAnnotation((63, 335, 535, 568), "bank_01"))
        >>> ann.save()
        >>> records = ann.to_records()          # one item per box, ready for split_dataset
    """

    def __init__(self, root: str | Path, images: dict[str, list[BoxAnnotation]] | None = None,
                 filename: str = ANNOTATIONS_FILE) -> None:
        """Create an annotation set (use :meth:`load` to read an existing file).

        Args:
            root: Image directory; also where the annotation file lives.
            images: Relative image path -> boxes (only *reviewed* images).
            filename: Annotation file name inside ``root``.
        """
        self.root = Path(root)
        self.filename = filename
        self.images: dict[str, list[BoxAnnotation]] = images or {}

    @property
    def path(self) -> Path:
        """Return the annotation file path."""
        return self.root / self.filename

    @classmethod
    def load(cls, root: str | Path, filename: str = ANNOTATIONS_FILE) -> AnnotationSet:
        """Read ``<root>/<filename>``; an absent file gives an empty set.

        Args:
            root: Image directory.
            filename: Annotation file name.

        Returns:
            The annotation set.

        Raises:
            FileNotFoundError: If ``root`` is not a directory.
            ValueError: If the file has an unknown format version.
        """
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory {root} does not exist.")
        path = root / filename
        if not path.exists():
            return cls(root, {}, filename)
        data = json.loads(path.read_text())
        if data.get("version", FORMAT_VERSION) != FORMAT_VERSION:
            raise ValueError(f"Unsupported annotation format version {data.get('version')}.")
        images = {rel: [BoxAnnotation.from_dict(b) for b in boxes] for rel, boxes in data.get("images", {}).items()}
        return cls(root, images, filename)

    def save(self, backup: bool = True) -> None:
        """Write the annotation file atomically (keeping the previous version as ``.bak``).

        Args:
            backup: Copy the existing file to ``<filename>.bak`` first.
        """
        if backup and self.path.exists():
            shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
        payload = {"version": FORMAT_VERSION,
                   "images": {rel: [b.to_dict() for b in boxes] for rel, boxes in sorted(self.images.items())}}
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # --- images -------------------------------------------------------------------

    def image_paths(self) -> list[str]:
        """Return all images under ``root`` (recursively) as sorted relative POSIX paths."""
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
        )

    def is_reviewed(self, rel: str) -> bool:
        """Return True if the image is in the annotation file (labelled, possibly with no boxes)."""
        return rel in self.images

    def mark_reviewed(self, rel: str) -> None:
        """Register an image as reviewed even if it has no boxes (a whole-image distractor)."""
        self.images.setdefault(rel, [])

    def unmark_reviewed(self, rel: str) -> None:
        """Remove an image without boxes from the annotation file (back to "not labelled yet").

        Raises:
            ValueError: If the image still has boxes.
        """
        if self.images.get(rel):
            raise ValueError(f"{rel} still has boxes; delete them first.")
        self.images.pop(rel, None)

    # --- boxes --------------------------------------------------------------------

    def boxes(self, rel: str) -> list[BoxAnnotation]:
        """Return the boxes of an image (empty if none or not reviewed)."""
        return list(self.images.get(rel, []))

    def add(self, rel: str, box: BoxAnnotation) -> int:
        """Add a box to an image (marking it reviewed).

        Args:
            rel: Relative image path.
            box: The annotation.

        Returns:
            The index of the new box within that image.
        """
        self.images.setdefault(rel, []).append(box)
        return len(self.images[rel]) - 1

    def remove(self, rel: str, index: int) -> BoxAnnotation:
        """Delete box ``index`` of an image (the image stays reviewed).

        Returns:
            The removed annotation.

        Raises:
            IndexError: If there is no such box.
        """
        return self.images[rel].pop(index)

    def relabel(self, rel: str, index: int, label: str | None, group: str | None = None) -> None:
        """Change the label and group of box ``index``.

        Raises:
            IndexError: If there is no such box.
        """
        old = self.images[rel][index]
        self.images[rel][index] = BoxAnnotation(old.box, label, group)

    # --- vocabulary and statistics ---------------------------------------------------

    def labels(self) -> list[str]:
        """Return all instance labels in use, sorted."""
        return sorted({b.label for boxes in self.images.values() for b in boxes if b.label})

    def groups(self) -> list[str]:
        """Return all groups in use, sorted."""
        return sorted({b.group for boxes in self.images.values() for b in boxes if b.group})

    def label_stats(self) -> pd.DataFrame:
        """Return one row per label: its group(s), number of boxes and images, and warnings.

        The ``warning`` column flags labels with a single box (they cannot be found
        again, so they don't count in evaluation) and labels that look like another
        label (possible spelling variants).

        Returns:
            DataFrame indexed by label, sorted by label.
        """
        rows: dict[str, dict[str, Any]] = {}
        for rel, boxes in self.images.items():
            for b in boxes:
                if not b.label:
                    continue
                r = rows.setdefault(b.label, {"groups": set(), "boxes": 0, "images": set()})
                r["boxes"] += 1
                r["images"].add(rel)
                if b.group:
                    r["groups"].add(b.group)
        labels = sorted(rows)
        out = []
        for lbl in labels:
            r = rows[lbl]
            warnings = []
            if r["boxes"] < 2:
                warnings.append("only 1 box")
            sim = similar_labels(lbl, labels)
            if sim:
                warnings.append("similar to " + ", ".join(sim))
            out.append({"label": lbl, "group": ", ".join(sorted(r["groups"])), "boxes": r["boxes"],
                        "images": len(r["images"]), "warning": "; ".join(warnings)})
        df = pd.DataFrame(out, columns=["label", "group", "boxes", "images", "warning"])
        return df.set_index("label")

    def summary(self) -> str:
        """Return a one-line progress summary."""
        n_img = len(self.image_paths())
        n_rev = sum(1 for rel in self.images if (self.root / rel).exists())
        n_box = sum(len(b) for b in self.images.values())
        return (f"{n_rev}/{n_img} images reviewed, {n_box} boxes, {len(self.labels())} labels, "
                f"{len(self.groups())} groups")

    # --- dataset ------------------------------------------------------------------------

    def to_records(self, include_distractor_images: bool = True) -> list[dict[str, Any]]:
        """Convert to raw dataset records: one per box, plus reviewed images without boxes.

        Each box record has id ``"<image>#<n>"``, the image path, the box's label (an
        unlabelled box is a distractor object), a box prompt, and extra keys
        ``image_id`` and (if set) ``group``. Images that are not reviewed are skipped.

        Args:
            include_distractor_images: Add reviewed images without boxes as whole-image
                distractors (``label=None``, no prompt).

        Returns:
            Records for :func:`~visual_search.data.split.split_dataset`, sorted by id.
        """
        records: list[dict[str, Any]] = []
        for rel, boxes in sorted(self.images.items()):
            path = self.root / rel
            if not path.exists():
                continue
            if not boxes:
                if include_distractor_images:
                    records.append({"id": rel, "input_image": path, "label": None, "image_id": rel})
                continue
            for i, b in enumerate(boxes):
                rec: dict[str, Any] = {"id": f"{rel}#{i}", "input_image": path, "label": b.label,
                                       "prompt": {"box": list(b.box)}, "image_id": rel}
                if b.group:
                    rec["group"] = b.group
                records.append(rec)
        return records
