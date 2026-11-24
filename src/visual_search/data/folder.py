"""Load datasets from plain image folders, or from mlcvtools' visual-search annotations.

Folder convention (no annotation file needed)::

    root/
      bank_01/          # one folder per physical instance; folder name = label
        a.jpg
        b.jpg
      billy_01/
        ...
      _distractors/     # label=None (also: "distractors", "singletons")
        room.jpg
      prompts.json      # optional: {"bank_01/a.jpg": {"box": [x0, y0, x1, y1]}, ...}

Item ids are the image paths relative to ``root`` (POSIX style), so they are stable
across machines and usable as cache keys and split seeds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
UNSUPPORTED_EXTENSIONS = frozenset({".heic", ".heif"})
DEFAULT_DISTRACTOR_DIRS = ("_distractors", "distractors", "singletons")


def _list_images(directory: Path) -> list[Path]:
    """Return the supported image files directly inside ``directory``, sorted, warning on HEIC."""
    images = []
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(p)
        elif ext in UNSUPPORTED_EXTENSIONS:
            logger.warning("Skipping %s: HEIC/HEIF is not supported, convert it to JPG.", p)
    return images


def load_folder_dataset(
    root: str | Path,
    distractor_dirs: Iterable[str] = DEFAULT_DISTRACTOR_DIRS,
    prompts_file: str | None = "prompts.json",
) -> list[dict[str, Any]]:
    """Build raw dataset records from a folder-per-instance directory tree.

    Every subfolder of ``root`` is one instance whose label is the folder name,
    except folders named in ``distractor_dirs``, whose images get ``label=None``.
    Images directly in ``root`` are also treated as distractors (with a warning).
    Empty folders and unsupported files are skipped with a warning. Instances with
    a single image are kept here; :func:`~visual_search.data.split.validate_dataset`
    reports them.

    Args:
        root: Dataset root directory.
        distractor_dirs: Folder names whose images are distractors.
        prompts_file: Optional JSON file in ``root`` mapping relative image paths to
            prompt dicts (``{"box": ...}`` / ``{"points": ...}``). ``None`` disables it.

    Returns:
        Raw records ``{"id", "input_image", "label", ["prompt"]}`` sorted by id, with
        ``input_image`` as a :class:`~pathlib.Path` (images are not loaded).

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
        ValueError: If the prompts file is not a JSON object.

    Example:
        >>> records = load_folder_dataset("data/furniture")
        >>> split = split_dataset(records, seed=0, strict=False)
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root {root} is not a directory.")
    distractor_names = set(distractor_dirs)

    records: list[dict[str, Any]] = []
    loose = _list_images(root)
    if loose:
        logger.warning("%d image(s) directly in %s are treated as distractors.", len(loose), root)
    for p in loose:
        records.append({"id": p.relative_to(root).as_posix(), "input_image": p, "label": None})

    for sub in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")):
        images = _list_images(sub)
        if not images:
            logger.warning("Skipping empty folder %s.", sub)
            continue
        label = None if sub.name in distractor_names else sub.name
        for p in images:
            records.append({"id": p.relative_to(root).as_posix(), "input_image": p, "label": label})

    if prompts_file is not None and (root / prompts_file).is_file():
        prompts = json.loads((root / prompts_file).read_text())
        if not isinstance(prompts, dict):
            raise ValueError(f"{root / prompts_file} must be a JSON object keyed by relative image path.")
        by_id = {r["id"]: r for r in records}
        for rel, prompt in prompts.items():
            if rel in by_id:
                by_id[rel]["prompt"] = prompt
            else:
                logger.warning("Prompt for unknown image %s in %s ignored.", rel, prompts_file)

    return sorted(records, key=lambda r: r["id"])


def load_image_folder(
    directory: str | Path,
    limit: int | None = None,
    seed: int = 0,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    """Load every image in a folder as a distractor (``label=None``).

    Use this for a separate distractor pool such as a COCO image directory.

    Args:
        directory: Folder with images.
        limit: If set, keep a random subset of this size (reproducible via ``seed``).
        seed: Seed for the subset selection.
        recursive: Also include images in subfolders.

    Returns:
        Raw records with ids ``"distractors/<relative path>"``, sorted by id.

    Raises:
        FileNotFoundError: If ``directory`` is not a directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Distractor folder {directory} is not a directory.")
    if recursive:
        paths = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
        )
    else:
        paths = _list_images(directory)
    if limit is not None and limit < len(paths):
        keep = np.random.default_rng(seed).choice(len(paths), size=limit, replace=False)
        paths = [paths[i] for i in sorted(keep)]
    return [
        {"id": f"distractors/{p.relative_to(directory).as_posix()}", "input_image": p, "label": None}
        for p in paths
    ]


def load_mlcvtools_annotations(path: str | Path, image_root: str | Path) -> list[dict[str, Any]]:
    """Convert annotations saved by mlcvtools' visual-search notebook into raw records.

    The mlcvtools format (``experimental.visual_search.save_annotations``) is a JSON
    list of ``{"image": <path relative to image_root>, "bbox": [x0, y0, x1, y1],
    "object_id": str | null}``. ``object_id`` becomes the label (null -> distractor)
    and ``bbox`` becomes a box prompt. Files are only read, never written.

    Args:
        path: The annotations JSON file.
        image_root: Directory the ``image`` paths are relative to.

    Returns:
        Raw records. Ids are the relative image path, suffixed with ``#<n>`` when an
        image has several boxes.

    Raises:
        ValueError: If an image path escapes ``image_root``.
        FileNotFoundError: If an annotated image does not exist.
    """
    root = Path(image_root).resolve()
    records: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for rec in json.loads(Path(path).read_text()):
        image_path = (root / rec["image"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"Annotated image {rec['image']} is outside {root}.")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rel = image_path.relative_to(root).as_posix()
        n = seen.get(rel, 0)
        seen[rel] = n + 1
        records.append(
            {
                "id": rel if n == 0 else f"{rel}#{n}",
                "input_image": image_path,
                "label": rec.get("object_id"),
                "prompt": {"box": [float(v) for v in rec["bbox"]]},
            }
        )
    return records
