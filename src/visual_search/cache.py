"""On-disk cache for per-image features, keyed by image content + model + config.

Layout: ``<root>/<namespace>/<config_hash>/<fp[:2]>/<fp>.npz`` where ``fp`` is a
fingerprint of the image content. Anything that changes the features (model id,
resolution, cache format version) must be part of the config dict, so a config
change automatically uses a fresh directory and stale entries are never read.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from visual_search.data.schema import ImageSource


def image_fingerprint(image: ImageSource) -> str:
    """Return a content hash of an image source.

    Paths are hashed by file bytes (no decoding, so cache hits stay cheap); arrays
    and PIL images by their pixel bytes and shape. The same picture given as a path
    and as an array therefore gets different fingerprints, which is harmless.

    Args:
        image: Path, ``np.ndarray`` or ``PIL.Image``.

    Returns:
        40-character hex SHA-1.
    """
    h = hashlib.sha1()
    if isinstance(image, (str, Path)):
        h.update(b"file:")
        with open(image, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    else:
        arr = np.ascontiguousarray(np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image))
        h.update(f"array:{arr.dtype}:{arr.shape}:".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    """Return a short stable hash of a JSON-serializable config dict.

    Args:
        config: Values that determine the cached features.

    Returns:
        12-character hex string.
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


class FeatureCache:
    """Store and load dicts of numpy arrays per image fingerprint.

    Writes are atomic (temp file + rename), so an interrupted run never leaves a
    corrupt entry. A ``config.json`` next to the entries records what produced them.

    Example:
        >>> cache = FeatureCache("cache", "dinov2", {"model": "facebook/dinov2-small", "res": 448})
        >>> if (feats := cache.get(fp)) is None:
        ...     cache.put(fp, {"cls": cls, "patches": patches})
    """

    def __init__(self, root: str | Path, namespace: str, config: dict[str, Any]) -> None:
        """Create (or reuse) the cache directory for one namespace + config.

        Args:
            root: Cache root directory (e.g. ``<repo>/cache``).
            namespace: Feature family, e.g. ``"dinov2"``.
            config: Everything that influences the features.
        """
        self.config = config
        self.dir = Path(root) / namespace / config_hash(config)
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "config.json"
        if not meta.exists():
            meta.write_text(json.dumps(config, indent=2, sort_keys=True, default=str))

    def _path(self, fingerprint: str) -> Path:
        """Return the file path for a fingerprint."""
        return self.dir / fingerprint[:2] / f"{fingerprint}.npz"

    def get(self, fingerprint: str) -> dict[str, NDArray] | None:
        """Load cached arrays for an image.

        Args:
            fingerprint: From :func:`image_fingerprint`.

        Returns:
            Name -> array, or ``None`` on a cache miss.
        """
        path = self._path(fingerprint)
        if not path.exists():
            return None
        with np.load(path) as data:
            return {k: data[k] for k in data.files}

    def put(self, fingerprint: str, arrays: dict[str, NDArray]) -> None:
        """Store arrays for an image (overwrites an existing entry).

        Args:
            fingerprint: From :func:`image_fingerprint`.
            arrays: Name -> array. Use float16 for large tensors to save disk.
        """
        path = self._path(fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def __contains__(self, fingerprint: str) -> bool:
        """Return True if an entry exists for ``fingerprint``."""
        return self._path(fingerprint).exists()
