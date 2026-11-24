"""Matcher interface: local features per image, correspondences per image pair.

Feature extraction is separated from matching so gallery features can be computed
once and cached. All coordinates are ``(x, y)`` in **original-image pixels**,
whatever resolution a matcher works at internally.

Adding a matcher = one module that subclasses :class:`Matcher` and registers
itself with :func:`~visual_search.matching.registry.register_matcher`. It must
pass the shared contract test (``tests/test_matching.py::test_matcher_contract``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Features:
    """Local features of one image.

    Attributes:
        keypoints: ``(N, 2)`` float32 ``(x, y)`` in original-image pixels.
        descriptors: ``(N, D)`` descriptors (dtype is matcher specific).
        scores: ``(N,)`` detection scores, or ``None``.
        image_hw: Original image ``(H, W)``.
    """

    keypoints: NDArray[np.float32]
    descriptors: NDArray
    scores: NDArray[np.float32] | None
    image_hw: tuple[int, int]

    def __len__(self) -> int:
        """Return the number of keypoints."""
        return len(self.keypoints)

    def subset(self, keep: NDArray[np.bool_]) -> Features:
        """Return the features selected by a boolean mask over keypoints.

        Args:
            keep: ``(N,)`` bool.

        Returns:
            A new :class:`Features` with only the kept keypoints.
        """
        return Features(
            self.keypoints[keep],
            self.descriptors[keep],
            None if self.scores is None else self.scores[keep],
            self.image_hw,
        )

    def to_arrays(self) -> dict[str, NDArray]:
        """Serialize to plain arrays (for the feature cache)."""
        out = {"keypoints": self.keypoints, "descriptors": self.descriptors, "image_hw": np.array(self.image_hw)}
        if self.scores is not None:
            out["scores"] = self.scores
        return out

    @classmethod
    def from_arrays(cls, arrays: dict[str, NDArray]) -> Features:
        """Inverse of :meth:`to_arrays`."""
        hw = arrays["image_hw"]
        return cls(arrays["keypoints"], arrays["descriptors"], arrays.get("scores"), (int(hw[0]), int(hw[1])))


@dataclass
class Correspondences:
    """Tentative point matches between a query and a candidate image.

    Attributes:
        pts_q: ``(M, 2)`` float32 points in the query image (pixels).
        pts_c: ``(M, 2)`` float32 corresponding points in the candidate image (pixels).
        confidence: ``(M,)`` float32 match confidence in [0, 1] (matcher specific).
        n_query_kpts: Keypoints considered in the query (after masking).
        n_cand_kpts: Keypoints considered in the candidate (after masking).
        extra: Matcher-specific artifacts (e.g. a dense warp), not used by core code.
    """

    pts_q: NDArray[np.float32]
    pts_c: NDArray[np.float32]
    confidence: NDArray[np.float32]
    n_query_kpts: int = 0
    n_cand_kpts: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of correspondences."""
        return len(self.pts_q)

    @classmethod
    def empty(cls, n_query_kpts: int = 0, n_cand_kpts: int = 0) -> Correspondences:
        """Return a correspondence set with no matches."""
        z = np.zeros((0, 2), np.float32)
        return cls(z, z.copy(), np.zeros(0, np.float32), n_query_kpts, n_cand_kpts)


def keypoints_in_mask(keypoints: NDArray[np.float32], mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return which keypoints fall inside a mask.

    Args:
        keypoints: ``(N, 2)`` ``(x, y)`` pixels.
        mask: ``(H, W)`` bool.

    Returns:
        ``(N,)`` bool.
    """
    h, w = mask.shape
    xs = np.clip(np.round(keypoints[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(keypoints[:, 1]).astype(int), 0, h - 1)
    return mask[ys, xs]


class Matcher(ABC):
    """Local-feature matcher (sparse or detector-free)."""

    #: Registry / report name.
    name: str = "matcher"

    @property
    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return everything that determines the extracted features (cache key)."""

    @abstractmethod
    def extract(self, image: NDArray[np.uint8]) -> Features:
        """Detect and describe local features on the full image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB.

        Returns:
            Features with keypoints in original-image pixels.
        """

    @abstractmethod
    def match_features(self, fq: Features, fc: Features) -> Correspondences:
        """Match two feature sets.

        Args:
            fq: Query features (already restricted to the query mask).
            fc: Candidate features (already restricted to the candidate mask).

        Returns:
            Tentative correspondences (before geometric verification).
        """

    def match(
        self,
        image_q: NDArray[np.uint8],
        image_c: NDArray[np.uint8],
        mask_q: NDArray[np.bool_] | None = None,
        mask_c: NDArray[np.bool_] | None = None,
    ) -> Correspondences:
        """Extract and match features for one image pair, restricted to the masks.

        Args:
            image_q: Query RGB image.
            image_c: Candidate RGB image.
            mask_q: Optional query object mask; keypoints outside are ignored.
            mask_c: Optional candidate mask.

        Returns:
            Tentative correspondences.
        """
        return self.match_extracted(self.extract(image_q), self.extract(image_c), mask_q, mask_c)

    def match_extracted(
        self,
        fq: Features,
        fc: Features,
        mask_q: NDArray[np.bool_] | None = None,
        mask_c: NDArray[np.bool_] | None = None,
    ) -> Correspondences:
        """Restrict precomputed features to the masks and match them.

        Args:
            fq: Full-image query features.
            fc: Full-image candidate features.
            mask_q: Optional query mask.
            mask_c: Optional candidate mask.

        Returns:
            Tentative correspondences.
        """
        if mask_q is not None:
            fq = fq.subset(keypoints_in_mask(fq.keypoints, mask_q))
        if mask_c is not None:
            fc = fc.subset(keypoints_in_mask(fc.keypoints, mask_c))
        if len(fq) == 0 or len(fc) == 0:
            return Correspondences.empty(len(fq), len(fc))
        corr = self.match_features(fq, fc)
        corr.n_query_kpts, corr.n_cand_kpts = len(fq), len(fc)
        return corr
