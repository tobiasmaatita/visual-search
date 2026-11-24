"""Brute-force cosine-similarity index.

One matrix-vector product is fast enough for galleries up to ~1e5 vectors
(e.g. 1e5 x 768 float32 is ~300 MB and well under 50 ms on CPU), so FAISS is not
needed at the planned scale.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def l2_normalize(x: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.float32]:
    """L2-normalize vectors along the last axis.

    Args:
        x: ``(..., D)`` array.
        eps: Guards against division by zero for all-zero vectors.

    Returns:
        float32 array of the same shape with unit-norm rows (zero rows stay zero).
    """
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


class BruteForceIndex:
    """Exact cosine-similarity search over a fixed set of vectors.

    Example:
        >>> index = BruteForceIndex(np.eye(3), ["a", "b", "c"])
        >>> index.search(np.array([0.0, 1.0, 0.1]), k=2)
        (['b', 'c'], array([0.995, 0.0995], dtype=float32))
    """

    def __init__(self, embeddings: NDArray[np.floating], ids: Sequence[str]) -> None:
        """Store normalized embeddings.

        Args:
            embeddings: ``(N, D)`` vectors, one per id.
            ids: ``N`` item ids.

        Raises:
            ValueError: If shapes and ids do not match.
        """
        emb = np.asarray(embeddings)
        if emb.ndim != 2 or emb.shape[0] != len(ids):
            raise ValueError(f"Expected ({len(ids)}, D) embeddings, got {emb.shape}.")
        self.embeddings = l2_normalize(emb)
        self.ids = list(ids)

    def __len__(self) -> int:
        """Return the number of indexed vectors."""
        return len(self.ids)

    def search(self, query: NDArray[np.floating], k: int) -> tuple[list[str], NDArray[np.float32]]:
        """Return the ``k`` most similar items to ``query``.

        Ties are broken by index order, so results are deterministic.

        Args:
            query: ``(D,)`` query vector (normalized internally).
            k: Number of results; clipped to the index size.

        Returns:
            ``(ids, scores)``: ids best first and their cosine similarities ``(k,)``.

        Raises:
            ValueError: If the query dimension does not match.
        """
        q = l2_normalize(np.asarray(query).reshape(-1))
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError(f"Query dim {q.shape[0]} != index dim {self.embeddings.shape[1]}.")
        sims = self.embeddings @ q
        k = min(k, len(self.ids))
        order = np.argsort(-sims, kind="stable")[:k]
        return [self.ids[i] for i in order], sims[order]
