"""Retriever interface (stage 1) and the shared candidate type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from visual_search.data.schema import Item
from visual_search.retrieval.index import BruteForceIndex


@dataclass
class Candidate:
    """One ranked gallery item for a query.

    Attributes:
        item_id: Id of the gallery item.
        score: Ranking score of the stage that produced this candidate (higher is better).
        features: Named score components (e.g. ``{"cosine": 0.83, "inliers": 41}``),
            filled by later stages for inspection and learned combiners.
    """

    item_id: str
    score: float
    features: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    """Stage-1 retriever: index a gallery once, then rank it for each query.

    Implementations must be deterministic for a fixed configuration so that
    evaluation results are reproducible.
    """

    #: Short identifier used in reports and cache keys.
    name: str = "retriever"

    @abstractmethod
    def fit(self, gallery: Sequence[Item]) -> None:
        """Index the gallery (e.g. compute and store embeddings).

        Args:
            gallery: All gallery items. Called once before any :meth:`search`.
        """

    @abstractmethod
    def search(self, query: Item, k: int) -> list[Candidate]:
        """Rank the indexed gallery for a query.

        Args:
            query: The query item; its ``prompt`` (if any) should focus the search.
            k: Number of candidates to return.

        Returns:
            Up to ``k`` candidates, best first.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """


class EmbeddingRetriever(Retriever):
    """Retriever based on one L2-normalized vector per image and cosine similarity.

    Subclasses implement :meth:`embed_query` and :meth:`embed_gallery`; indexing
    and brute-force search are shared.
    """

    def __init__(self) -> None:
        """Initialize with an empty index."""
        self._index: BruteForceIndex | None = None

    @abstractmethod
    def embed_query(self, query: Item) -> NDArray[np.float32]:
        """Compute the descriptor of a query, focused on its prompt.

        Args:
            query: The query item.

        Returns:
            ``(D,)`` float32 vector (normalization is done by the index).
        """

    @abstractmethod
    def embed_gallery(self, items: Sequence[Item]) -> NDArray[np.float32]:
        """Compute descriptors for gallery items.

        Args:
            items: Gallery items.

        Returns:
            ``(N, D)`` float32 array, one row per item, in order.
        """

    def fit(self, gallery: Sequence[Item]) -> None:
        """Embed the gallery and build the brute-force cosine index.

        Args:
            gallery: All gallery items.
        """
        self._index = BruteForceIndex(self.embed_gallery(gallery), [it.item_id for it in gallery])

    def search(self, query: Item, k: int) -> list[Candidate]:
        """Return the ``k`` gallery items with highest cosine similarity to the query.

        Args:
            query: The query item.
            k: Number of candidates.

        Returns:
            Candidates with ``score`` = cosine similarity, best first.

        Raises:
            RuntimeError: If :meth:`fit` was not called.
        """
        if self._index is None:
            raise RuntimeError(f"{type(self).__name__}.fit() must be called before search().")
        ids, scores = self._index.search(self.embed_query(query), k)
        return [Candidate(i, float(s), {"cosine": float(s)}) for i, s in zip(ids, scores)]
