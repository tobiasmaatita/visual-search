"""Trivial stage-1 baselines, used to validate the evaluation harness.

They set the floor any real method must beat: random ranking gives the chance
level, a color histogram shows what low-level appearance alone achieves.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from visual_search.data.schema import Item
from visual_search.imaging import crop_to_prompt
from visual_search.retrieval.base import Candidate, EmbeddingRetriever, Retriever


class RandomRetriever(Retriever):
    """Rank the gallery in a random order (chance level).

    The order depends only on ``seed`` and the query id, so repeated runs agree.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        """Create the retriever.

        Args:
            seed: Base seed for the per-query permutation.
        """
        self.seed = seed
        self._ids: list[str] | None = None

    def fit(self, gallery: Sequence[Item]) -> None:
        """Remember the gallery ids.

        Args:
            gallery: All gallery items.
        """
        self._ids = [it.item_id for it in gallery]

    def search(self, query: Item, k: int) -> list[Candidate]:
        """Return ``k`` gallery items in a random, reproducible order.

        Args:
            query: The query item (only its id is used, to seed the permutation).
            k: Number of candidates.

        Returns:
            Candidates with decreasing dummy scores.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._ids is None:
            raise RuntimeError("RandomRetriever.fit() must be called before search().")
        # Stable per-query seed: hash() of str is salted per process, so derive it from bytes.
        rng = np.random.default_rng([self.seed, *query.item_id.encode()])
        order = rng.permutation(len(self._ids))[:k]
        n = len(order)
        return [Candidate(self._ids[i], float(n - r) / n) for r, i in enumerate(order)]


class ColorHistogramRetriever(EmbeddingRetriever):
    """Rank by similarity of HSV color histograms.

    The query histogram is computed inside its prompt box (the whole image if no
    prompt); gallery histograms use the whole image, matching the image-level
    gallery of the real pipeline.
    """

    name = "color_hist"

    def __init__(self, bins: tuple[int, int, int] = (8, 4, 4), max_side: int = 256) -> None:
        """Create the retriever.

        Args:
            bins: Histogram bins for hue, saturation and value.
            max_side: Images are downscaled so the longer side is at most this (speed).
        """
        super().__init__()
        self.bins = bins
        self.max_side = max_side

    def _hist(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return the flattened, sum-normalized HSV histogram of an RGB image."""
        h, w = image.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1:
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256]).ravel()
        # Square root (Hellinger) makes cosine similarity behave like a proper histogram distance.
        return np.sqrt(hist / max(hist.sum(), 1.0)).astype(np.float32)

    def embed_query(self, query: Item) -> NDArray[np.float32]:
        """Histogram of the query's prompt region.

        Args:
            query: The query item.

        Returns:
            ``(prod(bins),)`` float32 vector.
        """
        return self._hist(crop_to_prompt(query.load_image(), query.prompt))

    def embed_gallery(self, items: Sequence[Item]) -> NDArray[np.float32]:
        """Whole-image histograms of gallery items.

        Args:
            items: Gallery items.

        Returns:
            ``(N, prod(bins))`` float32 array.
        """
        if not items:
            return np.zeros((0, int(np.prod(self.bins))), np.float32)
        return np.stack([self._hist(it.load_image()) for it in items])
