"""The retrieval cascade: stage-1 retrieval followed by optional re-ranking stages.

The same :class:`Cascade` object serves interactive single queries
(:meth:`Cascade.run`) and batch evaluation
(:func:`visual_search.eval.harness.evaluate`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from visual_search.data.schema import Item
from visual_search.retrieval.base import Candidate, Retriever
from visual_search.timing import StageTimer

RETRIEVAL_STAGE = "retrieval"


class RerankStage(ABC):
    """A re-ranking stage (e.g. sparse verification, dense matching).

    It receives the top ``top_k`` candidates of the previous stage and returns
    them re-scored and re-ordered. Candidates below ``top_k`` are not touched.
    """

    #: Stage name used in results, reports and timings; must be unique in a cascade.
    name: str = "rerank"

    def __init__(self, top_k: int) -> None:
        """Create the stage.

        Args:
            top_k: How many candidates of the previous stage to re-rank.

        Raises:
            ValueError: If ``top_k < 1``.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        self.top_k = top_k

    def prepare(self, gallery: Sequence[Item]) -> None:
        """Precompute per-gallery-item data (e.g. keypoints). Default: nothing.

        Args:
            gallery: All gallery items.
        """

    @abstractmethod
    def rerank(self, query: Item, candidates: list[Candidate], gallery: Mapping[str, Item]) -> list[Candidate]:
        """Re-score and re-order candidates.

        Args:
            query: The query item.
            candidates: The previous stage's top ``top_k`` candidates, best first.
            gallery: Item id -> gallery item, to access images.

        Returns:
            The same candidates (new ``score``/``features`` allowed), best first.
        """


@dataclass
class QueryResult:
    """Output of one cascade run.

    Attributes:
        query_id: Id of the query item.
        stages: Stage name -> full ranking after that stage, in cascade order. A
            re-ranking stage's list is its re-ordered head followed by the untouched
            tail of the previous stage, so Recall@k is comparable across stages.
        timings: Stage name -> seconds for this query.
    """

    query_id: str
    stages: dict[str, list[Candidate]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def final(self) -> list[Candidate]:
        """Return the ranking of the last stage."""
        return next(reversed(self.stages.values())) if self.stages else []


class Cascade:
    """Stage-1 retrieval plus a sequence of re-ranking stages.

    Example:
        >>> cascade = Cascade(ColorHistogramRetriever(), retrieval_k=100)
        >>> cascade.prepare(split.gallery)
        >>> result = cascade.run(split.queries[0])
        >>> [c.item_id for c in result.final[:5]]
    """

    def __init__(
        self,
        retriever: Retriever,
        retrieval_k: int = 100,
        stages: Sequence[RerankStage] = (),
        sync: Callable[[], None] | None = None,
    ) -> None:
        """Build the cascade.

        Args:
            retriever: Stage-1 retriever.
            retrieval_k: Candidates kept after stage 1.
            stages: Re-ranking stages, applied in order.
            sync: Optional device synchronization for accurate GPU timings.

        Raises:
            ValueError: If stage names are not unique or ``retrieval_k < 1``.
        """
        names = [RETRIEVAL_STAGE, *(s.name for s in stages)]
        if len(set(names)) != len(names):
            raise ValueError(f"Stage names must be unique, got {names}.")
        if retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1.")
        self.retriever = retriever
        self.retrieval_k = retrieval_k
        self.stages = list(stages)
        self.sync = sync
        self._gallery: dict[str, Item] | None = None

    @property
    def name(self) -> str:
        """Return a readable identifier, e.g. ``"color_hist@100 > sift@20"``."""
        parts = [f"{self.retriever.name}@{self.retrieval_k}"] + [f"{s.name}@{s.top_k}" for s in self.stages]
        return " > ".join(parts)

    def prepare(self, gallery: Sequence[Item]) -> dict[str, float]:
        """Index the gallery for the retriever and all stages (offline work).

        Args:
            gallery: All gallery items.

        Returns:
            Stage name -> seconds spent preparing.
        """
        timer = StageTimer(sync=self.sync)
        with timer.stage(RETRIEVAL_STAGE):
            self.retriever.fit(gallery)
        for st in self.stages:
            with timer.stage(st.name):
                st.prepare(gallery)
        self._gallery = {it.item_id: it for it in gallery}
        return timer.timings

    def run(self, query: Item) -> QueryResult:
        """Run the full cascade for one query.

        Args:
            query: The query item (its prompt focuses retrieval and matching).

        Returns:
            Rankings after each stage and per-stage timings.

        Raises:
            RuntimeError: If :meth:`prepare` was not called.
        """
        if self._gallery is None:
            raise RuntimeError("Cascade.prepare(gallery) must be called before run().")
        timer = StageTimer(sync=self.sync)
        result = QueryResult(query.item_id)
        with timer.stage(RETRIEVAL_STAGE):
            ranking = self.retriever.search(query, self.retrieval_k)
        result.stages[RETRIEVAL_STAGE] = ranking
        for st in self.stages:
            head, tail = ranking[: st.top_k], ranking[st.top_k :]
            with timer.stage(st.name):
                head = st.rerank(query, head, self._gallery)
            ranking = head + tail
            result.stages[st.name] = ranking
        result.timings = timer.timings
        return result
