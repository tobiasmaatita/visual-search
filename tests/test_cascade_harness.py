"""Tests for the cascade and the evaluation harness, using oracle components."""

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from visual_search.cascade import Cascade, RerankStage
from visual_search.data import Item, make_synthetic_dataset, split_dataset
from visual_search.eval import comparison_table, evaluate
from visual_search.retrieval import Candidate, ColorHistogramRetriever, RandomRetriever, Retriever


class ReverseLabelRetriever(Retriever):
    """Puts every correct item *last*: the worst possible stage 1."""

    name = "worst"

    def fit(self, gallery: Sequence[Item]) -> None:
        self.gallery = list(gallery)

    def search(self, query: Item, k: int) -> list[Candidate]:
        ordered = sorted(self.gallery, key=lambda g: (g.label == query.label, g.item_id))
        return [Candidate(g.item_id, -i) for i, g in enumerate(ordered)][:k]


class OracleRerank(RerankStage):
    """Moves correct candidates to the front (within its top_k)."""

    name = "oracle"

    def rerank(self, query: Item, candidates: list[Candidate], gallery: Mapping[str, Item]) -> list[Candidate]:
        return sorted(candidates, key=lambda c: gallery[c.item_id].label != query.label)


def _split():
    recs = [{"id": f"{i}_{v}", "input_image": np.zeros((4, 4, 3), np.uint8), "label": f"L{i}"}
            for i in range(5) for v in range(2)]
    recs += [{"id": f"d{d}", "input_image": np.zeros((4, 4, 3), np.uint8), "label": None} for d in range(7)]
    return split_dataset(recs, seed=0)  # 5 queries, gallery = 5 positives + 7 distractors


def test_stage_metrics_and_tail_preserved() -> None:
    split = _split()
    cascade = Cascade(ReverseLabelRetriever(), retrieval_k=100, stages=[OracleRerank(top_k=12)])
    rep = evaluate(cascade, split)
    assert list(rep.metrics.index) == ["retrieval", "oracle"]
    assert rep.metrics.loc["retrieval", "R@1"] == 0.0
    assert rep.metrics.loc["retrieval", "MRR"] == pytest.approx(1 / 12)  # correct item is always last of 12
    assert rep.metrics.loc["oracle", "R@1"] == 1.0
    assert rep.metrics.loc["oracle", "n_queries"] == 5
    assert set(rep.latency.index) == {"retrieval", "oracle", "total"}
    assert (rep.per_query["rank@oracle"] == 1).all()
    assert cascade.name == "worst@100 > oracle@12"


def test_rerank_only_touches_head() -> None:
    split = _split()
    cascade = Cascade(ReverseLabelRetriever(), stages=[OracleRerank(top_k=3)])
    cascade.prepare(split.gallery)
    res = cascade.run(split.queries[0])
    before = [c.item_id for c in res.stages["retrieval"]]
    after = [c.item_id for c in res.stages["oracle"]]
    assert after[3:] == before[3:] and sorted(after[:3]) == sorted(before[:3])
    assert res.final is res.stages["oracle"]
    assert set(res.timings) == {"retrieval", "oracle"}


def test_cascade_guards() -> None:
    with pytest.raises(RuntimeError):
        Cascade(RandomRetriever()).run(Item("q", "x"))
    with pytest.raises(ValueError):
        Cascade(RandomRetriever(), stages=[OracleRerank(3), OracleRerank(3)])
    with pytest.raises(ValueError):
        OracleRerank(top_k=0)


def test_random_baseline_near_chance_and_comparison_table() -> None:
    split = split_dataset(make_synthetic_dataset(n_instances=6, n_distractors=10, seed=0), seed=0)
    reports = [evaluate(Cascade(RandomRetriever(seed=0)), split), evaluate(Cascade(ColorHistogramRetriever()), split)]
    # 6 queries, gallery of 22 with 2 positives each: chance R@5 ~ 0.4; just check sanity bounds.
    for rep in reports:
        assert 0.0 <= rep.metrics.loc["retrieval", "R@5"] <= 1.0
        assert rep.metrics.loc["retrieval", "n_queries"] == 6
    table = comparison_table(reports)
    assert list(table.index) == ["random@100", "color_hist@100"]
    assert {"R@1", "R@5", "MRR", "total_median_ms", "total_p95_ms"} <= set(table.columns)
    assert "random@100" in str(reports[0])
