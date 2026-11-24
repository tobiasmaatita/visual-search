"""Tests for ranking metrics and latency utilities (hand-computed values)."""

import math
import time

import pytest

from visual_search.eval.metrics import first_correct_rank, mean_reciprocal_rank, recall_at_k
from visual_search.timing import LatencyStats, StageTimer


def test_first_correct_rank() -> None:
    assert first_correct_rank(["a", "b", "c"], {"c", "b"}) == 2
    assert first_correct_rank(["a", "b"], {"z"}) is None
    assert first_correct_rank([], {"a"}) is None


def test_recall_and_mrr_toy() -> None:
    ranks = [1, 3, None, 6]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert recall_at_k(ranks, 10) == 0.75
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 / 3 + 0 + 1 / 6) / 4)


def test_metric_edge_cases() -> None:
    assert math.isnan(recall_at_k([], 1))
    assert math.isnan(mean_reciprocal_rank([]))
    with pytest.raises(ValueError):
        recall_at_k([1], 0)


def test_latency_stats() -> None:
    st = LatencyStats.from_samples([1.0, 2.0, 3.0, 4.0, 100.0])
    assert st.n == 5 and st.median == 3.0 and st.max == 100.0
    assert st.p95 == pytest.approx(80.8)  # linear interpolation between 4 and 100
    assert math.isnan(LatencyStats.from_samples([]).median)


def test_stage_timer_accumulates_and_syncs() -> None:
    calls = []
    timer = StageTimer(sync=lambda: calls.append(1))
    with timer.stage("a"):
        time.sleep(0.01)
    with timer.stage("a"):
        time.sleep(0.01)
    with pytest.raises(RuntimeError), timer.stage("b"):
        raise RuntimeError
    assert timer.timings["a"] >= 0.02
    assert "b" in timer.timings  # recorded even when the block raises
    assert len(calls) == 6
    assert timer.total == pytest.approx(timer.timings["a"] + timer.timings["b"])
