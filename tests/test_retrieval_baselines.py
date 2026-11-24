"""Tests for the brute-force index, prompt cropping and the trivial baselines."""

import numpy as np
import pytest

from visual_search.data import Item, Prompt
from visual_search.imaging import crop_to_prompt, prompt_box
from visual_search.retrieval import BruteForceIndex, ColorHistogramRetriever, RandomRetriever


def test_index_search_and_ties() -> None:
    idx = BruteForceIndex(np.array([[1, 0], [0, 1], [1, 0], [3, 3]], np.float32), ["a", "b", "c", "d"])
    ids, scores = idx.search(np.array([1.0, 0.0]), k=3)
    assert ids == ["a", "c", "d"]  # tie between a and c broken by index order
    np.testing.assert_allclose(scores, [1, 1, np.sqrt(0.5)], rtol=1e-6)
    assert idx.search(np.array([0, 1.0]), k=99)[0][0] == "b"
    with pytest.raises(ValueError):
        idx.search(np.zeros(3), k=1)
    with pytest.raises(ValueError):
        BruteForceIndex(np.zeros((2, 2)), ["a"])


def test_prompt_box_padding_min_size_and_clipping() -> None:
    shape = (100, 200, 3)
    assert prompt_box(None, shape) == (0, 0, 200, 100)
    assert prompt_box(Prompt(box=(10, 10, 30, 50)), shape) == (10, 10, 30, 50)
    assert prompt_box(Prompt(box=(10, 10, 30, 50)), shape, pad=0.5) == (0, 0, 40, 70)
    one_click = Prompt(points=np.float32([[100, 50]]), point_labels=np.int64([1]))
    assert prompt_box(one_click, shape, min_size=20) == (90, 40, 110, 60)
    img = np.zeros(shape, np.uint8)
    assert crop_to_prompt(img, Prompt(box=(190, 90, 260, 140))).shape == (10, 10, 3)


def _solid(color: tuple[int, int, int], item_id: str, label: str | None = None) -> Item:
    return Item(item_id, np.full((32, 32, 3), color, np.uint8), label)


def test_color_histogram_finds_same_color() -> None:
    gallery = [_solid((255, 0, 0), "red"), _solid((0, 0, 255), "blue"), _solid((0, 255, 0), "green")]
    r = ColorHistogramRetriever()
    r.fit(gallery)
    res = r.search(_solid((250, 5, 5), "q"), k=2)
    assert len(res) == 2 and res[0].item_id == "red"
    assert res[0].score == pytest.approx(1.0, abs=1e-5)
    assert "cosine" in res[0].features


def test_color_histogram_uses_query_prompt() -> None:
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :20] = (255, 0, 0)
    img[:, 20:] = (0, 0, 255)
    gallery = [_solid((255, 0, 0), "red"), _solid((0, 0, 255), "blue")]
    r = ColorHistogramRetriever()
    r.fit(gallery)
    q = Item("q", img, prompt=Prompt(box=(22, 0, 40, 40)))
    assert r.search(q, k=1)[0].item_id == "blue"


def test_random_is_reproducible_permutation() -> None:
    gallery = [_solid((i, i, i), f"g{i}") for i in range(20)]
    r = RandomRetriever(seed=1)
    with pytest.raises(RuntimeError):
        r.search(gallery[0], 3)
    r.fit(gallery)
    a = [c.item_id for c in r.search(Item("q1", "x"), k=20)]
    assert sorted(a) == sorted(g.item_id for g in gallery)
    r2 = RandomRetriever(seed=1)
    r2.fit(gallery)
    assert a == [c.item_id for c in r2.search(Item("q1", "x"), k=20)]
    assert a !=[c.item_id for c in r.search(Item("q2", "x"), k=20)]
