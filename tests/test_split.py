"""Tests for dataset validation and the seeded query/gallery split."""

import pytest

from visual_search.data import DatasetValidationError, split_dataset, validate_dataset
from visual_search.data.schema import parse_dataset


def _records(n_instances: int = 10, per_instance: int = 4, n_distractors: int = 5) -> list[dict]:
    recs = [
        {"id": f"{i:02d}_{v}", "input_image": "x", "label": f"inst_{i:02d}", "group": f"g{i % 3}"}
        for i in range(n_instances)
        for v in range(per_instance)
    ]
    recs += [{"id": f"d_{d}", "input_image": "x", "label": None} for d in range(n_distractors)]
    return recs


def _ids(items: list) -> list[str]:
    return [it.item_id for it in items]


def test_reproducible_and_order_independent() -> None:
    recs = _records()
    a = split_dataset(recs, seed=3)
    b = split_dataset(list(reversed(recs)), seed=3)
    assert _ids(a.queries) == _ids(b.queries)
    assert _ids(a.gallery) == _ids(b.gallery)


def test_different_seed_changes_queries() -> None:
    recs = _records()
    assert _ids(split_dataset(recs, seed=0).queries) != _ids(split_dataset(recs, seed=1).queries)


def test_partition_and_roles() -> None:
    recs = _records()
    s = split_dataset(recs, queries_per_instance=1, seed=0)
    q, g = set(_ids(s.queries)), set(_ids(s.gallery))
    assert not q & g
    assert q | g == {r["id"] for r in recs}
    assert len(s.queries) == 10
    assert all(it.label is not None for it in s.queries)
    assert {it.item_id for it in s.gallery if it.label is None} == {f"d_{d}" for d in range(5)}
    # every query has at least one correct gallery item
    gallery_labels = {it.label for it in s.gallery}
    assert all(it.label in gallery_labels for it in s.queries)


def test_queries_per_instance_is_capped() -> None:
    s = split_dataset(_records(n_instances=3, per_instance=2), queries_per_instance=5, seed=0)
    assert len(s.queries) == 3  # one image per instance must stay in the gallery


def test_extra_keys_survive_split() -> None:
    s = split_dataset(_records(), seed=0)
    assert all(it.group is not None for it in s.queries)


def test_singleton_instance_strict_and_lenient() -> None:
    recs = _records(n_instances=2) + [{"id": "lonely", "input_image": "x", "label": "solo"}]
    with pytest.raises(DatasetValidationError, match="fewer than 2"):
        split_dataset(recs)
    s = split_dataset(recs, strict=False)
    assert "lonely" not in _ids(s.queries) + _ids(s.gallery)


def test_validation_report() -> None:
    rep = validate_dataset(parse_dataset(_records(n_distractors=0)))
    assert rep.ok
    assert rep.n_instances == 10 and rep.n_distractors == 0
    assert any("No distractors" in w for w in rep.warnings)
    assert not validate_dataset(parse_dataset([{"input_image": "x", "label": None}])).ok
