"""Tests for the smoke-test synthetic dataset."""

import cv2
import numpy as np
import pytest

from visual_search.data import Prompt, make_synthetic_dataset, split_dataset


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return make_synthetic_dataset(n_instances=4, views_per_instance=3, n_distractors=5, n_near_duplicates=2, seed=0)


def test_counts_and_split(records: list[dict]) -> None:
    assert len(records) == (4 + 2) * 3 + 5
    s = split_dataset(records, seed=0)
    assert len(s.queries) == 6
    assert sum(it.is_distractor for it in s.gallery) == 5


def test_deterministic(records: list[dict]) -> None:
    again = make_synthetic_dataset(n_instances=4, views_per_instance=3, n_distractors=5, n_near_duplicates=2, seed=0)
    for a, b in zip(records, again):
        assert a["id"] == b["id"]
        np.testing.assert_array_equal(a["input_image"], b["input_image"])
    other = make_synthetic_dataset(n_instances=4, views_per_instance=3, n_distractors=5, n_near_duplicates=2, seed=1)
    assert not np.array_equal(records[0]["input_image"], other[0]["input_image"])


def test_masks_and_prompts(records: list[dict]) -> None:
    for r in records:
        img = r["input_image"]
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
        if r["label"] is None:
            assert "gt_mask" not in r and "prompt" not in r
            continue
        mask = r["gt_mask"]
        assert mask.shape == img.shape[:2]
        assert mask.mean() > 0.02  # object is clearly visible
        assert Prompt.from_dict(r["prompt"]).bbox() == Prompt(mask=mask).bbox()


def test_homography_maps_texture_corner_into_mask(records: list[dict]) -> None:
    r = records[0]
    center = cv2.perspectiveTransform(np.float32([[[96, 96]]]), r["gt_homography"])[0, 0]
    x, y = int(round(center[0])), int(round(center[1]))
    assert r["gt_mask"][y, x]


def test_near_duplicates_share_group(records: list[dict]) -> None:
    groups = {r["label"]: r.get("group") for r in records if r["label"]}
    assert groups["syn_00"] == groups["syn_00_dup"] == "group_00"
    assert groups["syn_03"] is None


def test_base_images(records: list[dict]) -> None:
    base = [np.full((100, 120, 3), 200, np.uint8)]
    recs = make_synthetic_dataset(n_instances=2, views_per_instance=2, n_distractors=0, base_images=base, seed=0)
    assert len(recs) == 4


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        make_synthetic_dataset(views_per_instance=1)
    with pytest.raises(ValueError):
        make_synthetic_dataset(n_instances=1, n_near_duplicates=2)
