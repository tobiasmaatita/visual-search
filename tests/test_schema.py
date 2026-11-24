"""Tests for the dataset schema: prompts, items and the match rule."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_search.data import Item, Prompt, is_match, parse_dataset


class TestPrompt:
    def test_box(self) -> None:
        p = Prompt.from_dict({"box": [1, 2, 30, 40]})
        assert p.kind == "box"
        assert p.box == (1.0, 2.0, 30.0, 40.0)
        assert p.to_dict() == {"box": [1.0, 2.0, 30.0, 40.0]}

    def test_points_default_positive(self) -> None:
        p = Prompt.from_dict({"points": [[10, 20], [30, 40]]})
        assert p.kind == "points"
        assert p.point_labels.tolist() == [1, 1]

    def test_points_with_labels_and_triplets(self) -> None:
        a = Prompt.from_dict({"points": [[10, 20], [30, 40]], "point_labels": [1, 0]})
        b = Prompt.from_dict({"points": [[10, 20, 1], [30, 40, 0]]})
        assert a.point_labels.tolist() == b.point_labels.tolist() == [1, 0]
        np.testing.assert_array_equal(a.points, b.points)
        assert Prompt.from_dict(a.to_dict()).point_labels.tolist() == [1, 0]

    def test_bbox_uses_positive_points_only(self) -> None:
        p = Prompt.from_dict({"points": [[10, 20, 1], [30, 50, 1], [500, 500, 0]]})
        assert p.bbox() == (10.0, 20.0, 30.0, 50.0)

    def test_mask(self) -> None:
        m = np.zeros((10, 10), bool)
        m[2:5, 3:7] = True
        p = Prompt.from_dict({"mask": m})
        assert p.kind == "mask"
        assert p.bbox() == (3.0, 2.0, 7.0, 5.0)

    @pytest.mark.parametrize(
        "d",
        [
            {},
            {"box": [0, 0, 1, 1], "points": [[0, 0]]},
            {"box": [10, 0, 5, 5]},
            {"box": [0, 0, 1]},
            {"points": [[0, 0]], "point_labels": [0]},
            {"points": [[0, 0]], "point_labels": [2]},
            {"points": []},
            {"mask": np.zeros((4, 4), bool)},
        ],
    )
    def test_invalid(self, d: dict) -> None:
        with pytest.raises(ValueError):
            Prompt.from_dict(d)


class TestItem:
    def test_extra_keys_preserved(self) -> None:
        it = Item.from_dict({"input_image": "x.jpg", "label": "a", "group": "g1", "foo": 3}, index=7)
        assert it.item_id == "item_000007"
        assert it.group == "g1"
        assert it.extra == {"group": "g1", "foo": 3}
        assert it.to_dict()["group"] == "g1"

    def test_explicit_id_and_prompt(self) -> None:
        it = Item.from_dict({"id": "q1", "input_image": "x.jpg", "label": None, "prompt": {"box": [0, 0, 5, 5]}}, 0)
        assert it.item_id == "q1"
        assert it.is_distractor
        assert it.prompt.kind == "box"

    def test_bad_label_type(self) -> None:
        with pytest.raises(ValueError):
            Item.from_dict({"input_image": "x.jpg", "label": 3}, 0)

    def test_missing_image(self) -> None:
        with pytest.raises(KeyError):
            Item.from_dict({"label": "a"}, 0)

    def test_load_image_variants(self, tmp_path: Path) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, (8, 6, 3), dtype=np.uint8)
        gray = rgb[..., 0]
        rgba = np.dstack([rgb, np.full((8, 6), 255, np.uint8)])
        Image.fromarray(rgb).save(tmp_path / "a.png")
        for src in (rgb, Image.fromarray(rgb), tmp_path / "a.png", str(tmp_path / "a.png"), rgba):
            out = Item("i", src).load_image()
            assert out.shape == (8, 6, 3) and out.dtype == np.uint8
            np.testing.assert_array_equal(out, rgb)
        assert Item("i", gray).load_image().shape == (8, 6, 3)
        with pytest.raises(ValueError):
            Item("i", rgb.astype(np.float32)).load_image()

    def test_load_image_applies_exif_orientation(self, tmp_path: Path) -> None:
        img = Image.fromarray(np.zeros((10, 20, 3), np.uint8))  # width 20, height 10
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90° clockwise on display
        img.save(tmp_path / "r.jpg", exif=exif)
        assert Item("i", tmp_path / "r.jpg").load_image().shape == (20, 10, 3)


def test_is_match_rule() -> None:
    a1, a2, b, n1, n2 = (Item(str(i), "x", lbl) for i, lbl in enumerate(["a", "a", "b", None, None]))
    assert is_match(a1, a2)
    assert not is_match(a1, b)
    assert not is_match(a1, n1)
    assert not is_match(n1, n2)  # distractors never match, not even each other


def test_parse_dataset_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_dataset([{"id": "x", "input_image": "a"}, {"id": "x", "input_image": "b"}])
