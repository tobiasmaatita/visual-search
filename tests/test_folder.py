"""Tests for the folder-based loaders and the mlcvtools annotation importer."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_search.data import (
    load_folder_dataset,
    load_image_folder,
    load_mlcvtools_annotations,
    split_dataset,
)


def _img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(path)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for rel in ["bank_01/a.png", "bank_01/b.jpg", "billy_01/c.png", "singletons/d.png", "_distractors/e.png", "f.png"]:
        _img(tmp_path / rel)
    (tmp_path / "empty_01").mkdir()
    (tmp_path / "bank_01/.DS_Store").write_bytes(b"x")
    (tmp_path / "bank_01/notes.txt").write_text("x")
    (tmp_path / "bank_01/g.HEIC").write_bytes(b"x")
    (tmp_path / "prompts.json").write_text(json.dumps({"bank_01/a.png": {"box": [0, 0, 2, 2]}, "nope.png": {}}))
    return tmp_path


def test_folder_layout(tree: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        recs = load_folder_dataset(tree)
    by_id = {r["id"]: r for r in recs}
    assert sorted(by_id) == ["_distractors/e.png", "bank_01/a.png", "bank_01/b.jpg", "billy_01/c.png",
                             "f.png", "singletons/d.png"]
    assert by_id["bank_01/a.png"]["label"] == "bank_01"
    assert by_id["bank_01/a.png"]["prompt"] == {"box": [0, 0, 2, 2]}
    assert "prompt" not in by_id["bank_01/b.jpg"]
    for rid in ["_distractors/e.png", "singletons/d.png", "f.png"]:
        assert by_id[rid]["label"] is None
    log = caplog.text
    assert "HEIC" in log and "empty" in log and "nope.png" in log and "directly in" in log


def test_folder_split_drops_singleton_instance(tree: Path) -> None:
    s = split_dataset(load_folder_dataset(tree), strict=False, seed=0)
    assert [it.item_id for it in s.queries] in (["bank_01/a.png"], ["bank_01/b.jpg"])
    assert all(it.label != "billy_01" for it in s.gallery)
    assert s.queries[0].load_image().shape == (4, 4, 3)


def test_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_folder_dataset(tmp_path / "nope")


def test_image_folder_limit_is_seeded(tmp_path: Path) -> None:
    for i in range(10):
        _img(tmp_path / f"{i}.png")
    a = load_image_folder(tmp_path, limit=4, seed=1)
    assert len(a) == 4
    assert a == load_image_folder(tmp_path, limit=4, seed=1)
    assert all(r["label"] is None and r["id"].startswith("distractors/") for r in a)


def test_mlcvtools_annotations(tmp_path: Path) -> None:
    _img(tmp_path / "imgs/bank1.png")
    _img(tmp_path / "imgs/room.png")
    ann = [
        {"image": "bank1.png", "bbox": [1, 2, 3, 4], "object_id": "bank_01"},
        {"image": "room.png", "bbox": [0, 0, 2, 2], "object_id": None},
        {"image": "room.png", "bbox": [1, 1, 3, 3], "object_id": "bank_01"},
    ]
    (tmp_path / "ann.json").write_text(json.dumps(ann))
    recs = load_mlcvtools_annotations(tmp_path / "ann.json", tmp_path / "imgs")
    assert [r["id"] for r in recs] == ["bank1.png", "room.png", "room.png#1"]
    assert [r["label"] for r in recs] == ["bank_01", None, "bank_01"]
    assert recs[0]["prompt"] == {"box": [1.0, 2.0, 3.0, 4.0]}

    (tmp_path / "bad.json").write_text(json.dumps([{"image": "../x.png", "bbox": [0, 0, 1, 1]}]))
    with pytest.raises(ValueError):
        load_mlcvtools_annotations(tmp_path / "bad.json", tmp_path / "imgs")
