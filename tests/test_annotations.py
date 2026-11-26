"""Tests for the per-object annotation format and the labeling app."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("module://ipympl.backend_nbagg", force=True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from PIL import Image  # noqa: E402

from visual_search.data import AnnotationSet, BoxAnnotation, similar_labels, split_dataset  # noqa: E402
from visual_search.data.annotations import normalize_label  # noqa: E402
from visual_search.ui import LabelingApp  # noqa: E402


def _img(path: Path, h: int = 60, w: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), 120, np.uint8)).save(path)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for rel in ["a.jpg", "b.png", "sub/c.jpg", "d.jpg"]:
        _img(tmp_path / rel)
    (tmp_path / ".hidden.jpg").write_bytes(b"x")
    return tmp_path


class TestLabels:
    def test_normalize_and_similar(self) -> None:
        assert normalize_label("Bank_01") == normalize_label("bank-01") == normalize_label(" bank 01 ") == "bank01"
        assert similar_labels("Bank_01", ["bank_01", "billy_01", "chair"]) == ["bank_01"]
        assert "bank_01" in similar_labels("bank_1", ["bank_01", "chair"])
        assert similar_labels("bank_01", ["bank_01"]) == []  # the label itself is not "similar"
        assert similar_labels("lamp", ["bank_01", "chair"]) == []
        assert similar_labels("___", ["bank"]) == []
        assert similar_labels("bank_02", ["bank_01", "Bank_02"]) == ["Bank_02"]  # other numbers are other items
        assert similar_labels("chiar_01", ["chair_01", "chair_02"]) == ["chair_01"]  # typo
        assert similar_labels("bank_1", ["bank_01"]) == ["bank_01"]  # numbers compare by value


class TestAnnotationSet:
    def test_roundtrip_backup_and_reviewed(self, root: Path) -> None:
        ann = AnnotationSet.load(root)
        assert ann.image_paths() == ["a.jpg", "b.png", "d.jpg", "sub/c.jpg"]
        assert not ann.images and not ann.path.exists()
        ann.add("a.jpg", BoxAnnotation((1, 2, 30, 40), " bank_01 ", "sofa_x"))
        ann.add("a.jpg", BoxAnnotation((5, 5, 20, 20), "lamp"))
        ann.add("sub/c.jpg", BoxAnnotation((0, 0, 10, 10), "bank_01", ""))
        ann.mark_reviewed("b.png")
        ann.save()
        ann.relabel("a.jpg", 1, "lamp_02")
        ann.save()
        assert ann.path.with_name("annotations.json.bak").exists()
        data = json.loads(ann.path.read_text())
        assert data["version"] == 1 and data["images"]["b.png"] == []
        assert data["images"]["a.jpg"][0] == {"box": [1.0, 2.0, 30.0, 40.0], "label": "bank_01", "group": "sofa_x"}
        assert "group" not in data["images"]["sub/c.jpg"][0]  # empty group is dropped
        again = AnnotationSet.load(root)
        assert again.labels() == ["bank_01", "lamp_02"] and again.groups() == ["sofa_x"]
        assert again.is_reviewed("b.png") and not again.is_reviewed("d.jpg")
        assert "3/4 images reviewed, 3 boxes, 2 labels, 1 groups" == again.summary()
        with pytest.raises(ValueError):
            again.unmark_reviewed("a.jpg")
        again.unmark_reviewed("b.png")
        assert not again.is_reviewed("b.png")

    def test_invalid(self, root: Path) -> None:
        with pytest.raises(ValueError):
            BoxAnnotation((10, 0, 5, 5), "x")
        with pytest.raises(FileNotFoundError):
            AnnotationSet.load(root / "nope")
        (root / "annotations.json").write_text(json.dumps({"version": 99, "images": {}}))
        with pytest.raises(ValueError):
            AnnotationSet.load(root)

    def test_stats_and_records(self, root: Path) -> None:
        ann = AnnotationSet(root)
        ann.add("a.jpg", BoxAnnotation((0, 0, 10, 10), "bank_01", "sofa_x"))
        ann.add("sub/c.jpg", BoxAnnotation((0, 0, 20, 20), "bank_01"))
        ann.add("sub/c.jpg", BoxAnnotation((30, 30, 50, 50), "Bank_01"))
        ann.add("sub/c.jpg", BoxAnnotation((1, 1, 5, 5), None))
        ann.mark_reviewed("b.png")
        stats = ann.label_stats()
        assert stats.loc["bank_01", "boxes"] == 2 and stats.loc["bank_01", "images"] == 2
        assert stats.loc["bank_01", "group"] == "sofa_x"
        warning = stats.loc["Bank_01", "warning"]
        assert "only 1 box" in warning and "similar to bank_01" in warning
        recs = ann.to_records()
        by_id = {r["id"]: r for r in recs}
        assert sorted(by_id) == ["a.jpg#0", "b.png", "sub/c.jpg#0", "sub/c.jpg#1", "sub/c.jpg#2"]
        assert by_id["a.jpg#0"]["prompt"] == {"box": [0.0, 0.0, 10.0, 10.0]} and by_id["a.jpg#0"]["group"] == "sofa_x"
        assert by_id["b.png"]["label"] is None and "prompt" not in by_id["b.png"]
        assert by_id["sub/c.jpg#2"]["label"] is None  # unlabelled box = distractor object
        assert "d.jpg" not in by_id  # never reviewed -> not in the dataset
        assert len(ann.to_records(include_distractor_images=False)) == 4
        split = split_dataset(recs, seed=0, strict=False)  # 'Bank_01' (1 box) is dropped
        assert [q.label for q in split.queries] == ["bank_01"]
        assert all(it.extra["image_id"] for it in split.gallery)


def _drag(pw, start, end) -> None:
    for name, xy in [("button_press_event", start), ("motion_notify_event", end), ("button_release_event", end)]:
        pw.figure.canvas.draw()
        x, y = pw._ax.transData.transform(xy)
        MouseEvent(name, pw.figure.canvas, x, y, button=1)._process()


class TestLabelingApp:
    def test_label_flow(self, root: Path) -> None:
        app = LabelingApp(root)
        assert app.current == "a.jpg" and app.prompt_widget.mode.layout.display == "none"
        assert not app.add_box()  # nothing drawn yet
        _drag(app.prompt_widget, (5, 5), (40, 30))
        assert not app.add_box()  # no label yet
        app.label_input.value = "bank_01"
        app.group_input.value = "sofa_x"
        assert app.add_box()
        assert app.prompt_widget.prompt is None  # drawn box cleared after adding
        saved = AnnotationSet.load(root)
        assert saved.boxes("a.jpg")[0].label == "bank_01" and saved.boxes("a.jpg")[0].group == "sofa_x"
        assert app.label_input.options == ("bank_01",) and app.group_input.options == ("sofa_x",)
        assert "bank_01" in app.stats.value and app.reviewed.disabled
        assert len(app.prompt_widget._box_artists) == 2  # rectangle + text

        # A near-duplicate label needs a second press.
        app.show_image("b.png")
        _drag(app.prompt_widget, (10, 10), (50, 50))
        app.label_input.value = "Bank_01"
        assert "similar" in app.label_info.value
        assert not app.add_box() and app.add_button.description == "Add anyway"
        assert app.add_box()
        assert AnnotationSet.load(root).labels() == ["Bank_01", "bank_01"]

        # Relabel and delete via the list.
        app.box_list.value = 0
        assert app.label_input.value == "Bank_01"
        app.label_input.value = "bank_01"
        app.relabel_selected()
        assert AnnotationSet.load(root).boxes("b.png")[0].label == "bank_01"
        app.box_list.value = 0
        app.delete_selected()
        assert AnnotationSet.load(root).boxes("b.png") == [] and AnnotationSet.load(root).is_reviewed("b.png")

    def test_navigation_and_reviewed(self, root: Path) -> None:
        AnnotationSet(root, {"a.jpg": [BoxAnnotation((0, 0, 9, 9), "x")]}).save()
        app = LabelingApp(root)
        assert app.current == "b.png"  # starts at the first unlabelled image
        app.reviewed.value = True
        assert AnnotationSet.load(root).is_reviewed("b.png")
        app.next_unlabelled()
        assert app.current == "d.jpg"
        app._step(1)
        assert app.current == "sub/c.jpg"
        app._step(1)
        assert app.current == "a.jpg" and app.reviewed.value and app.reviewed.disabled
        app.image_select.value = "b.png"
        assert app.current == "b.png"
        app.reviewed.value = False
        assert not AnnotationSet.load(root).is_reviewed("b.png")
        assert app.image_select.options[0][0].startswith("✓ a.jpg (1)")

    def test_empty_directory(self, tmp_path: Path) -> None:
        app = LabelingApp(tmp_path)
        assert "No images" in app.status.value and app.current is None
        app.next_unlabelled()
