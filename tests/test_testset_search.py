"""Tests for searching and evaluating a labelled test set (annotations.json): per-box items."""

from pathlib import Path

import matplotlib

matplotlib.use("module://ipympl.backend_nbagg", force=True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from test_dino_retrieval import FakeBackbone  # noqa: E402

from visual_search.cascade import Cascade  # noqa: E402
from visual_search.data import AnnotationSet, BoxAnnotation, make_synthetic_dataset, split_dataset  # noqa: E402
from visual_search.data.schema import parse_dataset  # noqa: E402
from visual_search.eval import evaluate  # noqa: E402
from visual_search.retrieval import ColorHistogramRetriever, DinoRetriever  # noqa: E402
from visual_search.stages import SparseVerificationStage  # noqa: E402
from visual_search.ui import SearchApp  # noqa: E402
from visual_search.viz.gallery import box_thumbnail  # noqa: E402


@pytest.fixture(scope="module")
def testset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A labelled test set on disk: synthetic views with their object boxes, plus distractor images."""
    root = tmp_path_factory.mktemp("testset")
    ann = AnnotationSet(root)
    for r in make_synthetic_dataset(n_instances=4, views_per_instance=3, n_distractors=3, seed=0):
        name = r["id"].replace("/", "_") + ".png"
        Image.fromarray(r["input_image"]).save(root / name)
        if r["label"] is None:
            ann.mark_reviewed(name)
        else:
            ann.add(name, BoxAnnotation(tuple(r["prompt"]["box"]), r["label"]))
            ann.add(name, BoxAnnotation((0, 0, 40, 40), None))  # an unlabelled corner box: a distractor object
    ann.save()
    return root


@pytest.fixture(scope="module")
def items(testset: Path) -> list:
    return parse_dataset(AnnotationSet.load(testset).to_records())


def test_records_are_per_box(items: list) -> None:
    assert len(items) == 4 * 3 * 2 + 3
    boxed = [it for it in items if it.prompt is not None]
    assert len(boxed) == 24 and all(it.prompt.kind == "box" for it in boxed)
    assert sum(it.label is None for it in items) == 12 + 3


def test_crop_descriptors_cached_per_box(items: list, tmp_path: Path) -> None:
    boxed = [it for it in items if it.prompt is not None][:6]
    r1 = DinoRetriever(FakeBackbone(), pooling="crop_cls", cache_dir=tmp_path, batch_size=4)
    r1.fit(boxed)
    assert r1.n_forward == 6 and r1.backbone.calls == 2  # batched crops
    r2 = DinoRetriever(FakeBackbone(), pooling="crop_cls", cache_dir=tmp_path)
    r2.fit(boxed)
    assert r2.n_forward == 0
    np.testing.assert_allclose(r1._index.embeddings, r2._index.embeddings)
    same_image = [it for it in boxed if it.extra["image_id"] == boxed[0].extra["image_id"]]
    assert len(same_image) == 2  # two boxes on one image get different descriptors
    d = r2.crop_descriptors(same_image)
    assert not np.allclose(d[0], d[1])


def test_cascade_excludes_query_itself(items: list) -> None:
    cascade = Cascade(ColorHistogramRetriever())
    cascade.prepare(items)
    q = items[0]
    assert cascade.run(q).final[0].item_id == q.item_id  # without exclusion it finds itself
    res = cascade.run(q, exclude={q.item_id})
    assert q.item_id not in [c.item_id for c in res.final] and len(res.final) == len(items) - 1


def test_per_box_evaluation(items: list) -> None:
    split = split_dataset(items, seed=0)
    assert len(split.queries) == 4 and all(q.prompt is not None for q in split.queries)
    stage = SparseVerificationStage("sift", top_k=20)
    rep = evaluate(Cascade(ColorHistogramRetriever(), stages=[stage]), split)
    assert len(rep.per_query) == 4 and rep.per_query["query_id"].str.contains("#").all()
    assert rep.metrics.loc["sift-H-magsac", "R@1"] == 1.0


def test_search_app_on_testset(items: list, tmp_path: Path) -> None:
    queries = [it for it in items if it.label]
    cascade = Cascade(ColorHistogramRetriever(), stages=[SparseVerificationStage("sift", top_k=10)])
    app = SearchApp(cascade, items, queries, top_k=3)
    q = queries[0]
    assert app.query is q and app.expected_label.value == q.label
    assert app.prompt_widget.prompt.box == q.prompt.box
    assert set(app.expected_label.options) == {"syn_00", "syn_01", "syn_02", "syn_03"}
    res = app.search()
    ids = [c.item_id for c in res.final]
    assert q.item_id not in ids  # the query box never finds itself
    gallery = {it.item_id: it for it in items}
    assert gallery[ids[0]].label == q.label  # SIFT verification puts the same object first
    assert "expected label: syn_00" in app.info.value
    assert app._thumb(res.final[0]).shape[0] <= 256  # box thumbnail

    # A new image: correctness only when an expected label is given.
    path = tmp_path / "new.png"
    Image.fromarray(q.load_image()).save(path)
    app.load_path(path)
    assert app.expected_label.value == ""
    app.prompt_widget.set_prompt(q.prompt)
    app.search()
    assert "no expected label" in app.info.value
    app.expected_label.value = q.label
    app.search()
    assert f"expected label: {q.label}" in app.info.value


def test_box_thumbnail() -> None:
    img = np.zeros((400, 600, 3), np.uint8)
    t = box_thumbnail(img, (100, 100, 200, 150), pad=0.5, max_side=256)
    assert t.shape[:2] == (100, 200) and t.any()  # crop 200x100 incl. context, box drawn
