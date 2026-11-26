"""Tests for judges, leave-one-out evaluation, grounding evaluation and the notebook helpers."""

from pathlib import Path

import matplotlib

matplotlib.use("module://ipympl.backend_nbagg", force=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from test_dino_retrieval import FakeBackbone  # noqa: E402

from visual_search.cascade import Cascade  # noqa: E402
from visual_search.data import AnnotationSet, BoxAnnotation, Item, make_synthetic_dataset  # noqa: E402
from visual_search.data.schema import parse_dataset  # noqa: E402
from visual_search.eval import (  # noqa: E402
    GroundedJudge,
    GroupJudge,
    ItemJudge,
    box_iou,
    default_judges,
    evaluate_leave_one_out,
    run_evaluation,
)
from visual_search.eval.inspect import (  # noqa: E402
    failure_grid,
    latency_table,
    localization_stats,
    per_label_table,
    save_tables,
    worst_queries,
)
from visual_search.retrieval import Candidate, ColorHistogramRetriever, DinoRetriever, GroundedDinoRetriever  # noqa: E402
from visual_search.stages import SparseVerificationStage  # noqa: E402


@pytest.fixture(scope="module")
def testset(tmp_path_factory: pytest.TempPathFactory) -> AnnotationSet:
    """Synthetic test set on disk: 4 items x 3 views, 2 look-alike pairs sharing a group, 3 distractors."""
    root = tmp_path_factory.mktemp("evalset")
    ann = AnnotationSet(root)
    recs = make_synthetic_dataset(n_instances=4, views_per_instance=3, n_distractors=3, n_near_duplicates=2, seed=0)
    for r in recs:
        name = r["id"].replace("/", "_") + ".png"
        Image.fromarray(r["input_image"]).save(root / name)
        if r["label"] is None:
            ann.mark_reviewed(name)
        else:
            ann.add(name, BoxAnnotation(tuple(r["prompt"]["box"]), r["label"], r.get("group")))
    ann.save()
    return ann


@pytest.fixture(scope="module")
def items(testset: AnnotationSet) -> list:
    return parse_dataset(testset.to_records())


class TestJudges:
    def test_item_and_group(self) -> None:
        a = Item("a", "x", "sofa_1", extra={"group": "model_x"})
        b = Item("b", "x", "sofa_2", extra={"group": "model_x"})
        c = Item("c", "x", "chair_1")
        d = Item("d", "x", None)
        cand = Candidate("b", 1.0)
        assert not ItemJudge().correct(a, b, cand) and GroupJudge().correct(a, b, cand)
        assert not GroupJudge().correct(a, c, cand) and not GroupJudge().correct(d, d, cand)
        assert GroupJudge().correct(c, Item("e", "x", "chair_1"), cand)  # no group -> label stands in
        assert GroupJudge().has_positive(a, [b, c]) and not ItemJudge().has_positive(a, [b, c])

    def test_box_iou(self) -> None:
        assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
        assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
        assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0

    def test_grounded(self) -> None:
        boxes = {"img1": [((0, 0, 100, 100), "sofa_1", "model_x"), ((200, 0, 300, 100), "lamp", None)]}
        q = Item("q", "x", "sofa_1", extra={"group": "model_x", "image_id": "img0"})
        g = Item("img1", "x", None, extra={"image_id": "img1"})
        j = GroundedJudge(boxes, iou_threshold=0.3)
        assert j.correct(q, g, Candidate("img1", 1, region=(0, 0, 90, 100)))
        assert not j.correct(q, g, Candidate("img1", 1, region=(200, 0, 300, 100)))  # the lamp
        assert not j.correct(q, g, Candidate("img1", 1, region=(0, 0, 30, 30)))  # IoU 0.09
        assert j.correct(q, g, Candidate("img1", 1))  # no region: the image contains it
        q2 = Item("q2", "x", "sofa_2", extra={"group": "model_x"})
        assert not j.correct(q2, g, Candidate("img1", 1)) and GroundedJudge(boxes, level="model").correct(
            q2, g, Candidate("img1", 1, region=(0, 0, 100, 100)))
        assert j.has_positive(q, [g]) and not j.has_positive(q, [Item("img9", "x", None)])
        with pytest.raises(ValueError):
            GroundedJudge(boxes, level="x")


def test_leave_one_out_two_levels(items: list) -> None:
    cascade = Cascade(ColorHistogramRetriever(), stages=[SparseVerificationStage("sift", top_k=10)])
    rep = evaluate_leave_one_out(cascade, items, judges=default_judges())
    labelled = [it for it in items if it.label]
    assert rep.metrics.loc["sift-H-magsac", "n_queries"] == len(labelled) == 18  # every box is a query once
    assert {"R@1", "R@5", "MRR", "model R@1", "model R@5", "model MRR"} <= set(rep.metrics.columns)
    assert rep.levels == ("item", "model") and rep.ks == (1, 5)
    # model level is never stricter than item level
    assert (rep.metrics["model MRR"] >= rep.metrics["MRR"] - 1e-9).all()
    # SIFT ignores color, so it sometimes picks the hue-shifted twin: wrong item, right model
    assert rep.metrics.loc["sift-H-magsac", "model R@1"] == 1.0
    assert 0.7 <= rep.metrics.loc["sift-H-magsac", "R@1"] < 1.0
    # a query never finds anything from its own photo
    image_of = {it.item_id: it.extra["image_id"] for it in items}
    for _, row in rep.per_query.iterrows():
        assert image_of[row["top1@retrieval"]] != image_of[row["query_id"]]
    assert {"model rank@retrieval", "group"} <= set(rep.per_query.columns)


def test_skips_unanswerable_queries(items: list) -> None:
    lonely = items + [Item("lonely.png#0", items[0].image, "solo", items[0].prompt, {"image_id": "lonely.png"})]
    rep = evaluate_leave_one_out(Cascade(ColorHistogramRetriever()), lonely)
    assert rep.n_skipped == 1


def test_grounding_evaluation(testset: AnnotationSet, items: list, tmp_path: Path) -> None:
    images = parse_dataset(testset.image_records())
    boxes = testset.boxes_by_image()
    queries = [it for it in items if it.label]
    assert len(images) == 6 * 3 + 3 and all(it.prompt is None for it in images)
    judges = {"item": GroundedJudge(boxes, 0.3), "model": GroundedJudge(boxes, 0.3, level="model")}
    grounded = GroundedDinoRetriever(FakeBackbone(), cache_dir=tmp_path)
    rep = run_evaluation(Cascade(grounded), queries, images, judges)
    assert rep.metrics.loc["retrieval", "n_queries"] == 18
    whole = run_evaluation(Cascade(DinoRetriever(FakeBackbone(), cache_dir=tmp_path)), queries, images, judges)
    assert whole.metrics.loc["retrieval", "n_queries"] == 18
    stats = localization_stats(grounded, queries, images, judges["item"])
    assert set(stats) == {"queries", "median_iou", "share_iou_0.3", "share_iou_0.5"} and stats["queries"] == 18
    assert 0.0 <= stats["median_iou"] <= 1.0


def test_inspection_helpers(items: list, tmp_path: Path) -> None:
    cascade = Cascade(ColorHistogramRetriever())
    rep = evaluate_leave_one_out(cascade, items)
    labels = per_label_table(rep)
    assert set(labels.index) == {"syn_00", "syn_01", "syn_02", "syn_03", "syn_00_dup", "syn_01_dup"}
    assert {"queries", "R@1", "R@5", "MRR", "median rank", "not found"} <= set(labels.columns)
    assert labels["queries"].sum() == 18 and labels["MRR"].is_monotonic_increasing
    worst = worst_queries(rep, n=3)
    assert len(worst) == 3
    q = next(it for it in items if it.item_id == worst[0])
    png = failure_grid(cascade, q, items, ItemJudge(), k=4)
    assert png[:4] == b"\x89PNG"
    lat = latency_table([rep])
    assert {"retrieval median_ms", "total p95_ms"} <= set(lat.columns)
    run_dir = save_tables({"metrics": rep.metrics, "per label": labels}, tmp_path / "results", "my run!")
    assert run_dir.name.endswith("_my-run") and sorted(p.name for p in run_dir.iterdir()) == ["metrics.csv",
                                                                                            "per-label.csv"]
    assert pd.read_csv(run_dir / "metrics.csv").shape[0] == 1
    assert np.isfinite(labels["MRR"]).all()
