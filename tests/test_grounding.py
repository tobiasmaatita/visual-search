"""Tests for DINOv2 gallery grounding, region-aware stage 2, gallery loading and new-gallery search."""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("module://ipympl.backend_nbagg", force=True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from test_dino_retrieval import FakeBackbone, real_backbone  # noqa: E402, F401

from visual_search.cascade import Cascade  # noqa: E402
from visual_search.data import Item, Prompt, load_gallery, make_synthetic_dataset, split_dataset  # noqa: E402
from visual_search.data.schema import parse_dataset  # noqa: E402
from visual_search.eval import evaluate  # noqa: E402
from visual_search.retrieval import GroundedDinoRetriever, localize  # noqa: E402
from visual_search.stages import SparseVerificationStage  # noqa: E402
from visual_search.ui import SearchApp  # noqa: E402


def _patches_with_blobs(blobs: list[tuple[slice, slice, float]], g: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Patch grid: background along e1, blobs mix in the query direction e0 with a given strength."""
    q = np.array([1.0, 0.0, 0.0], np.float32)
    patches = np.tile(np.array([0.0, 1.0, 0.0], np.float32), (g, g, 1))
    for rows, cols, strength in blobs:
        patches[rows, cols] = [strength, 1 - strength, 0.0]
    return q, patches


class TestLocalize:
    def test_single_blob_box(self) -> None:
        q, p = _patches_with_blobs([(slice(2, 4), slice(4, 7), 1.0)])
        regions = localize(q, p, (80, 160), pad_patches=0)
        assert len(regions) == 1
        assert regions[0].box == (80.0, 20.0, 140.0, 40.0)  # patches are 20 px wide, 10 px high
        assert regions[0].n_patches == 6 and regions[0].score == pytest.approx(1.0)

    def test_two_objects_and_weak_blob(self) -> None:
        q, p = _patches_with_blobs([(slice(0, 2), slice(0, 2), 1.0), (slice(5, 7), slice(5, 7), 0.97),
                                    (slice(0, 1), slice(6, 7), 0.5)])
        regions = localize(q, p, (80, 80), max_regions=3, delta=0.05, region_delta=0.1, pad_patches=0)
        assert [r.n_patches for r in regions] == [4, 4]  # the weak blob is dropped
        assert regions[0].box == (0.0, 0.0, 20.0, 20.0)
        assert len(localize(q, p, (80, 80), max_regions=1, delta=0.05)) == 1

    def test_padding_is_clipped(self) -> None:
        q, p = _patches_with_blobs([(slice(0, 1), slice(0, 1), 1.0)])
        assert localize(q, p, (80, 80), pad_patches=2)[0].box == (0.0, 0.0, 30.0, 30.0)


def _scene(squares: list[tuple[int, int]], color=(255, 0, 0), size: int = 80) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (0, 0, 255)
    for y, x in squares:
        img[y : y + 20, x : x + 20] = color
    return img


class TestGroundedRetriever:
    def test_regions_per_object_and_box_items(self) -> None:
        gallery = [
            Item("two_red", _scene([(0, 0), (60, 60)])),
            Item("blue_only", _scene([])),
            Item("boxed", _scene([(40, 40)]), prompt=Prompt(box=(0, 0, 20, 20))),  # box on a blue corner
        ]
        r = GroundedDinoRetriever(FakeBackbone(), cache_dir=None, pad_patches=0)
        r.fit(gallery)
        query = Item("q", _scene([(20, 20)]), prompt=Prompt(box=(20, 20, 40, 40)))
        res = r.search(query, 10)
        two_red = [c for c in res if c.item_id == "two_red"]
        assert len(two_red) == 2  # one candidate per red square
        assert {c.region for c in two_red} == {(0.0, 0.0, 20.0, 20.0), (60.0, 60.0, 80.0, 80.0)}
        assert res[0].item_id == "two_red" and res[0].score == pytest.approx(1.0, abs=1e-3)
        boxed = [c for c in res if c.item_id == "boxed"]
        assert len(boxed) == 1 and boxed[0].region == (0, 0, 20, 20)  # its own box, not localized
        assert all({"cosine", "peak", "region_patches"} <= set(c.features) for c in res)
        assert len(r.search(query, 2)) == 2

    def test_requires_fit(self) -> None:
        with pytest.raises(RuntimeError):
            GroundedDinoRetriever(FakeBackbone()).search(Item("q", _scene([])), 3)


def test_sparse_stage_matches_inside_region() -> None:
    recs = make_synthetic_dataset(n_instances=2, views_per_instance=2, n_distractors=0, seed=0)
    q, c = parse_dataset(recs)[0], dataclasses.replace(parse_dataset(recs)[1], prompt=None)
    region = c.extra["gt_mask"].nonzero()
    box = (float(region[1].min()), float(region[0].min()), float(region[1].max()), float(region[0].max()))

    class FixedRegion:
        """Stage-1 stand-in returning the candidate with a chosen region."""

        name = "fixed"

        def fit(self, gallery):
            pass

        def search(self, query, k):
            from visual_search.retrieval import Candidate

            return [Candidate(c.item_id, 0.5, region=box)]

    cascade = Cascade(FixedRegion(), stages=[SparseVerificationStage("sift", top_k=5)])  # type: ignore[arg-type]
    cascade.prepare([c])
    cand = cascade.run(q).final[0]
    pts = cand.artifacts["correspondences"].pts_c
    assert cand.region == box and len(pts) > 0
    assert (pts[:, 0] >= box[0] - 1).all() and (pts[:, 0] <= box[2] + 1).all()
    assert cand.features["inliers"] > 0


def test_load_gallery(tmp_path: Path) -> None:
    for rel in ["a.jpg", "sub/b.png", "sub/c.txt", ".hidden.jpg"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".txt") or rel.startswith("."):
            p.write_text("x")
        else:
            Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(p)
    assert [r["id"] for r in load_gallery(tmp_path)] == ["a.jpg", "sub/b.png"]
    assert [r["id"] for r in load_gallery(tmp_path, recursive=False)] == ["a.jpg"]
    assert len(load_gallery(str(tmp_path / "**" / "*.png"))) == 1
    assert len(load_gallery([tmp_path / "a.jpg", str(tmp_path / "sub/b.png")])) == 2
    assert all(r["label"] is None for r in load_gallery(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_gallery([tmp_path / "nope.jpg"])
    with pytest.raises(FileNotFoundError):
        load_gallery(str(tmp_path / "*.gif"))


def test_search_app_on_new_gallery(tmp_path: Path) -> None:
    gallery = [Item("two_red", _scene([(0, 0), (60, 60)])), Item("blue_only", _scene([]))]
    cascade = Cascade(GroundedDinoRetriever(FakeBackbone(), pad_patches=0))
    app = SearchApp(cascade, gallery, top_k=12, show_expected_label=False)
    assert app.expected_label.layout.display == "none"
    path = tmp_path / "q.png"
    Image.fromarray(_scene([(20, 20)])).save(path)
    app.load_path(path)
    app.prompt_widget.set_prompt(Prompt(box=(20, 20, 40, 40)))
    res = app.search()
    assert res.final[0].region is not None
    assert app._thumb(res.final[0]).shape[:2] != (80, 80)  # a crop around the region
    assert "region" in app.detail_select.options[0][0]
    assert app.results.value[:4] == b"\x89PNG"  # 12 results wrap over two grid rows


def test_real_dinov2_localizes_synthetic_objects(real_backbone, tmp_path: Path) -> None:  # noqa: F811
    split = split_dataset(make_synthetic_dataset(n_instances=6, n_distractors=10, seed=0), seed=0)
    gt = {it.item_id: it.prompt.bbox() for it in split.gallery if it.prompt}
    gallery = [dataclasses.replace(it, prompt=None) for it in split.gallery]
    r = GroundedDinoRetriever(real_backbone, cache_dir=tmp_path)
    rep = evaluate(Cascade(r), dataclasses.replace(split, gallery=gallery))
    assert rep.metrics.loc["retrieval", "MRR"] > 0.3

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)

    ious = []
    for q in split.queries:
        best = next(c for c in r.search(q, 200) if c.item_id in gt and c.item_id.split("/")[1] == q.label)
        ious.append(iou(best.region, gt[best.item_id]))
    assert np.median(ious) > 0.3
