"""Tests for matchers (shared contract), geometric verification and the sparse re-ranking stage."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from visual_search.cascade import Cascade
from visual_search.data import make_synthetic_dataset, split_dataset
from visual_search.eval import evaluate
from visual_search.geometry import verify
from visual_search.matching import Correspondences, Matcher, available_matchers, create_matcher, register_matcher
from visual_search.retrieval import ColorHistogramRetriever
from visual_search.stages import SparseVerificationStage
from visual_search.viz.pairs import render_pair_detail


def _learned_available() -> bool:
    """True if kornia is installed and the learned-matcher weights are in the torch hub cache."""
    import importlib.util

    ckpt = Path(torch_hub_dir()) / "checkpoints"
    names = ["aliked-n16.pth", "aliked_lightglue_v0-1_arxiv-pth", "xfeat.pt", "xfeat-lighterglue.pt"]
    return importlib.util.find_spec("kornia") is not None and all((ckpt / n).exists() for n in names)


def torch_hub_dir() -> str:
    import torch

    return torch.hub.get_dir()


_needs_learned = pytest.mark.skipif(not _learned_available(), reason="needs the 'sparse' extra and downloaded weights")
# Every matcher must pass the contract test; learned ones only run when their weights are present.
MATCHERS = ["sift", pytest.param("aliked-lightglue", marks=_needs_learned),
            pytest.param("xfeat-lighterglue", marks=_needs_learned)]


@pytest.fixture(scope="module")
def matchers() -> dict:
    """One instance per matcher name (learned models are slow to load)."""
    return {}


def _get(matchers: dict, name: str):
    if name not in matchers:
        matchers[name] = create_matcher(name)
    return matchers[name]


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return make_synthetic_dataset(n_instances=6, views_per_instance=3, n_distractors=8, seed=0)


def _view(records: list[dict], label: str, v: int) -> dict:
    return next(r for r in records if r["label"] == label and r["id"].endswith(f"view_{v:02d}"))


@pytest.mark.parametrize("name", MATCHERS)
def test_matcher_contract(name: str, records: list[dict], matchers: dict) -> None:
    """Every registered matcher: shapes, pixel frame, mask restriction, determinism."""
    m = _get(matchers, name)
    assert isinstance(m, Matcher) and m.name == name and isinstance(m.identity, dict)
    img = _view(records, "syn_00", 0)["input_image"]
    big = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_LINEAR)
    f = m.extract(big)
    h, w = big.shape[:2]
    assert f.image_hw == (h, w) and f.keypoints.shape == (len(f), 2) and len(f.descriptors) == len(f)
    assert (f.keypoints >= -1).all() and (f.keypoints[:, 0] <= w) .all() and (f.keypoints[:, 1] <= h).all()
    # Matching an image with a 3x upscaled copy: points must agree in original pixels.
    corr = m.match(img, big)
    assert len(corr) > 50
    diff = corr.pts_c - (corr.pts_q * 3 + 1)
    assert np.abs(np.median(diff, axis=0)).max() < 1.5  # no systematic offset: same pixel convention
    assert np.median(np.linalg.norm(diff, axis=1)) < 5.0  # XFeat localizes on a stride-8 grid
    # Mask restriction.
    mask = np.zeros(img.shape[:2], bool)
    mask[:, : img.shape[1] // 2] = True
    cm = m.match(img, img, mask_q=mask)
    assert len(cm) > 0 and (cm.pts_q[:, 0] < img.shape[1] / 2 + 1).all()
    assert cm.n_query_kpts < corr.n_query_kpts or cm.n_query_kpts <= len(m.extract(img))
    # Determinism.
    np.testing.assert_array_equal(m.match(img, big).pts_q, corr.pts_q)


def _gt_cand_to_query(q: dict, c: dict) -> np.ndarray:
    return q["gt_homography"] @ np.linalg.inv(c["gt_homography"])


def _corner_error(H_est: np.ndarray, H_gt: np.ndarray, c: dict) -> float:
    ys, xs = np.nonzero(c["gt_mask"])
    pts = np.float64([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]])
    a = cv2.perspectiveTransform(pts[None], H_est)[0]
    b = cv2.perspectiveTransform(pts[None], H_gt)[0]
    return float(np.linalg.norm(a - b, axis=1).max())


@pytest.mark.parametrize("name", MATCHERS)
@pytest.mark.parametrize("method", ["magsac", "ransac"])
def test_matcher_recovers_known_homography(records: list[dict], method: str, name: str, matchers: dict) -> None:
    m = _get(matchers, name)
    # SIFT is rotation invariant and sub-pixel accurate; ALIKED-n16 is not rotation invariant (the synthetic
    # views rotate up to 35 degrees) and XFeat is coarser, so they get a looser bound and may miss one pair.
    max_err, min_ok = {"sift": (6.0, 3)}.get(name, (10.0, 2))
    n_ok = 0
    for label in ["syn_00", "syn_01", "syn_02"]:
        q, c = _view(records, label, 0), _view(records, label, 1)
        corr = m.match(q["input_image"], c["input_image"], q["gt_mask"], c["gt_mask"])
        v = verify(corr, "H", method, threshold=4.0)
        n_ok += v.ok and v.n_inliers >= 15 and _corner_error(v.model, _gt_cand_to_query(q, c), c) < max_err
    assert n_ok >= min_ok


@pytest.mark.parametrize("name", MATCHERS)
def test_different_instances_get_few_inliers(records: list[dict], name: str, matchers: dict) -> None:
    m = _get(matchers, name)
    q, same, other = _view(records, "syn_00", 0), _view(records, "syn_00", 1), _view(records, "syn_03", 0)
    pos = verify(m.match(q["input_image"], same["input_image"], q["gt_mask"]), "H")
    neg = verify(m.match(q["input_image"], other["input_image"], q["gt_mask"]), "H")
    assert pos.n_inliers > 3 * max(neg.n_inliers, 1)


def _synthetic_corr(H: np.ndarray, n: int = 200, outliers: float = 0.4, seed: int = 0) -> Correspondences:
    rng = np.random.default_rng(seed)
    pc = rng.uniform(0, 400, (n, 2)).astype(np.float32)
    pq = cv2.perspectiveTransform(pc[None].astype(np.float64), H)[0].astype(np.float32)
    pq += rng.normal(0, 0.5, pq.shape).astype(np.float32)
    bad = rng.random(n) < outliers
    pq[bad] = rng.uniform(0, 400, (bad.sum(), 2))
    return Correspondences(pq, pc, np.ones(n, np.float32))


@pytest.mark.parametrize("model_type", ["H", "A"])
@pytest.mark.parametrize("method", ["magsac", "ransac"])
def test_verify_models(model_type: str, method: str) -> None:
    H = np.array([[1.1, 0.05, 10], [-0.03, 0.95, -5], [0, 0, 1]], np.float64)  # affine, so both models fit
    corr = _synthetic_corr(H)
    v = verify(corr, model_type, method, threshold=3.0)
    assert v.ok and v.can_warp and 0.45 < v.inlier_ratio < 0.75
    assert np.median(v.residuals[v.inliers]) < 1.5
    corners = np.float64([[[0, 0], [400, 0], [400, 400], [0, 400]]])
    err = np.linalg.norm(cv2.perspectiveTransform(corners, v.model) - cv2.perspectiveTransform(corners, H), axis=2)
    assert err.max() < 2.0


def test_verify_fundamental_and_degenerate() -> None:
    rng = np.random.default_rng(1)
    X = np.c_[rng.uniform(-1, 1, (150, 2)), rng.uniform(4, 8, 150)]  # 3D points in front of two cameras
    K = np.array([[300, 0, 200], [0, 300, 200], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(np.array([0.0, 0.15, 0.0]))
    pq = (K @ X.T).T
    pc = (K @ (R @ X.T + np.array([[0.5], [0], [0]]))).T
    corr = Correspondences((pq[:, :2] / pq[:, 2:]).astype(np.float32), (pc[:, :2] / pc[:, 2:]).astype(np.float32),
                           np.ones(150, np.float32))
    v = verify(corr, "F", "magsac", threshold=1.0)
    assert v.ok and v.inlier_ratio > 0.9 and not v.can_warp
    assert np.median(v.residuals) < 0.5  # residual convention x_q^T F x_c = 0 matches the fit
    few = Correspondences(corr.pts_q[:5], corr.pts_c[:5], np.ones(5, np.float32))
    assert not verify(few, "F").ok and not verify(Correspondences.empty(), "H").ok
    with pytest.raises(ValueError):
        verify(corr, "X")  # type: ignore[arg-type]


def test_registry() -> None:
    assert "sift" in available_matchers()
    with pytest.raises(KeyError):
        create_matcher("nope")
    with pytest.raises(ValueError):
        register_matcher("sift")(type("Other", (Matcher,), {}))


def test_sparse_stage_improves_ranking_and_caches(records: list[dict], tmp_path: Path) -> None:
    split = split_dataset(records, seed=0)
    base = evaluate(Cascade(ColorHistogramRetriever()), split)
    stage = SparseVerificationStage("sift", top_k=20, cache_dir=tmp_path)
    rep = evaluate(Cascade(ColorHistogramRetriever(), stages=[stage]), split)
    assert rep.metrics.index.tolist() == ["retrieval", "sift-H-magsac"]
    assert rep.metrics.loc["retrieval", "MRR"] == pytest.approx(base.metrics.loc["retrieval", "MRR"])
    assert rep.metrics.loc["sift-H-magsac", "R@1"] >= 0.8 > base.metrics.loc["retrieval", "R@1"]
    assert stage.n_extracted == len(split.gallery) + len(split.queries)
    again = SparseVerificationStage("sift", top_k=20, cache_dir=tmp_path)
    again.prepare(split.gallery)
    assert again.n_extracted == 0  # all from the disk cache


def test_sparse_stage_artifacts_and_detail_view(records: list[dict]) -> None:
    split = split_dataset(records, seed=0)
    cascade = Cascade(ColorHistogramRetriever(), stages=[SparseVerificationStage("sift", top_k=5)])
    cascade.prepare(split.gallery)
    q = split.queries[0]
    top = cascade.run(q).final[0]
    assert {"inliers", "inlier_ratio", "matches", "residual_px", "stage1", "cosine"} <= set(top.features)
    corr, ver = top.artifacts["correspondences"], top.artifacts["verification"]
    gallery = {g.item_id: g for g in split.gallery}
    png = render_pair_detail(q.load_image(), gallery[top.item_id].load_image(), corr, ver,
                             mask_q=q.extra["gt_mask"], mask_c=gallery[top.item_id].extra.get("gt_mask"))
    assert png[:4] == b"\x89PNG"
    assert render_pair_detail(q.load_image(), q.load_image(), Correspondences.empty(), None)[:4] == b"\x89PNG"


def test_rejected_pair_has_no_warp_panel(records: list[dict]) -> None:
    from visual_search.ui import SearchApp

    split = split_dataset(records, seed=0)
    stage = SparseVerificationStage("sift", top_k=30, min_inliers=10_000)  # reject everything
    app = SearchApp(Cascade(ColorHistogramRetriever(), stages=[stage]), split.gallery, split.queries, top_k=3)
    res = app.search()
    assert all(c.artifacts["accepted"] is False and c.features["inliers"] == 0 for c in res.final[:3])
    assert len(app.detail_select.options) == 3
    app.detail_select.value = 0
    assert app.detail.value[:4] == b"\x89PNG"
