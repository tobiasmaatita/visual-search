"""Tests for warp models, animation frames, GIF export and the warp app."""

from pathlib import Path

import matplotlib

matplotlib.use("module://ipympl.backend_nbagg", force=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from visual_search.cascade import Cascade  # noqa: E402
from visual_search.data import Prompt, make_synthetic_dataset  # noqa: E402
from visual_search.data.schema import parse_dataset  # noqa: E402
from visual_search.matching import Correspondences, create_matcher  # noqa: E402
from visual_search.retrieval import ColorHistogramRetriever  # noqa: E402
from visual_search.stages import SparseVerificationStage  # noqa: E402
from visual_search.ui import WarpApp, warp_pair  # noqa: E402
from visual_search.viz.animation import (  # noqa: E402
    VIEWS,
    WarpPair,
    compare_frames,
    gif_bytes,
    save_gif,
    side_by_side,
    warp_frames,
    warp_stats,
)
from visual_search.warp import AffineWarp, HomographyWarp, TPSWarp, base_transform, fit_warps  # noqa: E402


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return make_synthetic_dataset(n_instances=3, views_per_instance=2, n_distractors=2, seed=0)


@pytest.fixture(scope="module")
def true_pair(records: list[dict]):
    q, c = records[0], records[1]  # two views of syn_00
    corr = create_matcher("sift").match(q["input_image"], c["input_image"], q["gt_mask"], c["gt_mask"])
    pair = WarpPair(q["input_image"], tuple(q["prompt"]["box"]), c["input_image"], tuple(c["prompt"]["box"]),
                    canvas_side=240)
    return q, c, pair, fit_warps(corr)


class TestModels:
    def test_base_transform(self) -> None:
        b = base_transform((10, 20, 110, 70), (0, 0, 50, 25))
        pts = cv2.perspectiveTransform(np.float64([[[0, 0], [50, 25]]]), b)[0]
        np.testing.assert_allclose(pts, [[10, 20], [110, 70]])

    def test_coarse_map_is_exact_at_nodes(self) -> None:
        h = np.array([[1.3, 0.2, 7], [-0.1, 0.9, 4], [2e-4, 1e-4, 1]], np.float64)
        fine = HomographyWarp(h).sample_map((97, 121), step=1)
        coarse = HomographyWarp(h).sample_map((97, 121), step=8)
        assert np.abs(fine[0] - coarse[0]).max() < 0.2 and np.abs(fine[1] - coarse[1]).max() < 0.2

    def test_homography_sample_map_matches_opencv(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (60, 80, 3), np.uint8)
        h = np.array([[1.1, 0.1, 5], [-0.05, 0.95, 3], [1e-4, 0, 1]], np.float64)  # candidate -> query
        mx, my = HomographyWarp(h).sample_map((60, 80), step=1)
        ours = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        ref = cv2.warpPerspective(img, h, (80, 60), flags=cv2.INTER_LINEAR)
        inner = (slice(10, 50), slice(10, 70))
        assert np.abs(ours[inner].astype(int) - ref[inner].astype(int)).mean() < 2.0

    def test_partial_warp_endpoints(self) -> None:
        h = np.array([[1.2, 0, 4], [0, 1.2, 2], [0, 0, 1]], np.float64)
        warp, base = AffineWarp(h), base_transform((0, 0, 50, 40), (0, 0, 25, 20))
        start = warp.sample_map((40, 50), t=0.0, base=base)
        end = warp.sample_map((40, 50), t=1.0, base=base)
        full = warp.sample_map((40, 50))
        np.testing.assert_allclose(end[0], full[0], atol=1e-4)
        np.testing.assert_allclose(start[0][0, :3], [0.0, 0.5, 1.0], atol=1e-4)  # base: x/2

    def test_tps_exact_and_smooth(self) -> None:
        rng = np.random.default_rng(1)
        q = rng.uniform(0, 100, (25, 2))
        c = q + 3 * np.sin(q / 20) + rng.normal(0, 0.5, q.shape)
        exact = TPSWarp(q, c, stiffness=0.0)
        assert np.abs(exact.to_candidate(q) - c).max() < 1e-6 and np.abs(exact.to_query(c) - q).max() < 1e-6
        smooth = TPSWarp(q, c, stiffness=1.0)
        err = np.abs(smooth.to_candidate(q) - c).max()
        assert 1e-3 < err < 5.0  # no longer through every point, but close
        with pytest.raises(ValueError):
            TPSWarp(q[:2], c[:2])

    def test_fit_warps(self, true_pair) -> None:
        q, c, _, fitted = true_pair
        assert set(fitted.warps) == {"H", "A", "TPS"}
        gt = q["gt_homography"] @ np.linalg.inv(c["gt_homography"])  # candidate -> query
        ys, xs = np.nonzero(c["gt_mask"])
        corners = np.float64([[xs.min(), ys.min()], [xs.max(), ys.max()]])
        est = fitted.warps["H"].to_query(corners)
        ref = cv2.perspectiveTransform(corners[None], gt)[0]
        assert np.abs(est - ref).max() < 15.0  # extrapolated to the object's far corners
        qi, ci = fitted.inliers["H"]
        inl_ref = cv2.perspectiveTransform(ci[None].astype(np.float64), gt)[0]
        assert np.median(np.linalg.norm(fitted.warps["H"].to_query(ci) - inl_ref, axis=1)) < 2.0
        empty = fit_warps(Correspondences.empty())
        assert not empty.warps and set(empty.notes) == {"H", "A", "TPS"}


class TestFrames:
    @pytest.mark.parametrize("view", VIEWS)
    def test_views(self, true_pair, view: str) -> None:
        _, _, pair, fitted = true_pair
        frames = warp_frames(pair, fitted.warps["TPS"], view, n_frames=6, hold=2, inliers=fitted.inliers["TPS"])
        assert all(f.shape == (*pair.shape, 3) and f.dtype == np.uint8 for f in frames)
        if view == "flicker":
            assert not np.array_equal(frames[0], frames[3])  # query vs warped candidate
        else:
            assert len(frames) == 8 and np.array_equal(frames[-1], frames[-2])  # held end frame
            assert not np.array_equal(frames[0], frames[5])  # something moves
        with pytest.raises(ValueError):
            warp_frames(pair, fitted.warps["H"], "spin")

    def test_alignment_reduces_residual(self, true_pair) -> None:
        _, _, pair, fitted = true_pair
        qg = cv2.cvtColor(pair.query_canvas, cv2.COLOR_RGB2GRAY).astype(float)

        def residual(t: float) -> float:
            w, valid = pair.warped(fitted.warps["H"], t)
            m = valid & pair._box_mask
            return float(np.abs(qg - cv2.cvtColor(w, cv2.COLOR_RGB2GRAY))[m].mean())

        assert residual(1.0) < 0.8 * residual(0.0)

    def test_compare_side_by_side_stats(self, true_pair) -> None:
        _, _, pair, fitted = true_pair
        frames = compare_frames(pair, fitted, "grid", n_frames=4, hold=1)
        assert len(frames) == 5 and frames[0].shape[1] == 3 * pair.shape[1] + 2 * 6
        sbs = side_by_side(pair, height=100)
        assert sbs.shape[0] == 100
        stats = warp_stats(pair, fitted)
        assert list(stats.columns) == ["points", "reproj_px", "strain", "residual", "note"]
        assert stats.loc["TPS", "reproj_px"] < stats.loc["H", "reproj_px"] + 1.0  # TPS fits its points closely
        assert (stats["residual"] >= 0).all()

    def test_gif_export(self, true_pair, tmp_path: Path) -> None:
        import imageio.v3 as iio

        _, _, pair, fitted = true_pair
        frames = warp_frames(pair, fitted.warps["H"], "morph", n_frames=5, hold=1)
        path = save_gif(frames, tmp_path / "sub" / "w.gif", fps=10)
        assert iio.imread(path, index=None).shape[0] >= 5  # Pillow merges the identical held end frame
        assert gif_bytes(frames)[:6] == b"GIF89a"
        with pytest.raises(ValueError):
            save_gif([], tmp_path / "x.gif")


def test_warp_pair_helper(records: list[dict]) -> None:
    q, c = records[0], records[1]
    pair, fitted = warp_pair(q["input_image"], tuple(q["prompt"]["box"]), c["input_image"], tuple(c["prompt"]["box"]))
    assert "H" in fitted.warps and pair.shape[1] <= 480


def test_warp_app(records: list[dict], tmp_path: Path) -> None:
    items = parse_dataset(records)
    queries = [it for it in items if it.label]
    cascade = Cascade(ColorHistogramRetriever(), stages=[SparseVerificationStage("sift", top_k=2)])
    app = WarpApp(cascade, items, queries, top_n=4, output_dir=tmp_path, n_frames=5, canvas_side=200)
    assert app.search_app.detail_select.layout.display == "none"
    assert app.warp() == [] and "Search first" in app.save_info.value
    app.search_app.search()
    assert len(app.candidate.options) == 4 and app.candidate.value is None
    app.candidate.value = 0  # a candidate that went through stage 2
    assert app.pair is not None and app.pair_image.value[:4] == b"\x89PNG"
    assert "stage-2 matches" in app.info.value and "<table" in app.stats.value
    assert ("unreliable" in app.info.value) == (max(len(app.fitted.inliers.get(m, ((),))[0]) for m in ("H", "A")) < 8)
    frames = app.warp(start=True)
    assert len(frames) == 5 + 8 and app.play.max == len(frames) - 1 and app.play.playing
    app.slider.value = 3
    assert app.frame_image.value == app._png[3]
    app.compare.value = True
    wide = app.warp()
    assert wide and wide[0].shape[1] > frames[0].shape[1]
    path = app.save()
    assert path is not None and path.exists() and path.suffix == ".gif"
    app.candidate.value = 3  # beyond stage 2's top-2: matches are computed on the spot
    assert "computed now" in app.info.value
    app.search_app.prompt_widget.set_prompt(Prompt(box=(0, 0, 50, 50)))
    app.search_app.search()  # a new search resets the candidate list
    assert app.candidate.value is None and app.frames == []
