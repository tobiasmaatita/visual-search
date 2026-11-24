"""Tests for the prompt widget and search app with the real ipympl backend and simulated mouse events."""

import matplotlib

matplotlib.use("module://ipympl.backend_nbagg", force=True)

import ipywidgets as widgets  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from PIL import Image  # noqa: E402

from visual_search.cascade import Cascade  # noqa: E402
from visual_search.data import Prompt, make_synthetic_dataset, split_dataset  # noqa: E402
from visual_search.masking import BoxMasker  # noqa: E402
from visual_search.retrieval import ColorHistogramRetriever  # noqa: E402
from visual_search.ui import PromptWidget, SearchApp  # noqa: E402
from visual_search.viz.gallery import render_ranking_grid  # noqa: E402


def _event(pw: PromptWidget, name: str, xy: tuple[float, float], button: int = 1, key: str | None = None) -> None:
    pw.figure.canvas.draw()
    x, y = pw._ax.transData.transform(xy)
    MouseEvent(name, pw.figure.canvas, x, y, button=button, key=key)._process()


def _drag(pw: PromptWidget, start: tuple[float, float], end: tuple[float, float]) -> None:
    _event(pw, "button_press_event", start)
    _event(pw, "motion_notify_event", end)
    _event(pw, "button_release_event", end)


def _img(h: int, w: int, v: int = 80) -> np.ndarray:
    return np.full((h, w, 3), v, np.uint8)


def test_box_drawing_and_canvas_reuse() -> None:
    seen = []
    pw = PromptWidget(on_change=seen.append)
    pw.set_image(_img(150, 250))
    canvas = pw.figure.canvas
    assert isinstance(canvas, widgets.DOMWidget) and pw._canvas_box.children == (canvas,)
    _drag(pw, (10, 12), (80, 90))
    assert pw.prompt.box == pytest.approx((10, 12, 80, 90), abs=0.6)
    assert seen[-1] is pw.prompt
    pw.set_image(_img(200, 300, 150))
    assert pw.figure.canvas is canvas and pw.prompt is None
    assert pw._image_artist.get_array().shape[:2] == (200, 300)
    _drag(pw, (290, 190), (200, 100))  # drag up-left
    assert pw.prompt.box == pytest.approx((200, 100, 290, 190), abs=0.6)
    _event(pw, "button_press_event", (50, 50))
    _event(pw, "button_release_event", (50, 50))  # a click in box mode is not a box
    assert pw.prompt.box == pytest.approx((200, 100, 290, 190), abs=0.6)


def test_points_positive_negative_and_clear() -> None:
    seen = []
    pw = PromptWidget(on_change=seen.append)
    pw.set_image(_img(100, 200))
    pw.mode.value = "points"
    _event(pw, "button_press_event", (20, 30), button=3)  # negative only: no valid prompt yet
    assert pw.prompt is None and not seen
    _event(pw, "button_press_event", (100, 50), button=1)
    _event(pw, "button_press_event", (150, 60), button=1, key="shift")
    assert pw.prompt.kind == "points"
    assert pw.prompt.point_labels.tolist() == [0, 1, 0]
    np.testing.assert_allclose(pw.prompt.points[1], [100, 50], atol=0.6)
    assert len(pw._pos_artist.get_offsets()) == 1 and len(pw._neg_artist.get_offsets()) == 2
    _drag(pw, (10, 10), (90, 90))  # rectangle selector inactive in points mode
    assert pw.prompt.kind == "points"
    pw.clear()
    assert pw.prompt is None and seen[-1] is None and len(pw._pos_artist.get_offsets()) == 0


def test_set_prompt_and_mask_preview() -> None:
    pw = PromptWidget()
    pw.set_image(_img(100, 200))
    pw.set_prompt(Prompt(points=np.float32([[10, 10]]), point_labels=np.int64([1])))
    assert pw.mode.value == "points"
    mask = np.zeros((100, 200), bool)
    mask[:50] = True
    pw.show_mask(mask)
    assert pw._mask_artist.get_visible() and pw._mask_artist.get_array().shape == (100, 200, 4)
    pw.show_mask(None)
    assert not pw._mask_artist.get_visible()


@pytest.fixture(scope="module")
def split():
    return split_dataset(make_synthetic_dataset(n_instances=4, views_per_instance=2, n_distractors=4, seed=0), seed=0)


def test_search_app_dataset_query(split) -> None:
    app = SearchApp(Cascade(ColorHistogramRetriever()), split.gallery, split.queries, masker=BoxMasker(), top_k=3)
    assert app.query is split.queries[0]
    assert app.prompt_widget.prompt.box == split.queries[0].prompt.box  # dataset prompt shown
    assert app.mask_result is not None and app.mask_result.source == "box"
    result = app.search()
    assert result is app.last_result and len(result.final) == len(split.gallery)
    assert set(app.timings) == {"mask", "retrieval"}
    assert app.results.value[:4] == b"\x89PNG"
    app.query_select.value = 1
    assert app.query is split.queries[1] and app.last_result is None


def test_search_app_path_query_and_drawn_box(split, tmp_path) -> None:
    Image.fromarray(_img(60, 90, 200)).save(tmp_path / "q.png")
    app = SearchApp(Cascade(ColorHistogramRetriever()), split.gallery, masker=BoxMasker())
    assert app.search() is None  # nothing loaded
    app.load_path(tmp_path / "nope.png")
    assert "Not a file" in app.info.value
    app.load_path(tmp_path / "q.png")
    assert app.query.label is None and app.prompt_widget.prompt is None
    _drag(app.prompt_widget, (5, 5), (40, 30))
    assert app.mask_result.mask[10, 10] and not app.mask_result.mask[50, 80]
    result = app.search()
    assert result.final and "✓" not in app.info.value


def test_render_ranking_grid() -> None:
    from visual_search.retrieval import Candidate

    png = render_ranking_grid(
        _img(40, 60), {"retrieval": [Candidate("a", 0.9, {"cosine": 0.9}), Candidate("b", 0.5)], "rerank": []},
        load_thumb=lambda i: _img(30, 30), is_correct=lambda i: i == "a", k=3, label_of=lambda i: i,
    )
    assert png[:4] == b"\x89PNG"
