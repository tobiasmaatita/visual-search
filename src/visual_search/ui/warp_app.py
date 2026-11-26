"""Warp app: run the cascade, pick a result from the top-n, and watch it warp onto the query.

The top part is the usual search app (query, box, Search, result grid). Below it:

1. pick a **candidate** from the top-n of the last search;
2. the pair is shown **side by side** (query box | candidate box or found region);
3. choose a **model** (TPS, homography H, affine A) and a **view** (morph, grid,
   flicker, residual), optionally "all models side by side";
4. press **Warp ▶**: frames are rendered and played; pause, scrub with the slider,
   change the speed, or **Save GIF**.

Matches come from the cascade's stage 2 when the candidate went through it,
otherwise they are computed on the spot with the app's matcher (SIFT by default).
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
from numpy.typing import NDArray

from visual_search.cascade import Cascade
from visual_search.data.schema import Item, Prompt
from visual_search.imaging import prompt_to_mask
from visual_search.masking.base import Masker
from visual_search.matching.base import Correspondences, Matcher
from visual_search.matching.registry import create_matcher
from visual_search.retrieval.base import Candidate
from visual_search.ui.search_app import SearchApp
from visual_search.viz.animation import (
    VIEWS,
    WarpPair,
    compare_frames,
    encode_png,
    save_gif,
    side_by_side,
    warp_frames,
    warp_stats,
)
from visual_search.warp.models import FittedWarps, fit_warps

MODELS = ("TPS", "H", "A")
MIN_RELIABLE_POINTS = 8  # same threshold as the sparse stage's min_inliers


class WarpApp:
    """Search, pick a result, and animate the warp of that result onto the query.

    Attributes:
        search_app: The embedded :class:`SearchApp` (query, box, Search, result grid).
        widget: The ipywidgets container to ``display``.
        frames: The frames of the last rendered animation.
        fitted: The warps fitted for the selected pair.

    Example:
        >>> app = WarpApp(cascade, gallery, queries, top_n=10)
        >>> app.widget
    """

    def __init__(
        self,
        cascade: Cascade,
        gallery: Sequence[Item],
        queries: Sequence[Item] = (),
        top_n: int = 10,
        masker: Masker | None = None,
        matcher: Matcher | str = "sift",
        output_dir: str | Path = "results/warps",
        n_frames: int = 24,
        canvas_side: int = 480,
        tps_stiffness: float = 0.05,
        show_expected_label: bool = True,
    ) -> None:
        """Build the app.

        Args:
            cascade: The pipeline to run (its gallery is indexed if needed).
            gallery: Gallery items.
            queries: Items offered in the query dropdown.
            top_n: How many results of the last search can be picked as candidate.
            masker: Turns the drawn box into a mask (default: the box itself).
            matcher: Matcher for pairs that did not go through a matching stage.
            output_dir: Where **Save GIF** writes.
            n_frames: Frames from unwarped to fully warped.
            canvas_side: Longer side of the animation in pixels.
            tps_stiffness: Smoothness of the thin-plate spline (0 = exact through all matches).
            show_expected_label: Show the search app's "Expected" field (✓/✗ marks).
        """
        self.top_n = top_n
        self.matcher = create_matcher(matcher) if isinstance(matcher, str) else matcher
        self.output_dir = Path(output_dir)
        self.n_frames = n_frames
        self.canvas_side = canvas_side
        self.tps_stiffness = tps_stiffness
        self.search_app = SearchApp(cascade, gallery, queries, masker=masker, top_k=top_n,
                                    show_expected_label=show_expected_label)
        self.search_app.detail_select.layout.display = "none"  # the warp panel replaces the detail view
        self.search_app.detail.layout.display = "none"
        self.gallery = self.search_app.gallery
        self.frames: list[NDArray[np.uint8]] = []
        self._png: list[bytes] = []
        self.fitted: FittedWarps | None = None
        self.pair: WarpPair | None = None
        self._cand: Candidate | None = None
        self._cache: dict[tuple, list[NDArray[np.uint8]]] = {}

        self.candidate = widgets.Dropdown(options=[], description="Candidate", layout=widgets.Layout(width="75%"))
        self.model = widgets.ToggleButtons(options=list(MODELS), value="TPS", description="Model",
                                           style={"button_width": "60px"})
        self.view = widgets.ToggleButtons(options=list(VIEWS), value="morph", description="View",
                                          style={"button_width": "80px"})
        self.compare = widgets.Checkbox(value=False, description="all models side by side", indent=False)
        self.warp_button = widgets.Button(description="Warp ▶", button_style="success")
        self.pair_image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))
        self.info = widgets.HTML()
        self.play = widgets.Play(value=0, min=0, max=0, step=1, interval=80, description="play")
        self.slider = widgets.IntSlider(value=0, min=0, max=0, description="frame", continuous_update=True)
        self.speed = widgets.IntSlider(value=12, min=2, max=30, description="fps")
        widgets.jslink((self.play, "value"), (self.slider, "value"))
        self.frame_image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))
        self.stats = widgets.HTML()
        self.save_button = widgets.Button(description="Save GIF", icon="download")
        self.save_info = widgets.HTML()

        self.widget = widgets.VBox([
            self.search_app.widget,
            widgets.HTML("<hr><b>Warp a result onto the query</b>"),
            self.candidate,
            self.pair_image,
            self.info,
            self.model,
            widgets.HBox([self.view, self.compare]),
            widgets.HBox([self.warp_button, self.play, self.slider, self.speed]),
            self.frame_image,
            self.stats,
            widgets.HBox([self.save_button, self.save_info]),
        ])
        self.search_app.results.observe(lambda _: self._refresh_candidates(), names="value")
        self.candidate.observe(lambda ch: self.select(ch["new"]), names="value")
        self.warp_button.on_click(lambda _: self.warp(start=True))
        self.slider.observe(lambda ch: self._show_frame(ch["new"]), names="value")
        self.speed.observe(lambda ch: setattr(self.play, "interval", int(1000 / ch["new"])), names="value")
        self.play.interval = int(1000 / self.speed.value)
        self.save_button.on_click(lambda _: self.save())

    # --- actions ------------------------------------------------------------------------

    def select(self, rank: int | None) -> None:
        """Choose the candidate at 0-based ``rank`` of the last search: show the pair and fit the warps.

        Args:
            rank: Position in the final ranking; ``None`` clears the panel.
        """
        self._clear_animation()
        result = self.search_app.last_result
        query = self.search_app.searched_query
        image_q = self.search_app.query_image
        if rank is None or result is None or query is None or image_q is None:
            self.pair_image.value, self.info.value, self.stats.value = b"", "", ""
            self.pair = self.fitted = self._cand = None
            return
        cand = result.final[rank]
        item = self.gallery[cand.item_id]
        image_c = item.load_image()
        hq, wq = image_q.shape[:2]
        hc, wc = image_c.shape[:2]
        query_box = query.prompt.bbox() if query.prompt is not None else (0.0, 0.0, float(wq), float(hq))
        if cand.region is not None:
            cand_box = cand.region
        elif item.prompt is not None:
            cand_box = item.prompt.bbox()
        else:
            cand_box = (0.0, 0.0, float(wc), float(hc))
        corr = cand.artifacts.get("correspondences")
        source = "stage-2 matches"
        if corr is None:
            mask_q = prompt_to_mask(query.prompt, (hq, wq))
            mask_c = prompt_to_mask(Prompt(box=cand_box), (hc, wc))
            corr = self.matcher.match(image_q, image_c, mask_q, mask_c)
            source = f"{self.matcher.name} matches (computed now)"
        self._cand = cand
        self.pair = WarpPair(image_q, query_box, image_c, cand_box, canvas_side=self.canvas_side)
        self.fitted = fit_warps(corr, tps_stiffness=self.tps_stiffness)
        self.pair_image.value = encode_png(side_by_side(self.pair))
        notes = "; ".join(f"<b>{k}</b>: {html.escape(v)}" for k, v in self.fitted.notes.items())
        best = max((len(q) for name, (q, _) in self.fitted.inliers.items() if name in ("H", "A")), default=0)
        warning = ("" if best >= MIN_RELIABLE_POINTS else
                   f"<br><span style='color:#c60'>Only {best} matches agree on a geometry: the warps below are "
                   f"unreliable (fewer than {MIN_RELIABLE_POINTS}). A wildly stretched warp usually means this is "
                   f"not the same item.</span>")
        self.info.value = (f"#{rank + 1} {html.escape(cand.item_id)} (label {html.escape(str(item.label))}) &mdash; "
                           f"{len(corr)} {source} &mdash; {notes}{warning}")
        available = [m for m in MODELS if m in self.fitted.warps]
        if available and self.model.value not in available:
            self.model.value = available[0]
        self.stats.value = (warp_stats(self.pair, self.fitted).to_html(float_format=lambda v: f"{v:.3f}", border=0)
                            if self.fitted.warps else "")

    def warp(self, model: str | None = None, view: str | None = None, start: bool = False) -> list[NDArray[np.uint8]]:
        """Render (or reuse) the animation for the selected pair and show its first frame.

        Args:
            model: ``"TPS"``, ``"H"`` or ``"A"``; default = the Model buttons.
            view: ``"morph"``, ``"grid"``, ``"flicker"`` or ``"residual"``; default = the View buttons.
            start: Start playing right away (what the **Warp ▶** button does).

        Returns:
            The frames (empty if nothing can be warped).
        """
        model = model or self.model.value
        view = view or self.view.value
        if self.pair is None or self.fitted is None:
            self.save_info.value = "Search first, then pick a candidate."
            return []
        key = (self._cand.item_id if self._cand else "", self._cand.region if self._cand else None, model, view,
               self.compare.value)
        if key not in self._cache:
            if self.compare.value:
                frames = compare_frames(self.pair, self.fitted, view, self.n_frames)
            elif model in self.fitted.warps:
                frames = warp_frames(self.pair, self.fitted.warps[model], view, self.n_frames,
                                     inliers=self.fitted.inliers.get(model))
            else:
                frames = []
            self._cache[key] = frames
        self.frames = self._cache[key]
        if not self.frames:
            self.save_info.value = f"<span style='color:#c00'>No {model} warp for this pair: " \
                                   f"{html.escape(self.fitted.notes.get(model, 'too few matches'))}</span>"
            return []
        self._png = [encode_png(f) for f in self.frames]
        self.play.playing = False
        self.play.max = self.slider.max = len(self.frames) - 1
        self.play.value = self.slider.value = 0
        self._show_frame(0)
        self.save_info.value = ""
        if start:
            self.play.playing = True
        return self.frames

    def save(self, path: str | Path | None = None) -> Path | None:
        """Save the current animation as a looping GIF.

        Args:
            path: Output file; default ``<output_dir>/<query>__<candidate>__<model>_<view>.gif``.

        Returns:
            The written path, or None if there is no animation yet.
        """
        if not self.frames:
            self.save_info.value = "Press Warp ▶ first."
            return None
        if path is None:
            query = self.search_app.searched_query
            tag = "all" if self.compare.value else self.model.value
            stem = f"{_safe(query.item_id if query else 'query')}__{_safe(self._cand.item_id if self._cand else '')}"
            path = self.output_dir / f"{stem}__{tag}_{self.view.value}.gif"
        written = save_gif(self.frames, path, fps=self.speed.value)
        self.save_info.value = f"Saved {html.escape(str(written))}"
        return written

    # --- internals ------------------------------------------------------------------------

    def _refresh_candidates(self) -> None:
        """Offer the top-n of the last search as candidates."""
        result = self.search_app.last_result
        self.candidate.options = [] if result is None else [
            (f"#{i + 1} {c.item_id}" + (f" ({self.gallery[c.item_id].label})" if self.gallery[c.item_id].label else "")
             + (" region" if c.region is not None else "") + f"  score {c.score:.3f}", i)
            for i, c in enumerate(result.final[: self.top_n])
        ]
        self.candidate.value = None
        self._cache.clear()

    def _clear_animation(self) -> None:
        """Stop playing and clear the frames."""
        self.play.playing = False
        self.frames, self._png = [], []
        self.frame_image.value = b""
        self.save_info.value = ""

    def _show_frame(self, i: int) -> None:
        """Display frame ``i`` (slider/player callback)."""
        if 0 <= i < len(self._png):
            self.frame_image.value = self._png[i]


def _safe(text: str) -> str:
    """Make a string safe for a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:60] or "x"


def warp_pair(
    query_image: NDArray[np.uint8],
    query_box: tuple[float, float, float, float],
    cand_image: NDArray[np.uint8],
    cand_box: tuple[float, float, float, float],
    corr: Correspondences | None = None,
    matcher: Matcher | str = "sift",
    **kwargs: Any,
) -> tuple[WarpPair, FittedWarps]:
    """Build a pair and fit its warps without the app (for scripts and notebook cells).

    Args:
        query_image: Query RGB image.
        query_box: Query object box.
        cand_image: Candidate RGB image.
        cand_box: Candidate object box.
        corr: Matches; computed with ``matcher`` inside the two boxes if None.
        matcher: Matcher name or instance.
        **kwargs: Passed to :func:`~visual_search.warp.models.fit_warps`.

    Returns:
        ``(pair, fitted_warps)``; feed them to ``warp_frames`` / ``save_gif``.
    """
    if corr is None:
        m = create_matcher(matcher) if isinstance(matcher, str) else matcher
        corr = m.match(query_image, cand_image, prompt_to_mask(Prompt(box=query_box), query_image.shape[:2]),
                       prompt_to_mask(Prompt(box=cand_box), cand_image.shape[:2]))
    return WarpPair(query_image, query_box, cand_image, cand_box), fit_warps(corr, **kwargs)
