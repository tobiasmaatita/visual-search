"""Interactive search app: pick a query, prompt it, preview the mask, run the cascade.

The app drives the same :class:`~visual_search.cascade.Cascade` object as the batch
evaluation, so an interactive query and a dataset query take the same code path:
the prompt is a :class:`~visual_search.data.schema.Prompt`, the masker output is
attached with :func:`~visual_search.masking.base.with_mask`, and the result is a
:class:`~visual_search.cascade.QueryResult`.
"""

from __future__ import annotations

import dataclasses
import html
import time
from collections.abc import Sequence
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from numpy.typing import NDArray

from visual_search.cache import image_fingerprint
from visual_search.cascade import Cascade, QueryResult
from visual_search.data.schema import Item, Prompt, is_match
from visual_search.imaging import prompt_to_mask
from visual_search.masking.base import BoxMasker, Masker, MaskResult, with_mask
from visual_search.ui.prompt_widget import PromptWidget
from visual_search.viz.gallery import render_ranking_grid, thumbnail
from visual_search.viz.overlay import overlay_mask
from visual_search.viz.pairs import render_pair_detail


class SearchApp:
    """JupyterLab app around a cascade.

    Attributes:
        widget: The ipywidgets container to ``display``.
        prompt_widget: The embedded :class:`PromptWidget`.
        query: The current query item (prompt not yet applied).
        mask_result: The latest mask preview, if any.
        last_result: The latest cascade result, if any.
        timings: Seconds per step of the latest run (mask + cascade stages).

    Example:
        >>> app = SearchApp(cascade, split.gallery, split.queries, masker=BoxMasker())
        >>> display(app.widget)
        >>> app.last_result.final[:5]   # after clicking "Search"
    """

    def __init__(
        self,
        cascade: Cascade,
        gallery: Sequence[Item],
        queries: Sequence[Item] = (),
        masker: Masker | None = None,
        top_k: int = 5,
        use_mask: bool = True,
        thumb_size: int = 256,
    ) -> None:
        """Build the app and index the gallery if the cascade is not prepared yet.

        Args:
            cascade: The pipeline to run.
            gallery: Gallery items (indexed with ``cascade.prepare`` if needed).
            queries: Items offered in the query dropdown (e.g. ``split.queries``).
            masker: Mask preview and query masking; default :class:`BoxMasker`.
            top_k: Candidates shown per stage.
            use_mask: Initial state of the "use mask" checkbox.
            thumb_size: Longer side of gallery thumbnails in pixels.
        """
        self.cascade = cascade
        self.gallery = {it.item_id: it for it in gallery}
        self.queries = list(queries)
        self.masker = masker or BoxMasker()
        self.top_k = top_k
        self.thumb_size = thumb_size
        self._thumbs: dict[str, NDArray[np.uint8]] = {}
        self.query: Item | None = None
        self._image: NDArray[np.uint8] | None = None
        self._image_key: str | None = None
        self.mask_result: MaskResult | None = None
        self.last_result: QueryResult | None = None
        self.timings: dict[str, float] = {}

        if not cascade.is_prepared:
            cascade.prepare(list(gallery))

        self.query_select = widgets.Dropdown(
            options=[(f"{q.item_id} [{q.label}]", i) for i, q in enumerate(self.queries)],
            value=None, description="Query", layout=widgets.Layout(width="60%"),
        )
        self.path_text = widgets.Text(placeholder="or an image path", layout=widgets.Layout(width="40%"))
        self.load_button = widgets.Button(description="Load path", layout=widgets.Layout(width="100px"))
        self.use_mask = widgets.Checkbox(value=use_mask, description=f"use mask ({self.masker.name})")
        self.search_button = widgets.Button(description="Search", button_style="primary")
        self.info = widgets.HTML()
        self.results = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))
        self.detail_select = widgets.Dropdown(options=[], value=None, description="Detail",
                                              layout=widgets.Layout(width="70%"))
        self.detail = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))
        self._query_item: Item | None = None
        self.prompt_widget = PromptWidget(on_change=self._prompt_changed)
        self.widget = widgets.VBox([
            widgets.HBox([self.query_select, self.path_text, self.load_button]),
            self.prompt_widget.widget,
            widgets.HBox([self.use_mask, self.search_button]),
            self.info,
            self.results,
            self.detail_select,
            self.detail,
        ])
        self.query_select.observe(self._query_selected, names="value")
        self.load_button.on_click(lambda _: self.load_path(self.path_text.value))
        self.search_button.on_click(lambda _: self.search())
        self.detail_select.observe(lambda ch: self.show_detail(ch["new"]), names="value")
        if self.queries:
            self.query_select.value = 0

    # --- actions ------------------------------------------------------------------

    def load_query(self, item: Item) -> None:
        """Show a query image; a dataset prompt (if any) becomes the initial prompt.

        Args:
            item: The query item.
        """
        self.query = item
        self._image = item.load_image()
        self._image_key = image_fingerprint(self._image)
        self.mask_result = None
        self.last_result = None
        self.results.value = b""
        self._reset_detail()
        self.prompt_widget.set_image(self._image)
        self.info.value = f"Query <b>{html.escape(item.item_id)}</b> (label: {html.escape(str(item.label))})"
        if item.prompt is not None and item.prompt.mask is None:
            self.prompt_widget.set_prompt(item.prompt)

    def load_path(self, path: str | Path) -> None:
        """Load any image from disk as an unlabelled query.

        Args:
            path: Image file path.
        """
        path = Path(str(path).strip()).expanduser()
        if not path.is_file():
            self.info.value = f"<span style='color:#c00'>Not a file: {html.escape(str(path))}</span>"
            return
        self.query_select.value = None
        self.load_query(Item(item_id=str(path), image=path, label=None))

    def search(self) -> QueryResult | None:
        """Run the cascade for the current query and prompt, and render the gallery.

        Returns:
            The cascade result (also stored in :attr:`last_result`), or None without a query.
        """
        if self.query is None or self._image is None:
            self.info.value = "Select or load a query first."
            return None
        item = dataclasses.replace(self.query, image=self._image, prompt=self.prompt_widget.prompt)
        timings = {k: v for k, v in self.timings.items() if k.startswith("mask")}
        if self.use_mask.value and self.mask_result is not None and item.prompt is not None:
            item = with_mask(item, self.mask_result)
        result = self.cascade.run(item)
        self.last_result = result
        self.timings = {**timings, **result.timings}
        self._query_item = item
        self._render(result, item)
        self._fill_detail_options(result)
        return result

    def show_detail(self, rank: int | None) -> None:
        """Render the pair detail view for the candidate at 0-based ``rank`` of the final ranking.

        Shows matches/inliers and the warp if the candidate went through a matching
        stage; otherwise only its scores.

        Args:
            rank: Position in ``last_result.final``; ``None`` clears the view.
        """
        if rank is None or self.last_result is None or self._query_item is None or self._image is None:
            self.detail.value = b""
            return
        cand = self.last_result.final[rank]
        corr = cand.artifacts.get("correspondences")
        if corr is None:
            self.detail.value = b""
            self.info.value += " &mdash; no match details for this candidate (not re-ranked by a matcher)"
            return
        item = self.gallery[cand.item_id]
        image_c = item.load_image()
        feats = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in cand.features.items())
        self.detail.value = render_pair_detail(
            self._image, image_c, corr, cand.artifacts.get("verification"),
            mask_q=prompt_to_mask(self._query_item.prompt, self._image.shape[:2]),
            mask_c=prompt_to_mask(item.prompt, image_c.shape[:2]),
            title=f"#{rank + 1} {cand.item_id} (label {item.label}) | {feats}",
            accepted=bool(cand.artifacts.get("accepted", True)),
        )

    def _reset_detail(self) -> None:
        """Clear the detail selector and view."""
        self.detail_select.options = []
        self.detail_select.value = None
        self.detail.value = b""

    def _fill_detail_options(self, result: QueryResult) -> None:
        """Offer the final top-k candidates in the detail dropdown."""
        self._reset_detail()
        self.detail_select.options = [
            (f"#{i + 1} {c.item_id} ({self.gallery[c.item_id].label}) score {c.score:.3f}", i)
            for i, c in enumerate(result.final[: self.top_k])
        ]

    # --- internals ----------------------------------------------------------------

    def _query_selected(self, change: dict) -> None:
        """Dropdown callback."""
        if change["new"] is not None:
            self.load_query(self.queries[change["new"]])

    def _prompt_changed(self, prompt: Prompt | None) -> None:
        """Prompt callback: update the mask preview and report its latency."""
        if self._image is None:
            return
        if prompt is None:
            self.mask_result = None
            self.timings = {}
            self.prompt_widget.show_mask(None)
            return
        t0 = time.perf_counter()
        self.mask_result = self.masker.mask(self._image, prompt, image_key=self._image_key)
        self.timings = {"mask": time.perf_counter() - t0}
        self.prompt_widget.show_mask(self.mask_result.mask)
        detail = ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in self.mask_result.timings.items())
        score = "" if self.mask_result.score is None else f", score {self.mask_result.score:.2f}"
        self.prompt_widget.status.value = (
            f"Mask ({html.escape(self.mask_result.source)}{score}): {self.timings['mask'] * 1000:.0f} ms"
            + (f" [{detail}]" if detail else "")
        )

    def _thumb(self, item_id: str) -> NDArray[np.uint8]:
        """Return (and memoize) a gallery thumbnail."""
        if item_id not in self._thumbs:
            self._thumbs[item_id] = thumbnail(self.gallery[item_id].load_image(), self.thumb_size)
        return self._thumbs[item_id]

    def _render(self, result: QueryResult, item: Item) -> None:
        """Draw the per-stage gallery and the timing line."""
        assert self._image is not None
        query_vis = self._image
        if item.prompt is not None and item.prompt.mask is not None:
            query_vis = overlay_mask(self._image, item.prompt.mask)
        labelled = item.label is not None

        def correct(item_id: str) -> bool | None:
            return is_match(item, self.gallery[item_id]) if labelled else None

        timing = " | ".join(f"{k} {v * 1000:.0f} ms" for k, v in self.timings.items())
        total = sum(self.timings.values()) * 1000
        self.results.value = render_ranking_grid(
            query_vis, result.stages, self._thumb, correct, k=self.top_k,
            label_of=lambda i: self.gallery[i].label, title=f"{item.item_id}: {timing} | total {total:.0f} ms",
        )
        self.info.value = (f"Query <b>{html.escape(item.item_id)}</b> (label: {html.escape(str(item.label))})"
                           f" &mdash; {html.escape(timing)} &mdash; total {total:.0f} ms")
