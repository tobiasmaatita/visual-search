"""Interactive prompt editor for JupyterLab: draw a box or click points on an image.

Built on ipympl (matplotlib's Jupyter widget backend) with matplotlib's
``RectangleSelector`` and canvas click events; no custom JavaScript. The canvas
is created once and reused when the image changes (replacing figures from widget
callbacks makes ipympl emit extra cell outputs). The approach and the ipympl #611
warning filter are ported from mlcvtools' ``demo/notebooks/visual-search.ipynb``.

Controls:
- **Box mode**: drag a rectangle with the left mouse button.
- **Points mode**: left click = positive point, right click or shift + click =
  negative point (excludes a region).
- Zoom/pan via the toolbar; clicks are ignored while a toolbar tool is active.

Every prompt change calls ``on_change(prompt)`` with a
:class:`~visual_search.data.schema.Prompt` in the dataset schema, so interactive
and dataset-driven runs share one code path.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import RectangleSelector
from numpy.typing import NDArray

from visual_search.data.schema import Prompt

logger = logging.getLogger(__name__)

MASK_RGBA = (0.0, 0.63, 1.0, 0.45)

# ipympl #611: its toolbar initializes fine, but traitlets' multiple-inheritance path warns first.
_IPYMPL_TOOLBAR_WARNING = (
    r"Passing unrecognized arguments to super\(Toolbar\).__init__\(\)\.\n"
    r"NavigationToolbar2WebAgg.__init__\(\) missing 1 required positional argument: 'canvas'"
)


class PromptWidget:
    """Image canvas on which the user draws a box or clicks points.

    Attributes:
        prompt: The current prompt (``None`` until the user draws something).
        widget: The ipywidgets container to ``display``.
        figure: The matplotlib figure (one canvas, reused across images).

    Example:
        >>> pw = PromptWidget(on_change=lambda p: print(p))
        >>> pw.set_image(rgb)
        >>> display(pw.widget)
    """

    def __init__(self, on_change: Callable[[Prompt | None], None] | None = None, figsize: tuple[float, float] = (7, 5)):
        """Create the widget (the canvas is created lazily on the first image).

        Args:
            on_change: Called with the new prompt (or ``None`` after clearing).
            figsize: Matplotlib figure size in inches.
        """
        self.on_change = on_change
        self.figsize = figsize
        self.prompt: Prompt | None = None
        self.image: NDArray[np.uint8] | None = None
        self._points: list[tuple[float, float, int]] = []
        self.figure: Any = None
        self._ax: Any = None
        self._image_artist: Any = None
        self._mask_artist: Any = None
        self._selector: RectangleSelector | None = None

        self.mode = widgets.ToggleButtons(options=[("Box", "box"), ("Points", "points")], value="box",
                                          description="Prompt", style={"button_width": "70px"})
        self.clear_button = widgets.Button(description="Clear", layout=widgets.Layout(width="80px"))
        self.status = widgets.HTML()
        self._canvas_box = widgets.Box()
        self.widget = widgets.VBox([widgets.HBox([self.mode, self.clear_button]), self._canvas_box, self.status])
        self.mode.observe(self._mode_changed, names="value")
        self.clear_button.on_click(lambda _: self.clear())

    # --- public API ---------------------------------------------------------------

    def set_image(self, image: NDArray[np.uint8]) -> None:
        """Show a new image and clear the prompt and mask.

        Args:
            image: ``(H, W, 3)`` uint8 RGB.
        """
        self.image = image
        h, w = image.shape[:2]
        if self.figure is None:
            self._create_figure(image)
        else:
            self._image_artist.set_data(image)
            self._image_artist.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
            self._ax.set_xlim(-0.5, w - 0.5)
            self._ax.set_ylim(h - 0.5, -0.5)
            toolbar = getattr(self.figure.canvas, "toolbar", None)
            if toolbar is not None:
                toolbar.update()  # reset the zoom history to the new image
        self.clear(notify=False)

    def set_prompt(self, prompt: Prompt | None, notify: bool = True) -> None:
        """Set the prompt programmatically (e.g. from the dataset) and draw it.

        Args:
            prompt: A box or points prompt, or ``None``. Mask prompts are shown as masks.
            notify: Call ``on_change``.
        """
        self.prompt = prompt
        self._points = []
        if prompt is not None and prompt.points is not None and prompt.point_labels is not None:
            self._points = [(float(x), float(y), int(lbl)) for (x, y), lbl in zip(prompt.points, prompt.point_labels)]
            self.mode.value = "points"
        elif prompt is not None and prompt.box is not None:
            self.mode.value = "box"
        self._draw_prompt()
        if notify:
            self._notify()

    def show_mask(self, mask: NDArray[np.bool_] | None) -> None:
        """Overlay a mask preview on the image (``None`` removes it).

        Args:
            mask: ``(H, W)`` bool matching the current image.
        """
        if self._ax is None or self.image is None:
            return
        if mask is None:
            if self._mask_artist is not None:
                self._mask_artist.set_visible(False)
        else:
            rgba = np.zeros((*mask.shape, 4), np.float32)
            rgba[mask] = MASK_RGBA
            h, w = mask.shape
            if self._mask_artist is None:
                self._mask_artist = self._ax.imshow(rgba, extent=(-0.5, w - 0.5, h - 0.5, -0.5), zorder=2)
            else:
                self._mask_artist.set_data(rgba)
                self._mask_artist.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
                self._mask_artist.set_visible(True)
        self.figure.canvas.draw_idle()

    def clear(self, notify: bool = True) -> None:
        """Remove the prompt, points and mask preview.

        Args:
            notify: Call ``on_change(None)``.
        """
        self.prompt = None
        self._points = []
        if self._selector is not None:
            self._selector.clear()
        self.show_mask(None)
        self._draw_prompt()
        if notify:
            self._notify()

    # --- internals ----------------------------------------------------------------

    def _create_figure(self, image: NDArray[np.uint8]) -> None:
        """Create the single figure, image artist, rectangle selector and click handler."""
        with plt.ioff(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_IPYMPL_TOOLBAR_WARNING, category=DeprecationWarning)
            self.figure, self._ax = plt.subplots(figsize=self.figsize)
        self._ax.set_axis_off()
        self.figure.tight_layout()
        self._image_artist = self._ax.imshow(image, zorder=1)
        self._pos_artist = self._ax.scatter([], [], c="#00d000", s=80, edgecolors="white", zorder=4)
        self._neg_artist = self._ax.scatter([], [], c="#e00000", s=80, marker="X", edgecolors="white", zorder=4)
        self._selector = RectangleSelector(self._ax, self._box_selected, button=[1], interactive=True,
                                           props={"edgecolor": "#ffd000", "fill": False, "linewidth": 2})
        self.figure.canvas.mpl_connect("button_press_event", self._clicked)
        canvas = self.figure.canvas
        # With the ipympl backend the canvas is a widget; with other backends (tests, scripts) it is not.
        self._canvas_box.children = (canvas,) if isinstance(canvas, widgets.DOMWidget) else ()
        self._mode_changed()

    def _mode_changed(self, change: Any = None) -> None:
        """Enable the rectangle selector only in box mode."""
        if self._selector is not None:
            self._selector.set_active(self.mode.value == "box")
            if self.mode.value == "points":
                self._selector.set_visible(False)
        self.status.value = (
            "Drag a box around the object." if self.mode.value == "box"
            else "Left click: positive point &middot; right click or shift+click: negative point."
        )
        if self.figure is not None:
            self.figure.canvas.draw_idle()

    def _box_selected(self, press: MouseEvent, release: MouseEvent) -> None:
        """RectangleSelector callback: store the drawn box as the prompt."""
        if self.mode.value != "box" or None in (press.xdata, press.ydata, release.xdata, release.ydata):
            return
        assert self.image is not None
        h, w = self.image.shape[:2]
        x0, x1 = sorted((press.xdata, release.xdata))
        y0, y1 = sorted((press.ydata, release.ydata))
        x0, y0 = max(x0, 0.0), max(y0, 0.0)
        x1, y1 = min(x1, float(w)), min(y1, float(h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return  # a click, not a drag
        self.prompt = Prompt(box=(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)))
        self._notify()

    def _clicked(self, event: MouseEvent) -> None:
        """Canvas click: add a positive/negative point in points mode."""
        if self.mode.value != "points" or event.inaxes is not self._ax or event.xdata is None:
            return
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # zoom/pan active
        negative = event.button == 3 or (event.button == 1 and event.key == "shift")
        if event.button not in (1, 3):
            return
        self._points.append((float(event.xdata), float(event.ydata), 0 if negative else 1))
        if not any(lbl == 1 for _, _, lbl in self._points):
            self.status.value = "Add at least one positive point (left click)."
            self._draw_prompt()
            return
        pts = np.array(self._points, np.float32)
        self.prompt = Prompt(points=pts[:, :2], point_labels=pts[:, 2].astype(np.int64))
        self._draw_prompt()
        self._notify()

    def _draw_prompt(self) -> None:
        """Redraw the click markers (the box is drawn by the selector)."""
        if self._ax is None:
            return
        pos = np.array([(x, y) for x, y, lbl in self._points if lbl == 1]).reshape(-1, 2)
        neg = np.array([(x, y) for x, y, lbl in self._points if lbl == 0]).reshape(-1, 2)
        self._pos_artist.set_offsets(pos)
        self._neg_artist.set_offsets(neg)
        if self.prompt is not None and self.prompt.box is not None and self._selector is not None:
            x0, y0, x1, y1 = self.prompt.box
            self._selector.extents = (x0, x1, y0, y1)
            self._selector.set_visible(True)
        self.figure.canvas.draw_idle()

    def _notify(self) -> None:
        """Call the change callback, reporting errors in the status line instead of raising."""
        if self.on_change is None:
            return
        try:
            self.on_change(self.prompt)
        except Exception as exc:  # keep the widget alive; show what went wrong
            logger.exception("Prompt callback failed")
            self.status.value = f"<span style='color:#c00'>Error: {type(exc).__name__}: {exc}</span>"
