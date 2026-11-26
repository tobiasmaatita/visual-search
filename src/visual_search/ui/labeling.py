"""Labeling app for JupyterLab: draw boxes on images, give them instance labels, save to ``annotations.json``.

Workflow per image: draw a box, type (or pick) a label, optionally a group, press
**Add box**. Every change is saved immediately (with a ``.bak`` of the previous
file). The label field autocompletes from all labels used in the set, a warning
appears when a new label looks like an existing one (``Bank_01`` vs ``bank_01``),
and a table lists every label with its box and image counts.

Images you have looked at but that contain nothing to label can be marked
**reviewed**: they become whole-image distractors. Images never touched are
"not labelled yet" and stay out of the dataset.
"""

from __future__ import annotations

import html
import logging
import zlib
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from matplotlib import colormaps

from visual_search.data.annotations import ANNOTATIONS_FILE, AnnotationSet, BoxAnnotation, similar_labels
from visual_search.data.schema import Item, Prompt
from visual_search.ui.prompt_widget import PromptWidget

logger = logging.getLogger(__name__)

_PALETTE = [colormaps["tab20"](i) for i in range(20)]
UNLABELLED_COLOR = (0.5, 0.5, 0.5, 1.0)


def label_color(label: str | None) -> tuple[float, float, float, float]:
    """Return a stable color per label (the same label always gets the same color)."""
    if not label:
        return UNLABELLED_COLOR
    return _PALETTE[zlib.crc32(label.encode()) % len(_PALETTE)]


class LabelingApp:
    """Interactive box labeling for one image directory.

    Attributes:
        annotations: The :class:`~visual_search.data.annotations.AnnotationSet` being edited.
        widget: The ipywidgets container to ``display``.
        current: Relative path of the image shown.

    Example:
        >>> app = LabelingApp("data/testset")
        >>> app.widget                      # last expression of a notebook cell
        >>> app.annotations.label_stats()   # label overview as a DataFrame
    """

    def __init__(self, root: str | Path, filename: str = ANNOTATIONS_FILE, similarity_cutoff: float = 0.8,
                 figsize: tuple[float, float] = (8, 6)) -> None:
        """Load (or start) the annotations of ``root`` and build the widgets.

        Args:
            root: Directory with the images (subfolders allowed).
            filename: Annotation file name inside ``root``.
            similarity_cutoff: Spelling-similarity threshold for the duplicate-label warning.
            figsize: Size of the drawing canvas in inches.

        Raises:
            FileNotFoundError: If ``root`` does not exist.
        """
        self.annotations = AnnotationSet.load(root, filename)
        self.paths = self.annotations.image_paths()
        self.similarity_cutoff = similarity_cutoff
        self.current: str | None = None
        self._confirm_label: str | None = None
        self._updating = False

        self.image_select = widgets.Dropdown(description="Image", layout=widgets.Layout(width="55%"))
        self.prev_button = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
        self.next_button = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
        self.next_todo_button = widgets.Button(description="Next unlabelled", layout=widgets.Layout(width="130px"))
        self.reviewed = widgets.Checkbox(description="reviewed (no more objects)", indent=False)
        self.progress = widgets.HTML()
        self.prompt_widget = PromptWidget(on_change=self._box_drawn, figsize=figsize)
        self.prompt_widget.mode.layout.display = "none"  # labeling uses boxes only
        self.label_input = widgets.Combobox(placeholder="instance label, e.g. bank_01", description="Label",
                                            ensure_option=False, layout=widgets.Layout(width="45%"))
        self.group_input = widgets.Combobox(placeholder="group / model (optional)", description="Group",
                                            ensure_option=False, layout=widgets.Layout(width="45%"))
        self.add_button = widgets.Button(description="Add box", button_style="primary")
        self.label_info = widgets.HTML()
        self.box_list = widgets.Select(options=[], rows=6, description="Boxes", layout=widgets.Layout(width="70%"))
        self.relabel_button = widgets.Button(description="Relabel selected")
        self.delete_button = widgets.Button(description="Delete selected", button_style="danger")
        self.status = widgets.HTML()
        self.stats = widgets.HTML()

        self.widget = widgets.VBox([
            widgets.HBox([self.image_select, self.prev_button, self.next_button, self.next_todo_button]),
            widgets.HBox([self.reviewed, self.progress]),
            self.prompt_widget.widget,
            widgets.HBox([self.label_input, self.group_input]),
            widgets.HBox([self.add_button, self.label_info]),
            widgets.HBox([self.box_list, widgets.VBox([self.relabel_button, self.delete_button])]),
            self.status,
            widgets.HTML("<b>Labels in this set</b>"),
            self.stats,
        ])

        self.image_select.observe(self._image_selected, names="value")
        self.prev_button.on_click(lambda _: self._step(-1))
        self.next_button.on_click(lambda _: self._step(1))
        self.next_todo_button.on_click(lambda _: self.next_unlabelled())
        self.reviewed.observe(self._reviewed_toggled, names="value")
        self.label_input.observe(lambda _: self._label_typed(), names="value")
        self.add_button.on_click(lambda _: self.add_box())
        self.box_list.observe(self._box_selected, names="value")
        self.relabel_button.on_click(lambda _: self.relabel_selected())
        self.delete_button.on_click(lambda _: self.delete_selected())

        self._refresh_vocabulary()
        self._refresh_image_options()
        if not self.paths:
            self.status.value = f"<span style='color:#c00'>No images found in {html.escape(str(root))}.</span>"
        else:
            first_todo = next((p for p in self.paths if not self.annotations.is_reviewed(p)), self.paths[0])
            self.show_image(first_todo)

    # --- navigation -------------------------------------------------------------------

    def show_image(self, rel: str) -> None:
        """Show an image with its existing boxes.

        Args:
            rel: Relative image path (one of ``self.paths``).
        """
        self.current = rel
        if self.image_select.value != rel:
            self._updating = True
            self.image_select.value = rel
            self._updating = False
        self.prompt_widget.set_image(Item(rel, self.annotations.root / rel).load_image())
        self._confirm_label = None
        self._refresh_boxes()
        self._refresh_reviewed()

    def next_unlabelled(self) -> None:
        """Jump to the next image (after the current one, wrapping around) that is not reviewed yet."""
        if not self.paths:
            return
        start = self.paths.index(self.current) + 1 if self.current in self.paths else 0
        order = self.paths[start:] + self.paths[:start]
        todo = next((p for p in order if not self.annotations.is_reviewed(p)), None)
        if todo is None:
            self.status.value = "All images are reviewed."
        else:
            self.show_image(todo)

    # --- editing ----------------------------------------------------------------------

    def add_box(self) -> bool:
        """Add the drawn box with the typed label and group, then save.

        A new label that looks like an existing one triggers a warning first; pressing
        **Add** again with the same label adds it anyway.

        Returns:
            True if a box was added.
        """
        prompt = self.prompt_widget.prompt
        if self.current is None or prompt is None or prompt.box is None:
            self._message("Draw a box first.", error=True)
            return False
        label = self.label_input.value.strip()
        if not label:
            self._message("Type a label first.", error=True)
            return False
        existing = self.annotations.labels()
        similar = similar_labels(label, existing, self.similarity_cutoff) if label not in existing else []
        if similar and self._confirm_label != label:
            self._confirm_label = label
            self.add_button.description = "Add anyway"
            self._message(f"'{label}' is new but looks like: {', '.join(similar)}. Pick one of those, "
                          f"or press <b>Add anyway</b> if it really is a different object.", error=True)
            return False
        self.annotations.add(self.current, BoxAnnotation(prompt.box, label, self.group_input.value))
        self._after_change(f"Added '{label}'.")
        self.prompt_widget.clear(notify=False)
        return True

    def delete_selected(self) -> None:
        """Delete the box selected in the list, then save."""
        idx = self.box_list.value
        if self.current is None or idx is None:
            self._message("Select a box in the list first.", error=True)
            return
        removed = self.annotations.remove(self.current, idx)
        self._after_change(f"Deleted box '{removed.label}'.")

    def relabel_selected(self) -> None:
        """Give the selected box the label and group currently typed, then save."""
        idx = self.box_list.value
        label = self.label_input.value.strip()
        if self.current is None or idx is None or not label:
            self._message("Select a box and type its new label.", error=True)
            return
        self.annotations.relabel(self.current, idx, label, self.group_input.value)
        self._after_change(f"Box relabelled to '{label}'.")

    # --- internals --------------------------------------------------------------------

    def _after_change(self, message: str) -> None:
        """Save, refresh everything and report."""
        try:
            self.annotations.save()
        except OSError as exc:
            logger.exception("Saving annotations failed")
            self._message(f"Could not save: {exc}", error=True)
            return
        self._confirm_label = None
        self.add_button.description = "Add box"
        self._refresh_vocabulary()
        self._refresh_image_options()
        self._refresh_boxes()
        self._refresh_reviewed()
        self._message(f"{html.escape(message)} Saved to {html.escape(self.annotations.path.name)}.")

    def _message(self, text: str, error: bool = False) -> None:
        """Show a status line (red for problems)."""
        self.status.value = f"<span style='color:#c00'>{text}</span>" if error else text

    def _refresh_vocabulary(self) -> None:
        """Update autocomplete options, the label table and the progress line."""
        self.label_input.options = tuple(self.annotations.labels())
        self.group_input.options = tuple(self.annotations.groups())
        stats = self.annotations.label_stats()
        self.stats.value = ("<i>No labels yet.</i>" if stats.empty
                            else stats.to_html(escape=True, border=0, justify="left"))
        self.progress.value = html.escape(self.annotations.summary())

    def _refresh_image_options(self) -> None:
        """Rebuild the image dropdown (✓ = reviewed, with box counts) without re-triggering navigation."""
        self._updating = True
        self.image_select.options = [
            (f"{'✓' if self.annotations.is_reviewed(p) else '·'} {p} ({len(self.annotations.boxes(p))})", p)
            for p in self.paths
        ]
        if self.current is not None:
            self.image_select.value = self.current
        self._updating = False

    def _refresh_boxes(self, highlight: int | None = None) -> None:
        """Redraw the current image's boxes and list them."""
        boxes = self.annotations.boxes(self.current) if self.current else []
        self.prompt_widget.show_boxes(
            [(b.box, b.label or "(unlabelled)", label_color(b.label)) for b in boxes], highlight=highlight)
        self._updating = True
        self.box_list.options = [
            (f"#{i} {b.label}{f' [{b.group}]' if b.group else ''}  ({', '.join(f'{v:.0f}' for v in b.box)})", i)
            for i, b in enumerate(boxes)
        ]
        self.box_list.value = highlight if highlight is not None and highlight < len(boxes) else None
        self._updating = False

    def _refresh_reviewed(self) -> None:
        """Sync the reviewed checkbox with the current image (locked while it has boxes)."""
        self._updating = True
        has_boxes = bool(self.current and self.annotations.boxes(self.current))
        self.reviewed.value = bool(self.current and self.annotations.is_reviewed(self.current))
        self.reviewed.disabled = has_boxes
        self._updating = False

    def _step(self, delta: int) -> None:
        """Previous/next image."""
        if not self.paths:
            return
        i = self.paths.index(self.current) if self.current in self.paths else 0
        self.show_image(self.paths[(i + delta) % len(self.paths)])

    def _image_selected(self, change: dict[str, Any]) -> None:
        """Dropdown callback."""
        if not self._updating and change["new"] is not None and change["new"] != self.current:
            self.show_image(change["new"])

    def _reviewed_toggled(self, change: dict[str, Any]) -> None:
        """Checkbox callback: mark/unmark an image without boxes as reviewed."""
        if self._updating or self.current is None:
            return
        if change["new"]:
            self.annotations.mark_reviewed(self.current)
            self._after_change(f"{self.current} marked as reviewed (no objects).")
        else:
            self.annotations.unmark_reviewed(self.current)
            self._after_change(f"{self.current} back to not labelled.")

    def _box_drawn(self, prompt: Prompt | None) -> None:
        """Canvas callback: a new box was drawn (or cleared)."""
        if prompt is not None and prompt.box is not None:
            self._message("Box drawn: type a label and press <b>Add box</b>.")

    def _box_selected(self, change: dict[str, Any]) -> None:
        """Box list callback: highlight the box and load its label/group into the fields."""
        if self._updating or change["new"] is None or self.current is None:
            return
        b = self.annotations.boxes(self.current)[change["new"]]
        self.label_input.value = b.label or ""
        self.group_input.value = b.group or ""
        self._refresh_boxes(highlight=change["new"])

    def _label_typed(self) -> None:
        """Label field callback: say whether the label exists, is new, or resembles another one."""
        label = self.label_input.value.strip()
        if self._confirm_label != label:
            self._confirm_label = None
            self.add_button.description = "Add box"
        if not label:
            self.label_info.value = ""
            return
        existing = self.annotations.labels()
        if label in existing:
            n = int(self.annotations.label_stats().loc[label, "boxes"])
            self.label_info.value = f"existing label ({n} box{'es' if n != 1 else ''})"
            return
        similar = similar_labels(label, existing, self.similarity_cutoff)
        hint = html.escape(", ".join(similar))
        warn = f"<span style='color:#c60'>new label; similar: {hint}</span>"
        self.label_info.value = warn if similar else "new label"
