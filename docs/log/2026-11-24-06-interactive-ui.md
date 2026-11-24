# Interactive prompting UI and stage-1 results gallery

- Date: 2026-11-24
- Type: feature

## What changed
- **`ui/prompt_widget.py`: `PromptWidget`.** An ipympl canvas that serves two modes:
  - **Box mode:** drag a box (matplotlib `RectangleSelector`).
  - **Points mode:** left click is a positive point; right click or shift + click is a negative point.
  - Every change produces a dataset-schema `Prompt` and calls a callback.
  - A mask preview is drawn as an overlay.
  - One canvas is reused across image switches, using mlcvtools' tested `region_editor` approach and its ipympl #611 warning filter.
  - Clicks are ignored while zoom or pan is active.
- **`ui/search_app.py`: `SearchApp`.**
  - **Query selection:** a dropdown over the query set, or any image path.
  - **Dataset prompts:** a query's own prompt from the dataset is shown as the initial prompt.
  - **Live mask preview:** every prompt change runs the `Masker` (box mask by default, SAM3 once it passes), and the status line shows the mask latency.
  - **Search:** runs the same `Cascade` as the batch evaluation. The masked query uses `with_mask`, the same path as dataset queries.
  - **Results gallery:** the top-k after each stage with score, label, main score features and ✓/✗ when the query is labelled.
  - **Timings:** per-step timings (mask, each stage, total) after every run.
  - After a run, `app.last_result` and `app.timings` hold the details.
- **`viz/gallery.py`:** `render_ranking_grid` renders the per-stage gallery to a PNG with a matplotlib `Figure` (no pyplot), so it can also be used outside the app.
- **Notebook:** a new "Interactive search" section, using your real data if it has queries and the synthetic set otherwise.
- **`Cascade.is_prepared`:** a new property, so the app indexes the gallery only once.
- **Dependencies:** matplotlib, ipywidgets and ipympl (all BSD or PSF style) were added to the base dependencies, since the notebook UI is a core deliverable.

## Why
TASK.md, step 5. The widget choice is ipympl + `RectangleSelector` + `mpl_connect`, as proposed in the plan:
- It needs no custom JavaScript.
- It is already proven in mlcvtools' notebook.
- It works in JupyterLab.

Alternatives considered:
- **anywidget:** nicer UX, but needs its own JavaScript to maintain.
- **ipycanvas:** another dependency, and the drawing has to be done by hand.
- **jupyter-bbox-widget:** boxes only, no clicks.

## Impact
- **Behavior:** you can now prompt a query and browse the stage-1 results yourself. 81 tests pass. They include simulated mouse drags and clicks on the real ipympl backend, canvas reuse, the points and negative-click logic, and the app flow for a dataset query, a path query and a drawn box. The notebook executes headless, including building the app on the real `data/` folder.
- **Not verified:** use in an actual JupyterLab browser session. Starting a local server for a browser check was blocked in this session, so the owner checks this manually.
- **Code/interfaces:** the detail view for a selected result (inlier matches, warped mask) comes with matching in step 6. The gallery already shows the score features of each candidate, which will fill up once re-ranking stages add them.
- **Follow-ups:**
  - Clickable result thumbnails for the detail view (step 6).
  - Showing SAM3's alternative masks so you can pick one.

## Evidence (optional)
`uv run pytest tests/test_ui.py`. `notebooks/cascade.ipynb` executed with nbconvert.
