# visual-search

Exploration week: visual search. This repo builds a cascade for finding the *same instance* of a visually prompted object:

- **Stage 1:** DINOv2 retrieval.
- **Stage 2:** sparse geometric verification.
- **Stage 3:** dense matching and scoring.

**Start with the [repository guide](docs/guide.md).** It covers how to use the code, what data to provide, what the notebook shows, which models and matchers exist, and how to add new ones.

The assignment is in [TASK.md](TASK.md), the working rules in [CLAUDE.md](CLAUDE.md), and the change log in [docs/log/](docs/log/README.md).

## Setup
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. [mlcvtools](../mlcvtools) must be checked out next to this repo; it's installed as an editable path dependency.

```bash
uv sync --extra onnx --extra sparse   # base + dev + SAM3 runtime + learned matchers (kornia)
uv run pytest                         # run the tests
uv run ruff check src tests
```

### Notebooks
- [notebooks/labeling.ipynb](notebooks/labeling.ipynb): label your own test set. Draw boxes, give them item labels (and optional groups) and save them to `annotations.json`.
- [notebooks/search.ipynb](notebooks/search.ipynb): free play. Search your test set (with ✓/✗) or any new photo collection.
- [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb): measure and compare approaches on the test set (item and model level, grounding, per label, failures, latency). Results are saved to `results/`.

JupyterLab is a dev dependency, so after `uv sync`:

```bash
uv run jupyter lab
```

Open a notebook from `notebooks/`. The default "Python 3" kernel is the project's `.venv`. The interactive widgets use ipympl (`%matplotlib widget`), which is included in the dependencies.

To use the environment from a JupyterLab installed elsewhere, register it as a named kernel instead:

```bash
uv run python -m ipykernel install --user --name visual-search
```

To run a notebook headless (a smoke check):

```bash
cd notebooks && uv run jupyter nbconvert --to notebook --execute --stdout evaluation.ipynb > /dev/null
```

### Evaluating a pipeline in code
```python
from visual_search.cascade import Cascade
from visual_search.eval import evaluate, comparison_table
from visual_search.models.dinov2 import Dinov2Backbone
from visual_search.retrieval import DinoRetriever

from visual_search.stages import SparseVerificationStage

retriever = DinoRetriever(Dinov2Backbone("facebook/dinov2-small"), pooling="crop_cls", cache_dir="cache")
stage2 = SparseVerificationStage("sift", top_k=20, model_type="H", method="magsac", cache_dir="cache")
report = evaluate(Cascade(retriever, retrieval_k=100, stages=[stage2]), split)
report.metrics      # R@1, R@5, MRR per stage
report.latency      # median / p95 ms per stage and total
report.per_query    # rank and top-1 per query, for failure inspection
```

## Data
Real images live in `data/`. This folder is gitignored and never committed. Use one folder per physical instance; the folder name is the label:

```
data/
  bank_01/          # >= 2 photos of the same physical object
    a.jpg
    b.jpg
  billy_01/
    ...
  singletons/       # label=None distractors (also accepted: _distractors/, distractors/)
    room.jpg
  prompts.json      # optional: {"bank_01/a.jpg": {"box": [x0, y0, x1, y1]}}
```

Keep original JPG or PNG files. Convert HEIC to JPG first.

```python
from visual_search.data import load_folder_dataset, split_dataset, make_synthetic_dataset

split = split_dataset(load_folder_dataset("data"), seed=0, strict=False)
smoke = split_dataset(make_synthetic_dataset(seed=0), seed=0)  # SMOKE TEST ONLY
```

To read annotations drawn in mlcvtools' `visual-search.ipynb` (`own-annotations.json`), use `load_mlcvtools_annotations(path, image_root)`.

## Models
ONNX models go in `models/` (gitignored):
- `models/sam3/`: SAM3, split into vision encoder, geometry encoder, decoder and text encoder.
- `models/rf-detr/`: RF-DETR small and seg-small.

SAM3 needs the optional ONNX extra. The fp16 export is expected as `models/sam3/{vision-encoder,geometry-encoder,decoder}-fp16.onnx`. The adapter hasn't been run on the real model yet, so check it on target hardware first:

```bash
uv sync --extra onnx
uv run python scripts/try_sam3.py data/bank_01/img-001.png --box 63 335 535 568
```

A plain `uv sync` without the extras uninstalls onnxruntime and kornia again. The learned matchers (`aliked-lightglue`, `xfeat-lighterglue`) download their weights (about 67 MB) to `~/.cache/torch/hub/checkpoints` on first use.

DINOv2 is loaded with PyTorch from Hugging Face (`facebook/dinov2-small` by default; it's downloaded on first use). Extracted features are cached in `cache/` (gitignored), keyed by image content and model configuration. See [docs/log/2026-11-24-01-latency-budget-and-rejections.md](docs/log/2026-11-24-01-latency-budget-and-rejections.md) for which model variants are in or out, and why.
