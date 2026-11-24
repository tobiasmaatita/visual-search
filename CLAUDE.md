# CLAUDE.md: standing rules for this project

These rules apply to every session. The actual assignment lives in `TASK.md`.

## Language and code style
- English everywhere: code, comments, docstrings, docs, notebook text.
- Type hints on all function signatures.
- **Docstrings on all methods.**
  - Public methods: thorough. Cover purpose, args, returns (with shapes/units), raises, and a short example where useful.
  - Private methods: brief, one or two lines.
- Comments only where the *why* is not obvious from the code.
- Small, focused modules. Prefer common abstract interfaces (e.g. `Retriever`, `Matcher`, `Verifier`, `Scorer`) so methods are pluggable.
- Keep configuration light: dataclasses or a small YAML file.

## Project structure conventions
- Main deliverable is an experimental Jupyter notebook that drives a small, modular Python package.
  - Notebook = orchestration, experiments, result tables.
  - Package = reusable logic. No heavy logic lives only in the notebook.
- Cache expensive outputs (embeddings, masks, keypoints, matches) on disk, keyed by image id + model + config, so experiments are fast to rerun.
- Dependency management uses **uv**, as in all my projects:
  - Dependencies live in `pyproject.toml` with a committed `uv.lock`. No `requirements.txt` and no bare `pip install`.
  - Add packages with `uv add` (dev-only tools with `uv add --dev`), run code with `uv run`, and create the environment with `uv sync`.
  - Put optional or heavy extras (e.g. MASt3R, FAISS) in optional dependency groups so the base install stays light.
  - Document the setup (`uv sync`, how to start the notebook kernel) in the README.
  - Adding mlcvtools as a dependency (e.g. `uv add --editable ../mlcvtools`) can write files into the mlcvtools directory. Ask me first, per the mlcvtools rules.
- The notebook is interactive (draw a box / click on the query image, browse results). Widget and display code lives in the package (e.g. `ui/`), not inline in notebook cells. Visual polish is out of scope for now; keep it functional and plain.

## mlcvtools (read-only dependency)
- `mlcvtools` is a local, in-development library that lives next to this repo (`../mlcvtools`). It is used for model loading and more. Read it freely to get oriented.
- **Never modify anything in mlcvtools without my explicit permission.** This covers code, notebooks, configs, tests and generated files, including side effects such as editable-install build artifacts.
- If you think a change in mlcvtools is needed, describe it (file, proposed diff, reason) and ask. Meanwhile, work around it with an adapter in this repo.
- Before running a command that could write into the mlcvtools directory (e.g. `pip install -e`, scripts with relative output paths), check where it writes. Never run state-changing git commands there (commit, checkout, stash, reset, clean).
- Prefer reusing mlcvtools over reimplementing. State reused vs new components in plans.

## Licenses
- Only open-source, research-friendly licenses.
- Verify the license of every model and library before adopting it, and list them in the plan.
- Anything non-commercial or restrictive (e.g. MASt3R) must be optional and clearly marked. Ask before adopting it.

## Working method
- Plan first, then build step by step. Never start coding a new phase without an approved plan.
- Stop after each step with a short summary: what works, what was verified, what is next.
- Ask me when a decision is mine to make (design choices, license doubts, ambiguous requirements) instead of guessing silently.
- Do not claim something works unless you ran it. Report results honestly, including failures and caveats.
- Use git: commit after each verified step with a clear message.

## Change log (`docs/log/`)
Every significant change gets a short log entry in `docs/log/`, written for a human who wants to understand what happened and why, not at code level.

**When to write an entry** (one per significant commit):
- a new component, pipeline stage, or matcher
- a change to an interface, the dataset schema, or defaults
- a dependency added, removed, or swapped
- a design decision, or a deviation from the approved plan
- an experiment or evaluation run with results worth keeping
- a bug fix that changes behavior or results
- a change to mlcvtools that I approved, or one you proposed and I declined or postponed

Skip it for typos, formatting, comment edits, and small refactors without behavior change.

**How:**
- One Markdown file per entry, in the same commit as the change it describes. Do not put the commit hash in the file; `git log -- docs/log/<file>` links the two.
- File name: `YYYY-MM-DD-NN-short-slug.md` (NN = running number for that day), e.g. `2026-11-24-01-dinov2-retrieval.md`.
- Also add one line per entry to the index `docs/log/README.md`, newest first: date, title, link.
- Keep entries short: a few sentences per section, bullets where useful. Numbers (metrics, latency) belong in the entry when they exist.

**Template:**
```markdown
# <Short title>

- Date: YYYY-MM-DD
- Type: feature | refactor | decision | experiment | fix | dependency

## What changed
Plain-language summary of the change.

## Why
The motivation or problem. Mention alternatives considered and why they were rejected, if relevant.

## Impact
- Behavior/results: what works differently, and metrics before/after if measured
- Code/interfaces: what other parts are affected
- Follow-ups: open issues, risks, next steps

## Evidence (optional)
Notebook, config, or cached-run reference, so the result can be reproduced.
```