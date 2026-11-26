"""Helpers for the evaluation notebook: breakdowns, failure grids, localization, latency, CSV export."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from visual_search.cascade import Cascade
from visual_search.data.schema import Item
from visual_search.eval.harness import TOTAL, EvalReport
from visual_search.eval.judges import GroundedJudge, Judge, box_iou
from visual_search.retrieval.base import Retriever
from visual_search.viz.gallery import box_thumbnail, render_ranking_grid, thumbnail


def per_label_table(report: EvalReport, stage: str | None = None) -> pd.DataFrame:
    """Break a report down per label: how well is each object found?

    Args:
        report: An evaluation report.
        stage: Stage to analyse; default = the last stage.

    Returns:
        One row per label (hardest first): number of queries, Recall@k for the
        report's cut-offs, MRR, median rank of the first correct result, and how
        many of its queries found nothing.
    """
    stage = stage or report.metrics.index[-1]
    col = f"rank@{stage}"
    pq = report.per_query
    if pq.empty:
        return pd.DataFrame()
    rows = {}
    for label, grp in pq.groupby("label"):
        r = grp[col]
        found = r.dropna()
        row = {"queries": len(r)}
        for k in report.ks:
            row[f"R@{k}"] = float((r <= k).sum()) / len(r)
        row["MRR"] = float((1.0 / found).sum()) / len(r)
        row["median rank"] = float(found.median()) if len(found) else float("nan")
        row["not found"] = int(r.isna().sum())
        rows[label] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "label"
    return df.sort_values(["MRR", "queries"], ascending=[True, False])


def worst_queries(report: EvalReport, n: int = 5, stage: str | None = None) -> list[str]:
    """Return the ids of the ``n`` queries with the worst rank (not found first)."""
    stage = stage or report.metrics.index[-1]
    pq = report.per_query.sort_values(f"rank@{stage}", ascending=False, na_position="first")
    return pq["query_id"].head(n).tolist()


def failure_grid(
    cascade: Cascade,
    query: Item,
    gallery: Sequence[Item],
    judge: Judge,
    k: int = 5,
    exclude_same_image: bool = True,
) -> bytes:
    """Re-run one query and render its result grid (query vs top-k per stage, with ✓/✗).

    The cascade must already be prepared on ``gallery``.

    Args:
        cascade: The prepared pipeline.
        query: The query item.
        gallery: The gallery it was evaluated against.
        judge: Decides the ✓/✗ marks.
        k: Results shown per stage.
        exclude_same_image: Leave out the query's own photo, as in the evaluation.

    Returns:
        PNG bytes.
    """
    by_id = {g.item_id: g for g in gallery}
    image_id = query.extra.get("image_id", query.item_id)
    exclude = {g.item_id for g in gallery if exclude_same_image and g.extra.get("image_id", g.item_id) == image_id}
    exclude |= {query.item_id} & by_id.keys()
    result = cascade.run(query, exclude=exclude)
    thumbs: dict = {}

    def load(c):
        key = (c.item_id, c.region)
        if key not in thumbs:
            item = by_id[c.item_id]
            image = item.load_image()
            box = c.region or (item.prompt.bbox() if item.prompt is not None and item.prompt.mask is None else None)
            thumbs[key] = box_thumbnail(image, box) if box is not None else thumbnail(image)
        return thumbs[key]

    q_image = query.load_image()
    q_vis = box_thumbnail(q_image, query.prompt.bbox(), pad=0.6, max_side=512) if query.prompt else q_image
    return render_ranking_grid(
        q_vis, result.stages, load, lambda c: judge.correct(query, by_id[c.item_id], c), k=k,
        label_of=lambda c: by_id[c.item_id].label, title=f"{query.item_id} (label {query.label})",
    )


def localization_stats(
    retriever: Retriever,
    queries: Sequence[Item],
    gallery: Sequence[Item],
    judge: GroundedJudge,
    max_candidates: int = 1000,
) -> dict[str, float]:
    """How well do grounded regions overlap the annotated boxes?

    For each query, take the highest-ranked candidate (stage 1) in another photo that
    contains the query's object, and the best IoU between its region and that
    object's boxes. Summarized over all queries.

    Args:
        retriever: A fitted grounding retriever (``GroundedDinoRetriever``).
        queries: Query items (test-set boxes).
        gallery: The gallery it was fitted on (test-set images without boxes).
        judge: Provides the annotated boxes per image (its threshold is not used here).
        max_candidates: How deep to search for the first candidate in a correct photo.

    Returns:
        ``{"queries", "median_iou", "share_iou_0.3", "share_iou_0.5"}``.
    """
    by_id = {g.item_id: g for g in gallery}
    ious = []
    for q in queries:
        image_id = q.extra.get("image_id", q.item_id)
        for c in retriever.search(q, max_candidates):
            item = by_id[c.item_id]
            if item.extra.get("image_id", item.item_id) == image_id:
                continue
            boxes = judge.matching_boxes(q, item)
            if boxes and c.region is not None:
                ious.append(max(box_iou(c.region, b) for b in boxes))
                break
    arr = np.array(ious)
    return {
        "queries": len(arr),
        "median_iou": float(np.median(arr)) if len(arr) else float("nan"),
        "share_iou_0.3": float((arr >= 0.3).mean()) if len(arr) else float("nan"),
        "share_iou_0.5": float((arr >= 0.5).mean()) if len(arr) else float("nan"),
    }


def latency_table(reports: Sequence[EvalReport]) -> pd.DataFrame:
    """Median and p95 latency per stage and in total (ms), one row per report.

    Args:
        reports: Evaluation reports (stage names may differ between them).

    Returns:
        DataFrame indexed by report name with ``<stage> median_ms`` / ``<stage> p95_ms`` columns.
    """
    rows = {}
    for rep in reports:
        row = {}
        for stage in [*[s for s in rep.latency.index if s != TOTAL], TOTAL]:
            row[f"{stage} median_ms"] = rep.latency.loc[stage, "median_ms"]
            row[f"{stage} p95_ms"] = rep.latency.loc[stage, "p95_ms"]
        rows[rep.name] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "pipeline"
    return df


def save_tables(tables: Mapping[str, pd.DataFrame], root: str | Path, run_name: str = "eval") -> Path:
    """Write tables as CSV into ``<root>/<date-time>_<run_name>/<table>.csv``.

    Args:
        tables: File stem -> table.
        root: Results directory (e.g. ``<repo>/results``, gitignored).
        run_name: Short name for this run.

    Returns:
        The run directory.
    """
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", run_name).strip("-") or "eval"
    run_dir = Path(root) / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for stem, table in tables.items():
        table.to_csv(run_dir / f"{re.sub(r'[^A-Za-z0-9_-]+', '-', stem)}.csv")
    return run_dir
