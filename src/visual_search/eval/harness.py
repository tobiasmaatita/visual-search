"""Batch evaluation of a cascade.

Reports Recall@k and MRR after every stage (so "stage 1 only" vs "after
re-ranking" comes for free), for one or more correctness levels (judges: exact
item, same model, grounded region), plus per-stage latency (median, p95).

Protocols: :func:`evaluate` (one query per label, from ``split_dataset``) and
:func:`evaluate_leave_one_out` (every labelled item is a query once).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from visual_search.cascade import Cascade
from visual_search.data.schema import Item
from visual_search.data.split import Split
from visual_search.eval.judges import GroupJudge, ItemJudge, Judge
from visual_search.eval.metrics import mean_reciprocal_rank, recall_at_k
from visual_search.timing import LatencyStats

logger = logging.getLogger(__name__)

TOTAL = "total"


@dataclass
class EvalReport:
    """Result of :func:`evaluate`.

    Attributes:
        name: Pipeline identifier (see :attr:`Cascade.name`) or a custom name.
        split_summary: Description of the evaluated split.
        metrics: One row per stage; columns ``R@k`` for each k and ``MRR`` for the primary
            judge, the same prefixed with the level for other judges (``"model R@1"``),
            and ``n_queries``.
        latency: One row per stage plus ``"total"``; columns ``median_ms``,
            ``p95_ms``, ``mean_ms``, ``max_ms``.
        per_query: One row per query: ``query_id``, ``label``, ``group``, ``rank@<stage>``
            (primary judge; ``"<level> rank@<stage>"`` for others), ``top1@<stage>``,
            ``ms@<stage>``. Use it to inspect failures.
        prepare_s: Stage name -> seconds spent indexing the gallery (offline).
        n_skipped: Queries without any correct gallery item (excluded from metrics).
        ks: The Recall@k cut-offs used.
        levels: Judge names, primary first (e.g. ``("item", "model")``).
    """

    name: str
    split_summary: str
    metrics: pd.DataFrame
    latency: pd.DataFrame
    per_query: pd.DataFrame
    prepare_s: dict[str, float]
    n_skipped: int = 0
    ks: tuple[int, ...] = (1, 5)
    levels: tuple[str, ...] = ("item",)

    def __str__(self) -> str:
        """Return a plain-text report with both tables."""
        return (
            f"== {self.name} | {self.split_summary}\n"
            f"{self.metrics.to_string(float_format=lambda v: f'{v:.3f}')}\n\n"
            f"{self.latency.to_string(float_format=lambda v: f'{v:.1f}')}"
        )


def _image_id(item: Item) -> str:
    """Photo an item belongs to (``extra["image_id"]`` for test-set boxes, else the item id)."""
    return item.extra.get("image_id", item.item_id)


def run_evaluation(
    cascade: Cascade,
    queries: Sequence[Item],
    gallery: Sequence[Item],
    judges: Mapping[str, Judge] | None = None,
    ks: Sequence[int] = (1, 5),
    name: str | None = None,
    prepare: bool = True,
    exclude_same_image: bool = True,
    description: str = "",
) -> EvalReport:
    """Run every query through ``cascade`` and compute metrics for one or more judges.

    The first judge is the **primary** one: its metrics get plain column names
    (``R@1``, ``R@5``, ``MRR``) and it decides which queries are answerable. Every
    other judge adds prefixed columns (``"model R@1"``, ...), so e.g. item-level and
    model-level results come from the same run.

    Args:
        cascade: The pipeline to evaluate.
        queries: Query items.
        gallery: Gallery items (indexed with ``cascade.prepare`` unless ``prepare=False``).
        judges: Level name -> judge; default ``{"item": ItemJudge()}``.
        ks: Cut-offs for Recall@k.
        name: Report name; defaults to ``cascade.name``.
        prepare: Index the gallery first (False reuses an already prepared cascade).
        exclude_same_image: Leave out every gallery item from the query's own photo
            (its own box, other boxes in it, and the photo itself), so a query must be
            found in *other* photos. Items without an ``image_id`` exclude only themselves.
        description: Text for the report header (e.g. the protocol).

    Returns:
        The evaluation report; queries without any correct gallery item (for the
        primary judge, after exclusion) are skipped and counted in ``n_skipped``.
    """
    judges = dict(judges) if judges else {"item": ItemJudge()}
    levels = list(judges)
    primary = levels[0]
    prepare_s = cascade.prepare(gallery) if prepare else {}
    gallery_by_id = {g.item_id: g for g in gallery}
    by_image: dict[str, list[str]] = {}
    for g in gallery:
        by_image.setdefault(_image_id(g), []).append(g.item_id)

    stage_names: list[str] | None = None
    ranks: dict[tuple[str, str], list[int | None]] = {}
    times: dict[str, list[float]] = {}
    rows: list[dict] = []
    n_skipped = 0
    for query in queries:
        if exclude_same_image:
            exclude = set(by_image.get(_image_id(query), [])) | ({query.item_id} & gallery_by_id.keys())
        else:
            exclude = {query.item_id} & gallery_by_id.keys()
        candidates = [g for g in gallery if g.item_id not in exclude]
        if not judges[primary].has_positive(query, candidates):
            logger.warning("Query %s has no correct gallery item; skipped.", query.item_id)
            n_skipped += 1
            continue
        result = cascade.run(query, exclude=exclude)
        if stage_names is None:
            stage_names = list(result.stages)
            ranks = {(lvl, s): [] for lvl in levels for s in stage_names}
            times = {s: [] for s in [*stage_names, TOTAL]}
        row: dict = {"query_id": query.item_id, "label": query.label, "group": query.group}
        for stage, ranking in result.stages.items():
            for lvl in levels:
                judge = judges[lvl]
                rank = next((i for i, c in enumerate(ranking, 1)
                             if judge.correct(query, gallery_by_id[c.item_id], c)), None)
                ranks[(lvl, stage)].append(rank)
                row[f"rank@{stage}" if lvl == primary else f"{lvl} rank@{stage}"] = rank
            row[f"top1@{stage}"] = ranking[0].item_id if ranking else None
            row[f"ms@{stage}"] = 1000 * result.timings.get(stage, 0.0)
            times[stage].append(result.timings.get(stage, 0.0))
        times[TOTAL].append(sum(result.timings.values()))
        rows.append(row)

    stage_names = stage_names or []
    columns: dict[str, list] = {}
    for lvl in levels:
        prefix = "" if lvl == primary else f"{lvl} "
        for k in ks:
            columns[f"{prefix}R@{k}"] = [recall_at_k(ranks[(lvl, s)], k) for s in stage_names]
        columns[f"{prefix}MRR"] = [mean_reciprocal_rank(ranks[(lvl, s)]) for s in stage_names]
    columns["n_queries"] = [len(ranks[(primary, s)]) for s in stage_names]
    metrics = pd.DataFrame(columns, index=pd.Index(stage_names, name="stage"))
    lat_rows = {}
    for stage, samples in times.items():
        st = LatencyStats.from_samples(samples)
        lat_rows[stage] = {"median_ms": 1000 * st.median, "p95_ms": 1000 * st.p95,
                           "mean_ms": 1000 * st.mean, "max_ms": 1000 * st.max}
    latency = pd.DataFrame.from_dict(lat_rows, orient="index", columns=["median_ms", "p95_ms", "mean_ms", "max_ms"])
    latency.index.name = "stage"
    return EvalReport(
        name=name or cascade.name,
        split_summary=description or f"{len(queries)} queries, {len(gallery)} gallery",
        metrics=metrics,
        latency=latency,
        per_query=pd.DataFrame(rows),
        prepare_s=prepare_s,
        n_skipped=n_skipped,
        ks=tuple(ks),
        levels=tuple(levels),
    )


def evaluate(
    cascade: Cascade,
    split: Split,
    ks: Sequence[int] = (1, 5),
    name: str | None = None,
    prepare: bool = True,
    judges: Mapping[str, Judge] | None = None,
) -> EvalReport:
    """Evaluate on a query/gallery split (one query per label, see ``split_dataset``).

    A query's correct items are the gallery items with the same non-None label
    (:class:`~visual_search.eval.judges.ItemJudge`), unless other ``judges`` are given.
    Gallery items from the query's own photo are excluded.

    Args:
        cascade: The pipeline to evaluate.
        split: Queries and gallery.
        ks: Cut-offs for Recall@k.
        name: Report name; defaults to ``cascade.name``.
        prepare: Call ``cascade.prepare(split.gallery)`` first.
        judges: Level name -> judge (first = primary); default item level only.

    Returns:
        The evaluation report.

    Example:
        >>> report = evaluate(Cascade(ColorHistogramRetriever()), split)
        >>> report.metrics
    """
    return run_evaluation(cascade, split.queries, split.gallery, judges, ks, name, prepare,
                          description=split.summary())


def evaluate_leave_one_out(
    cascade: Cascade,
    items: Sequence[Item],
    ks: Sequence[int] = (1, 5),
    name: str | None = None,
    prepare: bool = True,
    judges: Mapping[str, Judge] | None = None,
) -> EvalReport:
    """Evaluate with every labelled item as a query once, against all other items.

    The gallery is ``items`` itself; for each query, everything from its own photo is
    excluded (so it must be found in other photos). Uses all labels of a small test
    set, giving many more queries than one-per-label splitting.

    Args:
        cascade: The pipeline to evaluate.
        items: All test items (labelled boxes and distractors).
        ks: Cut-offs for Recall@k.
        name: Report name; defaults to ``cascade.name``.
        prepare: Index ``items`` as the gallery first.
        judges: Level name -> judge (first = primary), e.g.
            ``{"item": ItemJudge(), "model": GroupJudge()}``.

    Returns:
        The evaluation report (one per-query row per answerable labelled item).

    Example:
        >>> items = parse_dataset(AnnotationSet.load("data/testset").to_records())
        >>> report = evaluate_leave_one_out(cascade, items, judges=default_judges())
    """
    queries = [it for it in items if it.label is not None]
    return run_evaluation(cascade, queries, items, judges, ks, name, prepare,
                          description=f"leave-one-out: {len(queries)} queries, {len(items)} gallery items")


def default_judges() -> dict[str, Judge]:
    """Return the item- and model-level judges (``{"item": ..., "model": ...}``)."""
    return {"item": ItemJudge(), "model": GroupJudge()}


def comparison_table(reports: Sequence[EvalReport], stage: str | None = None) -> pd.DataFrame:
    """Combine several reports into one ablation table.

    Args:
        reports: Reports to compare (typically on the same split).
        stage: Stage row to take from each report; default = each report's last stage.

    Returns:
        One row per report: its metrics at ``stage`` plus total median/p95 latency (ms).
    """
    rows = {}
    for rep in reports:
        st = stage if stage is not None else rep.metrics.index[-1]
        row = rep.metrics.loc[st].to_dict()
        row["stage"] = st
        row["total_median_ms"] = rep.latency.loc[TOTAL, "median_ms"]
        row["total_p95_ms"] = rep.latency.loc[TOTAL, "p95_ms"]
        rows[rep.name] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "pipeline"
    return table
