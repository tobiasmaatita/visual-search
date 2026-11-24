"""Batch evaluation of a cascade on a query/gallery split.

Reports Recall@k and MRR after every stage (so "stage 1 only" vs "after
re-ranking" comes for free) and per-stage latency (median, p95).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from visual_search.cascade import Cascade
from visual_search.data.schema import is_match
from visual_search.data.split import Split
from visual_search.eval.metrics import first_correct_rank, mean_reciprocal_rank, recall_at_k
from visual_search.timing import LatencyStats

logger = logging.getLogger(__name__)

TOTAL = "total"


@dataclass
class EvalReport:
    """Result of :func:`evaluate`.

    Attributes:
        name: Pipeline identifier (see :attr:`Cascade.name`) or a custom name.
        split_summary: Description of the evaluated split.
        metrics: One row per stage; columns ``R@k`` for each k, ``MRR``, ``n_queries``.
        latency: One row per stage plus ``"total"``; columns ``median_ms``,
            ``p95_ms``, ``mean_ms``, ``max_ms``.
        per_query: One row per query: ``query_id``, ``label``, ``rank@<stage>``,
            ``top1@<stage>``, ``ms@<stage>``. Use it to inspect failures.
        prepare_s: Stage name -> seconds spent indexing the gallery (offline).
        n_skipped: Queries without any correct gallery item (excluded from metrics).
    """

    name: str
    split_summary: str
    metrics: pd.DataFrame
    latency: pd.DataFrame
    per_query: pd.DataFrame
    prepare_s: dict[str, float]
    n_skipped: int = 0

    def __str__(self) -> str:
        """Return a plain-text report with both tables."""
        return (
            f"== {self.name} | {self.split_summary}\n"
            f"{self.metrics.to_string(float_format=lambda v: f'{v:.3f}')}\n\n"
            f"{self.latency.to_string(float_format=lambda v: f'{v:.1f}')}"
        )


def evaluate(
    cascade: Cascade,
    split: Split,
    ks: Sequence[int] = (1, 5),
    name: str | None = None,
    prepare: bool = True,
) -> EvalReport:
    """Run every query of ``split`` through ``cascade`` and compute metrics.

    A query's correct items are the gallery items with the same non-None label
    (:func:`~visual_search.data.schema.is_match`). Queries without any correct
    gallery item are skipped with a warning (cannot happen for splits made by
    :func:`~visual_search.data.split.split_dataset`).

    Args:
        cascade: The pipeline to evaluate.
        split: Queries and gallery.
        ks: Cut-offs for Recall@k.
        name: Report name; defaults to ``cascade.name``.
        prepare: Call ``cascade.prepare(split.gallery)`` first. Set False to reuse
            an already prepared cascade (e.g. after an interactive session).

    Returns:
        The evaluation report.

    Example:
        >>> report = evaluate(Cascade(ColorHistogramRetriever()), split)
        >>> report.metrics
    """
    prepare_s = cascade.prepare(split.gallery) if prepare else {}
    stage_names: list[str] | None = None
    ranks: dict[str, list[int | None]] = {}
    times: dict[str, list[float]] = {}
    rows: list[dict] = []
    n_skipped = 0

    for query in split.queries:
        correct = {g.item_id for g in split.gallery if is_match(query, g)}
        if not correct:
            logger.warning("Query %s has no correct gallery item; skipped.", query.item_id)
            n_skipped += 1
            continue
        result = cascade.run(query)
        if stage_names is None:
            stage_names = list(result.stages)
            ranks = {s: [] for s in stage_names}
            times = {s: [] for s in [*stage_names, TOTAL]}
        row: dict = {"query_id": query.item_id, "label": query.label}
        for stage, ranking in result.stages.items():
            rank = first_correct_rank([c.item_id for c in ranking], correct)
            ranks[stage].append(rank)
            times[stage].append(result.timings.get(stage, 0.0))
            row[f"rank@{stage}"] = rank
            row[f"top1@{stage}"] = ranking[0].item_id if ranking else None
            row[f"ms@{stage}"] = 1000 * result.timings.get(stage, 0.0)
        times[TOTAL].append(sum(result.timings.values()))
        rows.append(row)

    stage_names = stage_names or []
    metrics = pd.DataFrame(
        {
            **{f"R@{k}": [recall_at_k(ranks[s], k) for s in stage_names] for k in ks},
            "MRR": [mean_reciprocal_rank(ranks[s]) for s in stage_names],
            "n_queries": [len(ranks[s]) for s in stage_names],
        },
        index=pd.Index(stage_names, name="stage"),
    )
    lat_rows = {}
    for stage, samples in times.items():
        st = LatencyStats.from_samples(samples)
        lat_rows[stage] = {"median_ms": 1000 * st.median, "p95_ms": 1000 * st.p95,
                           "mean_ms": 1000 * st.mean, "max_ms": 1000 * st.max}
    latency = pd.DataFrame.from_dict(lat_rows, orient="index", columns=["median_ms", "p95_ms", "mean_ms", "max_ms"])
    latency.index.name = "stage"
    return EvalReport(
        name=name or cascade.name,
        split_summary=split.summary(),
        metrics=metrics,
        latency=latency,
        per_query=pd.DataFrame(rows),
        prepare_s=prepare_s,
        n_skipped=n_skipped,
    )


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
