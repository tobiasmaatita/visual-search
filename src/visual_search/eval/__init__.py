"""Evaluation: ranking metrics, correctness judges, latency statistics and the batch harness."""

from visual_search.eval.harness import (
    EvalReport,
    comparison_table,
    default_judges,
    evaluate,
    evaluate_leave_one_out,
    run_evaluation,
)
from visual_search.eval.judges import GroundedJudge, GroupJudge, ItemJudge, Judge, box_iou
from visual_search.eval.metrics import first_correct_rank, mean_reciprocal_rank, recall_at_k

__all__ = [
    "EvalReport",
    "GroundedJudge",
    "GroupJudge",
    "ItemJudge",
    "Judge",
    "box_iou",
    "comparison_table",
    "default_judges",
    "evaluate",
    "evaluate_leave_one_out",
    "first_correct_rank",
    "mean_reciprocal_rank",
    "recall_at_k",
    "run_evaluation",
]
