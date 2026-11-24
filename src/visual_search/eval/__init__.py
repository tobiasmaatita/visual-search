"""Evaluation: ranking metrics, latency statistics and the batch harness."""

from visual_search.eval.harness import EvalReport, comparison_table, evaluate
from visual_search.eval.metrics import first_correct_rank, mean_reciprocal_rank, recall_at_k

__all__ = ["EvalReport", "comparison_table", "evaluate", "first_correct_rank", "mean_reciprocal_rank", "recall_at_k"]
