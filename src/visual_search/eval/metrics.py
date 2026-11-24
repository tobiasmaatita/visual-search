"""Ranking metrics: rank of the first correct result, Recall@k and MRR.

Conventions: ranks are 1-based. A query whose ranking contains no correct item
(e.g. because the correct items fell outside the top-K) has rank ``None``; it
counts as a miss for Recall@k and contributes 0 to MRR.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def first_correct_rank(ranked_ids: Sequence[str], correct_ids: Collection[str]) -> int | None:
    """Return the 1-based rank of the first correct item in a ranking.

    Args:
        ranked_ids: Gallery item ids, best first.
        correct_ids: Ids that count as correct for this query.

    Returns:
        The rank, or ``None`` if no correct item appears in ``ranked_ids``.

    Example:
        >>> first_correct_rank(["a", "b", "c"], {"c"})
        3
    """
    for i, item_id in enumerate(ranked_ids, start=1):
        if item_id in correct_ids:
            return i
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Return the fraction of queries with a correct item in the top ``k``.

    This is the "instance retrieval" Recall@k (hit rate): a query counts as a hit
    if *any* correct item is within the first ``k`` results.

    Args:
        ranks: Per-query first-correct ranks (``None`` = not found).
        k: Cut-off, >= 1.

    Returns:
        Recall@k in [0, 1]; NaN if ``ranks`` is empty.

    Raises:
        ValueError: If ``k < 1``.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if not ranks:
        return float("nan")
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Return the mean of ``1 / rank`` over queries (0 for not-found queries).

    Args:
        ranks: Per-query first-correct ranks (``None`` = not found).

    Returns:
        MRR in [0, 1]; NaN if ``ranks`` is empty.
    """
    if not ranks:
        return float("nan")
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)
