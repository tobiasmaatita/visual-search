"""Dataset validation and the reproducible query/gallery split."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visual_search.data.schema import Item, parse_dataset


class DatasetValidationError(ValueError):
    """Raised when a dataset violates the schema rules and validation is strict."""


@dataclass
class ValidationReport:
    """Result of :func:`validate_dataset`.

    Attributes:
        errors: Rule violations that make the split impossible or wrong.
        warnings: Suspicious but usable situations.
        n_items: Total number of items.
        n_instances: Number of distinct non-None labels.
        n_distractors: Number of items with ``label=None``.
        images_per_instance: Label -> number of images.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_items: int = 0
    n_instances: int = 0
    n_distractors: int = 0
    images_per_instance: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return True if there are no errors."""
        return not self.errors

    def summary(self) -> str:
        """Return a short human-readable summary.

        Returns:
            Multi-line string with counts, errors and warnings.
        """
        lines = [f"{self.n_items} items: {self.n_instances} instances, {self.n_distractors} distractors"]
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_dataset(items: Sequence[Item], min_images_per_instance: int = 2) -> ValidationReport:
    """Check a parsed dataset against the schema rules.

    Rules (errors): every instance has at least ``min_images_per_instance`` images
    (one or more become queries, the rest the gallery); the dataset has at least
    one instance. Warnings: no distractors; empty-string labels.

    Images are not loaded here, so this is cheap.

    Args:
        items: Parsed items (see :func:`~visual_search.data.schema.parse_dataset`).
        min_images_per_instance: Minimum images per non-None label.

    Returns:
        A report; check ``report.ok``.
    """
    report = ValidationReport(n_items=len(items))
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        if it.label is None:
            report.n_distractors += 1
        else:
            counts[it.label] += 1
    report.images_per_instance = dict(counts)
    report.n_instances = len(counts)

    if not counts:
        report.errors.append("No labelled instances.")
    too_few = sorted(lbl for lbl, n in counts.items() if n < min_images_per_instance)
    if too_few:
        report.errors.append(
            f"{len(too_few)} instance(s) have fewer than {min_images_per_instance} images: {too_few[:10]}"
        )
    if report.n_distractors == 0:
        report.warnings.append("No distractors (label=None); retrieval will be unrealistically easy.")
    if "" in counts:
        report.warnings.append("Empty-string label found; did you mean None?")
    return report


@dataclass
class Split:
    """A query/gallery split.

    Attributes:
        queries: Items used as queries (always labelled).
        gallery: Items searched over: the remaining instance images plus all distractors.
        seed: The seed that produced this split.
    """

    queries: list[Item]
    gallery: list[Item]
    seed: int

    def summary(self) -> str:
        """Return a one-line description of the split sizes.

        Returns:
            E.g. ``"12 queries, 40 gallery (10 distractors), seed=0"``.
        """
        n_dis = sum(it.is_distractor for it in self.gallery)
        return f"{len(self.queries)} queries, {len(self.gallery)} gallery ({n_dis} distractors), seed={self.seed}"


def split_dataset(
    records: Sequence[Mapping[str, Any] | Item],
    queries_per_instance: int = 1,
    seed: int = 0,
    min_images_per_instance: int = 2,
    strict: bool = True,
) -> Split:
    """Validate a dataset and split it into queries and gallery, reproducibly.

    Per instance, ``queries_per_instance`` images (capped so that at least one
    image stays in the gallery) are drawn at random as queries; the rest go to the
    gallery. All distractors go to the gallery. The result depends only on the
    item ids, labels and ``seed``, not on the input order.

    Args:
        records: Raw dataset dicts or already parsed items.
        queries_per_instance: Queries drawn per instance.
        seed: Random seed.
        min_images_per_instance: Instances with fewer images are invalid.
        strict: If True, raise on validation errors. If False, drop invalid
            instances (with too few images) and continue.

    Returns:
        The split, with queries and gallery sorted by item id.

    Raises:
        DatasetValidationError: If ``strict`` and the dataset is invalid.
        ValueError: If ``queries_per_instance < 1``.

    Example:
        >>> split = split_dataset(records, seed=42)
        >>> print(split.summary())
    """
    if queries_per_instance < 1:
        raise ValueError("queries_per_instance must be >= 1.")
    items = parse_dataset(records)  # type: ignore[arg-type]
    report = validate_dataset(items, min_images_per_instance)
    if strict and not report.ok:
        raise DatasetValidationError(report.summary())

    by_label: dict[str, list[Item]] = defaultdict(list)
    gallery: list[Item] = []
    for it in items:
        if it.label is None:
            gallery.append(it)
        else:
            by_label[it.label].append(it)

    rng = np.random.default_rng(seed)
    queries: list[Item] = []
    for label in sorted(by_label):
        members = sorted(by_label[label], key=lambda it: it.item_id)
        if len(members) < min_images_per_instance:
            continue  # only reachable when strict=False
        n_q = min(queries_per_instance, len(members) - 1)
        q_idx = set(rng.choice(len(members), size=n_q, replace=False).tolist())
        for i, it in enumerate(members):
            (queries if i in q_idx else gallery).append(it)

    return Split(
        queries=sorted(queries, key=lambda it: it.item_id),
        gallery=sorted(gallery, key=lambda it: it.item_id),
        seed=seed,
    )
