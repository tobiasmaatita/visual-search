"""Wall-clock timing of pipeline stages and latency statistics."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StageTimer:
    """Collect wall-clock durations per named stage for one run.

    GPU work is asynchronous, so pass a ``sync`` callable (e.g.
    ``torch.cuda.synchronize``) to make the timings include queued kernels.

    Attributes:
        timings: Stage name -> seconds. Re-entering a stage adds to its total.
        sync: Optional callable invoked before reading the clock.

    Example:
        >>> timer = StageTimer()
        >>> with timer.stage("retrieval"):
        ...     run_retrieval()
        >>> timer.timings["retrieval"]
        0.012
    """

    timings: dict[str, float] = field(default_factory=dict)
    sync: Callable[[], None] | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block and add it to ``timings[name]``.

        Args:
            name: Stage name.

        Yields:
            Nothing; the duration is recorded on exit, also when the block raises.
        """
        if self.sync:
            self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                self.sync()
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - t0

    @property
    def total(self) -> float:
        """Return the sum of all stage durations in seconds."""
        return sum(self.timings.values())


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics of a set of durations, in seconds.

    Attributes:
        n: Number of samples.
        median: 50th percentile.
        p95: 95th percentile (linear interpolation).
        mean: Arithmetic mean.
        max: Largest sample.
    """

    n: int
    median: float
    p95: float
    mean: float
    max: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> LatencyStats:
        """Compute statistics from duration samples.

        Args:
            samples: Durations in seconds.

        Returns:
            The statistics; all values are NaN if ``samples`` is empty.
        """
        if len(samples) == 0:
            nan = float("nan")
            return cls(0, nan, nan, nan, nan)
        a = np.asarray(samples, dtype=np.float64)
        return cls(len(a), float(np.median(a)), float(np.percentile(a, 95)), float(a.mean()), float(a.max()))
