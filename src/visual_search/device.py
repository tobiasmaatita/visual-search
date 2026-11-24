"""Torch device selection and synchronization for honest timings."""

from __future__ import annotations

from collections.abc import Callable

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a torch device.

    Args:
        preference: ``"auto"`` (CUDA, then Apple MPS, then CPU) or an explicit
            device string such as ``"cpu"``, ``"cuda:0"`` or ``"mps"``.

    Returns:
        The selected device.

    Raises:
        ValueError: If an explicitly requested accelerator is not available.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but not available.")
    return device


def device_sync(device: torch.device) -> Callable[[], None] | None:
    """Return a function that waits for queued work on ``device``, or None for CPU.

    Pass the result to :class:`~visual_search.timing.StageTimer` or
    :class:`~visual_search.cascade.Cascade` so timings include asynchronous GPU work.

    Args:
        device: The device used for inference.

    Returns:
        A synchronization callable, or ``None`` when no synchronization is needed.
    """
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        return torch.mps.synchronize
    return None
