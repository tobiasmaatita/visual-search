"""Registry of matchers by name, so configs and notebooks select matchers with a string.

Example:
    >>> @register_matcher("sift")
    ... class SiftMatcher(Matcher): ...
    >>> matcher = create_matcher("sift", root_sift=True)
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

from visual_search.matching.base import Matcher

M = TypeVar("M", bound=type[Matcher])

_REGISTRY: dict[str, type[Matcher]] = {}

#: Name -> module that registers it, so heavy matchers are only imported when requested.
_LAZY_MODULES: dict[str, str] = {
    "sift": "visual_search.matching.sift",
    "aliked-lightglue": "visual_search.matching.lightglue",
    "xfeat-lighterglue": "visual_search.matching.lightglue",
}


def register_matcher(name: str) -> Callable[[M], M]:
    """Class decorator that registers a matcher under ``name``.

    Args:
        name: Unique registry name.

    Returns:
        The decorator (returns the class unchanged).

    Raises:
        ValueError: If ``name`` is already registered by a different class.
    """

    def decorator(cls: M) -> M:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Matcher name {name!r} already registered by {existing.__name__}.")
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def available_matchers() -> list[str]:
    """Return all known matcher names (registered or lazily importable)."""
    return sorted(set(_REGISTRY) | set(_LAZY_MODULES))


def create_matcher(name: str, **kwargs: Any) -> Matcher:
    """Instantiate a matcher by name.

    Args:
        name: Registry name (see :func:`available_matchers`).
        **kwargs: Passed to the matcher's constructor.

    Returns:
        The matcher.

    Raises:
        KeyError: If the name is unknown.
    """
    if name not in _REGISTRY and name in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[name])
    if name not in _REGISTRY:
        raise KeyError(f"Unknown matcher {name!r}; available: {available_matchers()}")
    return _REGISTRY[name](**kwargs)
