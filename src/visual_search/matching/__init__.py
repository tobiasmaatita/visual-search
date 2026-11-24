"""Local-feature matchers (stage 2 sparse, stage 3 dense) behind one interface."""

from visual_search.matching.base import Correspondences, Features, Matcher, keypoints_in_mask
from visual_search.matching.registry import available_matchers, create_matcher, register_matcher

__all__ = [
    "Correspondences",
    "Features",
    "Matcher",
    "available_matchers",
    "create_matcher",
    "keypoints_in_mask",
    "register_matcher",
]
