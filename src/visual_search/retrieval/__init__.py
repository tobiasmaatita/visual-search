"""Stage 1: candidate retrieval."""

from visual_search.retrieval.base import Candidate, EmbeddingRetriever, Retriever
from visual_search.retrieval.baselines import ColorHistogramRetriever, RandomRetriever
from visual_search.retrieval.dino import POOLINGS, DinoRetriever
from visual_search.retrieval.grounding import GroundedDinoRetriever, Region, localize
from visual_search.retrieval.index import BruteForceIndex, l2_normalize

__all__ = [
    "BruteForceIndex",
    "Candidate",
    "ColorHistogramRetriever",
    "DinoRetriever",
    "EmbeddingRetriever",
    "GroundedDinoRetriever",
    "POOLINGS",
    "RandomRetriever",
    "Region",
    "Retriever",
    "l2_normalize",
    "localize",
]
