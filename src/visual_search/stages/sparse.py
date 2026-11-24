"""Stage 2: sparse matching + robust geometric verification as a re-ranking stage."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from visual_search.cache import FeatureCache, image_fingerprint
from visual_search.cascade import RerankStage
from visual_search.data.schema import Item
from visual_search.geometry.verify import Method, ModelType, verify
from visual_search.imaging import prompt_to_mask
from visual_search.matching.base import Features, Matcher
from visual_search.matching.registry import create_matcher
from visual_search.retrieval.base import Candidate


class SparseVerificationStage(RerankStage):
    """Re-rank candidates by the number of geometrically verified matches.

    For each of the top-``top_k`` candidates: match local features (restricted to
    the query mask and, if the gallery item has a prompt, its mask), fit a robust
    model and count inliers. The new score is ``inliers + stage-1 score``: the
    inlier count dominates and the previous score only breaks ties (it is in
    [-1, 1] for cosine retrieval). Candidates whose model fails or has fewer than
    ``min_inliers`` inliers get 0 inliers. Step 7 replaces this by a combiner over
    several score features.

    Gallery features are extracted once in :meth:`prepare` and cached on disk.

    Per candidate, ``features`` gets ``matches``, ``inliers``, ``inlier_ratio``,
    ``residual_px`` (mean inlier error) and ``stage1``; ``artifacts`` gets the
    correspondences and the verification for the detail view.
    """

    def __init__(
        self,
        matcher: Matcher | str = "sift",
        top_k: int = 20,
        model_type: ModelType = "H",
        method: Method = "magsac",
        threshold: float = 4.0,
        min_inliers: int = 8,
        use_masks: bool = True,
        cache_dir: str | Path | None = None,
        keep_artifacts: bool = True,
        name: str | None = None,
    ) -> None:
        """Create the stage.

        Args:
            matcher: A matcher or a registry name (``"sift"``, ...).
            top_k: Candidates to re-rank.
            model_type: ``"H"``, ``"A"`` or ``"F"`` (see :mod:`visual_search.geometry.verify`).
            method: ``"magsac"`` or ``"ransac"``.
            threshold: Inlier threshold in pixels.
            min_inliers: Fewer verified inliers count as no match.
            use_masks: Restrict keypoints to the query (and gallery) prompt masks.
            cache_dir: Root of the on-disk feature cache; ``None`` keeps features in memory.
            keep_artifacts: Store correspondences/verification on candidates (detail view).
            name: Stage name; default ``"<matcher>-<model>-<method>"``.
        """
        super().__init__(top_k)
        self.matcher = create_matcher(matcher) if isinstance(matcher, str) else matcher
        self.model_type = model_type
        self.method = method
        self.threshold = threshold
        self.min_inliers = min_inliers
        self.use_masks = use_masks
        self.keep_artifacts = keep_artifacts
        self.name = name or f"{self.matcher.name}-{model_type}-{method}"
        self.cache = (FeatureCache(cache_dir, f"kp-{self.matcher.name}", self.matcher.identity)
                      if cache_dir is not None else None)
        self._memory: dict[str, Features] = {}
        self._fp_by_id: dict[str, str] = {}
        self._query_lru: OrderedDict[str, Features] = OrderedDict()
        self.n_extracted = 0  # images actually run through the extractor

    def features_for(self, item: Item, image: NDArray[np.uint8] | None = None) -> Features:
        """Return full-image features of an item (disk cache, then memory, then extract).

        Args:
            item: The item.
            image: Its pixels, if already loaded.

        Returns:
            The features.
        """
        if image is not None:
            fp = image_fingerprint(image)
        else:
            # Gallery items are looked up for every query; hash their source only once.
            if item.item_id not in self._fp_by_id:
                self._fp_by_id[item.item_id] = image_fingerprint(item.image)
            fp = self._fp_by_id[item.item_id]
        if fp in self._memory:
            return self._memory[fp]
        if fp in self._query_lru:
            self._query_lru.move_to_end(fp)
            return self._query_lru[fp]
        cached = self.cache.get(fp) if self.cache is not None else None
        if cached is not None:
            feats = Features.from_arrays(cached)
        else:
            feats = self.matcher.extract(image if image is not None else item.load_image())
            self.n_extracted += 1
            if self.cache is not None:
                self.cache.put(fp, feats.to_arrays())
        self._query_lru[fp] = feats
        while len(self._query_lru) > 8:
            self._query_lru.popitem(last=False)
        return feats

    def prepare(self, gallery: Sequence[Item]) -> None:
        """Extract (or load from cache) features of every gallery image.

        Without a disk cache the features are kept in memory.

        Args:
            gallery: All gallery items.
        """
        for item in gallery:
            feats = self.features_for(item)
            if self.cache is None:
                self._memory[self._fp_by_id[item.item_id]] = feats

    def rerank(self, query: Item, candidates: list[Candidate], gallery: Mapping[str, Item]) -> list[Candidate]:
        """Match, verify and re-sort the candidates.

        Args:
            query: The query (its prompt, ideally a mask, restricts the keypoints).
            candidates: Top candidates of the previous stage.
            gallery: Item id -> gallery item.

        Returns:
            New candidates sorted by verified inliers (stable for ties).
        """
        image_q = query.load_image()
        fq = self.features_for(query, image_q)
        mask_q = prompt_to_mask(query.prompt, fq.image_hw) if self.use_masks else None
        out = []
        for cand in candidates:
            item = gallery[cand.item_id]
            fc = self.features_for(item)
            mask_c = prompt_to_mask(item.prompt, fc.image_hw) if self.use_masks else None
            corr = self.matcher.match_extracted(fq, fc, mask_q, mask_c)
            ver = verify(corr, self.model_type, self.method, self.threshold)
            n_inl = ver.n_inliers if ver.ok and ver.n_inliers >= self.min_inliers else 0
            res = ver.residuals[ver.inliers]
            features = {
                "inliers": n_inl,
                "inlier_ratio": ver.inlier_ratio if n_inl else 0.0,
                "matches": len(corr),
                "residual_px": float(res.mean()) if n_inl else float("nan"),
                "stage1": cand.score,
            }
            artifacts = ({"correspondences": corr, "verification": ver, "accepted": n_inl > 0}
                         if self.keep_artifacts else {})
            out.append(Candidate(cand.item_id, n_inl + cand.score, {**cand.features, **features},
                                 {**cand.artifacts, **artifacts}))
        return sorted(out, key=lambda c: -c.score)
