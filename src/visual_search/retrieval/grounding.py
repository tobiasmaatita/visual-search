"""Gallery grounding with DINOv2: find where the query object is in each gallery image.

For every gallery image, the query descriptor (the mean of the query's DINOv2 patch
features under its mask) is compared with each of the image's patch features
(already cached for stage 1). That gives a similarity heatmap on the patch grid
(32 x 32 at 448 px). The blobs around the strongest peaks become **regions**:

1. patches with similarity >= ``peak - delta`` are switched on;
2. connected blobs (8-neighbourhood) are ranked by their own peak;
3. up to ``max_regions`` blobs whose peak is within ``region_delta`` of the image's
   best peak are kept; each becomes a box (plus ``pad_patches`` of context).

Each region is scored by the cosine between the query descriptor and the mean of
the patch features inside the blob, so a region competes as an *object*, not as a
whole scene. One gallery image can yield several candidates (one per region).

Cost per query: one matrix product over all gallery patches plus a tiny
connected-component pass per image (about 1 ms per image on CPU). Memory: all
gallery patch grids are kept in RAM (about 0.75 MB per image at 448 px, float16).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from visual_search.data.schema import Item, Prompt
from visual_search.imaging import prompt_patch_mask
from visual_search.retrieval.base import Candidate, Retriever
from visual_search.retrieval.dino import DinoRetriever, PatchBackbone
from visual_search.retrieval.index import l2_normalize


@dataclass
class Region:
    """A localized object region in one image.

    Attributes:
        box: ``(x0, y0, x1, y1)`` in original-image pixels.
        score: Cosine between the query descriptor and the mean feature of the region's patches.
        peak: Highest single-patch similarity inside the region.
        n_patches: Number of patches in the blob.
        patch_mask: ``(gh, gw)`` bool, the blob on the patch grid.
    """

    box: tuple[float, float, float, float]
    score: float
    peak: float
    n_patches: int
    patch_mask: NDArray[np.bool_]


def localize(
    query: NDArray[np.floating],
    patches: NDArray[np.floating],
    image_hw: tuple[int, int],
    max_regions: int = 3,
    delta: float = 0.1,
    region_delta: float = 0.1,
    pad_patches: int = 1,
    sim: NDArray[np.floating] | None = None,
) -> list[Region]:
    """Find the regions of an image that look most like the query.

    Args:
        query: ``(D,)`` query descriptor (normalized internally).
        patches: ``(gh, gw, D)`` patch features of the image (normalized internally).
        image_hw: Original image ``(H, W)``; the patch grid covers the whole image.
        max_regions: Maximum number of regions returned.
        delta: Patches with similarity ``>= peak - delta`` are switched on, where
            ``peak`` is the image's best patch similarity.
        region_delta: Secondary blobs are kept only if their peak is within
            ``region_delta`` of the best peak.
        pad_patches: Context added around each blob, in patches.
        sim: Precomputed ``(gh, gw)`` cosine map (saves the matrix product).

    Returns:
        Regions, best score first (at least one).

    Example:
        >>> regions = localize(query_vec, feats["patches"], (h, w))
        >>> regions[0].box
    """
    gh, gw, _ = patches.shape
    q = l2_normalize(np.asarray(query, np.float32).reshape(-1))
    p = l2_normalize(np.asarray(patches, np.float32))
    if sim is None:
        sim = p @ q
    peak = float(sim.max())
    on = (sim >= peak - delta).astype(np.uint8)
    n, labels = cv2.connectedComponents(on, connectivity=8)
    blobs = []
    for lbl in range(1, n):
        m = labels == lbl
        blobs.append((float(sim[m].max()), m))
    blobs.sort(key=lambda t: -t[0])
    h, w = image_hw
    sy, sx = h / gh, w / gw
    regions = []
    for bpeak, m in blobs[:max_regions]:
        if bpeak < peak - region_delta:
            break
        ys, xs = np.nonzero(m)
        r0, r1 = max(ys.min() - pad_patches, 0), min(ys.max() + 1 + pad_patches, gh)
        c0, c1 = max(xs.min() - pad_patches, 0), min(xs.max() + 1 + pad_patches, gw)
        box = (float(c0 * sx), float(r0 * sy), float(c1 * sx), float(r1 * sy))
        desc = l2_normalize(p[m].mean(axis=0))
        regions.append(Region(box, float(desc @ q), bpeak, int(m.sum()), m))
    return sorted(regions, key=lambda r: -r.score)


class GroundedDinoRetriever(Retriever):
    """Stage 1 at object level for unlabelled galleries: localize, then score regions.

    For each gallery image the object is localized with :func:`localize`, and every
    region becomes its own :class:`~visual_search.retrieval.base.Candidate` (with
    ``region`` set). A gallery item that already has a box prompt is not localized:
    its box is used as its only region.

    The query descriptor is the mean of the query's patch features under its mask
    (the same feature space as the gallery patches). Features come from the same
    on-disk cache as :class:`~visual_search.retrieval.dino.DinoRetriever`.

    Example:
        >>> retriever = GroundedDinoRetriever(Dinov2Backbone(), cache_dir="cache")
        >>> cascade = Cascade(retriever, retrieval_k=100, stages=[SparseVerificationStage("sift")])
    """

    name = "dinov2-grounded"

    def __init__(
        self,
        backbone: PatchBackbone,
        cache_dir: str | Path | None = None,
        max_regions: int = 3,
        delta: float = 0.1,
        region_delta: float = 0.1,
        pad_patches: int = 1,
        batch_size: int = 16,
    ) -> None:
        """Create the retriever.

        Args:
            backbone: DINOv2 backbone (e.g. ``Dinov2Backbone``).
            cache_dir: Feature cache root (shared with ``DinoRetriever``); ``None`` disables it.
            max_regions: Maximum regions (results) per gallery image.
            delta: Blob threshold below the image's peak similarity (see :func:`localize`).
            region_delta: Keep secondary blobs only if this close to the best peak.
            pad_patches: Context around each region, in patches.
            batch_size: Images per forward pass when indexing.
        """
        self._dino = DinoRetriever(backbone, pooling="masked_mean", cache_dir=cache_dir, batch_size=batch_size)
        self.max_regions = max_regions
        self.delta = delta
        self.region_delta = region_delta
        self.pad_patches = pad_patches
        model = str(backbone.identity.get("model_id", "dino")).split("/")[-1]
        self.name = f"{model}/grounded"
        self._ids: list[str] = []
        self._patches: NDArray[np.float16] | None = None  # (N, gh, gw, D), L2-normalized
        self._hw: list[tuple[int, int]] = []
        self._boxes: list[tuple[float, float, float, float] | None] = []

    @property
    def n_forward(self) -> int:
        """Images run through the backbone so far (cache diagnostics)."""
        return self._dino.n_forward

    def fit(self, gallery: Sequence[Item]) -> None:
        """Load (or compute) and keep the patch grids of all gallery images.

        Args:
            gallery: Gallery items; items with a box prompt keep that box as their region.
        """
        feats = self._dino.image_features(gallery)
        self._ids = [it.item_id for it in gallery]
        self._hw = [(int(f["hw"][0]), int(f["hw"][1])) for f in feats]
        self._boxes = [it.prompt.bbox() if it.prompt is not None else None for it in gallery]
        self._patches = (np.stack([l2_normalize(f["patches"]) for f in feats]).astype(np.float16)
                         if feats else None)

    def query_descriptor(self, query: Item) -> NDArray[np.float32]:
        """Mean of the query's patch features under its prompt (L2-normalized).

        Args:
            query: The query item.

        Returns:
            ``(D,)`` float32.
        """
        return l2_normalize(self._dino.describe(query))

    def search(self, query: Item, k: int) -> list[Candidate]:
        """Localize the query object in every gallery image and rank the regions.

        Args:
            query: The query item (its prompt or mask defines the object).
            k: Number of candidates (regions) to return.

        Returns:
            Candidates sorted by region score, each with ``region`` set; features
            ``cosine`` (region score), ``peak`` (best patch similarity) and ``region_patches``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._patches is None and not self._ids:
            raise RuntimeError("GroundedDinoRetriever.fit() must be called before search().")
        if self._patches is None:
            return []
        q = self.query_descriptor(query)
        sims = self._patches.astype(np.float32) @ q  # (N, gh, gw)
        out: list[Candidate] = []
        for i, item_id in enumerate(self._ids):
            hw, box = self._hw[i], self._boxes[i]
            if box is not None:
                m = prompt_patch_mask(Prompt(box=box), hw, sims[i].shape)
                desc = l2_normalize(self._patches[i][m].astype(np.float32).mean(axis=0))
                score = float(desc @ q)
                out.append(Candidate(item_id, score, {"cosine": score, "peak": float(sims[i][m].max()),
                                                       "region_patches": int(m.sum())}, region=box))
                continue
            for r in localize(q, self._patches[i], hw, self.max_regions, self.delta, self.region_delta,
                              self.pad_patches, sim=sims[i]):
                out.append(Candidate(item_id, r.score, {"cosine": r.score, "peak": r.peak,
                                                         "region_patches": r.n_patches}, region=r.box))
        out.sort(key=lambda c: -c.score)
        return out[:k]

