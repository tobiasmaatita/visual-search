"""Stage-1 retrieval with DINOv2 descriptors, focused by the visual prompt.

Pooling modes (how one vector per image is formed):

- ``"cls"``: the global CLS token of the full image; ignores the prompt.
- ``"crop_cls"``: the CLS token of the image cropped to the (padded) prompt box.
  Needs one extra forward pass per query.
- ``"masked_mean"``: the mean of the patch tokens under the prompt, from the
  full-image forward pass. Changing the prompt costs no extra forward pass, which
  keeps interactive refinement instant.

Gallery items without a prompt are described by their whole image (CLS, or the
mean of all patches for ``"masked_mean"``); gallery items *with* a prompt get the
same treatment as queries.

Full-image features (CLS + float16 patch grid + original size) are cached on disk
per image content and backbone configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from visual_search.cache import FeatureCache, image_fingerprint
from visual_search.data.schema import Item
from visual_search.imaging import crop_to_prompt, prompt_patch_mask
from visual_search.retrieval.base import EmbeddingRetriever

Pooling = Literal["cls", "crop_cls", "masked_mean"]
POOLINGS: tuple[str, ...] = ("cls", "crop_cls", "masked_mean")
CACHE_VERSION = 1


class PatchBackbone(Protocol):
    """What :class:`DinoRetriever` needs from a backbone (see ``Dinov2Backbone``)."""

    grid: int
    dim: int

    @property
    def identity(self) -> dict[str, Any]:
        """Return the config that determines the features."""
        ...

    def extract(self, images: Sequence[NDArray[np.uint8]]) -> tuple[NDArray[np.float32], NDArray[np.float16]]:
        """Return ``(B, D)`` CLS tokens and ``(B, g, g, D)`` patch tokens."""
        ...


class DinoRetriever(EmbeddingRetriever):
    """Cosine retrieval over DINOv2 descriptors with a choice of pooling.

    Example:
        >>> retriever = DinoRetriever(Dinov2Backbone(), pooling="crop_cls", cache_dir="cache")
        >>> report = evaluate(Cascade(retriever), split)
    """

    def __init__(
        self,
        backbone: PatchBackbone,
        pooling: Pooling = "crop_cls",
        cache_dir: str | Path | None = None,
        batch_size: int = 16,
        crop_pad: float = 0.1,
    ) -> None:
        """Create the retriever.

        Args:
            backbone: Feature extractor, e.g. ``Dinov2Backbone``.
            pooling: One of ``"cls"``, ``"crop_cls"``, ``"masked_mean"``.
            cache_dir: Root of the on-disk feature cache (e.g. ``<repo>/cache``); ``None`` disables it.
            batch_size: Images per forward pass when indexing the gallery.
            crop_pad: Relative padding around the prompt box for ``"crop_cls"``.

        Raises:
            ValueError: If ``pooling`` is unknown.
        """
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}.")
        self.backbone = backbone
        self.pooling = pooling
        self.batch_size = batch_size
        self.crop_pad = crop_pad
        cache_cfg = {**backbone.identity, "cache_version": CACHE_VERSION}
        self.cache = FeatureCache(cache_dir, "dinov2", cache_cfg) if cache_dir is not None else None
        model = str(backbone.identity.get("model_id", "dino")).split("/")[-1]
        self.name = f"{model}/{pooling}"
        self.n_forward = 0  # images actually run through the backbone (for cache diagnostics)

    def image_features(self, items: Sequence[Item]) -> list[dict[str, NDArray]]:
        """Return full-image features per item, using the cache where possible.

        Args:
            items: Items whose images to describe.

        Returns:
            Per item: ``{"cls": (D,) float32, "patches": (g, g, D) float16, "hw": (2,) int}``.
        """
        out: list[dict[str, NDArray] | None] = [None] * len(items)
        fps = [image_fingerprint(it.image) for it in items] if self.cache is not None else [None] * len(items)
        todo = []
        for i, fp in enumerate(fps):
            hit = self.cache.get(fp) if self.cache is not None and fp is not None else None
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start : start + self.batch_size]
            images = [items[i].load_image() for i in idx]
            cls, patches = self.backbone.extract(images)
            self.n_forward += len(idx)
            for j, i in enumerate(idx):
                feats = {"cls": cls[j], "patches": patches[j], "hw": np.array(images[j].shape[:2])}
                if self.cache is not None and fps[i] is not None:
                    self.cache.put(fps[i], feats)
                out[i] = feats
        return out  # type: ignore[return-value]

    def describe(self, item: Item, feats: dict[str, NDArray] | None = None) -> NDArray[np.float32]:
        """Compute the pooled descriptor of one item, honoring its prompt.

        Args:
            item: The item (query or gallery).
            feats: Its full-image features, if already computed.

        Returns:
            ``(D,)`` float32 descriptor (not normalized).
        """
        if self.pooling == "crop_cls" and item.prompt is not None:
            crop = np.ascontiguousarray(crop_to_prompt(item.load_image(), item.prompt, pad=self.crop_pad))
            self.n_forward += 1
            return self.backbone.extract([crop])[0][0]
        feats = feats if feats is not None else self.image_features([item])[0]
        if self.pooling in ("cls", "crop_cls"):
            return feats["cls"].astype(np.float32)
        patches = feats["patches"]
        hw = (int(feats["hw"][0]), int(feats["hw"][1]))
        mask = prompt_patch_mask(item.prompt, hw, patches.shape[:2])
        return patches[mask].astype(np.float32).mean(axis=0)

    def embed_query(self, query: Item) -> NDArray[np.float32]:
        """Descriptor of the query, focused by its prompt (see module docstring).

        Args:
            query: The query item.

        Returns:
            ``(D,)`` float32 vector.
        """
        return self.describe(query)

    def embed_gallery(self, items: Sequence[Item]) -> NDArray[np.float32]:
        """Descriptors of gallery items (batched forward passes, cached).

        Args:
            items: Gallery items.

        Returns:
            ``(N, D)`` float32 array.
        """
        if not items:
            return np.zeros((0, self.backbone.dim), np.float32)
        feats = self.image_features(items)
        return np.stack([self.describe(it, f) for it, f in zip(items, feats)])
