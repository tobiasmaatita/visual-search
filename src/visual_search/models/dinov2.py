"""DINOv2 feature extractor (PyTorch / Hugging Face transformers).

Works with every ``facebook/dinov2-*`` and ``facebook/dinov2-with-registers-*``
checkpoint; switching model is a config change (the embedding dimension follows
the checkpoint: 384 small, 768 base, 1024 large, 1536 giant).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from visual_search.device import device_sync, resolve_device

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class Dinov2Backbone:
    """Extract the CLS token and the patch-token grid of DINOv2.

    Images are resized (without cropping or padding, aspect ratio not kept) to a
    square ``resolution``, so every patch maps linearly back to the original image.
    This makes prompt-to-patch mapping exact and lets images of any size share a batch.

    Attributes:
        model_id: Hugging Face model id.
        resolution: Input side length in pixels (multiple of the patch size).
        patch_size: ViT patch size (14 for DINOv2).
        grid: Patch grid side, ``resolution // patch_size``.
        dim: Embedding dimension.
        device: Torch device in use.

    Example:
        >>> backbone = Dinov2Backbone("facebook/dinov2-small", resolution=448)
        >>> cls, patches = backbone.extract([rgb_uint8])
        >>> cls.shape, patches.shape
        ((1, 384), (1, 32, 32, 384))
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        resolution: int = 448,
        device: str = "auto",
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Load the model.

        Args:
            model_id: Any DINOv2 checkpoint on the Hugging Face hub.
            resolution: Square input size; must be a multiple of 14.
            device: ``"auto"`` or an explicit torch device string.
            revision: Optional hub revision (commit) to pin the weights.
            local_files_only: Only use the local Hugging Face cache (no network).

        Raises:
            ValueError: If ``resolution`` is not a multiple of the patch size.
        """
        from transformers import AutoModel  # heavy import, only when a model is built

        self.model_id = model_id
        self.revision = revision
        self.device = resolve_device(device)
        self.model = AutoModel.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
        self.model.eval().to(self.device)
        cfg = self.model.config
        self.patch_size = int(cfg.patch_size)
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
        self.dim = int(cfg.hidden_size)
        if resolution % self.patch_size:
            raise ValueError(f"resolution {resolution} must be a multiple of the patch size {self.patch_size}.")
        self.resolution = resolution
        self.grid = resolution // self.patch_size
        self.sync = device_sync(self.device)

    @property
    def identity(self) -> dict[str, Any]:
        """Return everything that determines the features (used as cache key)."""
        return {"model_id": self.model_id, "revision": self.revision, "resolution": self.resolution}

    def _preprocess(self, images: Sequence[NDArray[np.uint8]]) -> torch.Tensor:
        """Resize to the square input size and normalize; returns an NCHW float tensor."""
        batch = np.empty((len(images), self.resolution, self.resolution, 3), np.float32)
        for i, img in enumerate(images):
            shrink = max(img.shape[:2]) > self.resolution
            interp = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
            batch[i] = cv2.resize(img, (self.resolution, self.resolution), interpolation=interp)
        batch = (batch / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()

    @torch.inference_mode()
    def extract(self, images: Sequence[NDArray[np.uint8]]) -> tuple[NDArray[np.float32], NDArray[np.float16]]:
        """Run DINOv2 on a batch of RGB images.

        Args:
            images: ``(H, W, 3)`` uint8 RGB arrays of any sizes.

        Returns:
            ``(cls, patches)``: ``(B, D)`` float32 CLS tokens and ``(B, g, g, D)``
            float16 patch tokens (after the final layer norm; register tokens dropped).
        """
        x = self._preprocess(images).to(self.device)
        tokens = self.model(pixel_values=x).last_hidden_state
        cls = tokens[:, 0]
        patches = tokens[:, 1 + self.num_register_tokens :].reshape(len(images), self.grid, self.grid, self.dim)
        return cls.float().cpu().numpy(), patches.half().cpu().numpy()
