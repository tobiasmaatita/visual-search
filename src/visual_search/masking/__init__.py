"""Object masks from visual prompts (SAM3 adapter and simple fallbacks).

``Sam3Masker`` is imported lazily from ``visual_search.masking.sam3`` so that
using the fallbacks never requires onnxruntime.
"""

from visual_search.masking.base import BoxMasker, Masker, MaskResult, NoMasker, with_mask

__all__ = ["BoxMasker", "MaskResult", "Masker", "NoMasker", "with_mask"]
