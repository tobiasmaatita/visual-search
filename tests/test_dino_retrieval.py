"""Tests for the feature cache, prompt-to-patch mapping and the DINOv2 retriever.

Most tests use a tiny fake backbone; one integration test runs the real
``facebook/dinov2-small`` if it is in the local Hugging Face cache.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_search.cache import FeatureCache, config_hash, image_fingerprint
from visual_search.cascade import Cascade
from visual_search.data import Item, Prompt, make_synthetic_dataset, split_dataset
from visual_search.eval import evaluate
from visual_search.imaging import prompt_patch_mask
from visual_search.retrieval import DinoRetriever, RandomRetriever


class FakeBackbone:
    """Patch token = mean color of the patch region; CLS = mean color of the image."""

    grid = 4
    dim = 3

    def __init__(self, tag: str = "fake") -> None:
        self.tag = tag
        self.calls = 0

    @property
    def identity(self) -> dict:
        return {"model_id": f"test/{self.tag}", "resolution": 8}

    def extract(self, images: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        cls, patches = [], []
        for img in images:
            h, w, g = img.shape[0], img.shape[1], self.grid
            grid = np.zeros((g, g, 3), np.float32)
            for i in range(g):
                for j in range(g):
                    region = img[i * h // g : (i + 1) * h // g, j * w // g : (j + 1) * w // g]
                    grid[i, j] = region.reshape(-1, 3).mean(0)
            cls.append(img.reshape(-1, 3).mean(0))
            patches.append(grid)
        return np.array(cls, np.float32), np.array(patches, np.float16)


def test_fingerprint_and_config_hash(tmp_path: Path) -> None:
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    b[0, 0, 0] = 1
    assert image_fingerprint(a) == image_fingerprint(a.copy()) != image_fingerprint(b)
    Image.fromarray(a).save(tmp_path / "a.png")
    assert image_fingerprint(tmp_path / "a.png") == image_fingerprint(str(tmp_path / "a.png"))
    assert config_hash({"x": 1, "y": 2}) == config_hash({"y": 2, "x": 1}) != config_hash({"x": 2, "y": 2})


def test_feature_cache_roundtrip(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path, "ns", {"m": 1})
    assert cache.get("ab" * 20) is None
    cache.put("ab" * 20, {"v": np.arange(3)})
    assert "ab" * 20 in cache
    np.testing.assert_array_equal(cache.get("ab" * 20)["v"], np.arange(3))
    assert (cache.dir / "config.json").exists()
    assert FeatureCache(tmp_path, "ns", {"m": 2}).get("ab" * 20) is None  # other config, other dir
    assert not list(cache.dir.rglob("*.tmp"))


def test_prompt_patch_mask() -> None:
    hw, grid = (100, 200), (4, 4)
    assert prompt_patch_mask(None, hw, grid).all()
    m = prompt_patch_mask(Prompt(box=(0, 0, 100, 50)), hw, grid)  # left half, top half
    assert m.sum() == 4 and m[:2, :2].all()
    tiny = prompt_patch_mask(Prompt(points=np.float32([[190, 90]]), point_labels=np.int64([1])), hw, grid)
    assert tiny.sum() == 1 and tiny[3, 3]
    mask = np.zeros(hw, bool)
    mask[50:, 100:] = True
    assert prompt_patch_mask(Prompt(mask=mask), hw, grid)[2:, 2:].all()
    assert prompt_patch_mask(Prompt(mask=mask), hw, grid).sum() == 4


def _two_color(left: tuple, right: tuple, item_id: str, label: str | None = None, prompt=None) -> Item:
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :20], img[:, 20:] = left, right
    return Item(item_id, img, label, prompt)


def test_pooling_modes_use_prompt(tmp_path: Path) -> None:
    gallery = [Item("red", np.full((40, 40, 3), (255, 0, 0), np.uint8)),
               Item("blue", np.full((40, 40, 3), (0, 0, 255), np.uint8))]
    query = _two_color((255, 0, 0), (0, 0, 255), "q", prompt=Prompt(box=(22, 0, 40, 40)))  # prompt on blue half
    for pooling in ("masked_mean", "crop_cls"):
        r = DinoRetriever(FakeBackbone(), pooling=pooling, cache_dir=tmp_path)
        r.fit(gallery)
        assert r.search(query, 1)[0].item_id == "blue", pooling
    r = DinoRetriever(FakeBackbone(), pooling="cls", cache_dir=None)
    r.fit(gallery)
    res = r.search(query, 2)
    assert res[0].score == pytest.approx(res[1].score)  # CLS ignores the prompt: purple is equally far


def test_cache_avoids_recompute_and_respects_config(tmp_path: Path) -> None:
    items = [Item(f"g{i}", np.full((16, 16, 3), i * 20, np.uint8)) for i in range(5)]
    r1 = DinoRetriever(FakeBackbone(), cache_dir=tmp_path, batch_size=2)
    r1.fit(items)
    assert r1.n_forward == 5 and r1.backbone.calls == 3
    r2 = DinoRetriever(FakeBackbone(), cache_dir=tmp_path)
    r2.fit(items)
    assert r2.n_forward == 0  # all hits
    np.testing.assert_allclose(r1._index.embeddings, r2._index.embeddings)
    r3 = DinoRetriever(FakeBackbone(tag="other"), cache_dir=tmp_path)
    r3.fit(items)
    assert r3.n_forward == 5  # different backbone identity -> fresh cache


def test_invalid_pooling() -> None:
    with pytest.raises(ValueError):
        DinoRetriever(FakeBackbone(), pooling="max", cache_dir=None)


@pytest.fixture(scope="module")
def real_backbone():
    try:
        from visual_search.models.dinov2 import Dinov2Backbone

        return Dinov2Backbone("facebook/dinov2-small", resolution=224, local_files_only=True)
    except OSError:
        pytest.skip("facebook/dinov2-small not in the local Hugging Face cache")


def test_real_dinov2_beats_random_on_synthetic(real_backbone, tmp_path: Path) -> None:
    cls, patches = real_backbone.extract([np.zeros((50, 70, 3), np.uint8)])
    assert cls.shape == (1, 384) and patches.shape == (1, 16, 16, 384)
    split = split_dataset(make_synthetic_dataset(n_instances=6, n_distractors=10, seed=0), seed=0)
    dino = evaluate(Cascade(DinoRetriever(real_backbone, pooling="crop_cls", cache_dir=tmp_path)), split)
    rand = evaluate(Cascade(RandomRetriever()), split)
    assert dino.metrics.loc["retrieval", "MRR"] > rand.metrics.loc["retrieval", "MRR"]
