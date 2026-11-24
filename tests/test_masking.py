"""Tests for the maskers. SAM3 is tested only with fake models (the real export is never run here)."""

import numpy as np
import pytest

from visual_search.data import Item, Prompt
from visual_search.masking import BoxMasker, NoMasker, with_mask
from visual_search.masking.sam3 import (
    Sam3Config,
    Sam3Masker,
    box_iou,
    box_to_cxcywh_norm,
    points_pseudo_box,
    rank_candidates_for_box,
    select_candidates_for_points,
)

IMG = np.zeros((100, 200, 3), np.uint8)


def _points(*pts: tuple[float, float, int]) -> Prompt:
    a = np.array(pts, np.float32)
    return Prompt(points=a[:, :2], point_labels=a[:, 2].astype(np.int64))


class TestFallbacks:
    def test_no_masker(self) -> None:
        assert NoMasker().mask(IMG, Prompt(box=(0, 0, 5, 5))).mask.all()

    def test_box_masker(self) -> None:
        r = BoxMasker().mask(IMG, Prompt(box=(10, 20, 50, 60)))
        assert r.mask.sum() == 40 * 40 and r.mask[20, 10] and not r.mask[19, 10]
        assert r.source == "box"
        assert BoxMasker(min_size=10).mask(IMG, _points((100, 50, 1))).mask.sum() == 100

    def test_mask_prompt_passthrough(self) -> None:
        m = np.zeros((100, 200), bool)
        m[5:10, 5:10] = True
        r = BoxMasker().mask(IMG, Prompt(mask=m))
        assert r.source == "prompt" and (r.mask == m).all()
        with pytest.raises(ValueError):
            BoxMasker().mask(IMG, Prompt(mask=np.ones((3, 3), bool)))

    def test_with_mask_keeps_source_prompt(self) -> None:
        item = Item("q", IMG, "a", Prompt(box=(10, 20, 50, 60)), {"group": "g"})
        masked = with_mask(item, BoxMasker().mask(IMG, item.prompt))
        assert masked.prompt.kind == "mask"
        assert masked.extra["source_prompt"].box == (10, 20, 50, 60)
        assert masked.extra["group"] == "g" and item.prompt.kind == "box"


class TestSam3Helpers:
    def test_box_conversion(self) -> None:
        np.testing.assert_allclose(box_to_cxcywh_norm((50, 25, 150, 75), (100, 200)), [0.5, 0.5, 0.5, 0.5])

    def test_pseudo_box(self) -> None:
        box = points_pseudo_box(np.float32([[100, 50], [120, 60], [0, 0]]), np.int64([1, 1, 0]), (100, 200))
        assert box == pytest.approx((95.0, 47.5, 125.0, 62.5))  # 20x10 padded 25 % per side; negative ignored
        single = points_pseudo_box(np.float32([[100, 50]]), np.int64([1]), (100, 200), min_frac=0.2)
        assert single == pytest.approx((90.0, 40.0, 110.0, 60.0))

    def test_iou(self) -> None:
        boxes = np.array([[0, 0, 1, 1], [0.5, 0, 1.5, 1], [2, 2, 3, 3]], np.float32)
        np.testing.assert_allclose(box_iou(boxes, np.array([0, 0, 1, 1])), [1, 1 / 3, 0], atol=1e-6)

    def test_rank_for_box(self) -> None:
        boxes = np.array([[0.0, 0, 0.2, 0.2], [0.5, 0.5, 0.9, 0.9], [0.52, 0.5, 0.9, 0.9]], np.float32)
        scores = np.array([0.99, 0.6, 0.2], np.float32)
        prompt = np.array([0.5, 0.5, 0.9, 0.9], np.float32)
        assert rank_candidates_for_box(boxes, scores, prompt, 0.3) == [1, 0]  # best overlap wins over best score
        assert rank_candidates_for_box(boxes, np.zeros(3, np.float32), prompt, 0.3)[0] == 1  # none pass -> all

    def test_select_for_points(self) -> None:
        masks = np.zeros((3, 10, 10), np.float32)
        masks[0, :5, :5] = 1  # top-left object
        masks[1, :, :] = 1  # whole image (contains the negative click)
        masks[2, 5:, 5:] = 1  # bottom-right object
        scores = np.array([0.8, 0.9, 0.7], np.float32)
        pts = np.array([[0.2, 0.2], [0.7, 0.7], [0.2, 0.8]], np.float32)  # pos TL, pos BR, neg BL
        assert select_candidates_for_points(masks, scores, pts, np.int64([1, 1, 0]), 0.5) == [0, 2]
        assert select_candidates_for_points(masks, scores, pts[:1], np.int64([1]), 0.5) == [1]  # no negative
        assert select_candidates_for_points(masks, scores, pts, np.int64([1, 1, 0]), 0.95) == []


class FakeModel:
    """Mimics mlcvtools' tensor wrapper: named inputs/outputs, declared dtypes."""

    def __init__(self, inputs: dict, outputs: dict) -> None:
        self.input_names = list(inputs)
        self.input_dtypes = list(inputs.values())
        self.output_names = list(outputs)
        self._outputs = outputs
        self.calls: list[dict] = []

    def predict(self, x: dict) -> list[np.ndarray]:
        assert set(x) == set(self.input_names)
        for name, dtype in zip(self.input_names, self.input_dtypes):
            assert x[name].dtype == dtype, name
        self.calls.append(x)
        return [v() if callable(v) else v for v in self._outputs.values()]


def _fake_sam3(dtype=np.float16) -> dict:
    f = np.float32 if dtype is None else dtype
    feats = {"fpn_feat_0": np.zeros((1, 2, 8, 8), f), "fpn_feat_1": np.zeros((1, 2, 4, 4), f),
             "fpn_feat_2": np.zeros((1, 2, 2, 2), f), "fpn_pos_2": np.zeros((1, 2, 2, 2), f)}
    masks = np.full((1, 3, 16, 16), -10.0, f)
    masks[0, 0, :8, :8] = 10  # candidate 0: top-left quarter
    masks[0, 1, 8:, 8:] = 10  # candidate 1: bottom-right quarter
    boxes = np.array([[[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1], [0, 0, 1, 1]]], f)
    logits = np.array([[2.0, 2.0, -5.0]], f)
    return {
        "vision-encoder": FakeModel({"images": f}, feats),
        "geometry-encoder": FakeModel(
            {"input_boxes": f, "input_boxes_labels": np.int64, "fpn_feat_2": f, "fpn_pos_2": f},
            {"geometry_features": np.zeros((1, 2, 2), f), "geometry_mask": np.ones((1, 2), bool)},
        ),
        "decoder": FakeModel(
            {"fpn_feat_0": f, "fpn_feat_1": f, "fpn_feat_2": f, "fpn_pos_2": f,
             "prompt_features": f, "prompt_mask": np.bool_},
            {"pred_masks": masks, "pred_boxes": boxes, "pred_logits": logits,
             "presence_logits": np.array([[3.0]], f)},
        ),
    }


class TestSam3MaskerWithFakeModels:
    def test_box_prompt_picks_overlapping_candidate(self) -> None:
        models = _fake_sam3()
        masker = Sam3Masker(Sam3Config(), models=models)
        img = np.zeros((40, 80, 3), np.uint8)
        r = masker.mask(img, Prompt(box=(42, 22, 78, 38)), image_key="k")
        assert r.source == "sam3" and r.mask.shape == (40, 80)
        assert r.mask[30, 60] and not r.mask[10, 20]  # bottom-right quarter
        assert len(r.alternatives) == 1  # candidate 0 passes the score threshold, candidate 2 does not
        assert set(r.timings) == {"encode", "decode", "select"}
        geo_in = models["geometry-encoder"].calls[0]
        np.testing.assert_allclose(geo_in["input_boxes"][0, 0], [0.75, 0.75, 0.45, 0.4], atol=1e-3)
        assert geo_in["input_boxes_labels"].tolist() == [[1]]
        assert models["vision-encoder"].calls[0]["images"].shape == (1, 3, 1008, 1008)

    def test_encoder_cache_reused_across_prompts(self) -> None:
        models = _fake_sam3()
        masker = Sam3Masker(Sam3Config(cache_size=1), models=models)
        img = np.zeros((40, 80, 3), np.uint8)
        masker.mask(img, Prompt(box=(0, 0, 40, 20)), image_key="a")
        masker.mask(img, Prompt(box=(40, 20, 80, 40)), image_key="a")
        assert len(models["vision-encoder"].calls) == 1
        masker.mask(img, Prompt(box=(0, 0, 40, 20)), image_key="b")
        masker.mask(img, Prompt(box=(0, 0, 40, 20)), image_key="a")  # evicted by "b"
        assert len(models["vision-encoder"].calls) == 3

    def test_points_union_and_fallback(self) -> None:
        masker = Sam3Masker(Sam3Config(), models=_fake_sam3())
        img = np.zeros((40, 80, 3), np.uint8)
        r = masker.mask(img, _points((10, 5, 1), (70, 35, 1)))
        assert r.source == "sam3" and r.mask[5, 10] and r.mask[35, 70] and not r.mask[35, 10]
        r = masker.mask(img, _points((10, 35, 1)))  # bottom-left: no candidate contains it
        assert r.source == "box(sam3-fallback)" and r.mask[35, 10]

    def test_missing_model_files(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            Sam3Masker(Sam3Config(model_dir=tmp_path))


def test_overlay_helpers() -> None:
    from visual_search.viz import draw_prompt, overlay_mask

    img = np.zeros((50, 60, 3), np.uint8)
    m = np.zeros((50, 60), bool)
    m[10:20, 10:20] = True
    out = overlay_mask(img, m, contour=False)
    assert out[15, 15].any() and not out[30, 30].any() and not img.any()
    assert draw_prompt(img, Prompt(box=(5, 5, 40, 40)))[5, 20].any()
    assert draw_prompt(img, _points((30, 30, 1), (10, 40, 0)))[30, 30].any()
