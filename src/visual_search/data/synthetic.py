"""Synthetic dataset — FOR SMOKE TESTING ONLY.

!!! Do not use these numbers to judge methods. !!!

Each "instance" is a procedurally generated textured patch (or a crop of a user
supplied image). Views of an instance paste that patch into random cluttered
scenes under a random perspective warp, scale, rotation and photometric jitter.
Distractors are cluttered scenes without any instance. Every view carries the
ground-truth object mask (``extra["gt_mask"]``) and a box prompt, so later stages
(masking, matching, scoring) can be tested end to end without real data.

Instances are planar and rigid by construction; this does not cover non-rigid
objects, 3D viewpoint changes or real lighting. It only checks that the code
paths run and that obviously-easy cases are solved.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

TEXTURE_SIZE = 192


def _random_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """Return a random RGB color as ints."""
    return tuple(int(c) for c in rng.integers(0, 256, size=3))  # type: ignore[return-value]


def _draw_random_shapes(img: NDArray[np.uint8], rng: np.random.Generator, n: int) -> None:
    """Draw ``n`` random filled circles, rectangles, polygons and lines in place."""
    h, w = img.shape[:2]
    for _ in range(n):
        color = _random_color(rng)
        kind = rng.integers(0, 4)
        if kind == 0:
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.circle(img, center, int(rng.integers(4, max(5, w // 5))), color, -1)
        elif kind == 1:
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.rectangle(img, p0, p1, color, -1)
        elif kind == 2:
            pts = rng.integers(0, [w, h], size=(int(rng.integers(3, 7)), 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], color)
        else:
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(img, p0, p1, color, int(rng.integers(1, 6)))


def make_texture(rng: np.random.Generator, size: int = TEXTURE_SIZE) -> NDArray[np.uint8]:
    """Generate a unique, keypoint-rich RGB texture for one synthetic instance.

    Args:
        rng: Random generator; the texture is fully determined by its state.
        size: Side length in pixels.

    Returns:
        ``(size, size, 3)`` uint8 RGB texture.
    """
    tex = np.empty((size, size, 3), np.uint8)
    tex[:] = _random_color(rng)
    _draw_random_shapes(tex, rng, n=int(rng.integers(25, 45)))
    noise = rng.normal(0, 12, size=tex.shape)
    return np.clip(tex.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _texture_from_image(image: NDArray[np.uint8], rng: np.random.Generator, size: int) -> NDArray[np.uint8]:
    """Cut a random square crop from a user image and resize it to ``size``."""
    h, w = image.shape[:2]
    side = int(min(h, w) * rng.uniform(0.4, 0.8))
    y0, x0 = int(rng.integers(0, h - side + 1)), int(rng.integers(0, w - side + 1))
    return cv2.resize(image[y0 : y0 + side, x0 : x0 + side], (size, size), interpolation=cv2.INTER_AREA)


def make_background(rng: np.random.Generator, shape: tuple[int, int]) -> NDArray[np.uint8]:
    """Generate a cluttered RGB background scene.

    Args:
        rng: Random generator.
        shape: ``(height, width)``.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    h, w = shape
    small = rng.integers(0, 256, size=(4, 4, 3)).astype(np.uint8)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    _draw_random_shapes(bg, rng, n=int(rng.integers(5, 15)))
    return cv2.GaussianBlur(bg, (0, 0), sigmaX=float(rng.uniform(0.5, 3.0)))


def _random_homography(rng: np.random.Generator, tex_size: int, scene: tuple[int, int]) -> NDArray[np.float64]:
    """Return a homography placing the texture in the scene with random pose and perspective."""
    h, w = scene
    scale = rng.uniform(0.25, 0.5) * min(h, w) / tex_size
    angle = rng.uniform(-35, 35)
    side = tex_size * scale
    margin = side * 0.75
    cx, cy = rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)
    src = np.float32([[0, 0], [tex_size, 0], [tex_size, tex_size], [0, tex_size]])
    corners = (src - tex_size / 2) * scale
    # Perspective: jitter each corner independently by up to 15% of the side.
    corners += rng.uniform(-0.15, 0.15, size=corners.shape) * side
    rot = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
    dst = (corners @ rot.T + [cx, cy]).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _photometric(img: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    """Apply random brightness/contrast, per-channel gain, gamma, blur and noise."""
    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.15, 0.15)
    x = x * rng.uniform(0.85, 1.15, size=3)
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.5:
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=float(rng.uniform(0.3, 1.5)))
    x = x + rng.normal(0, rng.uniform(0, 0.03), size=x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def render_view(
    texture: NDArray[np.uint8], rng: np.random.Generator, shape: tuple[int, int]
) -> tuple[NDArray[np.uint8], NDArray[np.bool_], NDArray[np.float64]]:
    """Paste ``texture`` into a random scene under a random perspective warp.

    Args:
        texture: ``(S, S, 3)`` uint8 instance texture.
        rng: Random generator.
        shape: Scene ``(height, width)``.

    Returns:
        ``(image, mask, H)``: the ``(H, W, 3)`` uint8 scene, the ``(H, W)`` bool
        object mask, and the 3x3 homography mapping texture pixels to scene pixels.
    """
    h, w = shape
    H = _random_homography(rng, texture.shape[0], shape)
    warped = cv2.warpPerspective(texture, H, (w, h), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(np.full(texture.shape[:2], 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
    mask = alpha > 127
    scene = make_background(rng, shape)
    scene[mask] = warped[mask]
    return _photometric(scene, rng), mask, H


def _mask_box(mask: NDArray[np.bool_]) -> list[float]:
    """Return the ``[x0, y0, x1, y1]`` pixel box enclosing a non-empty mask."""
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def make_synthetic_dataset(
    n_instances: int = 6,
    views_per_instance: int = 3,
    n_distractors: int = 10,
    n_near_duplicates: int = 0,
    image_shape: tuple[int, int] = (360, 480),
    base_images: Sequence[NDArray[np.uint8]] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build a small synthetic dataset in the raw list-of-dicts format. SMOKE TEST ONLY.

    Args:
        n_instances: Number of distinct instances (labels ``syn_00``, ``syn_01``, ...).
        views_per_instance: Images per instance (must be >= 2 for a valid split).
        n_distractors: Scenes without any instance (``label=None``).
        n_near_duplicates: Number of instances that get a "sibling": the same texture
            with a hue shift, under a different label. Both share ``"group"`` so they
            act as hard negatives for each other (tests the schema extension).
        image_shape: Scene ``(height, width)``.
        base_images: Optional RGB uint8 images; if given, instance textures are crops
            from these instead of procedural patterns.
        seed: Seed; the dataset is fully deterministic given all arguments.

    Returns:
        Records with ``id``, ``input_image`` (uint8 RGB array), ``label``, a box
        ``prompt`` around the object, and extra keys ``gt_mask`` ((H, W) bool),
        ``gt_homography`` (texture -> scene) and ``synthetic=True``. Distractors have
        no prompt and no mask.

    Raises:
        ValueError: If ``views_per_instance < 2`` or ``n_near_duplicates > n_instances``.

    Example:
        >>> records = make_synthetic_dataset(seed=0)
        >>> split = split_dataset(records, seed=0)
    """
    if views_per_instance < 2:
        raise ValueError("views_per_instance must be >= 2.")
    if n_near_duplicates > n_instances:
        raise ValueError("n_near_duplicates cannot exceed n_instances.")
    rng = np.random.default_rng(seed)

    textures: dict[str, NDArray[np.uint8]] = {}
    groups: dict[str, str] = {}
    for i in range(n_instances):
        if base_images:
            tex = _texture_from_image(base_images[i % len(base_images)], rng, TEXTURE_SIZE)
        else:
            tex = make_texture(rng)
        textures[f"syn_{i:02d}"] = tex
    for i in range(n_near_duplicates):
        label = f"syn_{i:02d}"
        hsv = cv2.cvtColor(textures[label], cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + 30) % 180  # OpenCV hue range is [0, 180)
        textures[f"{label}_dup"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        groups[label] = groups[f"{label}_dup"] = f"group_{i:02d}"

    records: list[dict[str, Any]] = []
    for label, tex in textures.items():
        for v in range(views_per_instance):
            image, mask, H = render_view(tex, rng, image_shape)
            rec: dict[str, Any] = {
                "id": f"synthetic/{label}/view_{v:02d}",
                "input_image": image,
                "label": label,
                "prompt": {"box": _mask_box(mask)},
                "gt_mask": mask,
                "gt_homography": H,
                "synthetic": True,
            }
            if label in groups:
                rec["group"] = groups[label]
            records.append(rec)
    for d in range(n_distractors):
        records.append(
            {
                "id": f"synthetic/distractor_{d:03d}",
                "input_image": _photometric(make_background(rng, image_shape), rng),
                "label": None,
                "synthetic": True,
            }
        )
    return records
