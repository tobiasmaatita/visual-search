"""Manual go/no-go check for the SAM3 ONNX masker (fp16 by default).

Not run in CI or on the 8 GB dev machine. Run it on the target hardware:

    uv sync --extra onnx
    uv run python scripts/try_sam3.py data/bank_01/img-001.png --box 63 335 535 568
    uv run python scripts/try_sam3.py IMG --points 300,400,1 120,80,0 --providers CPUExecutionProvider

It loads the three model parts, then runs the same image three times: the first
run includes the vision encoder ("cold"), the next ones reuse its cached features
("warm", i.e. an interactive prompt refinement). It prints per-step timings and
writes an overlay PNG. Budget (docs/log/2026-11-24-01-...): <= 1 s per query mask.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from visual_search.cache import image_fingerprint
from visual_search.data import Item, Prompt
from visual_search.masking.sam3 import Sam3Config, Sam3Masker
from visual_search.viz import draw_prompt, overlay_mask


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    g.add_argument("--points", nargs="+", metavar="X,Y,LABEL", help="label 1 = positive, 0 = negative")
    p.add_argument("--model-dir", default="models/sam3")
    p.add_argument("--variant", default="fp16", help='file suffix: "fp16", "q8" or "" for fp32')
    p.add_argument("--providers", nargs="*", default=None, help="e.g. CUDAExecutionProvider CPUExecutionProvider")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", type=Path, default=None, help="overlay PNG (default: <stem>_sam3.png in cwd)")
    return p.parse_args()


def main() -> None:
    """Load SAM3, segment the prompt, report timings and save an overlay."""
    args = parse_args()
    if args.box:
        prompt = Prompt(box=tuple(args.box))
    else:
        pts = np.array([[float(v) for v in s.split(",")] for s in args.points], np.float32)
        prompt = Prompt(points=pts[:, :2], point_labels=pts[:, 2].astype(np.int64))

    cfg = Sam3Config(model_dir=args.model_dir, variant=args.variant, providers=args.providers)
    t0 = time.perf_counter()
    masker = Sam3Masker(cfg)
    print(f"load: {time.perf_counter() - t0:.2f} s from {cfg.model_dir} (variant {cfg.variant!r})")

    image = Item("q", args.image).load_image()
    key = image_fingerprint(image)
    result = None
    for run in range(args.runs):
        t0 = time.perf_counter()
        result = masker.mask(image, prompt, image_key=key)
        total = time.perf_counter() - t0
        steps = ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in result.timings.items())
        print(f"run {run} ({'cold' if run == 0 else 'warm'}): total {total * 1000:.0f} ms | {steps}")
    assert result is not None
    print(f"source={result.source} score={result.score} mask_area={result.mask.mean():.3f} "
          f"alternatives={len(result.alternatives)}")

    out = args.out or Path(f"{args.image.stem}_sam3.png")
    Image.fromarray(draw_prompt(overlay_mask(image, result.mask), prompt)).save(out)
    print(f"overlay written to {out}")


if __name__ == "__main__":
    main()
