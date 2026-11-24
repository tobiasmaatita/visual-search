# Latency budget and up-front rejections

- Date: 2026-11-24
- Type: decision

## What changed
We set a latency budget for the cascade and decided, before writing code, which methods are out of scope because they are too slow, too heavy, or blocked by licenses. This entry is the "avenues that remain open" list: if latency stops mattering, start here.

**Target hardware:** a 16 GB MacBook (CPU / CoreML / MPS) and a 20 GB NVIDIA MIG slice (CUDA). The 8 GB M2 used during development is not a target.

**Budget per interactive query** (warm caches, gallery features precomputed). This is an upper bound and may be tightened later:

| Stage | Budget |
|---|---|
| Query mask (SAM3 or fallback) | <= 1 s |
| Stage 1 retrieval (DINOv2, cached gallery) | <= 0.1 s |
| Stage 2 sparse matching, top-20 | <= 1 s (about 50 ms per pair) |
| Stage 3 dense matching, top-5 | <= 1 s |
| **Total** | **<= 2 s on MIG, <= 5 s on the Mac** |

**Rule:** anything expected at around 10 s per image or pair on the target hardware is rejected up front. Every new component gets a micro-benchmark (median / p95) before it is wired into the cascade.

## Decisions per component

| Component | Verdict | Reason / evidence |
|---|---|---|
| SAM3 **q8** ONNX (`models/sam3/*-q8.onnx`) | **Rejected** | Measured on the M2 CPU: vision encoder 11.4 s and decoder 1.35 s per image. The model uses int8 dynamic-quantization ops (`MatMulInteger`, `ConvInteger`, `DynamicQuantizeLinear`). The CoreML provider fails on it, and on CUDA these ops partly fall back to CPU. |
| SAM3 **fp16** ONNX | **Deferred** | The fp16 export is the only other variant available; there is no fp32. SAM3 code may be written, but it is not run or tested on the dev machine (RAM). The owner will benchmark fp16 manually on target hardware and decide go / no-go against the 1 s budget. Until then the cascade uses fallback masks (box mask, no mask). |
| SAM3 on every gallery image | **Rejected for the default path** | Cost scales with gallery size. The gallery is image-level by default, with optional RF-DETR-seg object proposals (0.26 s per image on CPU) as an ablation. |
| OmniGlue | **Not adopted** (license and dependency, *not* latency) | Its paper reports about 51 FPS (about 20 ms per pair) on an A40, so speed is fine on GPU. Blockers: it needs SuperPoint weights under Magic Leap's restrictive, non-commercial license, and it runs on TensorFlow (a second deep-learning stack next to PyTorch). Replacement: XFeat + LighterGlue, pending a license check of its code and weights. |
| MASt3R / DUSt3R | **Rejected** (latency and license) | Heavy ViT-L per-pair 3D reconstruction, seconds per pair on a Mac and large memory. CC BY-NC-SA (non-commercial). |
| VGGT | **Rejected** (latency and license) | 1B-parameter model. The default weights are non-commercial; VGGT-1B-Commercial has its own custom license. |
| RoMa (v1) | **Conditional** | DINOv2-L backbone plus dense decoder. Roughly sub-second per pair on a good GPU, much slower on MPS. Only on the top-5, in low-resolution mode; must pass its benchmark. Fallback: EfficientLoFTR. |
| RoMa v2, DINOv3 | **Later, license review** | RoMa v2 code is MIT, but it ships DINOv3 weights under Meta's custom DINOv3 License. |
| SIFT / RootSIFT, ALIKED + LightGlue, XFeat + LighterGlue, DINOv2-small retrieval, RF-DETR, MAGSAC++ | **Keep** | Expected tens of ms per pair or image; gallery features are cached. |

## Why
The owner needs an approach that runs considerably fast. Methods in the 10 s per image range are not usable for the intended application, so we don't spend time integrating them. They are documented here so they can be revisited if the constraints change.

## Impact
- Behavior/results: the first working cascade uses DINOv2-small retrieval and fallback masks; SAM3 is plugged in later behind the same `Masker` interface.
- Code/interfaces: none yet (decision only).
- Follow-ups: owner benchmarks SAM3 fp16 on the 16 GB Mac and the MIG box; license check for XFeat + LighterGlue before adoption.

## Evidence (optional)
ONNX inspection and the SAM3 / RF-DETR timings were run with onnxruntime 1.32 (CPU and CoreML providers) on an Apple M2 with 8 GB RAM, using random input at the model's native resolution, best of 2 runs after a warm-up.
