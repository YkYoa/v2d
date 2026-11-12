#!/usr/bin/env python3
"""So sánh fp32 vs int8 ONNX trên CPU/NPU: AbsRel depth, latency p50/p95, size. Chạy trên laptop hoặc board.
  python deploy/board_runtime/benchmark.py --fp32 a.onnx --int8 b.onnx --frames outputs/ep09/frames --n 20
"""
import argparse, time
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort


def prep(f, h, w):
    im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    return (cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def run(path, frames, provider):
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(path, so, providers=[provider]); inp = s.get_inputs()[0]; _, _, h, w = inp.shape
    extra = {i.name: np.array(1200, dtype=np.int64) for i in s.get_inputs()[1:] if i.name == "num_tokens"}
    outs, lat = [], []
    for f in frames:
        x = prep(f, h, w); t = time.perf_counter(); o = s.run(None, {inp.name: x, **extra}); lat.append(time.perf_counter() - t)
        outs.append(o[0][0, ..., 2] if o[0].ndim == 4 else o[0][0])
    return np.stack(outs), np.array(lat[1:]) * 1000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fp32", required=True); ap.add_argument("--int8", required=True)
    ap.add_argument("--frames", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args(); frames = sorted(Path(a.frames).glob("*.jpg"))[: a.n]
    d32, l32 = run(a.fp32, frames, a.provider); d8, l8 = run(a.int8, frames, a.provider)
    ok = np.isfinite(d32) & np.isfinite(d8) & (d32 > 0)
    absrel = np.mean(np.abs(d8[ok] - d32[ok]) / d32[ok])
    print("| model | size MB | p50 ms | p95 ms | FPS | AbsRel vs fp32 |\n|---|---|---|---|---|---|")
    print(f"| fp32 | {Path(a.fp32).stat().st_size/2**20:.0f} | {np.median(l32):.0f} | {np.percentile(l32,95):.0f} | {1000/np.median(l32):.1f} | 0 |")
    print(f"| int8 | {Path(a.int8).stat().st_size/2**20:.0f} | {np.median(l8):.0f} | {np.percentile(l8,95):.0f} | {1000/np.median(l8):.1f} | {absrel:.3f} |")


if __name__ == "__main__":
    main()
