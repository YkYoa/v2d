#!/usr/bin/env python3
"""Benchmark fp32 vs int8 ONNX bằng OpenVINO runtime (CPU Intel/AMD/ARM): AbsRel, latency. Dùng khi ORT không load được model QDQ.
  python deploy/board_runtime/benchmark_openvino.py --fp32 a.onnx --int8 b.onnx --frames outputs/ep09/frames --n 10
"""
import argparse, time
from pathlib import Path
import cv2, numpy as np, openvino as ov


def prep(f, h, w):
    im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    return (cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def run(path, frames, device, threads):
    core = ov.Core(); core.set_property(device, {"INFERENCE_NUM_THREADS": threads}) if device == "CPU" else None
    model = core.read_model(path); cm = core.compile_model(model, device)
    _, _, h, w = cm.inputs[0].shape; req = cm.create_infer_request()
    outs, lat = [], []
    for f in frames:
        x = prep(f, h, w); t = time.perf_counter(); r = req.infer({0: x}); lat.append(time.perf_counter() - t)
        o = list(r.values())[0]; outs.append(o[0, ..., 2] if o.ndim == 4 else o[0])
    return np.stack(outs), np.array(lat[1:]) * 1000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fp32", required=True); ap.add_argument("--int8", required=True)
    ap.add_argument("--frames", required=True); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default="CPU"); ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args(); frames = sorted(Path(a.frames).glob("*.jpg"))[: a.n]
    d32, l32 = run(a.fp32, frames, a.device, a.threads); d8, l8 = run(a.int8, frames, a.device, a.threads)
    ok = np.isfinite(d32) & np.isfinite(d8) & (d32 > 0); absrel = np.mean(np.abs(d8[ok] - d32[ok]) / d32[ok])
    print(f"OpenVINO {a.device} threads={a.threads}\n| model | size MB | p50 ms | p95 ms | FPS | AbsRel vs fp32 |\n|---|---|---|---|---|---|")
    print(f"| fp32 | {sum(p.stat().st_size for p in Path(a.fp32).parent.glob(Path(a.fp32).name+'*'))/2**20:.0f} | {np.median(l32):.0f} | {np.percentile(l32,95):.0f} | {1000/np.median(l32):.1f} | 0 |")
    print(f"| int8 | {Path(a.int8).stat().st_size/2**20:.0f} | {np.median(l8):.0f} | {np.percentile(l8,95):.0f} | {1000/np.median(l8):.1f} | {absrel:.3f} |")


if __name__ == "__main__":
    main()
