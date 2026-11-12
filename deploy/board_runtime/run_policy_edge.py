#!/usr/bin/env python3
"""Chạy policy ONNX trên CPU board, đo latency (mục tiêu < 5 ms/step cho 20 Hz).
  python deploy/board_runtime/run_policy_edge.py --onnx deploy/onnx/policy.onnx [--steps 1000]
Obs random; khi nối hardware-in-the-loop thay bằng obs nhận từ sim trên server (ZMQ/UDP).
"""
import argparse, time, numpy as np, onnxruntime as ort
ap = argparse.ArgumentParser(); ap.add_argument("--onnx", required=True); ap.add_argument("--steps", type=int, default=1000)
a = ap.parse_args()
s = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"]); inp = s.get_inputs()[0]
x = np.random.randn(1, inp.shape[1]).astype(np.float32); s.run(None, {inp.name: x})
lat = []
for _ in range(a.steps):
    t = time.perf_counter(); s.run(None, {inp.name: x}); lat.append((time.perf_counter() - t) * 1e3)
print(f"policy {a.onnx}: p50 {np.median(lat):.3f} ms, p95 {np.percentile(lat,95):.3f} ms, obs_dim {inp.shape[1]}")
