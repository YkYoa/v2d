#!/usr/bin/env python3
"""M4 — Kiểm tra vòng kín: chạy evaluate_policy.py của FlashCHORD nhưng hành động lấy từ file ONNX (onnxruntime CPU fp32)
thay cho actor JAX. So `mppe_cm`/success với bản JAX gốc để biết export có giữ được hành vi không.

Chạy trong venv FlashCHORD, từ thư mục flash_chord (tham số còn lại là override hydra của evaluate_policy.py):
  V2D_POLICY_ONNX=/path/policy.onnx python /path/eval_onnx_in_sim.py scripts/evaluate_policy.py \
      evaluation.checkpoint=/path/policy_xxx.safetensors evaluation.world_count=512 evaluation.metrics_output=/path/onnx.metrics.json
(checkpoint vẫn cần để khôi phục cấu hình môi trường; trọng số của nó KHÔNG được dùng để sinh hành động.)
"""
import os, runpy, sys, time

import numpy as np
import onnxruntime as ort
import jax
import jax.numpy as jnp

import flash_chord.evaluation.harness as harness

ONNX = os.environ["V2D_POLICY_ONNX"]
_stats = {"calls": 0, "sec": 0.0, "batch": 0}


def load_onnx_policy(checkpoint, training, observation, action_dim, *, deterministic, device=None):
    so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("V2D_ORT_THREADS", "4"))
    sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].shape[1] == observation.shape[1] and sess.get_outputs()[0].shape[1] == action_dim
    print(f"[onnx-in-sim] hành động lấy từ {ONNX} (CPUExecutionProvider), obs {observation.shape} → action {action_dim}")

    def policy_action(_state, obs, key):
        x = np.asarray(obs, dtype=np.float32); t = time.perf_counter()
        a = sess.run(None, {"obs": x})[0]
        _stats["sec"] += time.perf_counter() - t; _stats["calls"] += 1; _stats["batch"] = len(x)
        return jax.device_put(jnp.asarray(a), device), key

    return None, policy_action, jax.random.PRNGKey(0), {}


harness.load_flash_sac_policy = load_onnx_policy
script = sys.argv[1]; sys.argv = [script, *sys.argv[2:]]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    if _stats["calls"]:
        print(f"[onnx-in-sim] {_stats['calls']} lần gọi ONNX, batch {_stats['batch']}, "
              f"trung bình {1000 * _stats['sec'] / _stats['calls']:.2f} ms/lần")
