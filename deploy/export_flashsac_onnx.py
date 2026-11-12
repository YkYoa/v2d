#!/usr/bin/env python3
"""M4.3 — Export actor FlashSAC (FlashCHORD, JAX/flax) sang ONNX fp32 để chạy trên board không-NVIDIA.

Actor KHÔNG phải MLP thuần: BatchNorm đầu vào → Linear (không bias) → N khối inverted-residual
(Linear→BN→ReLU→Linear→BN→ReLU + skip) → RMSNorm → head mean (+gain, +bias) → tanh * action_scale.
Hai bước, vì venv FlashCHORD (JAX) không có gói `onnx`:

  # 1) trong venv FlashCHORD (server): khôi phục actor bằng loader chính thức, ghi trọng số có tên + cặp (obs, action) mẫu
  python export_flashsac_onnx.py dump --checkpoint outputs/sharpa_mini/policy_19999744.safetensors -o policy_dump.npz
  # 2) ở bất kỳ đâu có onnx + onnxruntime + numpy (vd `uv run --no-project --with onnx --with onnxruntime --with numpy`)
  python export_flashsac_onnx.py build --dump policy_dump.npz -o policy_sharpa_mini.onnx

Đầu vào ONNX: obs [N, obs_dim] float32 (đúng layout `policy_schema_json` trong checkpoint, ghi kèm vào metadata ONNX).
Đầu ra: action [N, action_dim] trong [-action_scale, action_scale] = hành động deterministic (tanh(mean)) như evaluate_policy.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- bước 1: dump (JAX)
def dump(args):
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    from safetensors import safe_open

    from flash_chord.training.flash_sac.checkpoint import load_actor_checkpoint
    from flash_chord.training.flash_sac.config import NetworkConfig
    from flash_chord.training.flash_sac.evaluation import create_actor_template
    from flash_chord.training.flash_sac.network import Actor, deterministic_action

    with safe_open(args.checkpoint, "np") as f:
        meta = f.metadata()
    cfg = json.loads(meta["user.resolved_config_json"])["training"]
    net_cfg = NetworkConfig(**{k: v for k, v in cfg["network"].items() if not k.startswith("_")})
    schema = json.loads(meta["user.policy_schema_json"])
    action_dim = max(b["end"] for b in schema["action"]["blocks"])

    def restore(obs_dim: int, mixed: bool):
        training = SimpleNamespace(network=net_cfg, seed=cfg["seed"], mixed_precision=mixed,
                                   compute_dtype=cfg["compute_dtype"])
        actor, template = create_actor_template(training, obs_dim, action_dim)
        state, _ = load_actor_checkpoint(args.checkpoint, template)
        return actor, state

    # obs_dim: thử khôi phục với kích thước lấy từ tensor đầu vào (running_mean của input_norm) — đọc thẳng từ file
    with safe_open(args.checkpoint, "np") as f:
        shapes = sorted({f.get_tensor(k).shape for k in f.keys()})
    hidden = net_cfg.actor_hidden_dim
    cand = [s[0] for s in shapes if len(s) == 2 and s[1] == hidden and s[0] not in (hidden * net_cfg.expansion,)]
    obs_dim = cand[0]
    actor, state = restore(obs_dim, cfg["mixed_precision"])

    rng = np.random.default_rng(0)
    mean = np.asarray(state.batch_stats["input_norm"]["running_mean"])
    std = np.sqrt(np.asarray(state.batch_stats["input_norm"]["running_variance"]))
    obs = (mean + std * rng.standard_normal((args.n, obs_dim))).astype(np.float32)  # obs "giống thật" theo thống kê BN

    def act(a, s):
        dist = a.apply({"params": s.params, "batch_stats": s.batch_stats}, jnp.asarray(obs), training=False)
        return np.asarray(deterministic_action(dist, action_transform=net_cfg.action_transform,
                                               action_scale=net_cfg.action_scale), dtype=np.float32)

    ref_native = act(actor, state)                       # đúng như evaluate_policy (mixed precision nếu bật)
    actor32 = Actor(action_dim=action_dim, config=net_cfg, dtype=jnp.float32)
    # fp32 "thật": trên GPU NVIDIA, XLA mặc định nhân ma trận bằng TF32 (mantissa 10 bit) → lệch ~4e-3 so với fp32/fp64
    with jax.default_matmul_precision("float32"):
        ref_fp32 = act(actor32, state)                   # cùng trọng số, tính toàn fp32

    out = {"p/" + "/".join(k): np.asarray(v, np.float32) for k, v in flatten_dict(dict(state.params)).items()}
    out |= {"s/" + "/".join(k): np.asarray(v, np.float32) for k, v in flatten_dict(dict(state.batch_stats)).items()}
    out |= {"obs": obs, "ref_native": ref_native, "ref_fp32": ref_fp32}
    info = {"obs_dim": obs_dim, "action_dim": action_dim, "blocks": net_cfg.actor_blocks,
            "bn_eps": net_cfg.batch_norm_epsilon, "rms_eps": net_cfg.rms_norm_epsilon,
            "action_transform": net_cfg.action_transform, "action_scale": net_cfg.action_scale,
            "compute_dtype": cfg["compute_dtype"] if cfg["mixed_precision"] else "float32",
            "environment_steps": meta.get("user.environment_steps"), "policy_schema": schema,
            "checkpoint": Path(args.checkpoint).name}
    np.savez(args.out, info=json.dumps(info), **out)
    print("dump →", args.out, {k: v for k, v in info.items() if k != "policy_schema"})
    print("  params:", sorted(k for k in out if k[:2] in ("p/", "s/")))
    print(f"  max|native({info['compute_dtype']}) - fp32| = {np.abs(ref_native - ref_fp32).max():.2e}")


# ----------------------------------------------------------------------------- bước 2: build (onnx)
def build(args):
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    d = np.load(args.dump, allow_pickle=False)
    info = json.loads(str(d["info"]))
    P = {k[2:]: d[k] for k in d.files if k.startswith("p/")}
    S = {k[2:]: d[k] for k in d.files if k.startswith("s/")}
    nodes, inits = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name)); return name

    def node(op, ins, name, **kw):
        nodes.append(helper.make_node(op, ins, [name], name=name, **kw)); return name

    def bn(x, scope):  # BatchNorm suy luận: (x - mean) * a + bias. KHÔNG gộp mean vào bias: đặc trưng có phương sai ~0
        # cho a rất lớn (~1/sqrt(eps)), x*a - mean*a mất chính xác fp32 (sai số 4e-3 ở đầu ra khi thử gộp).
        a = P[f"{scope}/unit_scale"] / np.sqrt(S[f"{scope}/running_variance"] + info["bn_eps"])
        tag = scope.replace("/", "_")
        c = node("Sub", [x, const(tag + "_mean", S[f"{scope}/running_mean"])], tag + "_sub")
        return node("Add", [node("Mul", [c, const(tag + "_a", a)], tag + "_mul"),
                            const(tag + "_b", P[f"{scope}/unit_bias"])], tag + "_out")

    def linear(x, scope):
        tag = scope.replace("/", "_")
        return node("MatMul", [x, const(tag + "_w", P[f"{scope}/unit_kernel"])], tag + "_out")

    x = linear(bn("obs", "input_norm"), "embed")
    for i in range(info["blocks"]):
        b = f"block_{i}"
        h = node("Relu", [bn(linear(x, f"{b}/expand"), f"{b}/expand_norm")], f"{b}_relu1")
        h = node("Relu", [bn(linear(h, f"{b}/contract"), f"{b}/contract_norm")], f"{b}_relu2")
        x = node("Add", [h, x], f"{b}_res")
    # RMSNorm: x / sqrt(mean(x^2) + eps) * scale
    ms = node("ReduceMean", [node("Mul", [x, x], "rms_sq")], "rms_mean", axes=[-1], keepdims=1)
    rms = node("Sqrt", [node("Add", [ms, const("rms_eps", np.float32(info["rms_eps"]))], "rms_add")], "rms_sqrt")
    x = node("Mul", [node("Div", [x, rms], "rms_div"), const("rms_scale", P["post_norm/rms_scale"])], "post_norm")
    # head mean: unit_kernel (+gain) hoặc Dense kernel; cộng mean_bias
    w = P["mean/unit_kernel"] if "mean/unit_kernel" in P else P["mean/kernel"]
    if "mean/gain" in P:
        w = w * P["mean/gain"][None, :]
    mean = node("Add", [node("MatMul", [x, const("mean_w", w)], "mean_mm"), const("mean_bias", P["mean_bias"])], "mean")
    if info["action_transform"] == "tanh":
        mean = node("Tanh", [mean], "tanh")
    if info["action_transform"] == "tanh" and info["action_scale"] != 1.0:
        mean = node("Mul", [mean, const("action_scale", np.float32(info["action_scale"]))], "scaled")
    nodes.append(helper.make_node("Identity", [mean], ["action"], name="action_out"))

    graph = helper.make_graph(
        nodes, "flashsac_actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", info["obs_dim"]])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["N", info["action_dim"]])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="v2d-deploy")
    model.ir_version = 9
    for k in ("checkpoint", "environment_steps", "action_transform", "action_scale", "policy_schema"):
        e = model.metadata_props.add(); e.key = k; e.value = json.dumps(info[k]) if k == "policy_schema" else str(info[k])
    onnx.checker.check_model(model)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); onnx.save(model, str(out))

    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"obs": d["obs"]})[0]
    e32, enat = np.abs(got - d["ref_fp32"]).max(), np.abs(got - d["ref_native"]).max()
    import time
    one = d["obs"][:1]; lat = []
    for _ in range(2000):
        t = time.perf_counter(); sess.run(None, {"obs": one}); lat.append(time.perf_counter() - t)
    lat = np.array(lat[200:]) * 1000
    print(f"ONNX → {out} ({out.stat().st_size / 2**20:.2f} MB), obs {info['obs_dim']} → action {info['action_dim']}, "
          f"{len(nodes)} node")
    print(f"  max|onnx - jax fp32|   = {e32:.2e}   (tiêu chí < 1e-4)")
    print(f"  max|onnx - jax {info['compute_dtype']}| = {enat:.2e}   (sai khác do chính sách train bằng mixed precision)")
    print(f"  latency 1 thread CPU: p50 {np.median(lat):.3f} ms, p95 {np.percentile(lat, 95):.3f} ms (tiêu chí < 5 ms)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="what", required=True)
    a = sub.add_parser("dump"); a.add_argument("--checkpoint", required=True); a.add_argument("-o", "--out", required=True)
    a.add_argument("--n", type=int, default=256)
    b = sub.add_parser("build"); b.add_argument("--dump", required=True); b.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    dump(args) if args.what == "dump" else build(args)
