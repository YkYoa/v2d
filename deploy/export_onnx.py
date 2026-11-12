#!/usr/bin/env python3
"""M4.3 — Export sang ONNX (static shape) để quantize/deploy board không-NVIDIA.

  python deploy/export_onnx.py moge   --model Ruicheng/moge-2-vits-normal --h 288 --w 384 -o deploy/onnx/moge2_vits_288x384.onnx
  python deploy/export_onnx.py policy --checkpoint outputs/sharpa_mini/policy_xxx.safetensors -o deploy/onnx/policy.onnx

MoGe: bản ONNX chính thức (Ruicheng/moge-2-*-onnx) là fp32 dynamic shape; ở đây export lại shape tĩnh
(bắt buộc cho RKNN/Hailo/QNN) từ PyTorch, trả points (H,W,3) + mask.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def export_moge(args):
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(args.model).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, image):  # [1,3,H,W] in [0,1]
            out = self.m.forward(image, num_tokens=args.num_tokens)
            return out["points"], out["mask"]

    dummy = torch.rand(1, 3, args.h, args.w)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Wrapper(model), dummy, str(out), opset_version=17,
                      input_names=["image"], output_names=["points", "mask"], dynamo=True, dynamic_shapes=None)
    _verify(out, {"image": dummy.numpy()}, lambda: Wrapper(model)(dummy))


def export_policy(args):
    """Policy MLP (FlashCHORD .safetensors hoặc RSL-RL .pt). Kiến trúc suy từ state_dict (giả định ELU)."""
    if args.checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(args.checkpoint)
    else:
        ck = torch.load(args.checkpoint, map_location="cpu")
        sd = ck.get("model_state_dict", ck)
    keys = [k for k in sd if k.endswith(".weight") and sd[k].ndim == 2 and any(t in k for t in ("actor", "policy", "pi"))]
    if not keys:
        keys = [k for k in sd if k.endswith(".weight") and sd[k].ndim == 2]
    keys = sorted(keys, key=lambda k: [int(t) if t.isdigit() else t for t in k.replace(".", " ").split()])
    layers = []
    for k in keys:
        w = sd[k]; b = sd.get(k[:-6] + "bias", torch.zeros(w.shape[0]))
        lin = torch.nn.Linear(w.shape[1], w.shape[0]); lin.weight.data = w.float(); lin.bias.data = b.float()
        layers += [lin, torch.nn.ELU()]
    net = torch.nn.Sequential(*layers[:-1]).eval()
    print("policy MLP:", [tuple(l.weight.shape) for l in net if isinstance(l, torch.nn.Linear)])
    dummy = torch.zeros(1, net[0].in_features)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, dummy, str(out), opset_version=17, input_names=["obs"], output_names=["action"], dynamo=False)
    _verify(out, {"obs": dummy.numpy()}, lambda: net(dummy))


def _verify(path: Path, feeds: dict, torch_fn):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    o = sess.run(None, feeds)
    with torch.no_grad():
        t = torch_fn()
    t = t if isinstance(t, (tuple, list)) else (t,)
    for a, b in zip(o, t):
        b = b.float().numpy() if torch.is_tensor(b) else np.asarray(b, dtype=np.float32)
        print(f"  max|onnx-torch| = {np.nanmax(np.abs(a.astype(np.float32) - b)):.2e}  shape={a.shape}")
    print(f"ONNX OK → {path} ({path.stat().st_size/2**20:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="what", required=True)
    m = sub.add_parser("moge"); m.add_argument("--model", default="Ruicheng/moge-2-vits-normal")
    m.add_argument("--h", type=int, default=288); m.add_argument("--w", type=int, default=384)
    m.add_argument("--num-tokens", type=int, default=1200); m.add_argument("-o", "--out", required=True)
    p = sub.add_parser("policy"); p.add_argument("--checkpoint", required=True); p.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    export_moge(a) if a.what == "moge" else export_policy(a)
