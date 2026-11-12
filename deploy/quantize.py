#!/usr/bin/env python3
"""M4.2 — Quantize ONNX int8 static (calibration = frame V2D) bằng ONNX Runtime, chạy được trên CPU.

  python deploy/quantize.py --onnx deploy/onnx/moge2_vits_288x384.onnx --calib outputs/ep09/frames --n 100 \
      -o deploy/onnx/moge2_vits_288x384_int8.onnx [--smooth 0.5] [--keep-fp32 LayerNormalization,Softmax,Gelu]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2, numpy as np, onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quant_pre_process, quantize_dynamic, quantize_static)


EXTRA_DEFAULTS = {"num_tokens": np.array(1200, dtype=np.int64)}


class FrameReader(CalibrationDataReader):
    def __init__(self, frames, h, w, name, extra=None):
        extra = extra or {}
        self.items = iter([{name: self._prep(f, h, w), **extra} for f in frames])
    @staticmethod
    def _prep(f, h, w):
        im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        return (cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    def get_next(self): return next(self.items, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--calib", required=True); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--smooth", type=float, default=0.5, help="SmoothQuant alpha (0 = tắt)")
    ap.add_argument("--keep-fp32", default="LayerNormalization,Softmax,Gelu,Erf,ReduceMean,Resize")
    ap.add_argument("--mode", choices=["static", "dynamic"], default="static",
                    help="dynamic = weight-only int8 MatMul/Gemm, không cần calibration (nhanh, chạy mọi CPU)")
    ap.add_argument("--skip-symbolic", action="store_true", help="bỏ symbolic shape inference ở bước pre-process")
    a = ap.parse_args()
    m = onnx.load(a.onnx)  # nạp cả external data (.onnx.data) rồi ghi lại 1 file để các bước sau không mất weights
    inline = Path(a.out).with_suffix(".inline.onnx")
    onnx.save(m, str(inline)); a.onnx = str(inline)
    inp = m.graph.input[0]
    _, _, h, w = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    frames = sorted(Path(a.calib).glob("*.jpg"))[: a.n]
    extra = {i.name: EXTRA_DEFAULTS[i.name] for i in m.graph.input[1:] if i.name in EXTRA_DEFAULTS}
    pre = Path(a.out).with_suffix(".pre.onnx")
    try:
        quant_pre_process(a.onnx, str(pre), skip_symbolic_shape=a.skip_symbolic)
    except Exception as e:  # graph có shape động (vd num_tokens là input) → bỏ symbolic shape inference
        print(f"quant_pre_process lỗi ({type(e).__name__}), thử lại skip_symbolic_shape=True")
        quant_pre_process(a.onnx, str(pre), skip_symbolic_shape=True)
    if a.mode == "dynamic":
        quantize_dynamic(str(pre), a.out, per_channel=True, weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        print(f"int8 dynamic (weight-only) → {a.out} ({Path(a.out).stat().st_size/2**20:.1f} MB)")
        return
    keep = [t for t in a.keep_fp32.split(",") if t]
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True, "CalibMovingAverage": True}
    if a.smooth > 0:
        extra.update({"SmoothQuant": True, "SmoothQuantAlpha": a.smooth, "SmoothQuantFolding": True})
    quantize_static(str(pre), a.out, FrameReader(frames, h, w, inp.name, extra),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, op_types_to_quantize=["MatMul", "Gemm", "Conv"],
                    nodes_to_exclude=[n.name for n in m.graph.node if n.op_type in keep], extra_options=extra)
    print(f"int8 → {a.out} ({Path(a.out).stat().st_size/2**20:.1f} MB), calib {len(frames)} frame, giữ fp32: {keep}")


if __name__ == "__main__":
    main()
