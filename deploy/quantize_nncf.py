#!/usr/bin/env python3
"""M4.2 (đường 2) — Static int8 PTQ bằng NNCF (OpenVINO) trên ONNX: quantize cả Conv lẫn MatMul, ra ONNX QDQ chạy được trên
ONNX Runtime mọi CPU và chuyển tiếp sang OpenVINO/RKNN. Dùng khi onnxruntime.quantization.quantize_static lỗi.

  python deploy/quantize_nncf.py --onnx deploy/onnx/moge2_vits_288x384_dynamo.onnx --calib outputs/ep09/frames --n 40 \
      -o deploy/onnx/moge2_vits_288x384_int8nncf.onnx [--preset mixed] [--ignore LayerNormalization,Softmax]
"""
import argparse
from pathlib import Path
import cv2, numpy as np, onnx, nncf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--calib", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--preset", choices=["performance", "mixed"], default="mixed",
                    help="mixed = weight symmetric + activation asymmetric (tốt cho transformer)")
    ap.add_argument("--ignore", default="", help="op types bỏ qua, vd LayerNormalization,Softmax")
    ap.add_argument("--fast-bias", action="store_true", default=True)
    a = ap.parse_args()
    m = onnx.load(a.onnx)
    inp = m.graph.input[0]; name = inp.name
    _, _, h, w = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    frames = sorted(Path(a.calib).glob("*.jpg"))[: a.n]

    def prep(f):
        im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        return {name: (cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]}

    ds = nncf.Dataset(frames, prep)
    ignored = nncf.IgnoredScope(types=[t for t in a.ignore.split(",") if t]) if a.ignore else None
    q = nncf.quantize(m, ds, preset=nncf.QuantizationPreset.MIXED if a.preset == "mixed" else nncf.QuantizationPreset.PERFORMANCE,
                      model_type=nncf.ModelType.TRANSFORMER, subset_size=len(frames), ignored_scope=ignored,
                      fast_bias_correction=a.fast_bias)
    onnx.save(q, a.out)
    print(f"NNCF int8 → {a.out} ({Path(a.out).stat().st_size/2**20:.1f} MB), calib {len(frames)} frame, preset {a.preset}")


if __name__ == "__main__":
    main()
