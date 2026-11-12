#!/usr/bin/env python3
"""M4.2 (đường 3) — Accuracy-aware int8 PTQ bằng NNCF trên OpenVINO IR: tự hoàn nguyên các layer nhạy về fp32 cho tới khi
AbsRel (so với fp32) ≤ max_drop. Output: IR .xml/.bin chạy bằng OpenVINO (Intel x86/NPU, ARM CPU).

  python deploy/quantize_nncf_ac.py --onnx deploy/onnx/moge2_vits_288x384_dynamo.onnx --calib outputs/ep09/frames \
      --n 40 --max-drop 0.10 -o deploy/onnx/moge2_vits_288x384_int8ac.xml
"""
import argparse
from pathlib import Path
import cv2, numpy as np, openvino as ov, nncf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--calib", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--val-n", type=int, default=12); ap.add_argument("--max-drop", type=float, default=0.10)
    a = ap.parse_args()
    core = ov.Core(); model = core.read_model(a.onnx)
    _, _, h, w = model.inputs[0].shape
    frames = sorted(Path(a.calib).glob("*.jpg"))
    calib, val = frames[: a.n], frames[-a.val_n:]

    def prep(f):
        im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        return (cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    ref_cm = core.compile_model(model, "CPU")
    ref = {str(f): list(ref_cm(prep(f)).values())[0][0, ..., 2] for f in val}  # z của point map fp32

    def validate(compiled, dataset):
        errs = []
        for x, f in dataset:
            z = list(compiled(x).values())[0][0, ..., 2]; r = ref[f]
            ok = np.isfinite(z) & np.isfinite(r) & (r > 0)
            errs.append(np.mean(np.abs(z[ok] - r[ok]) / r[ok]))
        return float(1.0 - np.mean(errs))  # higher = better; drop = AbsRel

    calib_ds = nncf.Dataset(calib, prep)
    val_ds = nncf.Dataset([(prep(f), str(f)) for f in val], lambda t: t)
    q = nncf.quantize_with_accuracy_control(
        model, calib_ds, val_ds, validation_fn=validate, max_drop=a.max_drop, drop_type=nncf.DropType.ABSOLUTE,
        preset=nncf.QuantizationPreset.MIXED, model_type=nncf.ModelType.TRANSFORMER, subset_size=len(calib))
    ov.save_model(q, a.out)
    size = sum(p.stat().st_size for p in Path(a.out).parent.glob(Path(a.out).stem + ".*")) / 2**20
    print(f"NNCF accuracy-control int8 → {a.out} ({size:.1f} MB), max_drop AbsRel {a.max_drop}")


if __name__ == "__main__":
    main()
