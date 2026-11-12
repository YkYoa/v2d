# M4 — Deploy lên board không-NVIDIA
Thứ tự: `export_onnx.py` (static shape) → `quantize.py` (int8 static + SmoothQuant, calib bằng frame V2D) →
`board_runtime/benchmark.py` (AbsRel/latency) → toolchain board (bảng M4.4 trong PLAN.md):
ARM CPU: ONNX Runtime/ExecuTorch · RK3588: rknn-toolkit2 → .rknn · Hailo: DFC → .hef · Qualcomm: AI Hub/QNN · Intel: OpenVINO+NNCF.
Kết quả ghi `docs/DEPLOY_REPORT.md`.
