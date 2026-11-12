# DEPLOY_REPORT — M4 (laptop CPU, cập nhật 2026-11-12)

Thiết bị đo: laptop AMD Ryzen 12 thread, **CPU only** (giả lập board ARM; chưa có board thật). Model MoGe-2 ViT-S, export dynamo
(torch 2.15 + onnxscript) shape tĩnh 1×3×288×384, num_tokens=1200; sai số export vs PyTorch 3.6e-6. Calibration: 40 frame V2D ep09.

## Bảng kết quả
| Model | Precision / toolchain | Runtime | Threads | Size MB | p50 ms | FPS | AbsRel vs fp32 | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| MoGe-2 ViT-S | fp32 ONNX | ONNX Runtime 1.31 | 4 | 129 | 752 | 1.3 | 0 | baseline ORT |
| MoGe-2 ViT-S | fp32 ONNX | ONNX Runtime | 12 | 129 | 636 | 1.6 | 0 | |
| MoGe-2 ViT-S | int8 dynamic (48 MatMul→MatMulInteger) | ONNX Runtime | 12 | 131 | 596 | 1.7 | 0.005 | `deploy/quantize.py --mode dynamic`; Conv vẫn fp32 → không nhanh hơn |
| MoGe-2 ViT-S | int8 static ORT `quantize_static` | – | – | – | – | – | – | **lỗi** calibrator ORT ("Unable to handle object of type bool") trên graph dynamo |
| MoGe-2 ViT-S | fp32 ONNX | **OpenVINO 2026.4** | 12 | 123 | 208 | 4.8 | 0 | OV nhanh hơn ORT 3× cùng model |
| MoGe-2 ViT-S | int8 NNCF (ONNX backend, SmoothQuant+bias corr) | OpenVINO | 12 | 33 | 137 | 7.3 | **0.192** | `deploy/quantize_nncf.py`; ORT không load được (Resize cubic); mất chính xác lớn |
| MoGe-2 ViT-S | **int8 NNCF accuracy-control (OV backend)** | OpenVINO | 12 | **34** | **134** | **7.5** | **0.005** | `deploy/quantize_nncf_ac.py --max-drop 0.10`; drop thực 0.003, không cần hoàn nguyên layer |
| Policy MLP (giả lập 128→512→256→128→30, ELU) | fp32 ONNX | ONNX Runtime | 1 | 0.9 | 0.015 | – | 3.7e-8 | `deploy/board_runtime/run_policy_edge.py`; checkpoint thật chờ M2 |
| **Policy FlashSAC thật** (`sharpa_mini`, 20M step; obs 504 → action 56; BN + 2 khối residual 256↔1024 + RMSNorm) | fp32 ONNX (opset 17, 36 node) | ONNX Runtime (CPU server 4090) | 1 | 4.6 | 0.095 | – | 6.4e-6 (max abs vs JAX fp32) | `deploy/export_flashsac_onnx.py`; vòng kín trong sim (`deploy/eval_onnx_in_sim.py`, 512 world): MPPE 0.616 cm, success 92.4 % vs JAX 0.642 cm, 93.4 % |

## Profile ORT fp32 (12 thread), % thời gian theo op
Conv 38 · Gemm 11 · Pad 10 · ReorderInput/Output 13 · ConvTranspose 5 · FusedMatMul 5.
→ Head DPT (Conv/ConvTranspose/Resize ở độ phân giải đầy) > 50 %; backbone ViT (MatMul) ≈ 15 %.

## Kết luận
1. **Đường khuyến nghị cho board CPU (ARM/x86):** export dynamo → `quantize_nncf_ac.py` (OpenVINO IR int8) → chạy OpenVINO runtime (hỗ trợ ARM64). Đạt 3.6× nhỏ hơn, 1.55× nhanh hơn fp32-OV, 4.7× nhanh hơn fp32-ORT, sai số 0.5 %.
2. Quantize chỉ MatMul (dynamic) vô ích với MoGe vì head Conv chiếm đa số thời gian; phải int8 cả Conv.
3. NNCF trên **ONNX backend** làm hỏng độ chính xác (19 %) và sinh Resize ORT không load được; **OV backend** cho kết quả tốt → ưu tiên OV backend; muốn ONNX QDQ cho RKNN/Hailo thì dùng ORT static sau khi vá graph (TODO) hoặc convert từ IR.
4. Mục tiêu M4 trên laptop: AbsRel ≤ 15 % ✔ (0.5 %), ≥ 2 FPS CPU ✔ (7.5 FPS).

## Việc tiếp theo (khi có board)
- **Board đích đã chốt: Renesas R-Car X5H** (chưa có board). Khi có: xác định toolchain AI của Renesas cho X5H, thử policy ONNX trước (chỉ MatMul/Mul/Add/Sub/Relu/Tanh/ReduceMean/Sqrt/Div), rồi MoGe int8. Các dòng board khác bên dưới chỉ còn để tham khảo.
- RK3588: `rknn-toolkit2` từ ONNX fp32 tĩnh (`moge2_vits_288x384_dynamo.onnx` + .data → gộp 1 file) với hybrid quant; so với số ở trên.
- Pi 5 + Hailo-8: thử Depth-Anything-V2-S có sẵn trong Hailo Model Zoo trước, rồi MoGe nếu DFC compile được ViT.
- Qualcomm: AI Hub compile ONNX fp32 → QNN w8a16.
- Intel/ARM Linux: copy `deploy/onnx/moge2_vits_288x384_int8ac.{xml,bin}` + `pip install openvino` và chạy `benchmark_openvino.py`.
