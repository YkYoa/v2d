# RUNLOG

| Ngày | Bước | Lệnh / việc | Kết quả | Ghi chú |
|---|---|---|---|---|
| 2026-11-12 | Bước 0 | Tạo repo ~/v2d, PLAN.md, CLAUDE.md, .env.example, server/ssh.sh | OK | |
| 2026-11-12 | M0.3 | `hf download nvidia/video_to_data_challenge` subset (ep09 video 11 MB, meta T1/T3) + submission kit + FAQ → `data/`, `docs/kit/` | OK | lần đầu thiếu `track_1/meta`, tải lại OK |
| 2026-11-12 | M0.2 | `uv venv .venv` + torch cu124 từ download.pytorch.org | FAIL | timeout pypi.nvidia.com → thử lại từ PyPI, UV_HTTP_TIMEOUT=900 |
| 2026-11-12 | M0.1 | driver laptop | CHỜ USER | cần `sudo apt install linux-modules-nvidia-595-open-7.0.0-38-generic && sudo reboot` |
| 2026-11-12 | M1 | viết `demo_track1/run_perception.py`, `viewer.py` | viết xong, chưa chạy | chờ env + driver |
| 2026-11-12 | M0.1 | `sudo apt update && sudo apt install linux-modules-nvidia-595-open-7.0.0-38-generic && sudo modprobe nvidia` (user gõ pass) | OK: RTX 4050 6141 MiB, driver 595.99.02 | lần 1 lỗi 404 gói meta → cần `apt update` trước; apt kéo thêm kernel 7.0.0-42 |
| 2026-11-12 | M0.2 | `uv pip install torch torchvision` (PyPI) + MoGe/SAM2/MediaPipe/transformers | OK torch 2.15+cu132, CUDA True | lần 1 timeout pypi.nvidia.com |
| 2026-11-12 | M1 | `python demo_track1/run_perception.py --episode 9` (60 frame, stride 5, 768px) | OK 51 s lần đầu (16 s cache), VRAM đỉnh 1754 MB; overlay.mp4, 120 .ply, keypoints3d.npz | v1: DINO chọn nhầm bàn xanh phía sau → sửa chọn box theo độ sâu gần người + khoảng cách cổ tay + prompt phụ "furniture"; tay 0→14/60 frame nhờ crop quanh cổ tay |
| 2026-11-12 | M4 | `deploy/export_onnx.py moge` (dynamo, cần onnxscript) → ONNX tĩnh 288×384, diff 3.6e-6 | OK | TorchScript exporter lỗi op `upsample_bilinear2d_aa`; bản ONNX chính thức có input num_tokens động → cố định shape bằng onnxruntime.tools.make_dynamic_shape_fixed |
| 2026-11-12 | M4 | `deploy/quantize.py --mode dynamic` + benchmark | int8 dyn: 691→596 ms (4→12 thread), AbsRel 0.005; fp32 752/636 ms | static ORT lỗi tensor bool; SmoothQuant ORT cần neural-compressor (đã cài); profile: Conv head > 50% thời gian |
| 2026-11-12 | M4 | NNCF ONNX backend (`quantize_nncf.py`) | int8 33 MB, OV 137 ms, AbsRel 0.192 ✗; ORT không load (Resize cubic) | SmoothQuant+FastBias trên ONNX làm hỏng độ chính xác |
| 2026-11-12 | M4 | NNCF accuracy-control OV backend (`quantize_nncf_ac.py --max-drop 0.10`) | **int8 IR 34 MB, OV 134 ms (7.5 FPS), AbsRel 0.005** ✔ | OV fp32 208 ms vs ORT fp32 636 ms; đạt tiêu chí M4 trên CPU laptop |
| 2026-11-12 | M4 | `export_onnx.py policy` (MLP giả lập 128→512→256→128→30) + `run_policy_edge.py` | ONNX diff 3.7e-8, latency p50 0.015 ms/step CPU | đường export policy sẵn sàng cho checkpoint FlashCHORD thật ở M2 |
