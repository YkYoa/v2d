# Plan: NVIDIA Video-to-Data (V2D) Challenge — tài liệu + lộ trình demo nhỏ nhưng chạy được

## Trạng thái milestone (Claude cập nhật)
- [x] Bước 0 — repo ~/v2d, PLAN.md, CLAUDE.md, .env.example, server/ssh.sh
- [x] M0.1 — driver laptop (module 595-open cho kernel 7.0.0-38 đã nạp, nvidia-smi OK)
- [x] M0.2 — uv venv + thư viện (torch 2.15+cu132, MoGe, SAM2, MediaPipe, transformers)
- [x] M0.3 — dataset subset + submission kit
- [ ] M0.4 — server 4090 audit (chờ .env)
- [x] M1 — demo perception Track 1 trên laptop (ep09: 60 frame, 16-51 s, VRAM 1.75 GB, outputs/ep09)
- [ ] M2 — FlashCHORD replay + train mini trên 4090
- [ ] M3 — Track 1 submission format (tùy chọn)
- [x] M4 (laptop) — export ONNX + int8 NNCF/OpenVINO đạt AbsRel 0.5%, 7.5 FPS CPU; [ ] M4 (board thật) chờ bạn chốt board

## Context

Folder `/home/hans/v2d` hiện **trống**. "V2D" = **NVIDIA Video to Data Challenge** (CoRL 2026): biến video người thao tác đồ vật thành dữ liệu/kỹ năng robot. 3 track:

| Track | Input → Output | Metric | Baseline chính thức |
|---|---|---|---|
| 1 Reconstruction | 1 video RGB góc thứ 3 → 4D người (MHR) + mesh + quỹ đạo vật | CD-H, CD-O, ACC-H, ACC-O, PEN (thấp hơn tốt) | CARI4D |
| 2 Robotic Grounding | quỹ đạo 4D (3 tier sạch/nhiễu/tự tái dựng) → policy robot G1+Dex3 trong sim | AUC, SP-SR, MP-SR, MPPE | CHORD |
| 3 Egocentric | video camera đội đầu → policy robot (Sharpa hand mặc định) | AUC, SP-SR, MP-SR, RPE, MPPE, CD-O | tissue-box e2e |

**Thực tế quan trọng:**
- Leaderboard đã **đóng 4/11/2026**; hôm nay 12/11 là ngày công bố winner. Mục tiêu bây giờ là **demo chạy được + học pipeline**, không phải thi. (Kaggle có thể vẫn nhận late submission để tự chấm — cần kiểm tra khi thực thi.)
- Pipeline chính thức yêu cầu **GPU 48 GB** (A6000/L40S), Docker, driver ≥580/CUDA 13, **không hỗ trợ Blackwell**.
- Tài nguyên có: **laptop RTX 4050 6 GB** (driver chưa load) + **server lab RTX 4090 24 GB** (user tự add SSH key sau; plan chừa chỗ).
- Các team cộng đồng đã làm Track 1 trên **3090 24 GB** (skytealang) và Track 1 baseline trên GPU ≥24 GB (Swithinraj) → 4090 đủ cho Track 1 đầy đủ, FlashCHORD replay/train nhỏ, và các module reconstruction lẻ.

Kết quả mong muốn: (1) một bộ tài liệu tham khảo có cấu trúc trong `~/v2d/docs/`, (2) **demo M1** chạy trên laptop (video → depth + mask + keypoints 3D), (3) **demo M2** trên 4090: FlashCHORD replay + train ngắn, có số MPPE, (4) tùy chọn **M3**: Track 1 pipeline ra file parquet đúng format submission.

---

## Tài liệu đã thu thập (đưa vào `~/v2d/docs/RESEARCH.md` khi thực thi)

### Chính thức (NVIDIA)
- Trang challenge (rules, timeline, submission, FAQ, kit): https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- FAQ PDF: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf
- Submission kit (eval scripts + agent skill, 718 KB): https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip
- Repo chính: https://github.com/nvidia-isaac/video_to_data (module `reconstruction/` 18 Docker module, `robotic_grounding/` Isaac Lab 2.3, `robotic_grounding/flash_chord/` Newton/Warp/JAX không cần Isaac Sim, `docs/e2e_workflow.md`)
- Dataset (CC-BY-4.0, 4.14 GB, 60 sequence): https://huggingface.co/datasets/nvidia/video_to_data_challenge
  - Track 1: 30 video exo 1536×1152 30fps, ~524 MB tổng, nhỏ nhất `episode_000009.mp4` 11 MB (white_desk). 10 vật × 3 clip. **Không có GT** (held-out).
  - Track 2: 30 episode LeRobot v2.1, Tier 1 sạch (SOMA-X) / Tier 2 nhiễu, video exo giống nhau.
  - Track 3: 20 episode, 3 stream ego (1 màu 2028×1520 + stereo mono 1280×800), mesh+URDF 10 vật, **có GT pose vật** trong parquet → chấm local được.
- HF collection: https://huggingface.co/collections/nvidia/video-to-data (task library, object assets, form-hoi)
- Checkpoint ví dụ Isaac Lab: `huggingface-cli download nvidia/video-to-data-robotic-grounding-example`
- CoRL 2026 tutorial: https://nvidia-isaac.github.io/video_to_data/v2d_corl_tutorial/ ; video: https://www.youtube.com/watch?v=U8GHIxXqHLQ
- Leaderboard data (JSON, cập nhật 9/11): https://github.com/MVerghese/v2d-leaderboard-data — Track 1: 59 team, top CD-H 11.4 / CD-O 16.3; baseline CARI4D CD-H 15.6 / CD-O 23.2. Track 2 T1: 28 team, CHORD baseline AUC 0.51. Track 3: 21 team, top AUC 0.985.
- Kaggle: 22 competition kiểu `v2d-challenge-track3-sp-sr`, `v2d-challenge-track2-tier3-auc`…
- Forum: https://forums.developer.nvidia.com/t/nvidia-video-to-data-v2d-challenge-turn-videos-into-robot-skills/384050 ; Discord Omniverse #v2d-challenge; email v2d_challenge@nvidia.com

### Paper
- **CARI4D** (CVPR 2026, baseline Track 1): https://arxiv.org/abs/2512.11988 — code https://github.com/NVlabs/CARI4D (Docker `xiexh20/cari4d`, cần SMPL-H + weights HF `nvidia/CARI4D`)
- **CHORD** (CoRL 2026, baseline Track 2/3): https://arxiv.org/abs/2607.00033 — https://nvidia-isaac.github.io/video_to_data/chord/
- Thành phần: MoGe-2 (https://github.com/microsoft/MoGe), SAM 2, Grounding DINO, FoundationPose, WiLoR/HaMeR (tay), WHAM (người), MHR (https://github.com/facebookresearch/MHR — format output Track 1), TRELLIS/SAM3D (mesh), Isaac Lab, Newton/Warp.

### Repo cộng đồng (bài học thực tế)
- https://github.com/skytealang/v2d-track1-reconstruction — Track 1 trên **8×3090 24 GB**, MoGe-2 + SAM2.1 + MediaPipe + MHR + FoundationPose; CD-H 33 cm. Bài học: khởi tạo root translation MHR, tối ưu temporal + contact.
- https://github.com/Swithinraj/v2d-track1-wham-foundationpose — Track 1 baseline ≥24 GB VRAM, Apptainer; WHAM→MHR, GroundingDINO+SAM2, MoGe-2, TRELLIS, FoundationPose; CD-H 21 / CD-O 29. **Có mô tả format parquet submission** (MHR pose [T,136], scales [68], shape [45], mesh GLB, Sim(3) track).
- https://github.com/priyam-saha-17/nvidia-v2d-challenge — Track 1 trên **Kaggle T4** (chứng minh chạy được với GPU nhỏ).
- https://github.com/aebatirel/v2d-track3 — Track 3, FlashCHORD, MediaPipe hands + stereo; **Colab notebook** giải thích; perception AUC 0.81 trên 20 episode public.
- https://github.com/the-simsons/ego2sim-nvidia-v2d — Track 3, uv + Python 3.11, FoundationStereo + SAM2 + WiLoR + FoundationPose + FlashSAC.
- https://github.com/Swaguto/NVIDIA-V2D-Challenge — Track 3 Isaac Lab; có **smoke test zero-download** (`dummy_agent.py --task Sharpa-V2D-v0-Play` với synthbox).
- https://github.com/tianny0078/v2d-track3 , https://github.com/ibrahim-1811/v2d-track2-robotic-grounding , https://github.com/civicW/v2d-challenge-asap (tham khảo thêm).

---

## Lộ trình thực thi

### Bước 0 — Lưu plan thành file trong repo để Claude đọc/chạy lại được (làm đầu tiên)
- Copy nguyên nội dung plan này sang `~/v2d/PLAN.md`.
- Tạo `~/v2d/CLAUDE.md` với nội dung: mô tả 1 dòng về dự án, "Đọc `PLAN.md` trước khi làm bất cứ gì; tiến độ ghi ở `docs/RUNLOG.md`; key/secret chỉ đọc từ `.env` (không commit)", và quy ước: mỗi milestone M0–M4 có checkbox trạng thái trong `PLAN.md` (`[ ]`/`[x]`), cập nhật sau khi hoàn thành.
- `git init` trong `~/v2d`, `.gitignore` gồm `data/`, `.env`, `outputs/`, `*.pt`, `*.onnx`, `*.rknn`.
- Từ đó về sau, trong phiên mới chỉ cần nói "đọc PLAN.md và tiếp tục milestone kế tiếp" là Claude chạy tiếp.

### M0 — Chuẩn bị môi trường (laptop + chừa chỗ server)

**M0.1 Sửa driver laptop.** Chẩn đoán: `nvidia-driver-595-open` 595.91.07 đã cài, nhưng kernel đang chạy là `7.0.0-38-generic` và chỉ có module cho `7.0.0-34`. Gói hợp lý nhất (khớp driver open 595 + kernel hiện tại, không đổi version userspace):
```bash
sudo apt install linux-modules-nvidia-595-open-7.0.0-38-generic
sudo reboot
```
Kiểm tra sau reboot: `nvidia-smi` thấy RTX 4050. Lý do không chọn `nvidia-dkms-595-open`: nó kéo 595.99 lệch với libs 595.91 đang cài, phải nâng cả stack. Ghi chú: mỗi lần Ubuntu lên kernel mới phải cài lại gói `linux-modules-nvidia-595-open-<kernel>` (hoặc chuyển hẳn sang dkms sau).

**M0.2 Cấu trúc repo `~/v2d`** (tạo mới):
```
v2d/
  docs/RESEARCH.md        # nội dung phần "Tài liệu" ở trên, có chú thích tiếng Việt
  docs/SETUP_SERVER.md    # checklist cho server 4090
  data/                   # HF dataset (gitignore)
  demo_track1/            # M1: script perception
  server/                 # M2/M3: script chạy trên 4090 (sync bằng rsync/git)
  .env.example            # SSH_HOST, SSH_USER, HF_TOKEN, KAGGLE_* (user tự điền)
```
`uv venv --python 3.11` + `uv pip install torch (cu12x) moge sam2 mediapipe opencv trimesh viser huggingface_hub pyarrow`.

**M0.3 Tải dữ liệu tối thiểu** (laptop, không cần token vì CC-BY public):
```bash
hf download nvidia/video_to_data_challenge --repo-type dataset \
  --include "track_1/meta/*" "track_1/videos/chunk-000/observation.images.exo_camera/episode_000009.mp4" \
  "track_3/public/meta/*" --local-dir data/v2d_challenge
```
Tải submission kit + FAQ vào `docs/kit/` và đọc `eval_*.py` để biết format chính xác.

**M0.4 Server 4090 (user add key rồi mới chạy).** `docs/SETUP_SERVER.md` ghi checklist: `nvidia-smi` (driver ≥580 cho FlashCHORD/Isaac; 4090 = SM 8.9 OK), `docker --version` + NVIDIA Container Toolkit, `git lfs`, disk ≥150 GB, `docker login nvcr.io` (NGC key — user tự làm), `uv`. Plan **không** chứa key; mọi lệnh server đi qua `ssh $SSH_USER@$SSH_HOST` đọc từ `.env`.

### M1 — Demo perception Track 1 trên laptop (mục tiêu "nhỏ nhất nhưng thành công")

Input: `episode_000009.mp4` (white_desk, 11 MB). Script `demo_track1/run_perception.py`:
1. Trích ~60 frame (mỗi 5 frame, resize 768 px) bằng OpenCV.
2. **MoGe-2 ViT-B** (`Ruicheng/moge-2-vitb`, fp16, ~2 GB VRAM) → depth metric + intrinsics + point map từng frame. (Trên 4090 đổi sang ViT-L.)
3. **SAM 2.1 hiera-small** video predictor với prompt box/điểm từ Grounding DINO text `object_prompt` lấy trong `meta/episodes_metadata.jsonl` ("a medium sized white desk") + prompt "person" → mask người & vật.
4. **MediaPipe** Pose + Hands → keypoint 2D, nâng lên 3D bằng depth MoGe.
5. Output: `outputs/ep09/`: `depth/*.png`, `masks/*.png`, `points/*.ply` (point cloud màu, tách người/vật), `keypoints3d.npz`, `overlay.mp4` (mask + skeleton vẽ đè), và `viewer.py` (viser) xem point cloud 4D.
Tiêu chí thành công M1: overlay.mp4 xem được, point cloud vật ở khoảng cách hợp lý (desk ~1–3 m), script chạy end-to-end < 10 phút trên 4050.

### M2 — Demo robot trên 4090: FlashCHORD replay + train ngắn (Track 2/3 side)

Dùng repo chính thức, **không cần Docker/Isaac Sim** (uv path):
```bash
git clone https://github.com/nvidia-isaac/video_to_data && cd video_to_data
git lfs install && git lfs pull
cd robotic_grounding/flash_chord
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[train,dev]"
# 1) replay reference tissue-box Sharpa trong viser (ít VRAM)
python scripts/debug/replay.py "task.parquet='${SHARPA_PARQUET}'" viewer.backend=viser +start_paused=true markers.enabled=true
# 2) train FlashSAC rút gọn cho 24 GB: giảm scene.world_count (vd 1024) + ít step
python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${SHARPA_PARQUET}'" reset.seed=42 output_dir=outputs/sharpa_mini scene.world_count=1024
# 3) evaluate → MPPE
python scripts/evaluate_policy.py "evaluation.checkpoint=${CHECKPOINT}" evaluation.world_count=512 "evaluation.metrics_output=${CHECKPOINT}.metrics.json"
```
**Không có robot thật → mô phỏng hoàn toàn là hợp lệ.** FAQ chính thức: challenge simulation-only, không yêu cầu deploy thật. Hai simulator có thể dùng, đều chạy trên server 4090 (Isaac Sim/Lab bắt buộc GPU RTX NVIDIA, ≥8 GB; laptop 4050 chỉ xem kết quả):
- FlashCHORD (Newton/Warp, không Isaac Sim) — mặc định cho M2.
- Isaac Lab 2.3.2 Docker (`nvcr.io/nvidia/isaac-lab:2.3.2`, cần `docker login nvcr.io` bằng NGC key user tự nhập) — task `Sharpa-V2D-v0`, `SonicG1-ReconHand-v0`; embodiment G1+Dex3, Sharpa Wave, floating hand. Xem robot chạy ngay bằng checkpoint mẫu: `huggingface-cli download nvidia/video-to-data-robotic-grounding-example --local-dir checkpoints/pretrained` rồi `python scripts/rsl_rl/play.py --task Sharpa-V2D-v0 --checkpoint ...`. Dùng khi muốn render đẹp/quay video hoặc đi tiếp sang GR00T finetune (`docs/e2e_workflow.md`) — đường sim-to-real sau này khi có robot thật.
- Với M4: board không-NVIDIA chạy policy đã export và nối với sim trên server qua mạng (hardware-in-the-loop) để kiểm tra latency/độ đúng trước khi có robot.

`SHARPA_PARQUET` = tissue-box parquet bundled dưới `robotic_grounding/assets` (git LFS). Tiêu chí thành công: replay hiển thị tay Sharpa bám reference; train mini ra checkpoint và `metrics.json` có MPPE (so với full-run ~0.8 cm của top team, mini chỉ cần ra số hợp lý < 20 cm). Fallback nếu uv path lỗi: Docker `docker/run.sh build && docker/run.sh start latest 0`. Fallback 2 (Isaac Lab): smoke test synthbox `dummy_agent.py --task Sharpa-V2D-v0-Play` (cần NGC login).

### M3 (tùy chọn) — Track 1 ra đúng format submission trên 4090

Hai đường, chọn theo kết quả M1/M2:
- **A (ưu tiên):** tái sử dụng Swithinraj baseline (≥24 GB, SM ≥8.0 ✔) với Apptainer hoặc chuyển sang Docker; chạy cho 1–3 episode → `pack.sh` ra parquet (MHR pose/scales/shape + GLB + Sim(3) track). Dùng `eval_reconstruction.py` trong kit để **validate format** (không có GT nên không ra điểm local).
- **B:** CARI4D Docker (`xiexh20/cari4d`) chạy demo trên video Track 1 (cần SMPL-H đăng ký MPI + weights HF).
Sau đó nếu Kaggle còn mở late submission → nộp để lấy điểm tham khảo (cần `KAGGLE_USERNAME/KEY` user tự điền; cần commit URL vì rule bắt buộc).

### M4 — Tối ưu weight / transformer, quantize và deploy lên board không phải NVIDIA

Mục tiêu: lấy 2 loại model sinh ra từ M1–M2 và chạy được trên board ARM/NPU bất kỳ (không CUDA, không TensorRT), có bảng so sánh độ chính xác và tốc độ so với FP32 trên 4090.

**M4.0 Chọn thứ để deploy (theo thứ tự dễ → khó)**
| Thành phần | Kiến trúc | Kích thước | Độ khó deploy |
|---|---|---|---|
| Policy robot (FlashCHORD/RSL-RL) | MLP 3 lớp (~[512,256,128]) | < 2 MB | Rất dễ: ONNX → chạy CPU bất kỳ, kể cả MCU |
| Hand/body keypoints | MediaPipe (đã TFLite int8 sẵn) | ~10 MB | Dễ: dùng LiteRT/TFLite có sẵn |
| Depth MoGe-2 ViT-S (35M) / ViT-B (104M) | ViT (DINOv2 backbone) | 140 / 400 MB fp32 | Trung bình: cần static shape + int8 |
| Mask SAM2.1 hiera-tiny | Hiera encoder + prompt decoder | ~150 MB | Trung bình-khó: tách encoder/decoder |
Không deploy: FoundationPose, WHAM, CARI4D, Isaac Lab (quá nặng cho edge; chạy ở server, board chỉ nhận kết quả).

**M4.1 Tối ưu weight / kiến trúc trước khi quantize** (script `deploy/optimize.py`)
- Đổi backbone nhỏ: MoGe-2 ViT-L → ViT-S; SAM2.1 large → hiera-tiny. Đo mất mát trên 100–300 frame V2D (AbsRel depth, IoU mask) để chọn.
- Giảm độ phân giải & cố định shape: 518×518 → 384×288 hoặc 256×256 (static input, không dynamic token), vì NPU (RKNN/Hailo/QNN) không hỗ trợ shape động.
- Chưng cất (distillation) khi cần: train student nhỏ hơn từ MoGe-2 ViT-L trên frame V2D (tham khảo Depth-Anything-Edge chạy 27–60 FPS trên Pi 5 + Hailo).
- Pruning có cấu trúc (bỏ head attention / MLP channel ít quan trọng) chỉ làm nếu còn chậm sau quantize; ưu tiên distill vì ViT nhạy với pruning.
- Policy MLP: không cần tối ưu, chỉ export ONNX opset 17 + fold BatchNorm/obs-normalizer vào graph.

**M4.2 Tối ưu riêng cho transformer khi quantize**
- Dùng **PTQ static int8 với SmoothQuant (alpha ≈ 0.5)** cho ViT; calibration set 200–500 frame từ chính dataset V2D (đủ đa dạng vật/ánh sáng).
- Giữ **LayerNorm, Softmax, GELU ở fp16/fp32** (mixed precision), chỉ int8 cho MatMul/Linear của Q/K/V/MLP; trên board không có fp16 thì int16 cho các op này (RKNN hỗ trợ hybrid quant).
- Fuse attention thành SDPA/MHA op trước export (`torch.onnx.export` với `dynamo=True` hoặc `onnxruntime.transformers.optimizer`), loại bỏ op Where/Gather không cần thiết.
- Kiểm tra outlier activation per-channel; nếu AbsRel tăng > 20% so với fp32 thì chuyển sang **QAT ngắn** (1–2 epoch trên frame V2D) hoặc int8 weight-only + fp16 activation.
- Lưu lại **bảng accuracy drop** cho từng biến thể (fp32 / fp16 / int8-dynamic / int8-static-SQ / QAT).

**M4.3 Export sang format trung gian** (`deploy/export_onnx.py`)
- MoGe-2: có sẵn ONNX fp32 `Ruicheng/moge-2-vits-normal-onnx` / `-vitb-` (opset ≥14, dynamic shape) → tự export lại bản static shape; post-processing (point map, intrinsics recovery) viết lại bằng numpy trên board.
- SAM2.1: dùng `samexporter` (https://github.com/vietanhdev/samexporter) tách `encoder.onnx` + `decoder.onnx`; encoder chạy 1 lần/frame, decoder chạy theo prompt.
- MediaPipe: dùng `.tflite` chính thức (đã int8), không export lại.
- Policy: Isaac Lab `export_policy_as_onnx()` (`isaaclab_rl.rsl_rl.exporter`) hoặc cho FlashCHORD: load `policy_*.safetensors` → `torch.onnx.export`, kèm `env_cfg.json` (obs scale, action scale).
- Verify: `onnxruntime` CPU fp32 trên laptop cho kết quả khớp PyTorch (max abs diff < 1e-3) trước khi quantize.

**M4.4 Toolchain theo board (user chọn 1 board thì thu hẹp; plan để ma trận)**
| Board / NPU | Toolchain | Quantize | Ghi chú đã xác minh |
|---|---|---|---|
| ARM CPU chung (Raspberry Pi 5, Orange Pi, bất kỳ Linux ARM) | **ONNX Runtime** (XNNPACK / ArmNN EP) hoặc **ExecuTorch** (XNNPACK, `.pte`) | ORT static int8 (`quantize_static`, SmoothQuant qua `onnxruntime.quantization`) | Arm đã công bố `depth-anything-v2-small-int8-xnnpack-executorch` → path đã được chứng minh cho ViT depth |
| Rockchip RK3588 (Orange Pi 5, Rock 5B, Radxa) | **RKNN-Toolkit2** (ONNX → `.rknn` trên PC x86) + `rknnlite` trên board | int8 asym per-channel, hybrid quant cho LayerNorm/Softmax; có hỗ trợ transformer/MatMul int4 | SAM2.1 RKNN có sẵn (`happyme531/Segment-Anything-2.1-RKNN2`), encoder tiny ~3 s/frame 1 core → cần giảm res + 3 core |
| Raspberry Pi 5 + Hailo-8/8L AI HAT | **Hailo DFC** (ONNX → `.har` → `.hef`) + HailoRT | int8 bắt buộc, calibration ảnh | Model Zoo v2.19 có Depth-Anything-V2-S 224×224; ViT cần chỉnh post-process ra ngoài; 26 TOPS |
| Qualcomm (QCS8550, Dragonwing, RB3) | **Qualcomm AI Hub** → QNN DLC / ONNX-QNN EP | int8/w8a16 | Depth-Anything-V2 chính thức trên AI Hub, ~77 ms trên QCS8550 |
| Intel (x86/NPU Core Ultra) | **OpenVINO** + NNCF | int8 PTQ (có `OpenVINO/dino_v2-int8-ov` tham khảo) | Dễ nhất nếu board là mini-PC Intel |
| Apple (Mac/iPhone) | CoreML | fp16/int8 palettize | `ebowwa/moge-2-vits-normal-coreml` đã có |
| AMD (Ryzen AI / ROCm) | ONNX Runtime VitisAI EP / ROCm | int8 | `AMD-PAVS-AI/sam2`, `/dinov2` đã tối ưu sẵn |
Đường chuẩn khuyến nghị: **ONNX Runtime int8 trên ARM CPU** làm baseline chạy được ở mọi board, rồi mới ép xuống NPU riêng (RKNN/Hailo/QNN) để tăng FPS.

**M4.5 Runtime trên board** (`deploy/board_runtime/`)
- `run_perception_edge.py`: camera/video → resize static → depth int8 → mask (encoder 1 lần, decoder theo prompt) → keypoints TFLite → point cloud nhỏ → gửi JSON/ZMQ về server hoặc ghi file.
- `run_policy_edge.py`: nhận obs (pose vật + joint state) → ONNX policy MLP → action, đo latency; mục tiêu < 5 ms/step trên ARM CPU để đủ 20 Hz control như trong CHORD.
- Benchmark script: FPS, latency p50/p95, RAM, nhiệt; so với fp32 trên 4090 và laptop.

**Tiêu chí thành công M4**
1. Policy ONNX chạy trên CPU ARM cho cùng action như PyTorch (max diff < 1e-4), latency < 5 ms.
2. Depth int8 trên board: AbsRel tăng ≤ 15% so với fp32, ≥ 2 FPS ở 384×288 trên ARM CPU (ORT), ≥ 10 FPS nếu có NPU.
3. Mask int8: IoU so với fp32 ≥ 0.9 trên 100 frame V2D.
4. Báo cáo `docs/DEPLOY_REPORT.md`: bảng model × precision × board × (accuracy, FPS, size).

Tham khảo thêm cho M4: NVIDIA Model-Optimizer ONNX quantization (dùng được trên PC để sinh ONNX int8 rồi chạy nơi khác), Ultralytics RKNN guide (https://docs.ultralytics.com/integrations/rockchip-rknn/), Hailo community "Deploy state-of-the-art ViT model", `rk-transformers` export guide (https://emapco.github.io/rk-transformers/user_guide/export.html).

---

## File quan trọng sẽ tạo/sửa
- `~/v2d/docs/RESEARCH.md`, `~/v2d/docs/SETUP_SERVER.md`, `~/v2d/.env.example`, `~/v2d/README.md`
- `~/v2d/demo_track1/run_perception.py`, `~/v2d/demo_track1/viewer.py`
- `~/v2d/server/m2_flashchord.sh`, `~/v2d/server/m3_track1.sh` (chạy qua ssh, đọc `.env`)
- `~/v2d/deploy/optimize.py`, `~/v2d/deploy/export_onnx.py`, `~/v2d/deploy/quantize.py`, `~/v2d/deploy/board_runtime/`, `~/v2d/docs/DEPLOY_REPORT.md`
- Tái sử dụng: module `v2d.moge`, `v2d.sam2`, `v2d.detectron2` trong `reconstruction/modules/` của repo chính thức khi có Docker (server); trên laptop dùng thẳng thư viện MoGe/SAM2.

## Verification
1. `nvidia-smi` trên laptop sau M0.1 hiện RTX 4050; `python -c "import torch;print(torch.cuda.is_available())"` = True.
2. M1: chạy `python demo_track1/run_perception.py --episode 9` → tồn tại `overlay.mp4`, ≥50 file `.ply`, `keypoints3d.npz`; mở `viewer.py` xem point cloud; kiểm tra độ sâu vật trong khoảng 0.5–5 m.
3. M2: `replay.py` mở viser trên server (port-forward ssh -L 8080) và hiển thị; `metrics.json` có khóa MPPE; log train không OOM trên 24 GB.
4. M3: parquet có đủ cột theo kit; `eval_reconstruction.py --dry-run`/validate pass.
5. M4: `onnxruntime` int8 trên laptop (CPU, giả lập ARM) ra depth/mask/action khớp fp32 trong ngưỡng ở "Tiêu chí thành công M4"; khi có board thật, chạy `deploy/board_runtime/benchmark.py` và điền bảng `DEPLOY_REPORT.md`.
6. Mọi bước ghi vào `~/v2d/docs/RUNLOG.md` (lệnh, thời gian chạy, VRAM đỉnh, lỗi gặp).

## Giả định / rủi ro (bổ sung M4)
- Chưa biết board cụ thể → plan giữ ma trận toolchain; khi user chốt board sẽ chỉ làm 1 cột. Nếu không có board thật, M4 vẫn xác minh được trên CPU laptop bằng ONNX Runtime int8.
- ViT int8 trên NPU (RKNN/Hailo) có thể cần thay op không hỗ trợ (LayerNorm, GELU) → fallback CPU cho op đó hoặc đổi sang Depth-Anything-V2-S đã có sẵn trong model zoo của board.

## Giả định / rủi ro
- Server 4090: giả định Linux + Docker + driver ≥580; nếu driver < 580, FlashCHORD/Isaac có thể không chạy → báo user nâng driver.
- 24 GB có thể OOM với FlashCHORD mặc định → giảm `world_count`; HOI object reconstruction chính thức (24+ GB) để sau.
- Track 1 không có GT nên "thành công" đo bằng chất lượng trực quan + format hợp lệ; muốn số liệu thật phải nộp Kaggle (nếu còn mở) hoặc làm Track 3 với GT public.
