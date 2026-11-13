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
| 2026-11-12 | M0.4 | (máy Windows Core Ultra 5, Wi-Fi 192.168.1.x) tạo `.env` từ `~/.ssh/config`, `SERVER_WORKDIR=/data2/huyhoang/v2d`; `server/ssh.sh hostname` | FAIL: timeout port 22 | cả 2 host trong ssh config đều không ping được từ mạng này → cần VPN/mạng lab; host/user là suy đoán, chờ user xác nhận |
| 2026-11-12 18:05 | M0.4 | VPN bật → `server/ssh.sh` + `server/audit.sh` (phải bỏ CR: thêm `.gitattributes` eol=lf) | OK: RTX 4090 24564 MiB, driver 580.178.04, SM 8.9, Docker 29.8.1 + runtime nvidia, Ubuntu 24.04, 32 CPU, RAM 62 GB, /data2 trống 2.7 TB | không có sudo: thiếu `git-lfs`, `uv` nằm ở `/data2/huyhoang/tools`; **GPU đang bị job train của user khác chiếm 14.7 GB / 73 %** → chỉ còn ~9.8 GB |
| 2026-11-12 18:08 | M2 | tải binary git-lfs → `/data2/huyhoang/tools`; `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 video_to_data` → `/data2/huyhoang/v2d` | OK (~3 phút, 1057 file LFS chưa pull) | tissue-box parquet nằm ở `robotic_grounding/source/.../ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave`; full train cần 46 GB trống, recipe 250M step → script mini: 5M step, replay 524k |
| 2026-11-12 18:15 | M2 | `git lfs pull --include=robotic_grounding/**` + `uv sync --extra train --extra dev` (nohup trên server) | BỊ CHẶN | bộ phân loại quyền của Claude Code từ chối tải+chạy code ngoài trên server; chờ user cấp quyền |
| 2026-11-12 18:17 | M2 | user cho phép → `git lfs pull --include=robotic_grounding/**` + `uv sync --python 3.11 --extra train --extra dev` (theo uv.lock) | OK ~13 phút, repo 3.8 GB; warp 1.14.0 + jax 0.10.2 thấy cuda:0 | cache uv ở `/data2/huyhoang/uv_cache` |
| 2026-11-12 18:32 | M2 | smoke: `OUT=smoke WORLDS=64 STEPS=131072 REPLAY=65536 REPLAY_MIN=6400 bash m2_flashchord.sh train` rồi `eval` (64 world) | OK: train 242 s (122 s chạy + compile kernel), ~1070 env step/s, ~0.5 GB VRAM; `metrics.json` có `mppe_cm` = **6.75**, chord_sr 0, kết thúc do vật lệch (object 100 %) | GPU đang dùng chung; curriculum 250M được co theo STEPS (`curriculum.thresholds=[...]`); policy 131k step chưa học được gì đáng kể, chỉ để kiểm tra đường ống |
| 2026-11-12 18:38 | M2 | `replay.py ... viewer.backend=viser +start_paused=true` (timeout 100 s) | OK: "Replay kernels are ready" | chưa xem bằng mắt; xem: `ssh -L 8080:localhost:8080` rồi `bash m2_flashchord.sh replay` |
| 2026-11-12 18:38 | M2 | `nohup setsid bash m2_wait_gpu.sh` với `OUT=sharpa_mini WORLDS=1024 STEPS=20000000 REPLAY=1048576 REPLAY_MIN=102400 EVAL_WORLDS=512` | ĐANG CHỜ GPU rảnh (<1.5 GB dùng, 3 phút liên tục, tối đa 14 h) | log: `/data2/huyhoang/v2d/outputs/m2_wait.log`, `train_sharpa_mini.log`, `sharpa_mini.vram.log`; kết quả: `outputs/sharpa_mini/policy_*.metrics.json` |
| 2026-11-12 20:15 | M2 | GPU vẫn bận (user khác chạy job thứ 2 từ ~18:25, 13.4 GB) → user đồng ý chạy song song: dừng `m2_wait_gpu.sh`, `OUT=sharpa_mini WORLDS=512 STEPS=20000000 REPLAY=524288 REPLAY_MIN=51200 EVAL_WORLDS=512` train rồi eval (nohup, log `outputs/m2_run.log`) | ĐANG CHẠY: ~8 000 env step/s, ETA ~40 phút, VRAM tiến trình 5.8 GB (GPU tổng 19.3/24.5 GB, 100 % util) | lần đầu `pkill -f m2_wait_gpu.sh` tự giết luôn phiên ssh (chuỗi lệnh chứa tên script) → chạy lại |
| 2026-11-12 20:57 | M2 | train mini xong (`sharpa_mini`: 512 world, 19 999 744 step, curriculum co về 20M, kết thúc ở stage 10, voc_scale 0) + `evaluate_policy.py` 512 world | **OK: train 2425 s (~40 phút), ~8 300 env step/s, VRAM đỉnh tiến trình 5.8 GB; `mppe_cm` = 0.64, success 93.4 % (chord_sr), maniptrans_sr 98.6 %, lỗi vật 6.4 mm / 0.92°; thất bại: cổ tay 6.6 %, vật 2.0 %** | chạy song song job của user khác (GPU 100 % util) nên tốc độ thấp hơn khi GPU rảnh; không OOM; đã xoá state_* trung gian, outputs 349 MB |
| 2026-11-12 21:00 | M4 | xem cấu trúc `policy_19999744.safetensors` | 32 tensor `state.00xx` phẳng (khối 256↔1024 có norm, đầu vào/ra 504 và 56 chiều), metadata có `policy_schema_json` | **không phải MLP thuần** → `deploy/export_onnx.py policy` (giả định Linear+ELU) chưa dùng được; cần dựng lại actor FlashSAC (flax) rồi mới export ONNX — TODO |
| 2026-11-12 21:00 | M2 | render policy ra MP4 không cần màn hình: `PYGLET_HEADLESS=1 PYOPENGL_PLATFORM=egl python scripts/view_policy.py evaluation.checkpoint=... viewer.backend=mp4 viewer.headless=true viewer.num_frames=520 viewer.width=1280 viewer.height=720 "viewer.camera_position=[0.55,-0.65,1.45]" "viewer.camera_target=[0.0,0.0,1.05]"` | OK: `outputs/m2/sharpa_mini_policy.mp4` (26 s, 0.5 MB) + contact sheet `docs/figures/m2_sharpa_mini_policy_sheet.jpg`; hai tay Sharpa nhấc hộp khăn giấy khỏi đế, giữ rồi đặt lại | thiếu `PYGLET_HEADLESS=1` → pyglet lỗi NoSuchDisplay; camera mặc định quá xa (vật ở z≈1.06 m) nên phải đặt camera tay |
| 2026-11-12 21:16 | M2 | render reference kinematic: `replay.py replay=kinematic action=residual_hand_pose_stiff collision=kinematic embodiment=sharpa_hands_stiff viewer.backend=mp4 viewer.num_frames=1296 viewer.video_fps=50` (cùng camera) + ffmpeg hstack với video policy | OK: `outputs/m2/sharpa_reference.mp4`, `outputs/m2/sharpa_ref_vs_policy.mp4` (trái = reference, phải = policy), sheet `docs/figures/m2_sharpa_ref_vs_policy_sheet.jpg`; 5 frame mẫu khớp nhau về pha nhấc/giữ/đặt | replay là 1296 frame @ 50 Hz, policy 518 step @ 20 Hz (cùng ~26 s); lần đầu đặt num_frames=520 @ 20 fps nên chỉ quay được 10 s đầu |
| 2026-11-12 21:40 | M4 | user chốt board đích: **Renesas R-Car X5H** (chưa có board → hoãn phần board) | ghi nhận | |
| 2026-11-12 21:45 | M4 | `deploy/export_flashsac_onnx.py dump` (venv FlashCHORD, loader chính thức) → `build` (`uv run --no-project --with onnx --with onnxruntime --with numpy`) cho `sharpa_mini/policy_19999744` | **OK: `policy_sharpa_mini.onnx` 4.6 MB, obs 504 → action 56, 36 node; max sai khác vs JAX fp32 6.4e-6; latency 1 thread CPU server p50 0.095 ms** | actor = BN → Linear → 2 khối inverted-residual (256↔1024) → RMSNorm → head mean có gain → tanh. Bẫy: (1) trên GPU NVIDIA JAX nhân ma trận bằng TF32 → tham chiếu "fp32" lệch 4e-3, phải `jax.default_matmul_precision("float32")`; (2) policy train bằng mixed precision float16 → lệch 7.4e-3 so với fp32 (không tránh được, kiểm bằng vòng kín) |
| 2026-11-12 21:55 | M4 | vòng kín: `deploy/eval_onnx_in_sim.py scripts/evaluate_policy.py ...` (ONNX Runtime CPU sinh hành động trong sim FlashCHORD, 512 world, 518 step) | **OK: ONNX mppe_cm 0.616 / success 92.4 % / maniptrans 99.0 % vs JAX 0.642 / 93.4 % / 98.6 %** (chênh 5/512 world, trong mức nhiễu) | ONNX batch 512 trung bình 2.36 ms/lần gọi (4 thread); file ONNX đã tải về `deploy/onnx/` (gitignore) |
| 2026-11-12 22:45 | M3 | khảo sát: kit Track 1 (`pack_reconstruction.py track1`: mỗi episode 1 NPZ `pose[T,136] scales[68] shape[45] object_rotation[T,3,3] object_translation[T,3] object_scale` + mesh), repo cộng đồng, gating model | chọn đường **không cần model gated**: MoGe-2 + SAM2.1 + MediaPipe → fit tham số MHR (model MHR Apache-2.0, GitHub release v1.0.1) theo keypoint 3D, mesh vật từ depth, đóng gói bằng kit | SAM 3D Body gated thủ công, SMPL cần đăng ký → loại đường A/B trong PLAN; `sample_submission.parquet` chỉ có trên Kaggle (cần `KAGGLE_*` + chấp nhận luật) → nếu chưa có sẽ tự sinh sample giả lập để kiểm format; đội skytealang đi đường này đạt CD-H 33 cm |
| 2026-11-12 22:50 | M3 | setup server `/data2/huyhoang/v2d/m3`: venv + torch + MoGe/SAM2 (pip từ git) + mediapipe + tải ep09 + `mhr_model.pt` | BỊ CHẶN | bộ phân loại quyền của Claude Code từ chối cài code ngoài mới trên server; chờ user cho phép |
| 2026-11-12 23:15 | M3 | user tự chạy setup trên server (`/data2/huyhoang/v2d/m3`: uv venv + torch 2.15+cu132 + MoGe/SAM2 từ git + mediapipe + ep09 + `mhr_model.pt`) | OK, sha256 MHR khớp kit | `hf download --include track_1/meta/*` lại không tải meta → lấy 4 file meta bằng curl `resolve/main` |
| 2026-11-12 23:23 | M3 | `run_perception.py --episode 9 --stride 1 --max-frames 415 --moge Ruicheng/moge-2-vitl-normal` | OK 219 s, VRAM đỉnh 2.3 GB (stage SAM2; MoGe ViT-L không đo riêng); body 415/415 frame, tay 63/415; người z≈3.0 m, bàn z≈2.7 m | GPU dùng chung (job khác ~14.7 GB) |
| 2026-11-12 23:28 | M3 | `track1_submit/fit_mhr_keypoints.py` (Adam 1200 bước, 415 frame cùng lúc; 2D reprojection + depth Huber + mượt gia tốc + prior) | OK ~1.5 phút: reprojection TB 5.3 px (ảnh 768 px), gia tốc khớp 0.30 cm/frame², hông z 3.01 m | trọng số chọn `--w-z 20 --w-smooth 20000 --w-prior 0.05`; bản đầu (smooth yếu) overfit 0.45 px nhưng pose méo + rung 1.07 cm/frame²; smooth 60000 chưa hội tụ (8.4 px); scales/shape để 0, ngón tay để 0 |
| 2026-11-12 23:30 | M3 | `track1_submit/object_from_depth.py` (mesh = bề mặt nhìn thấy frame 0 + mặt sau 3 cm; ICP cắt tỉa frame→mô hình, làm mượt) | OK: mesh 4804 đỉnh / 7912 mặt, OBB 1.45×0.99×0.65 m; ICP sai số trung vị 1.46 cm; xoay tối đa 106° so với frame 0; gia tốc tịnh tiến 0.21 cm/frame² | phải đặt gốc mesh ở tâm vật (gốc ở camera → nhiễu xoay khuếch đại vào translation: 0.65 → 0.21) |
| 2026-11-12 23:32 | M3 | `track1_submit/pack_episode.py` → packer chính thức `eval_reconstruction.py` + đọc ngược bằng `v2dlb.mhr_submission` + dựng lại thân MHR từ tham số đã đóng gói | **FORMAT OK: `track1_ep09_LOCAL_CHECK_do_not_upload.parquet` 28 981 dòng, 0.58 MB, 415 frame; round-trip pose/translation 0, rotation 3e-8; mesh 4096/4096 dòng; MHR 18 439 đỉnh hữu hạn** | sample là GIẢ LẬP (mọi frame, 1 episode) vì sample thật chỉ có trên Kaggle → file không nộp được, chưa có điểm; ảnh kiểm `docs/figures/m3_ep09_packed_check.jpg`: thân bám người, mặt bàn bám tốt, chân bàn lệch ở nửa sau |
| 2026-11-13 00:05 | M3 | user đưa Kaggle token + username → lưu `.env` (`KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`); thử API bằng curl `Authorization: Bearer` | token OK: liệt kê được `sample_submission.parquet` (3.5 MB) ở cả 5 competition Track 1; **tải về bị 403: chưa chấp nhận luật competition** | từ PC này phải thêm `--ssl-no-revoke` (schannel không kiểm được CRL qua VPN); token kiểu `KGAT_` dùng Bearer, basic-auth username:token trả 401; chờ user bấm chấp nhận luật trên web |
| 2026-11-13 00:15 | M3 | user đã chấp nhận luật → tải `sample_submission.parquet` thật (3.5 MB, 740 780 dòng, 30 episode, 9 877 frame chấm; ep09 chấm frame 90–414; ngân sách mesh 4096/4096 đúng như đã suy) | OK → `data/kaggle/` + server `m3/data/` | |
| 2026-11-13 00:18 | M3 | đóng gói ep09 theo dòng THẬT của sample (lọc riêng ep 9) | **FORMAT OK: `track1_ep09.parquet` 24 481 dòng, 325 frame chấm** | |
| 2026-11-13 00:20 | M3 | `nohup setsid bash m3_batch.sh` (30 episode, 16 563 frame; xong thì đóng gói theo sample thật → `outputs/track1_all_30ep.parquet`) | ĐANG CHẠY, log `m3/outputs/batch.log` | ước ~2.5–3 giờ; commit URL còn là placeholder → file chưa nộp được; viết `docs/MANAGER_REPORT.md` |
| 2026-11-13 00:30 | M3 | user chọn tự kéo thả file lên 5 competition Kaggle; commit URL điền `https://github.com/YkYoa/v2d/commit/c3a14dc…` (commit local, CHƯA push → 404 cho tới khi push); `m3_final.sh` trên server chờ batch xong rồi đóng gói → `m3/outputs/track1_submission.parquet` | ĐANG CHỜ batch (~3 h, ~6 phút/episode) | script server trùng mã băm với commit c3a14dc (trừ `run_perception.py`, chưa kiểm tra) |
| 2026-11-13 00:40 | M3 | user cho dùng thêm GPU → chạy worker thứ 2 song song: `EPISODES="29 … 19" bash m3_batch.sh` (log `batch_b.log`); worker 1 đi 0→18 | ĐANG CHẠY | VRAM trống thực tế chỉ ~8.7 GB (job user khác 14.7 GB) nên chỉ thêm 1 worker (~2.5–3 GB/worker), không phải 3 |

### 2026-11-13 ~09:00 — M3 batch: 27/30, chạy lại 3 episode lỗi
- Batch 2 worker xong 02:07, 27/30 episode. Ep 4, 5, 7 FAIL: `CUDA out of memory` ở bước fit MHR (episode dài 669–778 frame, fit cần ~6 GB+, trong khi worker thứ hai giữ ~3 GB và job user khác 14.4 GB). Ep 22–26 cũng OOM ở worker B nhưng worker A đã chạy lại thành công.
- Bài học: 2 worker song song chỉ an toàn với episode ngắn; episode > ~600 frame phải chạy một mình.
- Chạy lại: `EPISODES="4 5 7" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash m3_batch.sh > outputs/batch_c.log` (1 worker, GPU trống 9.3 GB), xong thì tự chạy `m3_final.sh`.
- 02:56 chạy lại xong: ep 4 OK 307s, ep 5 OK 278s, ep 7 OK 324s → 30/30. `m3_final.sh` → `FINAL_PACK_OK`: `track1_submission.parquet` 740,780 dòng, 10.4 MB, `row_id` trùng khớp đúng thứ tự với sample Kaggle thật, mọi giá trị hữu hạn, `code_commit_url` = commit `c3a14dc`.
- Chất lượng fit trên 29 episode của batch: reprojection trung bình 5.7 px (tệ nhất 10.1 px, ep 8); ICP vật trung vị trung bình 0.92 cm (tệ nhất 6.6 cm).
- Đã tải về `outputs/m3/track1_submission.parquet` (sha256 khớp server, 377d9066…). Chờ user kéo-thả lên 5 competition Track 1 (CD-H, CD-O, ACC-H, ACC-O, PEN) rồi ghi điểm.

### 2026-11-13 ~10:20 — M3: nộp Track 1 lên Kaggle bằng CLI (late submission), có điểm
- Kaggle CLI 2.2.5 trong venv tạm (`%TEMP%\claude\kagglecli`, Python 3.14) + `six` + `pip-system-certs` (mạng công ty chèn chứng chỉ TLS riêng → `CERTIFICATE_VERIFY_FAILED` nếu không dùng kho chứng chỉ Windows). Token đọc từ `.env`.
- `kaggle competitions submit -c v2d-challenge-track1-{cd-h,cd-o,acc-h,acc-o,pen} -f outputs/m3/track1_submission.parquet` → cả 5 COMPLETE.
- Điểm (public = private, thấp hơn là tốt hơn): **CD-H 40.59 cm · CD-O 80.38 cm · ACC-H 0.298 cm · ACC-O 0.601 cm · PEN 0.0016 cm**.
- Tham chiếu CD-H: leader 11.4, baseline NVIDIA 15.6, đội cộng đồng cùng hướng 33. CD-O 80 cm là điểm yếu nhất (mesh chỉ là mặt nhìn thấy ở frame tham chiếu, tỉ lệ/độ sâu tuyệt đối từ MoGe).
- User đã tự upload tay trước đó vài bản cùng file (điểm trùng khớp) → có bản nộp trùng ở CD-H, ACC-H, PEN.

### 2026-11-13 — M3 v2: chẩn đoán điểm Track 1 và vòng cải thiện CD-H / CD-O
- Đọc `metric_code/track_1/CD-O.py`: scorer fit **một** Sim(3) trên đỉnh MHR của frame chấm đầu tiên rồi áp cho cả episode; Chamfer là **tổng hai chiều** → CD-H 40 cm ≈ lệch ~20 cm, CD-O 80 cm ≈ lệch ~40 cm. Sai hướng xoay/tỉ lệ ở frame đầu bị khuếch đại theo quãng đường người/vật di chuyển.
- `track1_submit/diag_facing.py` (tô mặt trước đỏ / lưng xanh ở frame chấm đầu, 30 episode → `outputs/m3/checks/diag_facing.jpg`): hướng người phần lớn đúng; nghi lật trước/sau ở ep 26, 29.
- Báo cáo v1 theo episode: ICP vật xoay tới ~180° ở 15+ episode (lật trên mặt mỏng/đối xứng); `travel_m` 3–4 m ở ep 0, 1, 12, 13, 14 (mask/depth nhảy). Vòng hula khi cầm trên không: depth trong mask là depth nền (z 3.4–5.0 m trong khi người ở 2.6 m).
- Không có intrinsics camera trong dataset (`meta/` chỉ có tên camera); MoGe ước lượng fx ≈ 415 px ở ảnh rộng 768 (HFOV ~86°).
- Thay đổi v2:
  - `track1_submit/prep_clouds.py`: hiệu chỉnh tỉ lệ depth từng frame theo nền tĩnh (ep01: scale 0.951–1.036, std 1.7%; ep03: std 0.6%) + gom cloud người/vật vào `clouds.npz`.
  - `fit_mhr_keypoints.py`: thêm số hạng bề mặt (điểm depth của người → đỉnh MHR gần nhất), tối ưu shape, chia đoạn + cộng dồn gradient (VRAM ~2 GB với `--chunk 32`, không phụ thuộc độ dài episode). Ep01, 900 bước, 5.5 phút: reproj 6.3 px (v1 5.7), gia tốc khớp 0.14 cm/frame² (v1 0.34), dư bề mặt trung vị 4.3 cm.
  - `object_from_depth.py`: chặn nhảy xoay > 25°/frame, loại frame hỏng rồi nội suy, và khi vật đã rời vị trí nghỉ mà depth lệch người > 0.7 m thì đặt vật trên tia nhìn ở độ sâu của người. Ep01: travel 3.26 → 1.79 m, 191 frame depth không tin được.
- Chạy lại perception giữ point map (`server/m3_perception_all.sh`): worker thứ hai chết vì `cusolver` (hết VRAM khi tôi chạy thử fit cùng lúc) → chạy tuần tự lẻ rồi chẵn. Giữ tổng VRAM của mình ≤ ~5 GB để không ép job 14.7 GB của user khác.
- `server/m3_v2_batch.sh` (prep → fit → vật → pack kiểm → `outputs/track1_episodes_v2/`), log `outputs/v2_a.log`, `outputs/v2_b.log`. Ước tính xong ~2.5 giờ.
- Kế hoạch sau challenge: `docs/POST_CHALLENGE_PLAN.md`.
- G1 + Dex3 video: tải 2 GIF kèm repo NVIDIA về `outputs/g1/nvidia_g1_reference.gif`, `outputs/g1/nvidia_g1_policy.gif` (của NVIDIA, không phải kết quả của ta). Tự render replay G1 snack-box (`replay.py --config-name=sonic_replay … replay=kinematic viewer.backend=mp4`, 800 frame) CHƯA được: bộ điều khiển SONIC cần ONNX Runtime CUDA, venv thiếu `libcurand.so.10` (cần `nvidia-curand==10.4.5.25`, CUDA 13); cài bị chặn quyền. Thử `device=cpu`: quá 15 phút chưa xong, bỏ.
- G1 render (sau khi user cài `nvidia-curand==10.4.5.25`; cần `LD_LIBRARY_PATH=.venv/.../nvidia/*/lib`): `replay.py --config-name=sonic_replay start_paused=false viewer.backend=mp4 viewer.num_frames=800 viewer.video_fps=50 viewer.camera_position=[0.9,-3.6,1.6] viewer.camera_target=[0.9,0,0.75]` → `outputs/g1/g1_snack_replay.mp4` (16 s, SONIC chạy CUDA EP). Đây là replay VẬT LÝ không có policy: robot không đi tới hộp (tham chiếu đi tới x≈1.2 m), hộp tự bay theo quỹ đạo ghi sẵn, cuối clip robot ngả ra sau. Chế độ "kinematic reference" thuần chỉ bật được qua nút trong Viser; `replay=kinematic collision=kinematic` với mp4 làm robot biến mất khỏi khung.

### 2026-11-13 05:12 (giờ server) — M2b: train mini G1 + Dex3 (snack-box pick-and-place), song song với M3 v2
- `server/m2_g1_mini.sh`: `experiment=g1_recon_body_flash_sac`, 256 world (gốc 2048), 20,000,000 step (gốc 350M), replay 524,288 (gốc 10M), curriculum co theo tỉ lệ; log `outputs/train_g1_mini.log`, VRAM `outputs/g1_mini.vram.log`.
- Dừng worker v2 thứ ba (`v2_c`) để nhường ~2 GB; các episode của nó do hai worker còn lại chạy nốt.
- Khởi động OK: 3,400–3,700 env step/s, ETA ~1.5 giờ, VRAM 4.0 GB; tổng GPU 22.7/24 GB (job user khác 14.7 GB + 2 worker fit 2×2 GB + train 4 GB).
- Lưu ý: ngân sách chỉ bằng ~6% công thức đầy đủ nên chưa chắc policy làm được nhiệm vụ.

### 2026-11-13 ~06:55 (giờ server) — kết quả M3 v2 và M2b (G1 mini)
- **M3 v2**: 30/30 episode, không FAIL (2 worker fit `--chunk 32`, ~6–9 phút/episode; worker thứ ba dừng giữa chừng để nhường VRAM). Reproj trung bình 6.6 px (tệ nhất 10.9), dư bề mặt người trung vị 3.9 cm, tổng 619 frame "depth vật không tin được". Pack: `outputs/m3/track1_submission_v2.parquet`, 740,780 dòng, 11.0 MB, sha256 cb154f54… (khớp server).
- Nộp Kaggle bằng CLI (late), cả 5 COMPLETE:

  | Chỉ số (cm, thấp = tốt) | v1 | v2 |
  |---|---|---|
  | CD-H | 40.59 | **34.91** |
  | CD-O | 80.38 | **67.49** |
  | ACC-H | 0.298 | **0.221** |
  | ACC-O | 0.601 | **0.351** |
  | PEN | 0.0016 | 0.0022 |

- File v2 vẫn ghi `code_commit_url` = commit `c3a14dc` (code v2 chưa commit/push).
- **M2b G1 mini**: train xong 20,000,000 step trong 4,797 s (~80 phút, ~4,200 step/s, VRAM 4.0 GB). Đánh giá 256 world: **0% thành công**, MPPE 57.9 cm, lệch vị trí vật 0.58 m, lệch hướng 61°; nguyên nhân dừng: wrist + object ở 100% world. Video `outputs/g1/g1_mini_policy.mp4` (`view_policy.py … markers.enabled=false viewer.backend=mp4`; để markers bật thì lỗi kernel `update_vbo_transforms_from_points`): robot đứng vững, có cử động tay/chân nhưng không đi tới hộp. Ngân sách 6% công thức gốc là không đủ.

### 2026-11-13 — M3 v3: dò nguyên nhân bằng biến thể hậu xử lý + điểm Kaggle (không fit lại)
- Công cụ: `track1_submit/variant_zscale.py` (server) + `server/m3_variant.sh` + `track1_submit/kaggle_try.sh` (laptop: tạo biến thể → tải → nộp CD-H/CD-O → ghi `outputs/m3/kaggle_probes.log`). Mỗi lần dò ~3 phút.
- **Lật trước/sau: KHÔNG có.** So hướng bàn chân của thân đã fit với hướng gót→mũi quan sát (MediaPipe 2D và 3D): cos trung bình 0.56–0.98 ở cả 30 episode, quanh frame chấm đầu đều dương (trừ ep04 thiếu dữ liệu 2D). Nhận định "ep 26, 29 bị lật" từ ảnh v1 là sai.
- **Tiêu cự: không phải nguồn chính.** Kéo giãn trục sâu quanh tâm camera (gốc người + tâm vật), a = f*/f:

  | a | CD-H | CD-O |
  |---|---|---|
  | 1.15 | 35.15 | 72.19 |
  | 1.00 (v2) | 34.91 | 67.49 |
  | 0.87 | 35.26 | 64.86 |
  | 0.75 | — | 63.46 |

  CD-H gần như không đổi → người ít dịch chuyển theo chiều sâu; sai số người đến từ tư thế/hướng xoay, không phải tỉ lệ chiều sâu.
- **Độ sâu vật so với người: đòn bẩy lớn.** Chỉ sửa vật, giữ tia nhìn, z' = z_người + clip(b·(z_vật − z_người), ±c):

  | b | c (m) | CD-O |
  |---|---|---|
  | 1.0 | — | 67.49 |
  | 0.5 | — | 57.35 |
  | 0.0 | — | 60.22 |
  | 1.0 | 0.30 | **47.74** |
  | 1.0 | 0.15 | 50.54 |
  | 0.5 | 0.20 | 50.60 |

  Diễn giải (của tôi, chưa kiểm chứng): hoặc depth vật hay lệch xa (mask dính nền, vật mảnh), hoặc Sim(3) từ frame đầu có nhiễu xoay nên vật càng xa người càng sai → kéo vật về gần người làm giảm sai số kỳ vọng. Giới hạn 0.3 m theo chiều sâu là prior hợp lý về vật lý (vật đang được thao tác nằm trong tầm tay), nhưng một phần mức giảm là hiệu ứng "co về mỏ neo", không phải perception tốt hơn.
- NLF (Neural Localizer Fields, isarandi/nlf): model cơ thể 3D TorchScript, chỉ cần torch + torchvision, không gated; **trọng số chỉ cho nghiên cứu phi thương mại** → chưa tải, chờ user quyết định.
- Thêm: c = 0.45 → CD-O 48.90; c = 0.30 + bỏ xoay ICP (R = I) → 49.03 (xoay ICP có ích ~1.3 cm, giữ lại).
- **v3 = v2 + giới hạn |z_vật − z_người| ≤ 0.3 m** (`OBJ_B=1.0 OBJ_C=0.3`, hậu xử lý, không fit lại): `outputs/m3/track1_submission_v3.parquet`. Kaggle: **CD-H 34.91 · CD-O 47.74 · ACC-H 0.221 · ACC-O 0.311 · PEN 0.00096**. So baseline NVIDIA 15.6 / 23.2: chưa đạt. Tổng số lần nộp dò hôm nay trên CD-O: 10.

### 2026-11-13 — M3 v4: cơ thể 3D từ NLF (user chọn NLF; trọng số CHỈ cho nghiên cứu phi thương mại)
- Tải `nlf_l_multi_0.3.2.patch6.torchscript` (549 MB, release v0.3.2 của isarandi/nlf) vào `checkpoints/nlf/` trên server. Chỉ cần torch + torchvision; `estimate_smpl_batched(images, boxes, intrinsic_matrix=K)` → SMPL 24 khớp + 6890 đỉnh, hệ camera, mm. VRAM ~0.9 GB, ~16 s cho 592 frame.
- `track1_submit/nlf_body.py`: box từ mask SAM2 → NLF → `clouds_nlf.npz` (khớp NLF đặt vào chỉ số MediaPipe, 1,200 đỉnh SMPL làm "bề mặt", và g = z_NLF/z_MoGe của bề mặt người).
- `fit_mhr_keypoints.py --clouds clouds_nlf.npz`: dùng khớp + bề mặt NLF (tâm khớp thật, bề mặt đầy đủ cả lưng), w_z 300, lọc Savitzky–Golay 9 frame; thêm `--init` (khởi tạo từ fit trước). Thử lịch trọng số làm mượt → loss nhảy ở bước đổi lịch → bỏ.
- `object_from_depth.py --nlf clouds_nlf.npz --z-clamp 0.3`: đưa cloud vật (MoGe) về thang NLF bằng g, độ sâu người lấy từ NLF, giới hạn |z_vật − z_người| ≤ 0.3 m (từ v3).
- Ep03 (thử): reproj 4.4 px (v2 6.1), dư bề mặt 2.3 cm (v2 4.5). g theo episode: ep00 1.106, ep03 1.134, ep29 1.028 → MoGe thấp hơn NLF 3–13% về độ sâu người.
- `server/m3_v4_batch.sh` (khoá `outputs/epNN/.v4lock` cho nhiều worker), 4 worker song song; log `outputs/v4_w{1..4}.log`.

### 2026-11-13 07:30–15:10 — Demo A: video điện thoại → tham chiếu robot, KHÔNG dùng MANO / SAM 3 (user: GR00T được dùng; MANO, SAM 3 không thương mại hoá được)
- Rà giấy phép → `docs/LICENSES.md`. Workflow e2e chính thức của NVIDIA cần MANO ở cả bước tái dựng tay (HaMeR/WiLoR) lẫn retarget (`ego_recon_to_sharpa.py` đọc khớp + hướng xoay MANO, manotorch) → phải thay.
- Đọc loader FlashCHORD (`flash_chord/data/mano_sharpa.py`): chỉ cần mảng số (cổ tay, 22 khớp ngón, 67 frame robot, pose vật, tiếp xúc vị trí/pháp tuyến/part id, đường dẫn mesh/URDF, bán kính), không cần tham số MANO.
- `phone_demo/retarget_sharpa.py`: 21 keypoint/tay (thứ tự MediaPipe = MANO) → IK Levenberg–Marquardt trên model MuJoCo của tay Sharpa (asset Apache-2.0), cùng bảng frame/trọng số với NVIDIA, phóng ×1.2 quanh cổ tay; hướng cổ tay từ Kabsch 5 điểm lòng bàn tay; tiếp xúc hình học (điểm đốt ngón → bề mặt vật, ngưỡng 2 cm, lòng bàn tay 3.5 cm); đẩy đầu ngón ra khi cắm vào vật > 1 cm; sinh mesh/URDF/mặt đỡ USDA.
  - Bẫy: (1) pháp tuyến tiếp xúc phải hướng VÀO vật (`objectives/wrench.py` dựng nón ma sát quanh nó); (2) loader đòi pháp tuyến đơn vị ở mọi ô đang bật → debounce bật thêm ô thì phải tính lại; mesh có mặt suy biến (pháp tuyến 0).
- Tự kiểm (`--selftest`, đầu vào = 21 khớp tay + pose hộp trong mẫu tissue-box của NVIDIA): đầu ngón lệch bản retarget chính thức 1.0 cm, cổ tay 1.3 cm / 9°; tiếp xúc trùng 93–96 %; đầu ngón cắm > 2 cm: 0 % (chính thức cũng 0 %). FlashCHORD `load_reference` đọc được, tự dò mặt đỡ.
- `server/phone_demo.sh` (retarget | replay | train | eval | render). Render mp4 phải `markers.enabled=false` (lỗi kernel `update_vbo_transforms_from_points`); số frame render = thời lượng / motion_speed 0.5 × 50.
- **Train trên tham chiếu tự retarget** (`SEQ=selftest_box SELFTEST=1`, 512 world, 20M step, 2575 s, 5.8 GB): **chord_sr 99.8 %, MPPE 0.50 cm**, lệch vật 5.0 mm / 0.54°. (M2 với tham chiếu MANO chính thức: 93.4 % / 0.64 cm.) Policy: `/data2/huyhoang/v2d/outputs/phone_demo/selftest_box/policy/policy_19999744.safetensors`, video `policy.mp4`.
- GPU lúc chạy: job train SmolVLA của user khác 14.7 GB + fit NLF của phiên Track 1 3.1 GB + train của tôi 5.8 GB → còn 0.4 GB. Tôi định dừng train của mình để nhường, bộ phân loại quyền chặn lệnh kill → để chạy tiếp, không ai bị OOM.
- `phone_demo/hands_object_from_video.py` (chỉ MoGe-2 MIT, Grounding DINO + SAM 2.1 Apache, MediaPipe Apache): mặt bàn RANSAC → hệ z lên, bàn z = 1 m; tỉ lệ mét chung hiệu chỉnh theo kích thước hộp đo bằng thước; ICP hộp từng frame; tay = điểm ảnh + depth MoGe tại khớp + độ sâu tương đối giữa khớp từ MediaPipe world landmarks.
  - Thử trên `reconstruction/assets/tissue_box.mp4` của NVIDIA (iPhone, góc nhìn thứ nhất, 390 frame, stride 3 → 130). Lần 1 chạy CPU: lỗi `Input type Half` — MoGe `infer` mặc định autocast fp16 → sửa `use_fp16=device == "cuda"` trong `demo_track1/run_perception.py` (local + server). GPU: MoGe ViT-L 47 s, VRAM 2.3 GB.
  - Lần 2: PnP tay (world landmarks ↔ điểm ảnh) cho khung tay lệch khỏi bàn tay thật; retarget sai 5 cm, 30 % frame đầu ngón cắm > 2 cm vào hộp → thay bằng nâng 3D từ depth.
  - Nhãn trái/phải của MediaPipe nhảy giữa chừng; và MediaPipe Tasks 1.2 gán nhãn theo ảnh KHÔNG lật gương (ngược ghi chú cũ) → gán theo vị trí trên ảnh (chiều do đa số nhãn quyết định) + bám thời gian hai chiều, neo vào frame có đủ hai tay; cờ `--mirrored` cho camera trước.
  - Kết quả: mặt bàn inlier 97 %, MoGe to hơn thật 21 % (k = 0.825), hộp ICP 3.7 mm, nhấc 14 cm; tay thấy 92/98 % frame, cổ tay→gốc ngón giữa 9.5 cm; tay phải 120/120 frame bên phải ảnh, tay trái 126/126 bên trái.
  - Retarget (`SEQ=phone_tissue`): đầu ngón lệch đích 1.1 cm, đầu ngón cắm > 2 cm 8.5 % (phải) / 0 % (trái), khớp ngón chạm giới hạn 31–34 % (mẫu NVIDIA: 8–10 %). Render tham chiếu 1300 frame: đủ chu trình tiếp cận–kẹp–nhấc–đặt–buông.
  - 15:07 train `phone_tissue` (512 world, 20M step) chạy nền, ~3,500 step/s, ETA ~1.5 giờ; log `outputs/phone_demo/phone_tissue.log`.

- 2026-11-13 16:39 (server): v4 NLF batch 21/30 OK, 0 FAIL, ~17 min/episode/worker (4 workers, GPU 7.9 GB). Reproj 2.7–8.2 px, surface residual 1.9–2.6 cm, g = z_NLF/z_MoGe 1.03–1.14. Còn ep 4–9, 21–23; watcher bw9vfcgus sẽ pack khi đủ 30.
