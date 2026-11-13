# Demo A — tự quay bằng điện thoại → dữ liệu cho robot

Video điện thoại (hai tay nhấc một cái hộp) → quỹ đạo tay + hộp 3D → tham chiếu cho hai tay robot Sharpa Wave → train policy FlashCHORD trong mô phỏng.

**Không dùng MANO, SAM 3 / SAM 3D, WiLoR, HaMeR.** Chỉ dùng MoGe-2, Grounding DINO, SAM 2.1, MediaPipe Hands, MuJoCo, FlashCHORD (giấy phép: `docs/LICENSES.md`).

## Cách quay (quan trọng hơn mọi bước xử lý)

| Yêu cầu | Lý do |
|---|---|
| Điện thoại **đứng yên** (kẹp giá, tựa vào vật), không cầm tay | depth được hiệu chỉnh theo nền tĩnh; camera chuyển động thì cần SLAM (chưa làm) |
| Quay chéo từ trên xuống ~45°, cách hộp 40–70 cm, thấy rõ mặt bàn quanh hộp | mặt bàn dùng để dựng hệ toạ độ (z hướng lên) |
| Vật là **hộp chữ nhật cứng**, có hoạ tiết, **đo dài × rộng × cao bằng thước** | kích thước thật vừa cho mesh, vừa hiệu chỉnh tỉ lệ mét của MoGe (tissue box mẫu: MoGe đo to 21 %) |
| Hộp nằm yên trên bàn ở đầu clip, không có tay che | frame đầu dùng để khởi tạo pose hộp |
| Hai tay kẹp hai đầu hộp bằng cả bàn tay, nhấc 10–20 cm, đặt lại, rút tay ra | thao tác kiểu này retarget tốt nhất; ngón khéo (xoay, vặn) chưa làm được |
| 5–15 s, ánh sáng đều, không đeo găng, nền không có tay người khác | MediaPipe cần thấy rõ hai bàn tay |
| Camera sau, quay ngang 1080p 30 fps; nếu dùng camera trước (ảnh lật gương) thì thêm `--mirrored` | nhãn trái/phải |

## Chạy (trên server 4090)

```bash
# 0) copy video lên server
scp my_clip.mp4 $SSH_USER@$SSH_HOST:/data2/huyhoang/v2d/data/phone/

# 1) perception → recon.npz + overlay.mp4 (venv m3: torch, MoGe, SAM2, MediaPipe), ~2 phút cho 130 frame
cd /data2/huyhoang/v2d
HF_HOME=/data2/huyhoang/hf_cache m3/.venv/bin/python phone_demo/hands_object_from_video.py \
  --video data/phone/my_clip.mp4 --object "a cardboard box" --box-size 0.22 0.13 0.11 --stride 3 \
  --out outputs/phone_demo/my_clip_recon
#    → XEM outputs/phone_demo/my_clip_recon/overlay.mp4: khung vàng phải bám hộp, khung đỏ = tay phải, xanh = tay trái

# 2) retarget → render tham chiếu → train (~45–90 phút) → đánh giá → render policy (venv FlashCHORD)
SEQ=my_clip RECON=/data2/huyhoang/v2d/outputs/phone_demo/my_clip_recon/recon.npz bash phone_demo.sh retarget
SEQ=my_clip bash phone_demo.sh replay        # → outputs/phone_demo/my_clip/reference.mp4: XEM trước khi train
SEQ=my_clip nohup setsid bash -c "bash phone_demo.sh train && bash phone_demo.sh eval && bash phone_demo.sh render" \
  > outputs/phone_demo/my_clip.log 2>&1 &
```

`SEQ` phải là tên mới (không trùng `tissue_box_simple` hay vật có sẵn).

## Đọc báo cáo

`report.json` (perception):
- `metric_scale_from_box` — MoGe sai tỉ lệ bao nhiêu (1.0 = đúng). Ngoài 0.75–1.35 là đang chạm biên tìm kiếm → kiểm lại kích thước hộp.
- `icp_rms_mm_median` < 6 mm, `object_frames_valid_frac` > 0.9, `box_tilt_deg_frame0` < 5°.
- `hands.*.wrist_to_middle_mcp_cm_median` ≈ 8–10 cm (người lớn). Lệch xa = depth tay sai.
- `relabelled_detections` — số lần nhãn trái/phải của MediaPipe bị sửa theo vị trí; nhiều là bình thường ở góc nhìn mu bàn tay.

`task/retarget_report.json` (retarget):
- `tip_err_cm_mean` — đầu ngón robot cách đích bao xa (< 1.5 cm là tốt).
- `tip_inside_gt2cm_frac` — tỉ lệ frame có đầu ngón cắm sâu > 2 cm vào hộp. Mốc của NVIDIA: 0. Cao → policy học hất hộp.
- `joint_at_limit_frac` — khớp ngón chạm giới hạn; > 0.4 nghĩa là tư thế tay người khác xa tay robot.

## Đã kiểm chứng (13/11/2026)

| Bước | Dữ liệu | Kết quả |
|---|---|---|
| Retarget không MANO (`retarget_sharpa.py --selftest`) | 21 khớp tay + pose hộp trong mẫu tissue-box của NVIDIA | đầu ngón robot lệch bản retarget chính thức 1.0 cm; tên/số frame robot khớp đúng 67/67; tiếp xúc trùng 93–96 %; FlashCHORD đọc được |
| Train trên tham chiếu đó | 512 world, 20M step, 43 phút | **thành công 99.8 %, MPPE 0.50 cm** (cùng công thức, tham chiếu MANO chính thức: 93.4 %, 0.64 cm) |
| Perception trên video điện thoại | `reconstruction/assets/tissue_box.mp4` của NVIDIA (góc nhìn thứ nhất, 13 s) | mặt bàn khớp 97 %, hộp ICP 3.7 mm, nhấc 14 cm; tay thấy 92–98 % frame, cổ tay→gốc ngón giữa 9.5 cm |
| Retarget từ video | như trên | đầu ngón lệch đích 1.1 cm; đầu ngón cắm > 2 cm: 8.5 % frame (phải), 0 % (trái) |

Lưu ý: dòng 1–2 chỉ kiểm bước retarget + train (đầu vào là khớp tay do NVIDIA tái dựng); dòng 3–4 là đường video đầy đủ.

## Giới hạn hiện tại

- Chỉ vật hình hộp (kích thước nhập tay). Vật khác cần mesh từ nguồn cho phép thương mại (quét 3D, CAD).
- Camera phải đứng yên.
- Depth tay lấy từ MoGe tại vị trí khớp: khi ngón khuất sau hộp, depth đọc trúng mặt hộp → ngón bị kéo ra trước. Bộ lọc 6 cm xử lý phần lớn, chưa hết.
- Ngón tay robot bám đầu ngón người; tư thế ngón khác xa tay robot (khớp chạm giới hạn ~30 % frame).
- Policy dùng trạng thái mô phỏng (pose vật), chưa dùng camera. Bước tiếp theo là sinh demo trong sim + fine-tune GR00T N1.7 (workflow chính thức `collect` → `finetune`, cần Isaac Lab + 48 GB GPU theo tài liệu NVIDIA).
