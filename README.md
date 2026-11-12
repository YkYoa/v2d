# v2d — NVIDIA Video-to-Data Challenge: tài liệu + demo

- `PLAN.md` — kế hoạch M0–M4 và trạng thái (Claude đọc file này trước).
- `docs/RESEARCH.md` — toàn bộ tài liệu/paper/repo đã thu thập.
- `docs/kit/` — submission kit + FAQ chính thức.
- `demo_track1/` — M1: video → depth + mask + keypoint 3D (laptop).
- `server/` — M2/M3: script chạy trên server 4090 qua `server/ssh.sh`.
- `track1_submit/` — M3: fit MHR từ keypoint, mesh + quỹ đạo vật từ depth, đóng gói bằng kit chính thức.
- `deploy/` — M4: export ONNX, quantize, runtime cho board không-NVIDIA.

```bash
cp .env.example .env   # điền SSH/HF/Kaggle
source .venv/bin/activate
python demo_track1/run_perception.py --episode 9
```
