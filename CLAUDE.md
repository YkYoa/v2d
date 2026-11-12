# V2D — NVIDIA Video-to-Data Challenge demo (laptop RTX 4050 + server RTX 4090)

Dự án học/làm demo cho NVIDIA Video to Data Challenge (CoRL 2026): video người thao tác → dữ liệu/kỹ năng robot trong mô phỏng, rồi tối ưu/quantize để deploy lên board không-NVIDIA.

## Quy ước bắt buộc
- **Đọc `PLAN.md` trước khi làm bất cứ gì.** Milestone M0–M4 có checkbox trạng thái ở đầu `PLAN.md`; cập nhật `[x]` ngay khi xong một mục.
- Tiến độ, lệnh đã chạy, thời gian, VRAM đỉnh, lỗi gặp → ghi vào `docs/RUNLOG.md` (append, có ngày giờ).
- Key/secret (SSH host/user, HF token, Kaggle, NGC) **chỉ** đọc từ `.env` (copy từ `.env.example`). Không bao giờ commit `.env`, không in giá trị key ra chat/log.
- Lệnh chạy trên server 4090 đi qua `server/ssh.sh` (đọc `.env`). Nếu `.env` chưa có `SSH_HOST` thì bỏ qua các bước server và báo lại.
- Dữ liệu lớn để trong `data/` (gitignore). Output demo trong `outputs/` (gitignore).
- Python env: `uv` tại `.venv` (Python 3.11). Kích hoạt: `source .venv/bin/activate`.
- Khi mở phiên mới: "đọc PLAN.md và tiếp tục milestone kế tiếp".

## Tài liệu
- `docs/RESEARCH.md` — toàn bộ link/paper/repo cộng đồng đã thu thập, có chú thích.
- `docs/SETUP_SERVER.md` — checklist chuẩn bị server 4090.
- `docs/kit/` — submission kit + FAQ chính thức.
