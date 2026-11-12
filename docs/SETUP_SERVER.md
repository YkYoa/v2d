# Checklist server lab RTX 4090 (M0.4)

Điền `.env` (SSH_HOST, SSH_USER, SSH_KEY) rồi chạy `./server/ssh.sh 'bash -s' < server/audit.sh`.

## Yêu cầu
| Mục | Cần | Kiểm tra |
|---|---|---|
| GPU | RTX 4090 24 GB (SM 8.9, không phải Blackwell) | `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` |
| Driver | ≥ 580 (CUDA 13) cho FlashCHORD/Isaac Lab 2.3 | `nvidia-smi` |
| Docker + NVIDIA Container Toolkit | cho `reconstruction/` modules, Isaac Lab, CARI4D | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` |
| NGC login | chỉ cho Isaac Lab image | `docker login nvcr.io` (user: `$oauthtoken`, pass: NGC key) — **user tự làm** |
| git-lfs | assets robot | `git lfs version` |
| uv + Python 3.11 | FlashCHORD không Docker | `uv --version` |
| Disk | ≥ 150 GB trống (container ~60 GB, weights ~25 GB, dataset 4 GB, outputs) | `df -h ~` |
| RAM | ≥ 32 GB khuyến nghị | `free -h` |
| Port | 8080 (viser) forward: `ssh -L 8080:localhost:8080 ...` | |

## Thứ tự làm trên server
1. `git clone https://github.com/nvidia-isaac/video_to_data && cd video_to_data && git lfs install && git lfs pull`
2. FlashCHORD: `cd robotic_grounding/flash_chord && uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[train,dev]"`
3. Chạy `server/m2_flashchord.sh` (replay → train mini → evaluate).
4. (M3) Docker modules reconstruction hoặc Swithinraj baseline.
