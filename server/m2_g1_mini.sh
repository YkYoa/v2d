#!/usr/bin/env bash
# M2b — chạy TRÊN SERVER: train FlashSAC bản MINI cho Unitree G1 + Dex3 (SONIC) trên motion snack-box pick-and-place kèm repo.
#   WORLDS=256 STEPS=20000000 OUT=g1_mini nohup setsid bash m2_g1_mini.sh > outputs/train_g1_mini.log 2>&1 &
# Công thức đầy đủ của NVIDIA: 2048 world, 350M step, replay 10M. Ở đây co curriculum về đúng STEPS và thu nhỏ replay cho vừa GPU dùng chung.
# Cần nvidia-curand trong venv (ONNX Runtime CUDA cho SONIC) và LD_LIBRARY_PATH trỏ vào nvidia/*/lib.
set -euo pipefail
WORK="${SERVER_WORKDIR:-/data2/huyhoang/v2d}"; REPO="$WORK/video_to_data"
cd "$REPO/robotic_grounding/flash_chord"; . .venv/bin/activate
export LD_LIBRARY_PATH="$(ls -d "$PWD"/.venv/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
WORLDS="${WORLDS:-256}"; STEPS="${STEPS:-20000000}"; OUT="${OUT:-g1_mini}"
G1="$(find "$REPO/robotic_grounding/source" -path '*snack_box_pick_and_place_01/robot_name=g1/data.parquet' | head -1)"
[ -n "$G1" ] || { echo "không tìm thấy parquet G1 snack-box"; exit 1; }
STEPS=$(( STEPS / WORLDS * WORLDS ))
THR="$(python -c "
T=[35000320,70000640,104998912,139999232,174999552,209999872,288749568,306249728,349999104]
S,W=$STEPS,$WORLDS; print('['+','.join(str(max(W,round(t/T[-1]*S/W)*W)) if t!=T[-1] else str(S) for t in T)+']')")"
echo ">>> G1 mini: world_count=$WORLDS steps=$STEPS out=$OUT"; echo "curriculum.thresholds=$THR"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
( while sleep 30; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; done > "$WORK/outputs/$OUT.vram.log" 2>/dev/null ) & MON=$!
SECONDS=0
python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac "task.parquet='${G1}'" reset.seed=42 logging.mode=disabled \
  output_dir="$WORK/outputs/$OUT" scene.world_count="$WORLDS" "curriculum.thresholds=$THR" training.total_environment_steps="$STEPS" \
  training.replay.capacity="${REPLAY:-524288}" training.replay.minimum_size="${REPLAY_MIN:-51200}" \
  training.checkpoint.save_interval_environment_steps="$(( STEPS / 4 / WORLDS * WORLDS ))" || { kill $MON 2>/dev/null; echo "TRAIN_FAIL"; exit 1; }
kill $MON 2>/dev/null || true; echo "train wall time: ${SECONDS}s"; echo "G1_TRAIN_DONE"
