#!/usr/bin/env bash
# M2 — chạy TRÊN SERVER 4090: FlashCHORD replay → train mini → evaluate. Dùng: bash server/m2_flashchord.sh [replay|train|eval|all]
set -euo pipefail
STEP="${1:-all}"
WORK="${SERVER_WORKDIR:-$HOME/v2d}"; REPO="$WORK/video_to_data"; FC="$REPO/robotic_grounding/flash_chord"
WORLDS="${WORLDS:-1024}"           # 24 GB: giảm từ 4096 mặc định; tăng dần nếu không OOM
mkdir -p "$WORK"
if [ ! -d "$REPO" ]; then git clone https://github.com/nvidia-isaac/video_to_data "$REPO"; fi
cd "$REPO"; git lfs install >/dev/null; git lfs pull
cd "$FC"
if [ ! -d .venv ]; then uv venv --python 3.11 && . .venv/bin/activate && uv pip install -e ".[train,dev]"; else . .venv/bin/activate; fi
# tìm parquet tissue-box Sharpa (bundled, LFS)
SHARPA_PARQUET="$(find "$REPO/robotic_grounding" -type d -path '*ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave' | head -1)"
[ -n "$SHARPA_PARQUET" ] || { echo "không tìm thấy tissue_box_simple parquet — kiểm tra git lfs pull"; exit 1; }
echo "SHARPA_PARQUET=$SHARPA_PARQUET"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
case "$STEP" in
  replay|all)
    echo ">>> replay (viser http://localhost:8080 — ssh -L 8080:localhost:8080)"
    [ "$STEP" = all ] && TMO="timeout 120" || TMO=""
    $TMO python scripts/debug/replay.py "task.parquet='${SHARPA_PARQUET}'" viewer.backend=viser +start_paused=true markers.enabled=true || true ;;&
  train|all)
    echo ">>> train mini (world_count=$WORLDS)"
    python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${SHARPA_PARQUET}'" reset.seed=42 \
      logging.mode=disabled output_dir="$WORK/outputs/sharpa_mini" scene.world_count="$WORLDS" 2>&1 | tee "$WORK/outputs/train_sharpa_mini.log" ;;&
  eval|all)
    CK="$(ls -t "$WORK"/outputs/sharpa_mini/policy_*.safetensors 2>/dev/null | head -1)"
    [ -n "$CK" ] || { echo "chưa có checkpoint"; exit 1; }
    echo ">>> evaluate $CK"
    python scripts/evaluate_policy.py "evaluation.checkpoint=${CK}" evaluation.world_count=512 \
      "evaluation.metrics_output=${CK}.metrics.json" logging.mode=disabled
    cat "${CK}.metrics.json" ;;
esac
