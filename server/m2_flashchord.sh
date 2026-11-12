#!/usr/bin/env bash
# M2 — chạy TRÊN SERVER 4090: FlashCHORD replay → train mini → evaluate. Dùng: bash server/m2_flashchord.sh [replay|train|eval|all]
set -euo pipefail
STEP="${1:-all}"
WORK="${SERVER_WORKDIR:-/data2/huyhoang/v2d}"
# server lab: không có sudo → uv/git-lfs nằm trong tools/, cache uv để ở /data2 (ổ / gần đầy)
export PATH="/data2/huyhoang/tools:$PATH" UV_CACHE_DIR="${UV_CACHE_DIR:-/data2/huyhoang/uv_cache}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/data2/huyhoang/uv_python}"
REPO="$WORK/video_to_data"; FC="$REPO/robotic_grounding/flash_chord"
STEPS="${STEPS:-5000000}"         # mini: 5M env step (full = 250M, cần 46 GB trống)
OUT="${OUT:-sharpa_mini}"
WORLDS="${WORLDS:-1024}"           # 24 GB: giảm từ 4096 mặc định; GPU dùng chung → kiểm tra VRAM trống, giảm tiếp nếu cần
mkdir -p "$WORK"
if [ ! -d "$REPO" ]; then GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/nvidia-isaac/video_to_data "$REPO"; fi
cd "$REPO"; git lfs install --local >/dev/null; git lfs pull --include="robotic_grounding/**"   # ~1040 file LFS, bỏ reconstruction/
cd "$FC"
[ -d .venv ] || uv sync --python 3.11 --extra train --extra dev   # theo uv.lock (đúng bản pin của container)
. .venv/bin/activate
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
    STEPS=$(( STEPS / WORLDS * WORLDS ))
    # co curriculum 250M (voc_scale 1.0 → 0: giảm dần trợ lực ảo lên vật) về đúng STEPS, nếu không policy mini chỉ học ở stage có trợ lực
    THR="$(python -c "
T=[19660800,34406400,49152000,63897600,78643200,93388800,108134400,122880000,137625600,200007680,250003456]
S,W=$STEPS,$WORLDS; print('['+','.join(str(max(W,round(t/T[-1]*S/W)*W)) if t!=T[-1] else str(S) for t in T)+']')")"
    echo ">>> train mini (world_count=$WORLDS, steps=$STEPS, out=$OUT)"; echo "curriculum.thresholds=$THR"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    ( while sleep 20; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '
' ';')"; done > "$WORK/outputs/$OUT.vram.log" 2>/dev/null ) & MON=$!
    SECONDS=0
    python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${SHARPA_PARQUET}'" reset.seed=42       logging.mode=disabled output_dir="$WORK/outputs/$OUT" scene.world_count="$WORLDS" "curriculum.thresholds=$THR"       training.total_environment_steps="$STEPS" training.replay.capacity="${REPLAY:-524288}" training.replay.minimum_size="${REPLAY_MIN:-51200}"       training.checkpoint.save_interval_environment_steps="$(( STEPS / 4 / WORLDS * WORLDS ))" 2>&1 | tee "$WORK/outputs/train_$OUT.log"
    kill $MON 2>/dev/null || true; echo "train wall time: ${SECONDS}s" | tee -a "$WORK/outputs/train_$OUT.log" ;;&
  eval|all)
    CK="$(ls "$WORK"/outputs/"$OUT"/policy_*.safetensors 2>/dev/null | sort -t_ -k2 -V | tail -1)"
    [ -n "$CK" ] || { echo "chưa có checkpoint"; exit 1; }
    echo ">>> evaluate $CK"
    python scripts/evaluate_policy.py "evaluation.checkpoint=${CK}" evaluation.world_count="${EVAL_WORLDS:-512}"       "evaluation.metrics_output=${CK}.metrics.json" logging.mode=disabled
    cat "${CK}.metrics.json" ;;
esac
