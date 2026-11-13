#!/usr/bin/env bash
# Demo A — chạy TRÊN SERVER: keypoint tay + pose vật → tham chiếu Sharpa (không MANO) → render → train FlashCHORD mini → eval → render policy.
#   bash phone_demo.sh [retarget|replay|train|eval|render|all]
# Biến môi trường:
#   SEQ=my_clip                 tên chuỗi (KHÔNG trùng tên vật có sẵn như tissue_box_simple)
#   RECON=/path/recon.npz       đầu ra của phone_demo/hands_object_from_video.py   (hoặc SELFTEST=1: dùng parquet mẫu của NVIDIA)
#   WORLDS=512 STEPS=20000000   quy mô train (như M2: 40 phút, ~5.8 GB VRAM)
# Chạy nền:  SEQ=my_clip RECON=... nohup setsid bash phone_demo.sh all > outputs/phone_demo/my_clip.log 2>&1 &
set -euo pipefail
STEP="${1:-all}"; WORK="${SERVER_WORKDIR:-/data2/huyhoang/v2d}"; FC="$WORK/video_to_data/robotic_grounding/flash_chord"
SEQ="${SEQ:?đặt SEQ=<tên chuỗi>}"; OUT="$WORK/outputs/phone_demo/$SEQ"; TASK="$OUT/task"
REF="$TASK/processed/sequence_id=$SEQ/robot_name=sharpa_wave"
WORLDS="${WORLDS:-512}"; STEPS="${STEPS:-20000000}"
CAM=("viewer.camera_position=${CAM_POS:-[0.55,-0.65,1.45]}" "viewer.camera_target=${CAM_TGT:-[0.0,0.0,1.05]}" viewer.width=1280 viewer.height=720 viewer.headless=true)
mkdir -p "$OUT"; cd "$FC"; . .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYGLET_HEADLESS=1 PYOPENGL_PLATFORM=egl

if [[ "$STEP" == retarget || "$STEP" == all ]]; then
  if [ -n "${SELFTEST:-}" ]; then
    SRC="$(find "$WORK/video_to_data/robotic_grounding" -type d -path '*ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave' | head -1)"
    python "$WORK/phone_demo/retarget_sharpa.py" --selftest "$SRC" --seq "$SEQ" --out "$TASK" ${RETARGET_ARGS:-}
  else
    python "$WORK/phone_demo/retarget_sharpa.py" --recon "${RECON:?đặt RECON=/path/recon.npz}" --seq "$SEQ" --out "$TASK" ${RETARGET_ARGS:-}
  fi
fi
if [[ "$STEP" == replay || "$STEP" == all ]]; then
  N="$(python -c "import pyarrow.parquet as pq,glob;r=pq.read_table(glob.glob('$REF/*.parquet')[0],columns=['fps','object_body_position']).to_pylist()[0];print(int(len(r['object_body_position'])/r['fps']/0.5*50))")"   # task.motion_speed mặc định 0.5 → phát chậm 2 lần
  echo ">>> render tham chiếu động học ($N frame @ 50 Hz) → $OUT/reference.mp4"
  python scripts/debug/replay.py "task.parquet='$REF'" replay=kinematic action=residual_hand_pose_stiff collision=kinematic embodiment=sharpa_hands_stiff \
    markers.enabled=false viewer.backend=mp4 viewer.num_frames="$N" viewer.video_fps=50 "viewer.output_path=$OUT/reference.mp4" "${CAM[@]}"
fi
if [[ "$STEP" == train || "$STEP" == all ]]; then
  STEPS=$(( STEPS / WORLDS * WORLDS ))
  THR="$(python -c "
T=[19660800,34406400,49152000,63897600,78643200,93388800,108134400,122880000,137625600,200007680,250003456]
S,W=$STEPS,$WORLDS; print('['+','.join(str(max(W,round(t/T[-1]*S/W)*W)) if t!=T[-1] else str(S) for t in T)+']')")"
  echo ">>> train (world_count=$WORLDS, steps=$STEPS)"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  ( while sleep 30; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; done > "$OUT/vram.log" 2>/dev/null ) & MON=$!
  SECONDS=0
  python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$REF'" reset.seed=42 logging.mode=disabled \
    output_dir="$OUT/policy" scene.world_count="$WORLDS" "curriculum.thresholds=$THR" training.total_environment_steps="$STEPS" \
    training.replay.capacity="${REPLAY:-524288}" training.replay.minimum_size="${REPLAY_MIN:-51200}" \
    training.checkpoint.save_interval_environment_steps="$(( STEPS / 4 / WORLDS * WORLDS ))" ${TRAIN_ARGS:-} || { kill $MON 2>/dev/null; echo TRAIN_FAIL; exit 1; }
  kill $MON 2>/dev/null || true; echo "train wall time: ${SECONDS}s"; rm -f "$OUT"/policy/state_*   # state trung gian rất nặng
fi
CK="$(ls "$OUT"/policy/policy_*.safetensors 2>/dev/null | sort -t_ -k2 -V | tail -1 || true)"
if [[ "$STEP" == eval || "$STEP" == all ]]; then
  [ -n "$CK" ] || { echo "chưa có checkpoint"; exit 1; }
  python scripts/evaluate_policy.py "evaluation.checkpoint=$CK" evaluation.world_count="${EVAL_WORLDS:-512}" "evaluation.metrics_output=$CK.metrics.json" logging.mode=disabled
  cat "$CK.metrics.json"; echo
fi
if [[ "$STEP" == render || "$STEP" == all ]]; then
  [ -n "$CK" ] || { echo "chưa có checkpoint"; exit 1; }
  python scripts/view_policy.py "evaluation.checkpoint=$CK" markers.enabled=false viewer.backend=mp4 viewer.num_frames="${RENDER_FRAMES:-520}" "viewer.output_path=$OUT/policy.mp4" "${CAM[@]}"
fi
echo "PHONE_DEMO_${STEP}_DONE"
