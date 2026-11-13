#!/usr/bin/env bash
# M3 v2 — chạy TRÊN SERVER sau m3_perception_all.sh: với mỗi episode trong $EPISODES (chờ perception.done):
#   prep_clouds (hiệu chỉnh tỉ lệ depth theo nền) → fit MHR theo bề mặt + shape → vật (lọc frame hỏng) → pack kiểm → gom vào $ALL.
#   EPISODES="0 3 6" CHUNK=48 nohup setsid bash m3_v2_batch.sh > outputs/v2_a.log 2>&1 &
set -uo pipefail
M3="${SERVER_WORKDIR:-/data2/huyhoang/v2d}/m3"; cd "$M3"; . .venv/bin/activate
ALL="${ALL:-outputs/track1_episodes_v2}"; mkdir -p "$ALL"; CHUNK="${CHUNK:-48}"
for EP in ${EPISODES:-$(seq 0 29)}; do
  EP2="$(printf %02d "$EP")"; ST="episode_$(printf %06d "$EP")"; RUN="outputs/ep$EP2"
  if [ -f "$ALL/$ST.npz" ]; then echo "$(date +%T) ep $EP: đã có"; continue; fi
  until [ -f "$RUN/perception.done" ]; do sleep 20; done
  SECONDS=0
  if { python track1_submit/prep_clouds.py --run "$RUN" &&
       python track1_submit/fit_mhr_keypoints.py --run "$RUN" --model checkpoints/mhr/mhr_model.pt -o "$RUN/mhr_fit.npz" --chunk "$CHUNK" ${FIT_ARGS:-} &&
       python track1_submit/object_from_depth.py --run "$RUN" -o "$RUN/object" ${OBJ_ARGS:-} &&
       python track1_submit/pack_episode.py --run "$RUN" --episode "$EP" --kit kit/v2d_submission_kit --model checkpoints/mhr/mhr_model.pt; } > "outputs/v2_ep${EP2}.log" 2>&1; then
    cp "$RUN/track1_episodes/$ST.npz" "$RUN/track1_episodes/${ST}_object.glb" "$ALL/"
    echo "$(date +%T) ep $EP: OK ${SECONDS}s $(grep -o '"reproj_px_mean": [0-9.]*' "outputs/v2_ep${EP2}.log" | tail -1) $(grep -o '"surface_residual_cm_median": [0-9.]*' "outputs/v2_ep${EP2}.log" | tail -1) $(grep -o '"bad_depth_frames": [0-9]*' "outputs/v2_ep${EP2}.log" | tail -1) $(grep -o '"valid_frames": [0-9]*' "outputs/v2_ep${EP2}.log" | tail -1)"
  else
    echo "$(date +%T) ep $EP: FAIL ${SECONDS}s — $(grep -v '^W0000\|^I0000' "outputs/v2_ep${EP2}.log" | tail -1 | cut -c1-200)"
  fi
done
echo "$(date +%T) V2_BATCH_DONE"
