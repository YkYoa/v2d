#!/usr/bin/env bash
# M3 v4 — chạy TRÊN SERVER: NLF (cơ thể 3D, trọng số phi thương mại) → fit MHR theo khớp + bề mặt NLF → vật (thang NLF, |z_vật − z_người| ≤ 0.3 m) → pack kiểm.
# Nhiều worker chạy song song an toàn (khoá thư mục outputs/epNN/.v4lock; xoá khoá nếu muốn chạy lại episode lỗi).
# Cần outputs/epNN/clouds.npz (từ m3_v2_batch.sh) và checkpoints/nlf/nlf_l_multi_0.3.2.patch6.torchscript.
#   EPISODES="0 2 4" nohup setsid bash m3_v4_batch.sh > outputs/v4_a.log 2>&1 &
set -uo pipefail
M3="${SERVER_WORKDIR:-/data2/huyhoang/v2d}/m3"; cd "$M3"; . .venv/bin/activate
ALL="${ALL:-outputs/track1_episodes_v4}"; mkdir -p "$ALL"; CHUNK="${CHUNK:-48}"
NLF="${NLF:-checkpoints/nlf/nlf_l_multi_0.3.2.patch6.torchscript}"
for EP in ${EPISODES:-$(seq 0 29)}; do
  EP2="$(printf %02d "$EP")"; ST="episode_$(printf %06d "$EP")"; RUN="outputs/ep$EP2"
  if [ -f "$ALL/$ST.npz" ]; then echo "$(date +%T) ep $EP: đã có"; continue; fi
  mkdir "$RUN/.v4lock" 2>/dev/null || { echo "$(date +%T) ep $EP: worker khác đang làm"; continue; }   # khoá: nhiều worker không đụng nhau
  SECONDS=0
  if { python track1_submit/nlf_body.py --run "$RUN" --model "$NLF" &&
       python track1_submit/fit_mhr_keypoints.py --run "$RUN" --model checkpoints/mhr/mhr_model.pt -o "$RUN/mhr_fit_v4.npz" \
         --clouds clouds_nlf.npz --chunk "$CHUNK" ${FIT_ARGS:-} &&
       python track1_submit/object_from_depth.py --run "$RUN" -o "$RUN/object_v4" --nlf clouds_nlf.npz --z-clamp 0.3 &&
       cp "$RUN/object_v4.glb" "$RUN/object.glb" && cp "$RUN/object_v4_track.npz" "$RUN/object_track.npz" &&
       python track1_submit/pack_episode.py --run "$RUN" --episode "$EP" --kit kit/v2d_submission_kit --model checkpoints/mhr/mhr_model.pt \
         --fit "$RUN/mhr_fit_v4.npz"; } > "outputs/v4_ep${EP2}.log" 2>&1; then
    cp "$RUN/track1_episodes/$ST.npz" "$RUN/track1_episodes/${ST}_object.glb" "$ALL/"
    echo "$(date +%T) ep $EP: OK ${SECONDS}s $(grep -o 'g = z_NLF/z_MoGe = [0-9.]*' "outputs/v4_ep${EP2}.log" | tail -1) $(grep -o '"reproj_px_mean": [0-9.]*' "outputs/v4_ep${EP2}.log" | tail -1) $(grep -o '"surface_residual_cm_median": [0-9.]*' "outputs/v4_ep${EP2}.log" | tail -1)"
  else
    echo "$(date +%T) ep $EP: FAIL ${SECONDS}s — $(grep -v '^W0000\|^I0000' "outputs/v4_ep${EP2}.log" | tail -1 | cut -c1-200)"
  fi
done
echo "$(date +%T) V4_BATCH_DONE"
