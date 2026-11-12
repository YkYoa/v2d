#!/usr/bin/env bash
# M3 — chạy TRÊN SERVER: cả 30 episode Track 1 → 1 parquet theo sample THẬT của Kaggle.
#   nohup setsid bash m3_batch.sh > outputs/batch.log 2>&1 &
# Mỗi episode lỗi thì ghi FAIL và đi tiếp; xong mỗi episode xoá point map MoGe (~2.5 GB/episode) để đỡ đĩa.
set -uo pipefail
M3="${SERVER_WORKDIR:-/data2/huyhoang/v2d}/m3"; cd "$M3"; . .venv/bin/activate
export HF_HOME="${HF_HOME:-/data2/huyhoang/hf_cache}"
SAMPLE="${SAMPLE:-$M3/data/track_1_sample_submission.parquet}"
COMMIT_URL="${COMMIT_URL:-https://github.com/local/check/commit/0000000000000000000000000000000000000000}"
hf download nvidia/video_to_data_challenge --repo-type dataset --include "track_1/videos/*" --local-dir data/v2d_challenge >/dev/null
ALL=outputs/track1_episodes_all; mkdir -p "$ALL"
for EP in ${EPISODES:-$(seq 0 29)}; do
  EP2="$(printf %02d "$EP")"; ST="episode_$(printf %06d "$EP")"
  if [ -f "$ALL/$ST.npz" ] && [ -f "$ALL/${ST}_object.glb" ]; then echo "$(date +%T) ep $EP: đã có, bỏ qua"; continue; fi
  SECONDS=0
  if bash m3_track1.sh "$EP" > "outputs/ep${EP2}.log" 2>&1; then
    cp "outputs/ep$EP2/track1_episodes/$ST.npz" "outputs/ep$EP2/track1_episodes/${ST}_object.glb" "$ALL/"
    echo "$(date +%T) ep $EP: OK ${SECONDS}s $(grep -o '"reproj_px_mean": [0-9.]*' "outputs/ep${EP2}.log" | tail -1) $(grep -o '"icp_err_cm_median": [0-9.]*' "outputs/ep${EP2}.log" | tail -1)"
  else
    echo "$(date +%T) ep $EP: FAIL ${SECONDS}s — $(grep -v '^W0000\|^I0000' "outputs/ep${EP2}.log" | tail -1 | cut -c1-200)"
  fi
  rm -rf "outputs/ep$EP2/moge" "outputs/ep$EP2/points" "outputs/ep$EP2/overlay"
done
N="$(ls "$ALL"/episode_*.npz 2>/dev/null | wc -l)"; echo "$(date +%T) xong vòng lặp: $N/30 episode có kết quả"
if [ "$N" -eq 30 ]; then
  OUT="$M3/outputs/track1_all_30ep.parquet"
  (cd kit/v2d_submission_kit && python eval_reconstruction.py --episodes "$M3/$ALL" --sample "$SAMPLE" --commit "$COMMIT_URL" --out "$OUT") && echo "PACK_ALL_OK $OUT"
fi
echo "$(date +%T) M3_BATCH_DONE"
