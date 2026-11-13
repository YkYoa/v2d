#!/usr/bin/env bash
# M3 — chạy TRÊN SERVER: chỉ bước perception (MoGe-2 + SAM2.1 + MediaPipe) cho các episode trong $EPISODES, GIỮ point map MoGe
# (~2.5 GB/episode) để lặp nhanh bước fit người/vật mà không phải chạy lại perception.
#   EPISODES="0 2 4" nohup setsid bash m3_perception_all.sh > outputs/perc_a.log 2>&1 &
set -uo pipefail
M3="${SERVER_WORKDIR:-/data2/huyhoang/v2d}/m3"; cd "$M3"; . .venv/bin/activate
export HF_HOME="${HF_HOME:-/data2/huyhoang/hf_cache}"
for EP in ${EPISODES:-$(seq 0 29)}; do
  EP2="$(printf %02d "$EP")"; RUN="outputs/ep$EP2"
  if [ -f "$RUN/perception.done" ]; then echo "$(date +%T) ep $EP: đã có"; continue; fi
  V="data/v2d_challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_$(printf %06d "$EP").mp4"
  N="$(python -c "import cv2,sys;print(int(cv2.VideoCapture(sys.argv[1]).get(7)))" "$V")"; SECONDS=0
  if python demo_track1/run_perception.py --episode "$EP" --stride 1 --max-frames "$N" --moge Ruicheng/moge-2-vitl-normal > "outputs/perc_ep${EP2}.log" 2>&1; then
    rm -rf "$RUN/points" "$RUN/overlay"; touch "$RUN/perception.done"; echo "$(date +%T) ep $EP: OK ${SECONDS}s"
  else echo "$(date +%T) ep $EP: FAIL ${SECONDS}s"; fi
done
echo "$(date +%T) PERCEPTION_DONE"
