#!/usr/bin/env bash
# M3 — chạy TRÊN SERVER 4090: Track 1 (1 episode) → parquet đúng format submission, đường không model gated.
#   bash m3_track1.sh [episode=9]     (thư mục làm việc: $SERVER_WORKDIR/m3, đã có .venv + data + checkpoints/mhr — xem docs/RUNLOG.md)
# Thêm SAMPLE=/path/track_1_sample_submission.parquet COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> để ra file nộp được.
set -euo pipefail
EP="${1:-9}"; EP2="$(printf %02d "$EP")"
M3="${SERVER_WORKDIR:-/data2/huyhoang/v2d}/m3"; cd "$M3"; . .venv/bin/activate
export HF_HOME="${HF_HOME:-/data2/huyhoang/hf_cache}"
RUN="outputs/ep$EP2"; N="$(python -c "import cv2,sys;print(int(cv2.VideoCapture(sys.argv[1]).get(7)))" "data/v2d_challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_$(printf %06d "$EP").mp4")"
python demo_track1/run_perception.py --episode "$EP" --stride 1 --max-frames "$N" --moge Ruicheng/moge-2-vitl-normal
python track1_submit/fit_mhr_keypoints.py --run "$RUN" --model checkpoints/mhr/mhr_model.pt -o "$RUN/mhr_fit.npz" --w-z 20 --w-smooth 20000 --w-prior 0.05
python track1_submit/object_from_depth.py --run "$RUN" -o "$RUN/object"
python track1_submit/pack_episode.py --run "$RUN" --episode "$EP" --kit kit/v2d_submission_kit --model checkpoints/mhr/mhr_model.pt \
  ${SAMPLE:+--sample "$SAMPLE"} ${COMMIT_URL:+--commit "$COMMIT_URL"}
