#!/usr/bin/env bash
# M3 v3 — tạo biến thể z-scale + đóng gói parquet:  bash m3_variant.sh <tên> <a>   (a: số, hoặc back=..,front=..,left=..,right=..)
set -euo pipefail
M3="${SERVER_WORKDIR:-/data2/huyhoang/v2d}/m3"; cd "$M3"; . .venv/bin/activate
NAME="$1"; A="$2"; SRC="${SRC:-outputs/track1_episodes_v2}"
CUDA_VISIBLE_DEVICES= python track1_submit/variant_zscale.py --src "$SRC" --runs outputs --model checkpoints/mhr/mhr_model.pt --a "$A" ${OBJ_B:+--obj-b "$OBJ_B"} ${OBJ_C:+--obj-clamp "$OBJ_C"} ${ROT_I:+--rot-identity} -o "outputs/var_$NAME" 2>/dev/null
(cd kit/v2d_submission_kit && python eval_reconstruction.py --episodes "$M3/outputs/var_$NAME" --sample "$M3/data/track_1_sample_submission.parquet" \
  --commit "${COMMIT_URL:-https://github.com/YkYoa/v2d/commit/c3a14dc60ea79d1eddfd82f0bbaa86f42cbe1004}" --out "$M3/outputs/var_$NAME.parquet" | tail -1)
