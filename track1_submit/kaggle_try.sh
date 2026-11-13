#!/usr/bin/env bash
# Chạy TRÊN LAPTOP: tạo biến thể trên server, tải về, nộp CD-H + CD-O (hoặc $METRICS) bằng Kaggle CLI, in điểm.
#   bash track1_submit/kaggle_try.sh z115 1.15
set -euo pipefail
cd "$(dirname "$0")/.."; NAME="$1"; A="$2"; K="${KAGGLE_BIN:-$TEMP/claude/kagglecli/Scripts/kaggle}"
set -a; . ./.env; set +a
bash server/ssh.sh "OBJ_B=${OBJ_B:-} OBJ_C=${OBJ_C:-} ROT_I=${ROT_I:-} SRC=${SRC:-outputs/track1_episodes_v2} bash /data2/huyhoang/v2d/m3/m3_variant.sh $NAME $A" >/dev/null
bash server/ssh.sh "cat /data2/huyhoang/v2d/m3/outputs/var_$NAME.parquet" > "outputs/m3/var_$NAME.parquet"
for m in ${METRICS:-cd-h cd-o}; do "$K" competitions submit -c v2d-challenge-track1-$m -f "outputs/m3/var_$NAME.parquet" -m "probe $NAME a=$A" >/dev/null 2>&1; done
for i in $(seq 1 12); do ping -n 20 127.0.0.1 >/dev/null
  o="$(for m in ${METRICS:-cd-h cd-o}; do "$K" competitions submissions -c v2d-challenge-track1-$m 2>/dev/null | grep "probe $NAME " | head -1 | awk -v m=$m '{print m, $(NF-2), $(NF-1), $NF}'; done)"
  echo "$o" | grep -q PENDING || break; done
o="$(echo "$o" | sed "s/SubmissionStatus.COMPLETE//;s/a=[^ ]* //")"
echo "$NAME a=$A: $(echo "$o" | tr '\n' ' ')" | tee -a outputs/m3/kaggle_probes.log
