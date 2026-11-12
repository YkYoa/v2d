#!/usr/bin/env bash
# Chờ GPU rảnh (job người khác xong) rồi mới train mini + eval. Chạy TRÊN SERVER:
#   nohup setsid bash m2_wait_gpu.sh > outputs/m2_wait.log 2>&1 &
# Biến: FREE_BELOW (MiB đang dùng tối đa để coi là rảnh), DEADLINE_H (số giờ chờ tối đa), + biến của m2_flashchord.sh
set -uo pipefail
cd "$(dirname "$0")"
FREE_BELOW="${FREE_BELOW:-1500}"; END=$(( $(date +%s) + ${DEADLINE_H:-14} * 3600 )); ok=0
while [ "$(date +%s)" -lt "$END" ]; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
  if [ "$used" -lt "$FREE_BELOW" ]; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && break            # rảnh liên tục 3 phút
  sleep 60
done
[ "$ok" -ge 3 ] || { echo "$(date) hết hạn chờ, GPU vẫn bận (${used} MiB) → không train"; exit 3; }
echo "$(date) GPU rảnh (${used} MiB) → bắt đầu"
bash m2_flashchord.sh train && bash m2_flashchord.sh eval
echo "$(date) M2_DONE rc=$?"
