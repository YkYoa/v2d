#!/usr/bin/env bash
# Chạy lệnh trên server 4090: ./server/ssh.sh "nvidia-smi"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] || { echo "Chưa có .env (copy từ .env.example)"; exit 2; }
set -a; source "$ROOT/.env"; set +a
[ -n "${SSH_HOST:-}" ] || { echo "SSH_HOST trống trong .env"; exit 2; }
exec ssh -p "${SSH_PORT:-22}" -i "${SSH_KEY/#\~/$HOME}" "${SSH_USER}@${SSH_HOST}" "$@"
