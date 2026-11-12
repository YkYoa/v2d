#!/usr/bin/env bash
# Chạy trên server: kiểm tra nhanh điều kiện M0.4
echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>&1
echo "== docker =="; docker --version 2>&1; docker info 2>/dev/null | grep -i -E "runtimes|nvidia" ; (docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L 2>&1 | head -3)
echo "== tools =="; git --version; git lfs version 2>&1; uv --version 2>&1; python3 --version
echo "== disk/ram/cpu =="; df -h ~ | tail -1; free -h | head -2; nproc
echo "== os =="; cat /etc/os-release | head -2; uname -r
