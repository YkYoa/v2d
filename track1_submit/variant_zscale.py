#!/usr/bin/env python3
"""M3 v3 — thử giả thuyết SAI TIÊU CỰ bằng biến thể hậu xử lý (không fit lại), dùng điểm Kaggle làm thước đo.

Nếu tiêu cự thật f* = a·f (f = ước lượng MoGe) thì mọi độ sâu thật = a·z, còn toạ độ ngang không đổi (x = u·z/f).
Người là vật rắn nên không kéo giãn thân: chỉ dời GỐC người và TÂM vật theo z' = a·z (quanh tâm camera).
  a có thể là 1 số chung, hoặc theo camera: --a back=1.1,front=0.9,left=1.0,right=1.0

  python variant_zscale.py --src outputs/track1_episodes_v2 --runs outputs --model checkpoints/mhr/mhr_model.pt --a 1.15 -o outputs/var_z115
"""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--a", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--rot-identity", action="store_true"); ap.add_argument("--obj-b", type=float, default=None); ap.add_argument("--obj-clamp", type=float, default=None)
    ap.add_argument("--meta", default="data/v2d_challenge/track_1/meta/episodes_metadata.jsonl")
    a = ap.parse_args()
    cams = {json.loads(l)["episode_index"]: json.loads(l)["camera"].split("_")[0] for l in open(a.meta)}
    amap = {k: float(v) for k, v in (kv.split("=") for kv in a.a.split(","))} if "=" in a.a else None
    model = torch.jit.load(a.model, map_location="cpu").float().eval()
    flip = np.array([1, -1, -1]) / 100.0

    def pelvis(p):
        with torch.no_grad():
            return model(torch.zeros(1, 45), torch.tensor(np.concatenate([p, np.zeros(68)])[None], dtype=torch.float32),
                         torch.zeros(1, 72), True)[1][0, 1, :3].numpy() * flip
    p0 = pelvis(np.zeros(136)); unit = np.stack([pelvis(np.eye(136)[i]) - p0 for i in range(3)])   # d_cam = unit.T @ d_param
    Minv = np.linalg.inv(unit.T)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.src).glob("episode_*.npz")):
        e = int(f.stem.split("_")[1]); d = dict(np.load(f)); s = amap[cams[e]] if amap else float(a.a)
        z = np.load(Path(a.runs) / f"ep{e:02d}" / "mhr_fit.npz")["joints"][: len(d["pose"]), 1, 2]
        dcam = np.zeros((len(z), 3)); dcam[:, 2] = (s - 1.0) * z
        d["pose"] = d["pose"].copy(); d["pose"][:, :3] += (dcam @ Minv.T).astype(np.float32)
        d["object_translation"] = d["object_translation"].copy(); d["object_translation"][:, 2] *= s
        if a.obj_b is not None:        # kéo độ sâu vật về phía độ sâu người dọc theo tia nhìn: z' = z_người + b·(z_vật − z_người)
            t = d["object_translation"]; zh = z * s; dz = a.obj_b * (t[:, 2] - zh)
            if a.obj_clamp is not None:
                dz = np.clip(dz, -a.obj_clamp, a.obj_clamp)
            zn = zh + dz
            d["object_translation"] = (t * (zn / t[:, 2])[:, None]).astype(np.float32)
        if a.rot_identity:             # bỏ xoay ICP: giữ hướng vật như ở frame tham chiếu (mesh vốn ở hệ camera của frame đó)
            d["object_rotation"] = np.tile(np.eye(3, dtype=np.float32), (len(d["object_rotation"]), 1, 1))
        np.savez(out / f.name, **d); shutil.copy(f.with_name(f.stem + "_object.glb"), out / (f.stem + "_object.glb"))
    print("→", out, "a =", a.a)


if __name__ == "__main__":
    main()
