#!/usr/bin/env python3
"""M3 v2 — Gom dữ liệu depth của 1 episode thành 1 file nhỏ `clouds.npz` + hiệu chỉnh tỉ lệ depth theo NỀN TĨNH.

MoGe ước lượng từng frame độc lập nên tỉ lệ mét nhấp nháy/trôi theo frame (vài %; ở 3 m là ~10 cm). Camera Track 1 đứng yên,
nên nền (không phải người/vật) phải có depth không đổi:
  ref     = trung vị theo thời gian của depth nền trên lưới thưa
  s_t     = trung vị(ref / depth_t) trên pixel nền  → nhân mọi điểm 3D của frame t với s_t
Ghi: scale [T], person [T,NP,3], object [T,NO,3] (đã nhân s_t; NaN = thiếu), body3d [T,33,3] (keypoint MediaPipe nâng 3D, đã nhân s_t),
     intrinsics (fx, fy, cx, cy px), size (w, h).

  python prep_clouds.py --run outputs/ep09
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--n-person", type=int, default=800)
    ap.add_argument("--n-object", type=int, default=2500); ap.add_argument("--grid", type=int, default=8)
    a = ap.parse_args()
    run = Path(a.run); frames = sorted((run / "frames").glob("*.jpg")); T = len(frames)
    rng = np.random.default_rng(0)
    person = np.full((T, a.n_person, 3), np.nan, np.float32); obj = np.full((T, a.n_object, 3), np.nan, np.float32)
    zbg, Ks = [], []
    k5 = np.ones((5, 5), np.uint8); k25 = np.ones((25, 25), np.uint8)
    for i, p in enumerate(frames):
        d = np.load(run / "moge" / f"{p.stem}.npz"); pts, ok = d["points"], d["mask"]; Ks.append(d["intrinsics"])
        lab = cv2.imread(str(run / "sam2" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if lab is None:
            lab = np.zeros(ok.shape, np.uint8)
        if lab.shape != ok.shape:
            lab = cv2.resize(lab, ok.shape[::-1], interpolation=cv2.INTER_NEAREST)
        ok = ok & np.isfinite(pts).all(-1)
        for lid, dst, n in ((1, person, a.n_person), (2, obj, a.n_object)):
            m = cv2.erode((lab == lid).astype(np.uint8), k5).astype(bool) & ok
            q = pts[m]
            if len(q) >= 30:
                z = q[:, 2]; med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-6
                q = q[np.abs(z - med) < max(3 * 1.4826 * mad, 0.25 if lid == 2 else 0.6)]   # bỏ điểm dính nền
                q = q[rng.choice(len(q), min(n, len(q)), replace=False)]
                dst[i, :len(q)] = q
        bg = ok & (cv2.dilate((lab > 0).astype(np.uint8), k25) == 0)
        z = np.where(bg, pts[..., 2], np.nan)[::a.grid, ::a.grid]
        zbg.append(z.astype(np.float32))
    zbg = np.stack(zbg); h, w = cv2.imread(str(frames[0])).shape[:2]
    ref = np.nanmedian(zbg[:: max(1, T // 120)], 0)
    use = np.isfinite(ref) & (ref < np.nanpercentile(ref, 80))          # bỏ vùng xa (trần/tường xa nhiễu hơn)
    scale = np.ones(T, np.float32)
    for i in range(T):
        r = ref[use] / zbg[i][use]; r = r[np.isfinite(r)]
        if len(r) > 200:
            scale[i] = np.median(r)
    K = np.median(np.stack(Ks), 0); intr = np.array([K[0, 0] * w, K[1, 1] * h, K[0, 2] * w, K[1, 2] * h], np.float32)
    fxs = np.stack(Ks)[:, 0, 0] * w
    body3d = np.load(run / "keypoints3d.npz")["body"].astype(np.float32)
    s3 = scale[:, None, None]
    np.savez_compressed(run / "clouds.npz", scale=scale, person=person * s3, object=obj * s3, body3d=body3d * s3,
                        intrinsics=intr, size=np.array([w, h]))
    print(f"T={T}; scale nền: trung vị {np.median(scale):.4f}, min {scale.min():.4f}, max {scale.max():.4f}, std {scale.std():.4f}; "
          f"fx theo frame: {np.median(fxs):.1f} ± {fxs.std():.1f} px; điểm người/frame {int(np.median(np.isfinite(person[..., 0]).sum(1)))}, "
          f"vật {int(np.median(np.isfinite(obj[..., 0]).sum(1)))} → {run / 'clouds.npz'}")


if __name__ == "__main__":
    main()
