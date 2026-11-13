#!/usr/bin/env python3
"""M3 chẩn đoán — thân MHR ở frame chấm điểm ĐẦU TIÊN có quay đúng mặt không?

Scorer fit MỘT Sim(3) trên đỉnh MHR của frame chấm đầu tiên (tương ứng đỉnh-đỉnh) rồi áp cho cả episode,
nên nếu thân bị lật trước/sau ở frame đó thì cả người lẫn vật bị quay ~180° quanh trục đứng.
Tô đỉnh mặt TRƯỚC (ngực/mặt) màu đỏ, mặt SAU màu xanh dương, vẽ xa→gần (painter) lên ảnh, ghép 30 episode thành 1 tấm.

  python diag_facing.py --outputs outputs --sample data/track_1_sample_submission.parquet --model checkpoints/mhr/mhr_model.pt -o outputs/diag_facing.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True); ap.add_argument("--sample", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--fit-name", default="mhr_fit.npz")
    a = ap.parse_args()
    ids = pd.read_parquet(a.sample, columns=["row_id"])["row_id"].str.split("_", expand=True)
    ep, fr = ids[1].astype(int), ids[2].astype(int)
    first = fr[fr != 999999].groupby(ep[fr != 999999]).min()
    model = torch.jit.load(a.model, map_location="cpu").float().eval()
    flip = np.array([1, -1, -1]) / 100.0
    with torch.no_grad():
        v0, s0 = model(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72), True)
    v0, j0 = v0[0].numpy(), s0[0, :, :3].numpy()
    fwd = j0[113] - 0.5 * (j0[75] + j0[39]); fwd[1] = 0; fwd /= np.linalg.norm(fwd)     # mũi − giữa hai vai, bỏ thành phần đứng
    mid = 0.5 * (j0[75] + j0[39])
    front = ((v0 - mid) @ fwd) > 0
    tiles = []
    for e, f in first.items():
        run = Path(a.outputs) / f"ep{e:02d}"
        try:
            fit = np.load(run / a.fit_name)
        except FileNotFoundError:
            continue
        with torch.no_grad():
            p = torch.tensor(np.concatenate([fit["pose"][f], fit["scales"]])[None], dtype=torch.float32)
            v, _ = model(torch.tensor(fit["shape"], dtype=torch.float32)[None], p, torch.zeros(1, 72), True)
        V = v[0].numpy() * flip
        fx, fy, cx, cy = fit["intrinsics"]
        im = cv2.imread(str(sorted((run / "frames").glob("*.jpg"))[f]))
        raw = im.copy()
        u = (fx * V[:, 0] / V[:, 2] + cx).astype(int); w = (fy * V[:, 1] / V[:, 2] + cy).astype(int)
        for i in np.argsort(-V[:, 2])[::2]:                                           # xa → gần
            cv2.circle(im, (u[i], w[i]), 2, (0, 0, 255) if front[i] else (255, 80, 0), -1)
        x0, x1, y0, y1 = u.min() - 30, u.max() + 30, w.min() - 30, w.max() + 30
        x0, y0 = max(x0, 0), max(y0, 0)
        pair = np.hstack([cv2.resize(c[y0:y1, x0:x1], (200, 400)) for c in (raw, im)])
        cv2.putText(pair, f"ep{e} f{f}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(pair)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80]); print("→", a.out, sheet.shape)


if __name__ == "__main__":
    main()
