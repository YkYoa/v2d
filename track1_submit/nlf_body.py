#!/usr/bin/env python3
"""M3 v4 — Cơ thể 3D từ NLF (Neural Localizer Fields, isarandi/nlf; TRỌNG SỐ CHỈ CHO NGHIÊN CỨU PHI THƯƠNG MẠI).

Với mỗi frame: box người lấy từ mask SAM2 (nhãn 1) → `estimate_smpl_batched` với intrinsics của MoGe → khớp SMPL 24 điểm + lưới 6890 đỉnh
trong hệ camera (mm → m). Ghi `clouds_nlf.npz` cùng khuôn với `clouds.npz` để `fit_mhr_keypoints.py --clouds clouds_nlf.npz` dùng lại:
  person  [T,NV,3]  : tập con cố định các đỉnh SMPL (bề mặt ĐẦY ĐỦ, kể cả lưng) — thay cho point cloud depth
  body3d  [T,33,3]  : khớp NLF đặt vào đúng chỉ số MediaPipe (vai/khuỷu/cổ tay/hông/gối/mắt cá/mũi chân); chỗ khác NaN
  body2d  [T,33,3]  : chiếu của body3d + visibility 1
  g, g_t            : tỉ lệ (độ sâu NLF) / (độ sâu MoGe) của bề mặt người — để đưa VẬT (từ MoGe) về cùng thang với người (từ NLF)

  python nlf_body.py --run outputs/ep09 --model checkpoints/nlf/nlf_l_multi_0.3.2.patch6.torchscript
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401  (phải import thì TorchScript của NLF mới load được)
from scipy.spatial import cKDTree

SMPL2MP = {16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16, 1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28, 10: 31, 11: 32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--n-verts", type=int, default=1200)
    a = ap.parse_args()
    run = Path(a.run); frames = sorted((run / "frames").glob("*.jpg")); T = len(frames)
    c = np.load(run / "clouds.npz"); fx, fy, cx, cy = (float(v) for v in c["intrinsics"]); w, h = (int(v) for v in c["size"])
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device="cuda")
    model = torch.jit.load(a.model).cuda().eval()
    vid = np.sort(np.random.default_rng(0).choice(6890, a.n_verts, replace=False))
    J = np.full((T, 24, 3), np.nan, np.float32); V = np.full((T, a.n_verts, 3), np.nan, np.float32)
    betas = np.full((T, 10), np.nan, np.float32); g_t = np.full(T, np.nan, np.float32); unc = np.full(T, np.nan, np.float32)
    person = c["person"]; prev = None
    for s in range(0, T, a.batch):
        idx = list(range(s, min(s + a.batch, T))); ims, boxes = [], []
        for i in idx:
            ims.append(torchvision.io.read_image(str(frames[i])))
            lab = cv2.imread(str(run / "sam2" / f"{frames[i].stem}.png"), cv2.IMREAD_GRAYSCALE)
            box = prev
            if lab is not None:
                if lab.shape != (h, w):
                    lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.nonzero(lab == 1)
                if len(xs) > 200:
                    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); m = 0.06 * max(x1 - x0, y1 - y0)
                    box = [max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, w - 1) - max(x0 - m, 0), min(y1 + m, h - 1) - max(y0 - m, 0)]
            prev = box
            boxes.append(torch.tensor([box if box is not None else [0, 0, w, h]], dtype=torch.float32, device="cuda"))
        with torch.inference_mode(), torch.device("cuda"):
            p = model.estimate_smpl_batched(torch.stack(ims).cuda(), boxes, intrinsic_matrix=K[None].expand(len(idx), -1, -1))
        for k, i in enumerate(idx):
            if len(p["joints3d"][k]) == 0:
                continue
            J[i] = (p["joints3d"][k][0] / 1000).cpu().numpy(); v = (p["vertices3d"][k][0] / 1000).cpu().numpy(); V[i] = v[vid]
            betas[i] = p["betas"][k][0].cpu().numpy(); unc[i] = float(p["joint_uncertainties"][k][0].mean())
            q = person[i]; q = q[np.isfinite(q[:, 0])]
            if len(q) > 100:                                   # bề mặt TRƯỚC của NLF tại cùng pixel so với depth MoGe
                uv = np.stack([fx * v[:, 0] / v[:, 2] + cx, fy * v[:, 1] / v[:, 2] + cy], 1)
                near = cKDTree(uv).query_ball_point(np.stack([fx * q[:, 0] / q[:, 2] + cx, fy * q[:, 1] / q[:, 2] + cy], 1), r=4.0)
                r = [v[n, 2].min() / z for n, z in zip(near, q[:, 2]) if len(n)]
                if len(r) > 50:
                    g_t[i] = np.median(r)
    ok = np.isfinite(J[:, 0, 0]); g = float(np.nanmedian(g_t))
    body3d = np.full((T, 33, 3), np.nan, np.float32)
    for sj, mp in SMPL2MP.items():
        body3d[:, mp] = J[:, sj]
    body2d = np.stack([fx * body3d[..., 0] / body3d[..., 2] + cx, fy * body3d[..., 1] / body3d[..., 2] + cy,
                       np.where(np.isfinite(body3d[..., 0]), 1.0, np.nan)], -1).astype(np.float32)
    np.savez_compressed(run / "clouds_nlf.npz", person=V, body3d=body3d, body2d=body2d, intrinsics=c["intrinsics"], size=c["size"],
                        nlf=np.array(1), g=np.float32(g), g_t=g_t, betas=betas, joints_smpl=J, uncertainty=unc, scale=c["scale"])
    jz = J[ok, 0, 2]
    print(f"T={T}; NLF có kết quả {int(ok.sum())}/{T} frame; pelvis z trung vị {np.median(jz):.2f} m; nhiễu z khung-khung "
          f"{np.median(np.abs(np.diff(jz))) * 100:.2f} cm; g = z_NLF/z_MoGe = {g:.4f} (p10 {np.nanpercentile(g_t, 10):.3f}, p90 {np.nanpercentile(g_t, 90):.3f}); "
          f"bất định khớp TB {np.nanmean(unc):.0f}; VRAM đỉnh {torch.cuda.max_memory_allocated() / 1e6:.0f} MB → {run / 'clouds_nlf.npz'}")


if __name__ == "__main__":
    main()
