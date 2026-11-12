#!/usr/bin/env python3
"""M3 — Mesh vật + quỹ đạo Sim(3) từ mask SAM2 + point map MoGe (không TRELLIS/FoundationPose).

  mesh : bề mặt NHÌN THẤY của vật ở frame tham chiếu (lưới tam giác trên pixel trong mask, bỏ cạnh nhảy depth),
         thêm mặt sau lùi `--thickness` m dọc tia nhìn → xấp xỉ 2.5D có bề dày. Toạ độ mesh = hệ camera frame tham chiếu.
  pose : với mỗi frame t, ICP cắt tỉa (frame → mô hình tham chiếu), khởi tạo từ pose frame trước; sau đó làm mượt.
         point_scene(t) = s * R_t @ point_mesh + t_t, s = 1 (mesh đã là mét).

  python object_from_depth.py --run outputs/ep09 -o outputs/ep09/object   # → object.glb + object_track.npz
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def object_points(run: Path, stem: str, label: int = 2, erode: int = 3):
    lab = cv2.imread(str(run / "sam2" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    d = np.load(run / "moge" / f"{stem}.npz")
    pts, ok = d["points"], d["mask"]
    if lab is None:
        return None, None, None
    if lab.shape != ok.shape:
        lab = cv2.resize(lab, ok.shape[::-1], interpolation=cv2.INTER_NEAREST)
    m = cv2.erode((lab == label).astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool) & ok & np.isfinite(pts).all(-1)
    return pts, m, lab


def clean(p: np.ndarray) -> np.ndarray:
    """Bỏ điểm lạc theo độ sâu (mask dính nền): giữ trong ±3 MAD quanh trung vị z."""
    if len(p) < 50:
        return p
    z = p[:, 2]; med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-6
    return p[np.abs(z - med) < max(3 * 1.4826 * mad, 0.25)]


def kabsch(a: np.ndarray, b: np.ndarray):
    """R, t sao cho R @ a + t ≈ b."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1, 1, d]) @ u.T
    return R, cb - R @ ca


def icp(model: np.ndarray, tree_unused, frame: np.ndarray, R: np.ndarray, t: np.ndarray, iters: int = 25, keep: float = 0.7):
    """Tìm (R, t): R @ model + t ≈ frame. Tương ứng: mỗi điểm frame → điểm mô hình đã biến đổi gần nhất, giữ `keep` tốt nhất."""
    err = np.inf
    for _ in range(iters):
        mt = model @ R.T + t
        dist, idx = cKDTree(mt).query(frame)
        sel = dist <= np.quantile(dist, keep)
        dR, dt = kabsch(mt[idx[sel]], frame[sel])
        R, t = dR @ R, dR @ t + dt
        new = float(dist[sel].mean())
        if abs(err - new) < 1e-5:
            err = new; break
        err = new
    return R, t, err


def surface_mesh(pts: np.ndarray, m: np.ndarray, step: int, thickness: float, jump: float = 0.08):
    """Lưới tam giác trên pixel trong mask (bước `step`), bỏ tam giác có cạnh nhảy depth > jump; thêm mặt sau."""
    H, W = m.shape
    ys, xs = np.arange(0, H, step), np.arange(0, W, step)
    grid = m[np.ix_(ys, xs)]; P = pts[np.ix_(ys, xs)]
    vid = -np.ones(grid.shape, int); vid[grid] = np.arange(grid.sum())
    V = P[grid]
    faces = []
    for (dy0, dx0), (dy1, dx1), (dy2, dx2) in (((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))):
        a = vid[dy0:grid.shape[0] - 1 + dy0, dx0:grid.shape[1] - 1 + dx0]
        b = vid[dy1:grid.shape[0] - 1 + dy1, dx1:grid.shape[1] - 1 + dx1]
        c = vid[dy2:grid.shape[0] - 1 + dy2, dx2:grid.shape[1] - 1 + dx2]
        ok = (a >= 0) & (b >= 0) & (c >= 0)
        f = np.stack([a[ok], b[ok], c[ok]], 1)
        z = V[f][..., 2]
        faces.append(f[(z.max(1) - z.min(1)) < jump])
    F = np.concatenate(faces)
    rays = V / np.linalg.norm(V, axis=1, keepdims=True)
    Vb = V + rays * thickness                                  # mặt sau: lùi dọc tia nhìn
    Fb = F[:, ::-1] + len(V)
    return np.concatenate([V, Vb]), np.concatenate([F, Fb])


def smooth_rotations(R: np.ndarray, window: int) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()
    for i in range(1, len(q)):                                # cùng bán cầu để lọc tuyến tính được
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = savgol_filter(q, window, 2, axis=0, mode="nearest")
    return Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--ref", type=int, default=-1, help="frame tham chiếu (-1: frame có nhiều pixel vật nhất trong 30 frame đầu)")
    ap.add_argument("--thickness", type=float, default=0.03); ap.add_argument("--mesh-step", type=int, default=2)
    ap.add_argument("--n-model", type=int, default=4000); ap.add_argument("--n-frame", type=int, default=2500)
    ap.add_argument("--smooth", type=int, default=15)
    a = ap.parse_args()
    run = Path(a.run); stems = [p.stem for p in sorted((run / "frames").glob("*.jpg"))]; T = len(stems)
    rng = np.random.default_rng(0)

    clouds = []
    for s in stems:
        pts, m, _ = object_points(run, s)
        clouds.append(clean(pts[m]) if pts is not None and m.sum() > 50 else np.zeros((0, 3)))
    n = np.array([len(c) for c in clouds])
    ref = int(np.argmax(n[:30])) if a.ref < 0 else a.ref
    print(f"T={T}, điểm vật/frame: trung vị {int(np.median(n))}, min {n.min()}, frame thiếu {(n < 50).sum()}; frame tham chiếu {ref} ({n[ref]} điểm)")

    pts, m, _ = object_points(run, stems[ref])
    zc = clean(pts[m]); zlo, zhi = zc[:, 2].min(), zc[:, 2].max()
    m &= (pts[..., 2] >= zlo) & (pts[..., 2] <= zhi)
    V, F = surface_mesh(pts, m, a.mesh_step, a.thickness)
    # đặt gốc mesh tại tâm vật: nếu để gốc ở camera (cách ~3 m) thì nhiễu xoay bị khuếch đại vào translation,
    # mà ACC-O chấm đúng sai phân bậc 2 của object_translation
    center = clouds[ref].mean(0); V = V - center
    import trimesh
    mesh = trimesh.Trimesh(V, F, process=False)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out.with_suffix(".glb")))
    ext = mesh.bounding_box_oriented.primitive.extents
    print(f"mesh: {len(V)} đỉnh, {len(F)} mặt, kích thước OBB {np.round(np.sort(ext)[::-1], 2)} m → {out.with_suffix('.glb')}")

    model = clouds[ref][rng.choice(len(clouds[ref]), min(a.n_model, len(clouds[ref])), replace=False)] - center
    R = np.tile(np.eye(3), (T, 1, 1)); t = np.zeros((T, 3)); err = np.full(T, np.nan)

    def sweep(order, R0, t0):
        Rc, tc = R0, t0
        for i in order:
            c = clouds[i]
            if len(c) < 50:
                R[i], t[i] = Rc, tc; continue
            f = c[rng.choice(len(c), min(a.n_frame, len(c)), replace=False)]
            # khởi tạo: pose trước + dịch theo tâm (vật bị bê đi nhanh hơn ICP hội tụ)
            guess_t = tc + (f.mean(0) - (model @ Rc.T + tc).mean(0))
            Rc, tc, err[i] = icp(model, None, f, Rc, guess_t)
            R[i], t[i] = Rc, tc

    sweep(range(ref, T), np.eye(3), center.copy())
    sweep(range(ref - 1, -1, -1), np.eye(3), center.copy())
    w = min(a.smooth, T - (1 - T % 2)); w -= 1 - w % 2
    Rs = smooth_rotations(R, w); ts = savgol_filter(t, w, 2, axis=0, mode="nearest")
    acc = np.linalg.norm(ts[2:] - 2 * ts[1:-1] + ts[:-2], axis=1).mean() * 100
    ang = np.degrees(Rotation.from_matrix(Rs).magnitude())
    report = {"frames": T, "ref_frame": ref, "icp_err_cm_median": float(np.nanmedian(err) * 100),
              "icp_err_cm_p95": float(np.nanpercentile(err, 95) * 100), "translation_accel_cm_per_frame2": float(acc),
              "max_rotation_from_ref_deg": float(ang.max()), "travel_m": float(np.linalg.norm(ts.max(0) - ts.min(0))),
              "mesh_vertices": int(len(V)), "mesh_faces": int(len(F))}
    print("report:", json.dumps(report))
    np.savez(out.parent / (out.name + "_track.npz"), object_rotation=Rs.astype(np.float32),
             object_translation=ts.astype(np.float32), object_scale=np.float32(1.0), icp_err=err, report=json.dumps(report))


if __name__ == "__main__":
    main()
