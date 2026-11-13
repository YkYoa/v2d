#!/usr/bin/env python3
"""M3 — Fit tham số MHR (pose [T,136], scales [68], shape [45]) vào keypoint MediaPipe + depth MoGe của M1.

Đường "không model gated": không dùng SAM 3D Body / SMPL. MHR TorchScript (Apache-2.0, release v1.0.1) khả vi nên
tối ưu trực tiếp bằng Adam:
  - L2D : reprojection khớp MHR ↔ keypoint 2D MediaPipe (trọng số = visibility)
  - Lz  : độ sâu khớp ↔ depth MoGe tại pixel đó + bù bề mặt→tâm khớp (Huber; depth là bề mặt nhìn thấy, không phải tâm khớp)
  - Lsm : làm mượt gia tốc khớp (sai phân bậc 2) — ACC-H chấm đúng đại lượng này
  - Lpr : prior tư thế nghỉ trên tham số thân (ngón tay giữ 0)
Hệ toạ độ đầu ra = hệ GT Track 1 = diag(1,-1,-1) @ MHR / 100 (mét) = hệ camera MoGe (x phải, y xuống, z ra xa).
Scorer tự căn Sim(3) theo người ở frame chấm đầu tiên nên sai tỉ lệ depth toàn cục không bị phạt.

  python fit_mhr_keypoints.py --run outputs/ep09 --model checkpoints/mhr/mhr_model.pt -o outputs/ep09/mhr_fit.npz
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

# MediaPipe Pose (33) → chỉ số khớp MHR (127), theo tên khớp trong v2dlb/mhr_metrics.py
MP2MHR = {11: 75, 12: 39,    # vai trái/phải  → l_uparm / r_uparm
          13: 76, 14: 40,    # khuỷu          → l_lowarm / r_lowarm
          15: 78, 16: 42,    # cổ tay         → l_wrist / r_wrist
          23: 2, 24: 18,     # hông           → l_upleg / r_upleg
          25: 3, 26: 19,     # gối            → l_lowleg / r_lowleg
          27: 4, 28: 20,     # mắt cá         → l_foot / r_foot
          31: 8, 32: 24,     # mũi bàn chân   → l_ball / r_ball
          0: 113}            # mũi            → c_head (xấp xỉ, trọng số thấp)
SURF_OFFSET = {75: .05, 39: .05, 76: .035, 40: .035, 78: .025, 42: .025, 2: .09, 18: .09,
               3: .05, 19: .05, 4: .035, 20: .035, 8: .02, 24: .02, 113: .10}  # m: bề mặt → tâm khớp
JOINT_W = {113: 0.3, 8: 0.5, 24: 0.5}
HAND_COLS = list(range(68, 122))
FLIP = torch.tensor([1.0, -1.0, -1.0])


def load_inputs(run: Path, clouds: str = "clouds.npz"):
    k2 = np.load(run / "mediapipe" / "keypoints2d.npz")["body"]
    if (run / clouds).exists():                                          # v2: depth đã hiệu chỉnh tỉ lệ theo nền tĩnh (prep_clouds.py)
        c = np.load(run / clouds)
        if "body2d" in c.files:                                          # v4: khớp 3D + bề mặt đầy đủ từ NLF (nlf_body.py)
            k2 = fill_nan_linear(c["body2d"])
            return k2, fill_nan_linear(c["body3d"]), tuple(float(v) for v in c["intrinsics"]), tuple(int(v) for v in c["size"]),                 np.nan_to_num(fill_nan_linear(c["person"]), nan=np.nan), True
        return k2, c["body3d"], tuple(float(v) for v in c["intrinsics"]), tuple(int(v) for v in c["size"]), c["person"], False
    return load_inputs_v1(run) + (None, False)


def load_inputs_v1(run: Path):
    k2 = np.load(run / "mediapipe" / "keypoints2d.npz")["body"]          # [T,33,3] x,y px, visibility
    k3 = np.load(run / "keypoints3d.npz")["body"]                        # [T,33,3] m, camera frame (NaN nếu thiếu)
    frames = sorted((run / "frames").glob("*.jpg"))
    import cv2
    h, w = cv2.imread(str(frames[0])).shape[:2]
    Ks = np.stack([np.load(p)["intrinsics"] for p in sorted((run / "moge").glob("*.npz"))[:: max(1, len(frames) // 40)]])
    K = np.median(Ks, 0)                                                  # intrinsics chuẩn hoá [0,1]
    fx, fy, cx, cy = K[0, 0] * w, K[1, 1] * h, K[0, 2] * w, K[1, 2] * h
    return k2, k3, (fx, fy, cx, cy), (w, h)


def fill_nan_linear(a: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính theo thời gian cho từng cột; đầu/cuối lấy giá trị gần nhất."""
    a = a.copy(); t = np.arange(len(a))
    flat = a.reshape(len(a), -1)
    for c in range(flat.shape[1]):
        ok = np.isfinite(flat[:, c])
        if ok.any() and not ok.all():
            flat[~ok, c] = np.interp(t[~ok], t[ok], flat[ok, c])
    return flat.reshape(a.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--model", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--device", default="cuda")
    ap.add_argument("--w-z", type=float, default=20.0); ap.add_argument("--w-smooth", type=float, default=20000.0)
    ap.add_argument("--w-prior", type=float, default=0.05)
    ap.add_argument("--surface-steps", type=int, default=900, help="v2: số bước tinh chỉnh theo bề mặt người (cần clouds.npz); 0 = tắt")
    ap.add_argument("--w-surf", type=float, default=300.0); ap.add_argument("--w-shape", type=float, default=0.005)
    ap.add_argument("--init", default=None, help="mhr_fit.npz của lần fit trước để khởi tạo tư thế")
    ap.add_argument("--clouds", default="clouds.npz"); ap.add_argument("--w-z-nlf", type=float, default=300.0)
    ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--n-obs", type=int, default=400)
    a = ap.parse_args()
    dev = torch.device(a.device); run = Path(a.run)
    k2, k3, (fx, fy, cx, cy), (w, h), person, nlf = load_inputs(run, a.clouds)
    if nlf:                                                               # khớp NLF là TÂM khớp thật → không bù bề mặt, tin depth hơn
        for k in SURF_OFFSET:
            SURF_OFFSET[k] = 0.0
        a.w_z = a.w_z_nlf
        from scipy.signal import savgol_filter                          # NLF ước lượng từng frame → lọc nhẹ theo thời gian trước khi fit
        if len(k2) >= 9:
            k3 = savgol_filter(k3, 9, 2, axis=0, mode="nearest"); person = savgol_filter(person, 9, 2, axis=0, mode="nearest")
            k2[..., :2] = savgol_filter(k2[..., :2], 9, 2, axis=0, mode="nearest")
    T = len(k2); mp_ids = list(MP2MHR); mhr_ids = [MP2MHR[i] for i in mp_ids]
    uv = torch.tensor(k2[:, mp_ids, :2], dtype=torch.float32, device=dev)
    vis = np.nan_to_num(k2[:, mp_ids, 2], nan=0.0)
    seen = np.isfinite(k2[:, mp_ids, 0])
    wj = torch.tensor([JOINT_W.get(j, 1.0) for j in mhr_ids], dtype=torch.float32, device=dev)
    w2d = torch.tensor(np.where(seen, np.clip(vis, 0.05, 1.0), 0.0), dtype=torch.float32, device=dev) * wj
    uv = torch.nan_to_num(uv)
    z_obs = k3[:, mp_ids, 2] + np.array([SURF_OFFSET[j] for j in mhr_ids])[None]
    # depth chỉ tin khi khớp nhìn rõ và nằm trong khoảng hợp lý quanh trung vị của người (loại depth rơi vào vật/nền)
    zmed = np.nanmedian(np.where(vis > 0.5, z_obs, np.nan), 1, keepdims=True)
    zok = np.isfinite(z_obs) & (vis > 0.6) & (np.abs(z_obs - zmed) < 0.6)
    wz = torch.tensor(zok, dtype=torch.float32, device=dev) * wj
    z_t = torch.tensor(np.nan_to_num(z_obs), dtype=torch.float32, device=dev)
    print(f"T={T}, ảnh {w}x{h}, fx={fx:.0f} fy={fy:.0f}; khớp 2D dùng được {int(seen.sum())}/{seen.size}, depth tin được {int(zok.sum())}")

    model = torch.jit.load(a.model, map_location=dev).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    flip = FLIP.to(dev)
    zeros_shape = torch.zeros(1, 45, device=dev); zeros_scale = torch.zeros(1, 68, device=dev)
    expr = torch.zeros(1, 72, device=dev)

    def joints_of(pose):  # pose [N,136] → khớp [N,127,3] mét, hệ camera
        n = len(pose)
        params = torch.cat([pose, zeros_scale.expand(n, -1)], 1)
        _, sk = model(zeros_shape.expand(n, -1).contiguous(), params, expr.expand(n, -1).contiguous(), True)
        return sk[..., :3] * flip / 100.0

    def project(j):
        z = j[..., 2].clamp(min=0.2)
        return torch.stack([fx * j[..., 0] / z + cx, fy * j[..., 1] / z + cy], -1)

    # ---- khởi tạo gốc: đặt trung điểm hông MHR trùng trung điểm hông quan sát (trừ vị trí hông khi translation = 0)
    hips3 = fill_nan_linear(np.nanmean(k3[:, [23, 24]], 1))               # [T,3]
    if not np.isfinite(hips3).all():
        raise SystemExit("không có depth hông ở frame nào — kiểm tra keypoints3d.npz")
    hips3[:, 2] += SURF_OFFSET[2]
    hips3_t = torch.tensor(hips3, dtype=torch.float32, device=dev)
    with torch.no_grad():
        j0 = joints_of(torch.zeros(1, 136, device=dev))
        hip0 = j0[0, [2, 18]].mean(0)                                     # mét, hệ camera, khi translation = 0
        # kiểm đơn vị translation: +1 ở tham số thứ i dịch khớp bao nhiêu mét (kỳ vọng 0.1 m, dấu theo FLIP)
        unit = torch.stack([joints_of(torch.eye(136, device=dev)[i][None])[0, 1] - j0[0, 1] for i in range(3)])
    print("đơn vị translation (m / đơn vị tham số), hệ camera:\n", unit.cpu().numpy().round(3))
    Minv = torch.linalg.inv(unit.T)                                       # d_cam = unit.T @ d_param

    def init_pose(yaw):
        p = torch.zeros(T, 136, device=dev)
        p[:, 4] = yaw                                                     # tham số xoay gốc quanh trục dọc (kiểm bằng multi-start)
        with torch.no_grad():
            hip = joints_of(p[:1])[0, [2, 18]].mean(0)
        p[:, :3] = (hips3_t - hip) @ Minv.T
        return p

    def loss_terms(pose, sel=slice(None)):
        j = joints_of(pose)
        jm = j[:, mhr_ids]
        e2 = ((project(jm) - uv[sel]) * (100.0 / w)).pow(2).sum(-1)       # đơn vị: 1 % bề rộng ảnh
        l2d = (e2 * w2d[sel]).sum() / w2d[sel].sum().clamp(min=1)
        ez = torch.nn.functional.huber_loss(jm[..., 2], z_t[sel], delta=0.15, reduction="none")
        lz = (ez * wz[sel]).sum() / wz[sel].sum().clamp(min=1)
        return j, l2d, lz

    init = np.load(a.init)["pose"] if a.init else None
    if init is not None and len(init) != T:
        init = None
    # ---- multi-start hướng người ở 40 frame đầu: thử 4 yaw × 3 trục euler, chọn cái reprojection thấp nhất
    best = None
    sub = slice(0, min(T, 40))
    for axis in ((3, 4, 5) if init is None else ()):
        for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
            if axis != 4 and yaw == 0.0:
                continue
            p = torch.zeros(T, 136, device=dev); p[:, axis] = yaw
            with torch.no_grad():
                hip = joints_of(p[:1])[0, [2, 18]].mean(0)
                p[:, :3] = (hips3_t - hip) @ Minv.T
            q = p[sub].clone().requires_grad_(True)
            opt = torch.optim.Adam([q], lr=0.02)
            mask = torch.zeros(136, device=dev); mask[:6] = 1                # chỉ tối ưu gốc
            for _ in range(80):
                _, l2d, lz = loss_terms(q, sub); loss = l2d + a.w_z * lz
                opt.zero_grad(); loss.backward(); q.grad *= mask; opt.step()
            print(f"  init trục {axis} góc {yaw:+.2f}: l2d={l2d.item():.5f} lz={lz.item():.4f}")
            if best is None or loss.item() < best[0]:
                best = (loss.item(), axis, yaw, q.detach().mean(0)[3:6])
    if init is None:
        _, axis, yaw, rot0 = best
        print(f"chọn init: trục {axis}, góc {yaw:+.2f}")
        pose = torch.zeros(T, 136, device=dev); pose[:, 3:6] = rot0
        with torch.no_grad():
            hip = joints_of(pose[:1])[0, [2, 18]].mean(0)
            pose[:, :3] = (hips3_t - hip) @ Minv.T
    else:                                                                # khởi tạo từ lần fit trước (tư thế đã hội tụ), chỉ dời gốc về hông quan sát mới
        axis, yaw = -1, 0.0
        pose = torch.tensor(init, dtype=torch.float32, device=dev)
        with torch.no_grad():
            hip = torch.cat([joints_of(pose[i:i + a.chunk])[:, [2, 18]].mean(1) for i in range(0, T, a.chunk)])
            pose[:, :3] += (hips3_t - hip) @ Minv.T
        print(f"khởi tạo từ {a.init}")
    pose.requires_grad_(True)

    body_mask = torch.ones(136, device=dev); body_mask[HAND_COLS] = 0       # ngón tay giữ 0
    prior_w = torch.ones(136, device=dev); prior_w[:6] = 0
    stages = [(a.steps // 4, 0.03, True), (a.steps - a.steps // 4, 0.02, False)]  # (số bước, lr, chỉ-gốc?)
    if person is not None and a.surface_steps > 0:
        stages = []                                                        # v2 làm cả hai giai đoạn trong vòng chia đoạn bên dưới
    for n, lr, root_only in stages:
        opt = torch.optim.Adam([pose], lr=lr)
        m = body_mask.clone()
        if root_only:
            m[6:] = 0
        for it in range(n):
            j, l2d, lz = loss_terms(pose)
            acc = j[2:, mhr_ids + [1, 35, 36, 37, 110]] - 2 * j[1:-1, mhr_ids + [1, 35, 36, 37, 110]] + j[:-2, mhr_ids + [1, 35, 36, 37, 110]]
            lsm = acc.pow(2).sum(-1).mean()
            lpr = (pose.pow(2) * prior_w).sum(-1).mean()
            loss = l2d + a.w_z * lz + a.w_smooth * lsm + a.w_prior * lpr
            opt.zero_grad(); loss.backward(); pose.grad *= m; opt.step()
            if it % 100 == 0 or it == n - 1:
                print(f"  [{'gốc' if root_only else 'toàn thân'}] {it:4d} loss={loss.item():.5f} l2d={l2d.item():.5f} "
                      f"lz={lz.item():.4f} lsm={lsm.item():.2e} lpr={lpr.item():.3f}")

    shape = torch.zeros(45, device=dev); surf_cm = float("nan")
    if person is not None and a.surface_steps > 0:
        # ---- v2: tinh chỉnh theo BỀ MẶT: mỗi điểm depth của người (mask SAM2) phải nằm gần một đỉnh MHR (một chiều: quan sát → mô hình,
        # đúng cho bề mặt nhìn thấy một phần). Cố định hướng xoay thân/độ sâu tốt hơn 15 keypoint, và cho phép tối ưu shape (cỡ người).
        # Chia đoạn theo thời gian + cộng dồn gradient (= full batch) để VRAM không phụ thuộc độ dài episode.
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            nv = model(zeros_shape, torch.zeros(1, 204, device=dev), expr, True)[0].shape[1]
        vid = torch.randperm(nv, generator=g)[:3000].to(dev)
        P = torch.tensor(person[:, :a.n_obs], dtype=torch.float32, device=dev); Pok = torch.isfinite(P).all(-1)
        P = torch.nan_to_num(P)
        shape.requires_grad_(True)
        pose = pose.detach().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [pose], "lr": 0.04}, {"params": [shape], "lr": 0.01}])
        root_steps = a.surface_steps // 6 if init is None else a.surface_steps // 20; root_mask = body_mask.clone(); root_mask[6:] = 0
        sm_ids = mhr_ids + [1, 35, 36, 37, 110]
        W2, WZ, NS = w2d.sum().clamp(min=1), wz.sum().clamp(min=1), Pok.sum().clamp(min=1)
        starts = list(range(0, T, a.chunk))

        def chunk_loss(s0):
            lo, hi = max(s0 - 2, 0), min(s0 + a.chunk, T)                 # 2 frame gối đầu cho số hạng gia tốc
            n = hi - lo; own = slice(s0 - lo, n)
            params = torch.cat([pose[lo:hi], zeros_scale.expand(n, -1)], 1)
            v, sk = model(shape[None].expand(n, -1).contiguous(), params, expr.expand(n, -1).contiguous(), True)
            j = sk[..., :3] * flip / 100.0; v = v[:, vid] * flip / 100.0
            jm = j[own][:, mhr_ids]; sel = slice(s0, hi)
            l2d = (((project(jm) - uv[sel]) * (100.0 / w)).pow(2).sum(-1) * w2d[sel]).sum() / W2
            lz = (torch.nn.functional.huber_loss(jm[..., 2], z_t[sel], delta=0.15, reduction="none") * wz[sel]).sum() / WZ
            dmin = torch.cdist(P[sel], v[own]).min(-1).values             # [n_own, n_obs]
            ls = (torch.nn.functional.huber_loss(dmin, torch.zeros_like(dmin), delta=0.05, reduction="none") * Pok[sel]).sum() / NS
            js = j[:, sm_ids]; acc = js[2:] - 2 * js[1:-1] + js[:-2]
            lsm = acc.pow(2).sum(-1).sum() / ((T - 2) * len(sm_ids))
            return l2d, lz, ls, lsm, dmin.detach()[Pok[sel]]

        for it in range(a.surface_steps):
            opt.zero_grad(); tot = np.zeros(4); res = []
            for s0 in starts:
                l2d, lz, ls, lsm, dm = chunk_loss(s0)
                (l2d + a.w_z * lz + a.w_surf * ls + a.w_smooth * lsm).backward()
                tot += [l2d.item(), lz.item(), ls.item(), lsm.item()]; res.append(dm)
            lpr = (pose.pow(2) * prior_w).sum(-1).mean() * a.w_prior + a.w_shape * shape.pow(2).sum()
            lpr.backward()
            if it == root_steps:
                opt.param_groups[0]["lr"] = 0.02
            if it < root_steps:                                            # giai đoạn đầu: chỉ gốc (tịnh tiến + xoay), chưa đụng shape
                pose.grad *= root_mask; shape.grad.zero_()
            else:
                pose.grad *= body_mask
            opt.step()
            surf_cm = float(torch.cat(res).median() * 100)
            if it % 50 == 0 or it == a.surface_steps - 1:
                print(f"  [bề mặt] {it:4d} l2d={tot[0]:.5f} lz={tot[1]:.4f} ls={tot[2]:.2e} lsm={tot[3]:.2e} "
                      f"|shape|={shape.norm().item():.2f} surf_med={surf_cm:.2f} cm")
        shape = shape.detach(); pose = pose.detach()

        def joints_of(pose_):                                              # từ đây dùng shape đã fit, chia đoạn
            out = []
            for s0 in range(0, len(pose_), a.chunk):
                q = pose_[s0:s0 + a.chunk]; n = len(q)
                _, sk = model(shape[None].expand(n, -1).contiguous(), torch.cat([q, zeros_scale.expand(n, -1)], 1),
                              expr.expand(n, -1).contiguous(), True)
                out.append(sk[..., :3] * flip / 100.0)
            return torch.cat(out)

        def loss_terms(pose_, sel=slice(None)):
            j = joints_of(pose_); jm = j[:, mhr_ids]
            l2d = (((project(jm) - uv[sel]) * (100.0 / w)).pow(2).sum(-1) * w2d[sel]).sum() / w2d[sel].sum().clamp(min=1)
            lz = (torch.nn.functional.huber_loss(jm[..., 2], z_t[sel], delta=0.15, reduction="none") * wz[sel]).sum() / wz[sel].sum().clamp(min=1)
            return j, l2d, lz

    with torch.no_grad():
        j, l2d, lz = loss_terms(pose)
        px = ((project(j[:, mhr_ids]) - uv).norm(dim=-1) * (w2d > 0)).sum() / (w2d > 0).sum()
        acc = (j[2:, list(MP2MHR.values())] - 2 * j[1:-1, list(MP2MHR.values())] + j[:-2, list(MP2MHR.values())]).norm(dim=-1).mean()
    report = {"frames": T, "reproj_px_mean": float(px), "image_width": w, "depth_huber": float(lz),
              "joint_accel_cm_per_frame2": float(acc * 100), "init_axis": axis, "init_angle": float(yaw),
              "pelvis_depth_m_median": float(j[:, 1, 2].median()),
              "surface_residual_cm_median": surf_cm, "shape_norm": float(shape.norm())}
    print("report:", json.dumps(report))
    np.savez(a.out, pose=pose.detach().cpu().numpy().astype(np.float32), scales=np.zeros(68, np.float32),
             shape=shape.cpu().numpy().astype(np.float32), joints=j.cpu().numpy().astype(np.float32),
             intrinsics=np.array([fx, fy, cx, cy], np.float32), report=json.dumps(report))
    print("→", a.out)
    # ảnh kiểm tra: khớp MHR chiếu lên ảnh (đỏ) so với MediaPipe (xanh) ở 8 frame
    import cv2
    frames = sorted((run / "frames").glob("*.jpg")); pj = project(j[:, mhr_ids]).cpu().numpy(); uvn = uv.cpu().numpy()
    bones = [(75, 76), (76, 78), (39, 40), (40, 42), (75, 39), (2, 18), (75, 2), (39, 18), (2, 3), (3, 4), (18, 19), (19, 20), (4, 8), (20, 24)]
    idx = {m: i for i, m in enumerate(mhr_ids)}; tiles = []
    for t in np.linspace(0, T - 1, 8).astype(int):
        im = cv2.imread(str(frames[t]))
        for a_, b_ in bones:
            cv2.line(im, tuple(int(v) for v in pj[t, idx[a_]]), tuple(int(v) for v in pj[t, idx[b_]]), (0, 0, 255), 2)
            cv2.line(im, tuple(int(v) for v in uvn[t, idx[a_]]), tuple(int(v) for v in uvn[t, idx[b_]]), (0, 255, 0), 1)
        cv2.putText(im, f"f{t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2); tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(Path(a.out).with_suffix(".jpg")), cv2.resize(sheet, None, fx=0.6, fy=0.6))


if __name__ == "__main__":
    main()
