#!/usr/bin/env python3
"""Demo A — video điện thoại (camera ĐỨNG YÊN, quay bàn + hai tay + một cái hộp) → recon.npz cho retarget_sharpa.py.

Chỉ dùng model cho phép thương mại hoá: MoGe-2 (MIT) · Grounding DINO (Apache-2.0) · SAM 2.1 (Apache-2.0) · MediaPipe Hands (Apache-2.0).
Không MANO, không SAM 3 / SAM 3D, không WiLoR/HaMeR, không FoundationPose. Vật là HỘP đo bằng thước nên không cần model dựng mesh.

  python hands_object_from_video.py --video clip.mp4 --object "a cardboard box" --box-size 0.22 0.13 0.11 --out outputs/phone_demo/my_clip

Các bước:
  1. trích frame → MoGe-2: point map mét + tiêu cự;
  2. Grounding DINO (frame đầu) + SAM 2.1 → mask hộp mọi frame;
  3. MediaPipe Hands → 21 điểm ảnh + 21 điểm "world" (hình dạng bàn tay, mét, gốc ở tay) + trái/phải;
  4. tỉ lệ depth: theo nền tĩnh từng frame, rồi một hệ số chung sao cho cạnh hộp đo được = cạnh hộp thật;
  5. mặt bàn (RANSAC quanh hộp) → hệ thế giới: z hướng lên, mặt bàn ở z = 1.0 m, gốc dưới tâm hộp lúc đầu;
  6. pose hộp từng frame: ICP giữa điểm depth trong mask và hộp kích thước thật;
  7. tay 3D: điểm ảnh + depth MoGe tại khớp, độ sâu tương đối giữa các khớp từ MediaPipe; trái/phải gán theo vị trí + liên tục.
Ghi: recon.npz, report.json, overlay.mp4 (khung hộp + khớp tay vẽ đè để kiểm bằng mắt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp

for _d in ("demo_track1", "m3/demo_track1"):                                # repo local / bố cục trên server
    if (Path(__file__).resolve().parents[1] / _d / "run_perception.py").exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _d)); break
import run_perception as rp  # noqa: E402  (extract_frames, run_moge, run_sam2, download, log)

log = rp.log
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


# ----------------------------------------------------------------------------- phát hiện hộp + tay
def detect_object(frame: Path, prompt: str, device: str, thr: float = 0.25):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid); model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
    img = Image.open(frame).convert("RGB")
    inputs = proc(images=img, text=prompt.rstrip(".") + " .", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=thr, text_threshold=thr,
                                                      target_sizes=[img.size[::-1]])[0]
    del model
    if not len(res["boxes"]):
        raise SystemExit(f"Grounding DINO không thấy '{prompt}' ở frame đầu — đổi --object hoặc quay lại cho hộp rõ hơn")
    W, H = img.size; cand = []
    for b, s in zip(res["boxes"].tolist(), res["scores"].tolist()):
        if (b[2] - b[0]) * (b[3] - b[1]) < 0.6 * W * H:                     # bỏ box trùm cả ảnh (thường là mặt bàn)
            cand.append((s, b))
    s, b = max(cand or list(zip(res["scores"].tolist(), res["boxes"].tolist())))
    log(f"detect '{prompt}': {len(res['boxes'])} ứng viên, chọn score {s:.2f} box {[round(v) for v in b]}")
    return b


def run_hands(frames: list[Path], out: Path):
    """MediaPipe Hands, tối đa 2 tay/frame, CHƯA gán trái/phải (nhãn của MediaPipe nhảy lung tung ở góc nhìn mu bàn tay).
    → uv [T,2,21,2] px, world [T,2,21,3] m (gốc ở tay), label [T,2] (1 = MediaPipe nói "Right", 0 = "Left", -1 = trống), score [T,2]."""
    if out.exists():
        d = np.load(out)
        if "label" in d:
            return d["uv"], d["world"], d["label"], d["score"]
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    model = rp.download(rp.MP_HAND_URL, rp.MODELS / "mediapipe" / "hand_landmarker.task")
    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))
    T = len(frames); uv = np.full((T, 2, 21, 2), np.nan, np.float32); world = np.full((T, 2, 21, 3), np.nan, np.float32)
    label = np.full((T, 2), -1, np.int8); score = np.zeros((T, 2), np.float32)
    for i, p in enumerate(frames):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB); h, w = img.shape[:2]
        r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), int(i * 1000 / 30))
        for j, (lm, wl, hd) in enumerate(list(zip(r.hand_landmarks, r.hand_world_landmarks, r.handedness))[:2]):
            uv[i, j] = [(q.x * w, q.y * h) for q in lm]; world[i, j] = [(q.x, q.y, q.z) for q in wl]
            label[i, j] = int(hd[0].category_name == "Right"); score[i, j] = hd[0].score
    det.close(); np.savez_compressed(out, uv=uv, world=world, label=label, score=score)
    return uv, world, label, score


def assign_sides(uv, world, label, swap: bool, width: int):
    """Gán mỗi phát hiện cho tay phải (0) / trái (1) của NGƯỜI.
    Frame có 2 tay: theo thứ tự trái-phải trên ảnh, chiều do đa số nhãn MediaPipe trên cả clip quyết định (không đổi giữa chừng).
    Frame có 1 tay: gán cho bên có cổ tay lần cuối nằm gần nhất (<= 20 frame trước), không có thì dùng nhãn.
    swap: chỉ bật khi video bị lật gương. MediaPipe Tasks 1.2 gán nhãn theo ảnh KHÔNG lật (kiểm 13/11 trên clip tissue_box của NVIDIA:
    đa số frame 2 tay, tay bên trái ảnh góc nhìn thứ nhất được gán "Left") — ngược với ghi chú cũ trong tài liệu MediaPipe."""
    T = len(uv); out_uv = np.full((T, 2, 21, 2), np.nan, np.float32); out_w = np.full((T, 2, 21, 3), np.nan, np.float32)
    is_right = (label == 0) if swap else (label == 1)
    two = (label >= 0).all(1)
    votes = [np.sign(uv[i, 0, 0, 0] - uv[i, 1, 0, 0]) * (1 if is_right[i, 0] else -1)
             for i in np.flatnonzero(two) if is_right[i, 0] != is_right[i, 1]]
    right_on_larger_x = (np.sum(votes) >= 0) if votes else True
    def sweep(order):
        """→ {frame: ({det: side}, độ tin)} khi đi theo thứ tự `order`. Độ tin = số frame tới frame có ĐỦ HAI tay gần nhất
        mà chuỗi bám vị trí dẫn về (0 = chính nó có hai tay; vô cùng = phải dựa vào nhãn MediaPipe)."""
        last = [None, None]; last_t = [None, None]; last_d = [np.inf, np.inf]; dec = {}
        for i in order:
            dets = [j for j in range(2) if label[i, j] >= 0]
            if len(dets) == 2:
                big, small = sorted(dets, key=lambda j: -uv[i, j, 0, 0])
                side_of, dist = ({big: 0, small: 1} if right_on_larger_x else {big: 1, small: 0}), 0.0
            elif len(dets) == 1:
                j = dets[0]; wr = uv[i, j, 0]
                near = [np.linalg.norm(wr - last[s]) if last[s] is not None and abs(i - last_t[s]) <= 20 else np.inf for s in range(2)]
                s_near = int(np.argmin(near))
                if near[s_near] < 0.15 * width and near[1 - s_near] - near[s_near] > 0.05 * width:
                    side_of, dist = {j: s_near}, last_d[s_near] + abs(i - last_t[s_near])      # bám theo tay gần nhất vừa thấy
                elif np.isfinite(near[s_near]) and not np.isfinite(near[1 - s_near]) and near[s_near] > 0.25 * width:
                    side_of, dist = {j: 1 - s_near}, last_d[s_near] + abs(i - last_t[s_near])  # xa tay đã biết → tay còn lại
                else:
                    side_of, dist = {j: 0 if is_right[i, j] else 1}, np.inf
            else:
                continue
            dec[i] = (side_of, dist)
            for j, sd in side_of.items():
                last[sd] = uv[i, j, 0]; last_t[sd] = i; last_d[sd] = dist
        return dec

    fwd, bwd = sweep(range(T)), sweep(range(T - 1, -1, -1)); flips = 0
    for i in fwd:
        side_of = (fwd[i] if fwd[i][1] <= bwd[i][1] else bwd[i])[0]          # lấy chiều quét có chuỗi về frame hai tay ngắn hơn
        for j, sd in side_of.items():
            flips += int(is_right[i, j] != (sd == 0))
            out_uv[i, sd] = uv[i, j]; out_w[i, sd] = world[i, j]
    log(f"gán tay: tay phải ở {'bên phải' if right_on_larger_x else 'bên trái'} ảnh khi thấy cả hai; "
        f"{flips} phát hiện trái nhãn MediaPipe đã đổi bên")
    return out_uv, out_w, flips


# ----------------------------------------------------------------------------- hình học
def fit_plane(pts: np.ndarray, iters=300, thr=0.008, seed=0):
    """RANSAC → (n đơn vị, d) với n·x + d = 0, và tỉ lệ inlier."""
    rng = np.random.default_rng(seed); best = (0, None)
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a); ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln; inl = np.abs((pts - a) @ n) < thr
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    p = pts[best[1]]; c = p.mean(0); n = np.linalg.svd(p - c)[2][-1]
    return n, -float(n @ c), best[0] / len(pts)


def box_samples(size, n=4000, seed=0):
    """Điểm rải đều trên 6 mặt hộp (gốc ở tâm)."""
    rng = np.random.default_rng(seed); h = np.asarray(size, float) / 2
    area = np.array([h[1] * h[2], h[0] * h[2], h[0] * h[1]]); out = []
    for ax in range(3):
        k = int(n * area[ax] / area.sum() / 2) + 1
        for sgn in (-1, 1):
            p = rng.uniform(-1, 1, (k, 3)) * h; p[:, ax] = sgn * h[ax]; out.append(p)
    return np.concatenate(out)


def icp_box(obs, tree, model, R, t, iters=25, trim=0.85):
    """Căn hộp (R, t: hệ hộp → thế giới) vào điểm quan sát (một phần bề mặt). Đối ứng: mỗi điểm quan sát → điểm hộp gần nhất."""
    err = np.inf
    for _ in range(iters):
        loc = (obs - t) @ R                                                  # thế giới → hệ hộp
        d, i = tree.query(loc); keep = d <= np.quantile(d, trim)
        a, b = model[i[keep]], obs[keep]                                     # R a + t ≈ b
        ca, cb = a.mean(0), b.mean(0); u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        s = np.sign(np.linalg.det(vt.T @ u.T)); Rn = vt.T @ np.diag([1, 1, s]) @ u.T
        tn = cb - Rn @ ca; new = float(np.sqrt((d[keep] ** 2).mean()))
        R, t = Rn, tn
        if abs(err - new) < 1e-5:
            err = new; break
        err = new
    return R, t, err


def init_box(obs, size, table_z):
    """Hộp nằm trên bàn ở frame đầu: thử 3 mặt úp xuống × các góc yaw, lấy nghiệm ICP tốt nhất."""
    model = box_samples(size); tree = cKDTree(model); best = None
    c = obs.mean(0); xy = obs[:, :2] - c[:2]; ang = np.arctan2(*np.linalg.svd(xy, full_matrices=False)[2][0][::-1])
    for up in range(3):
        base = np.eye(3)[:, [(up + 1) % 3, (up + 2) % 3, up]]                # trục `up` của hộp → z thế giới
        if np.linalg.det(base) < 0:
            base[:, 0] *= -1
        base = base.T
        for yaw in ang + np.radians([0, 45, 90, 135]):
            R0 = Rot.from_euler("z", yaw).as_matrix() @ base
            t0 = np.array([c[0], c[1], table_z + size[up] / 2])
            R, t, e = icp_box(obs, tree, model, R0, t0, iters=40)
            tilt = np.degrees(np.arccos(np.clip(abs((R @ np.eye(3)[up])[2]), 0, 1)))
            cost = e + 0.002 * tilt + abs((t[2] - size[up] / 2) - table_z)   # hộp đang nằm yên: phẳng và chạm bàn
            if best is None or cost < best[0]:
                best = (cost, R, t, e, up, tilt)
    return best[1], best[2], best[3], best[4], best[5], model, tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--object", default="a box", help="mô tả vật cho Grounding DINO")
    ap.add_argument("--box-size", type=float, nargs=3, required=True, metavar=("DAI", "RONG", "CAO"), help="kích thước hộp đo bằng thước (m)")
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--max-frames", type=int, default=900); ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--moge", default="Ruicheng/moge-2-vitl-normal"); ap.add_argument("--sam2", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--device", default=None); ap.add_argument("--table-z", type=float, default=1.0)
    ap.add_argument("--mirrored", action="store_true", help="video bị lật gương (camera trước/selfie) → đảo nhãn trái/phải của MediaPipe")
    ap.add_argument("--no-box-scale", action="store_true", help="không hiệu chỉnh tỉ lệ mét theo kích thước hộp")
    ap.add_argument("--max-step-deg", type=float, default=20.0)
    a = ap.parse_args()
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.out); run.mkdir(parents=True, exist_ok=True); size = np.asarray(a.box_size, float)

    frames = rp.extract_frames(Path(a.video), run / "frames", a.stride, a.max_frames, a.width)
    meta = json.load(open(run / "frames" / "frames.json")); fps = meta["source_fps"] / meta["stride"]; T = len(frames)
    rp.run_moge(frames, run / "moge", a.moge, dev)
    if not all((run / "sam2" / f"{p.stem}.png").exists() for p in frames):
        rp.run_sam2(run / "frames", frames, run / "sam2", {"object": detect_object(frames[0], a.object, dev)}, a.sam2, dev)
    H, W = cv2.imread(str(frames[0])).shape[:2]
    uv_raw, hw_raw, lab_raw, _ = run_hands(frames, run / "hands2d.npz")
    uv, hw, n_relabel = assign_sides(uv_raw, hw_raw, lab_raw, a.mirrored, W)

    # ---- gom dữ liệu từng frame: điểm hộp, depth nền (lưới thưa), depth tại khớp tay
    obj_pts, zbg, Ks, zhand = [], [], [], np.full((T, 2, 21), np.nan)
    k5 = np.ones((5, 5), np.uint8); k31 = np.ones((31, 31), np.uint8); rng = np.random.default_rng(0)
    ring_pts = None
    for i, p in enumerate(frames):
        d = np.load(run / "moge" / f"{p.stem}.npz"); pts, ok = d["points"], d["mask"] & np.isfinite(d["points"]).all(-1); Ks.append(d["intrinsics"])
        lab = cv2.imread(str(run / "sam2" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        lab = np.zeros((H, W), np.uint8) if lab is None else lab
        hand_m = np.zeros((H, W), np.uint8)
        for s in range(2):
            if np.isfinite(uv[i, s]).all():
                cv2.fillConvexPoly(hand_m, cv2.convexHull(uv[i, s].astype(np.int32)), 1)
                xi = np.clip(uv[i, s, :, 0].round().astype(int), 0, W - 1); yi = np.clip(uv[i, s, :, 1].round().astype(int), 0, H - 1)
                z = pts[yi, xi, 2]; z[~ok[yi, xi]] = np.nan; zhand[i, s] = z
        hand_d = cv2.dilate(hand_m, k31)
        m = cv2.erode((lab == 1).astype(np.uint8), k5).astype(bool) & ok & (hand_d == 0)
        q = pts[m]
        if len(q) > 3000:
            q = q[rng.choice(len(q), 3000, replace=False)]
        obj_pts.append(q.astype(np.float64))
        bg = ok & (cv2.dilate((lab > 0).astype(np.uint8), k31) == 0) & (hand_d == 0)
        zbg.append(np.where(bg, pts[..., 2], np.nan)[::8, ::8].astype(np.float32))
        if i == 0:                                                           # vành quanh hộp ở frame đầu = mặt bàn
            ring = (cv2.dilate((lab == 1).astype(np.uint8), np.ones((max(41, W // 12) | 1,) * 2, np.uint8)) == 1) & (lab == 0) & ok & (hand_d == 0)
            ring_pts = pts[ring].astype(np.float64)
    K = np.median(np.stack(Ks), 0); fx, fy, cx, cy = K[0, 0] * W, K[1, 1] * H, K[0, 2] * W, K[1, 2] * H
    zbg = np.stack(zbg); ref = np.nanmedian(zbg[:: max(1, T // 120)], 0); use = np.isfinite(ref)
    scale = np.ones(T)
    for i in range(T):
        r = (ref / zbg[i])[use]; r = r[np.isfinite(r)]
        if len(r) > 200:
            scale[i] = np.median(r)
    obj_pts = [q * scale[i] for i, q in enumerate(obj_pts)]; zhand = zhand * scale[:, None, None]

    # ---- mặt bàn + hệ thế giới (tạm, chưa hiệu chỉnh tỉ lệ theo hộp)
    assert ring_pts is not None and len(ring_pts) > 500, "không đủ điểm mặt bàn quanh hộp ở frame đầu"
    n, d0, inl = fit_plane(ring_pts[rng.choice(len(ring_pts), min(20000, len(ring_pts)), replace=False)] * scale[0])
    if n @ np.array([0, 0, 1.0]) > 0:                                        # camera ở gốc phải nằm PHÍA TRÊN mặt bàn
        n, d0 = -n, -d0
    assert d0 > 0, "mặt bàn tìm được nằm phía trên camera?"
    ex = np.array([1.0, 0, 0]) - n * n[0]; ex /= np.linalg.norm(ex); ey = np.cross(n, ex)
    Rwc = np.stack([ex, ey, n])                                              # camera → thế giới (chỉ phần xoay)

    def to_world(p, k=1.0, origin=np.zeros(3)):
        q = (p * k) @ Rwc.T; q[..., 2] += d0 * k + a.table_z; return q - origin

    # ---- hệ số mét chung theo kích thước hộp thật: thử k quanh 1, lấy k cho sai số ICP frame đầu nhỏ nhất
    ks = [1.0] if a.no_box_scale else np.linspace(0.75, 1.35, 25); trials = []
    q0 = obj_pts[0]; assert len(q0) > 200, "mask hộp ở frame đầu quá ít điểm"
    for k in ks:
        R, t, e, up, tilt, model, tree = init_box(to_world(q0, k), size, a.table_z)
        trials.append((e / k, k, R, t, up, tilt))                            # e/k: so sánh công bằng giữa các tỉ lệ
    e0, k, R, t, up, tilt = min(trials, key=lambda x: x[0]); e0 *= k
    origin = np.array([t[0], t[1], 0.0]); t = t - origin
    log(f"mặt bàn: inlier {inl:.0%}, camera cao {d0 * k:.2f} m; tỉ lệ mét theo hộp k={k:.3f}; hộp frame đầu: mặt úp trục {up}, "
        f"nghiêng {tilt:.1f}°, ICP rms {e0 * 1000:.1f} mm")
    model = box_samples(size); tree = cKDTree(model)

    # ---- theo dõi hộp
    Rs = np.tile(np.eye(3), (T, 1, 1)); ts = np.zeros((T, 3)); err = np.full(T, np.nan); ok_f = np.zeros(T, bool)
    Rc, tc = R, t; c_prev = None
    for i in range(T):
        q = to_world(obj_pts[i], k, origin)
        if len(q) >= 150:
            c_now = np.median(q, 0)                                          # hộp bị bê đi nhanh hơn ICP hội tụ → dịch theo tâm điểm thấy được
            guess = tc if c_prev is None else tc + (c_now - c_prev)
            c_prev = c_now
            Rn, tn, e = icp_box(q, tree, model, Rc, guess)
            if Rot.from_matrix(Rn @ Rc.T).magnitude() > np.radians(a.max_step_deg):
                Rn, tn, e = Rc, guess, np.nan                                # hộp đối xứng → ICP lật; giữ hướng cũ
            Rc, tc = Rn, tn; err[i] = e; ok_f[i] = np.isfinite(e) and e < 0.012
        Rs[i], ts[i] = Rc, tc
    ok_f[0] = True; vi = np.flatnonzero(ok_f)
    if len(vi) >= 2 and not ok_f.all():
        ti = np.clip(np.arange(T), vi[0], vi[-1])
        ts = np.stack([np.interp(ti, vi, ts[vi, c]) for c in range(3)], 1); Rs = Slerp(vi, Rot.from_matrix(Rs[vi]))(ti).as_matrix()
    w = min(11, T - (1 - T % 2)); w -= 1 - w % 2
    if w >= 5:
        ts = savgol_filter(ts, w, 2, axis=0, mode="nearest")
        rv = Rot.from_matrix(Rs[0]).inv() * Rot.from_matrix(Rs); Rs = (Rot.from_matrix(Rs[0]) * Rot.from_rotvec(savgol_filter(rv.as_rotvec(), w, 2, axis=0, mode="nearest"))).as_matrix()

    # ---- tay 3D: điểm ảnh MediaPipe + depth MoGe. Mỗi khớp: z = depth bề mặt tay (trung vị các khớp có depth tin được) + 8 mm
    # (tâm khớp nằm dưới da) + độ sâu TƯƠNG ĐỐI giữa các khớp lấy từ "world landmarks" của MediaPipe; x, y chiếu ngược từ điểm ảnh.
    # Khớp đúng 2D theo cấu trúc; tỉ lệ mét đi theo depth (đã nhân k cùng hộp) nên tay và hộp cùng một thang đo.
    zw = hw[..., 2]; g_all = np.isfinite(zhand) & np.isfinite(zw)
    za = (zhand - np.nanmedian(np.where(g_all, zhand, np.nan), -1, keepdims=True))[g_all]
    zb = (zw - np.nanmedian(np.where(g_all, zw, np.nan), -1, keepdims=True))[g_all]
    sgn = 1.0 if (not len(za) or np.sum(za * zb) >= 0) else -1.0            # quy ước dấu z của MediaPipe: tự kiểm theo depth
    cam = np.full((T, 2, 21, 3), np.nan)
    for i in range(T):
        for s in range(2):
            if not np.isfinite(uv[i, s]).all():
                continue
            zm = zhand[i, s]
            if not np.isfinite(zm).any():
                continue
            med = np.nanmedian(zm); g = np.isfinite(zm) & (np.abs(zm - med) < 0.06 / k)   # bỏ khớp có depth rơi ra nền / vật xa
            if g.sum() < 6:
                continue
            rel = sgn * (zw[i, s] - np.median(zw[i, s][g])) / k              # world landmarks là mét thật -> chia k về thang MoGe
            z = np.median(zm[g]) + 0.008 / k + rel
            cam[i, s] = np.stack([(uv[i, s, :, 0] - cx) / fx * z, (uv[i, s, :, 1] - cy) / fy * z, z], -1)
    kp = {}; rep_h = {}
    from scipy.ndimage import median_filter
    for s, name in enumerate(("right", "left")):
        kp[name] = to_world(cam[:, s], k, origin)
        vis = np.isfinite(kp[name][:, 0, 0])
        if vis.sum() >= 9:                                                   # bỏ frame cổ tay nhảy xa (depth sai, gán nhầm bên)
            idx = np.flatnonzero(vis); wr = kp[name][idx, 0]
            bad = np.linalg.norm(wr - median_filter(wr, size=(9, 1), mode="nearest"), axis=1) > 0.08
            kp[name][idx[bad]] = np.nan; vis = np.isfinite(kp[name][:, 0, 0])
        bone = np.linalg.norm(kp[name][:, 9] - kp[name][:, 0], axis=-1)      # cổ tay -> gốc ngón giữa, người lớn ~ 8-10 cm
        rep_h[name] = {"visible_frac": round(float(vis.mean()), 3), "detected_frac": round(float(np.isfinite(uv[:, s, 0, 0]).mean()), 3),
                       "wrist_to_middle_mcp_cm_median": round(float(np.nanmedian(bone) * 100), 1) if vis.any() else None,
                       "wrist_above_table_cm_median": round(float(np.nanmedian(kp[name][:, 0, 2]) - a.table_z) * 100, 1) if vis.any() else None}
    rep_h["relabelled_detections"] = int(n_relabel); rep_h["mediapipe_z_sign"] = sgn

    q_wxyz = Rot.from_matrix(Rs).as_quat(scalar_first=True)
    np.savez(run / "recon.npz", fps=fps, kp_right=kp["right"], kp_left=kp["left"], obj_pos=ts, obj_wxyz=q_wxyz, box_size=size, table_z=a.table_z,
             intrinsics=np.array([fx, fy, cx, cy]), cam_to_world_R=Rwc, cam_height=d0 * k, metric_scale=k, origin=origin)
    lift = ts[:, 2] - ts[0, 2]
    rep = {"frames": T, "fps": round(fps, 2), "image": [W, H], "fx_px": round(float(fx), 1), "hfov_deg": round(float(np.degrees(2 * np.arctan(W / 2 / fx))), 1),
           "depth_scale_std": round(float(scale.std()), 4), "table_inlier_frac": round(inl, 3), "camera_height_m": round(d0 * k, 3),
           "metric_scale_from_box": round(float(k), 3), "box_up_axis": int(up), "box_tilt_deg_frame0": round(float(tilt), 1),
           "icp_rms_mm_median": round(float(np.nanmedian(err) * 1000), 1), "object_frames_valid_frac": round(float(ok_f.mean()), 3),
           "object_lift_max_cm": round(float(lift.max() * 100), 1), "object_travel_m": round(float(np.linalg.norm(ts.max(0) - ts.min(0))), 3),
           "object_bottom_minus_table_cm_frame0": round(float(ts[0, 2] - size[up] / 2 - a.table_z) * 100, 2), "hands": rep_h}
    json.dump(rep, open(run / "report.json", "w"), indent=1); print(json.dumps(rep, indent=1))

    # ---- overlay kiểm tra bằng mắt: khung hộp (vàng) + khớp tay (phải đỏ, trái xanh), chiếu ngược về ảnh
    def project(pw):
        pc = ((pw + origin) - np.array([0, 0, d0 * k + a.table_z])) @ Rwc / k
        return np.stack([fx * pc[..., 0] / pc[..., 2] + cx, fy * pc[..., 1] / pc[..., 2] + cy], -1)

    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size / 2
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(corners[i] != corners[j]) == 1]
    vw = cv2.VideoWriter(str(run / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i, p in enumerate(frames):
        img = cv2.imread(str(p)); c2 = project(corners @ Rs[i].T + ts[i])
        for u, v in edges:
            cv2.line(img, tuple(int(x) for x in c2[u]), tuple(int(x) for x in c2[v]), (0, 220, 255) if ok_f[i] else (128, 128, 128), 2)
        for name, col in (("right", (0, 0, 255)), ("left", (255, 128, 0))):
            if np.isfinite(kp[name][i]).all():
                h2 = project(kp[name][i])
                for u, v in HAND_EDGES:
                    cv2.line(img, tuple(int(x) for x in h2[u]), tuple(int(x) for x in h2[v]), col, 2)
        cv2.putText(img, f"{i} z_box={ts[i, 2] - a.table_z:+.3f} m", (10, 28), 0, 0.8, (255, 255, 255), 2); vw.write(img)
    vw.release(); log(f"→ {run / 'recon.npz'}, {run / 'overlay.mp4'}")


if __name__ == "__main__":
    main()
