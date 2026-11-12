#!/usr/bin/env python3
"""M1 — Demo perception Track 1 trên laptop (RTX 4050 6 GB).

Video góc thứ 3 (episode Track 1) → mỗi frame: depth metric + intrinsics (MoGe-2),
mask người/vật (Grounding DINO → SAM 2.1 video), keypoint người/tay 2D (MediaPipe Tasks)
nâng lên 3D bằng depth → point cloud .ply tách người/vật + overlay.mp4 + keypoints3d.npz.

Các stage ghi ra đĩa và tự bỏ qua nếu đã có kết quả (cache theo kiểu repo chính thức).

    python demo_track1/run_perception.py --episode 9 [--max-frames 60] [--stride 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2d_challenge" / "track_1"
MODELS = ROOT / "data" / "models"


# ----------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_mem_mb() -> float:
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def episode_meta(ep: int) -> dict:
    path = DATA / "meta" / "episodes_metadata.jsonl"
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if int(d.get("episode_index", -1)) == ep:
                return d
    raise SystemExit(f"episode {ep} không có trong {path}")


def download(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"tải {url} → {dst}")
        urllib.request.urlretrieve(url, dst)
    return dst


# ----------------------------------------------------------------------------- stage 1: frames
def extract_frames(video: Path, out: Path, stride: int, max_frames: int, width: int) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.jpg"))
    if existing and len(existing) >= min(max_frames, 1):
        log(f"frames: dùng lại {len(existing)} frame trong {out}")
        return existing
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"không mở được video {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"video {video.name}: {n_total} frame @ {fps:.2f} fps, "
        f"{int(cap.get(3))}x{int(cap.get(4))}")
    paths, i, kept = [], 0, 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            p = out / f"{i:06d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(p)
            kept += 1
        i += 1
    cap.release()
    json.dump({"source_fps": fps, "stride": stride, "n_total": n_total}, open(out / "frames.json", "w"))
    log(f"frames: trích {len(paths)} frame → {out}")
    return paths


# ----------------------------------------------------------------------------- stage 2: MoGe-2
def run_moge(frames: list[Path], out: Path, model_name: str, device: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in frames if not (out / f"{p.stem}.npz").exists()]
    if not todo:
        log("moge: đã có depth cho mọi frame, bỏ qua")
        return
    import torch
    from moge.model.v2 import MoGeModel

    log(f"moge: load {model_name} trên {device}")
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    dtype = torch.float16 if device == "cuda" else torch.float32
    for p in todo:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        t = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"):
            o = model.infer(t)
        depth = o["depth"].float().cpu().numpy()
        mask = o["mask"].cpu().numpy() if "mask" in o else np.isfinite(depth)
        K = o["intrinsics"].float().cpu().numpy()  # normalized [0,1] intrinsics
        points = o["points"].float().cpu().numpy()  # H,W,3 metric, camera frame
        np.savez_compressed(out / f"{p.stem}.npz", depth=depth.astype(np.float32),
                            mask=mask.astype(bool), intrinsics=K.astype(np.float32),
                            points=points.astype(np.float32))
        vis = depth.copy(); vis[~mask] = np.nan
        lo, hi = np.nanpercentile(vis, 2), np.nanpercentile(vis, 98)
        vis8 = np.nan_to_num((np.clip(vis, lo, hi) - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{p.stem}_depth.png"), cv2.applyColorMap(vis8, cv2.COLORMAP_TURBO))
    log(f"moge: xong {len(todo)} frame, VRAM đỉnh {gpu_mem_mb():.0f} MB")
    del model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stage 3: detect + SAM2
GENERIC = {("desk", "table", "cart", "stool", "chair", "shelf"): "furniture",
           ("bowl", "pan", "cup", "pot", "lid", "plate"): "kitchenware",
           ("hoop", "roller", "roll", "block", "iron", "box"): "object"}


def generic_words(name: str) -> list[str]:
    n = name.lower()
    return [g for keys, g in GENERIC.items() if any(k in n for k in keys)] or ["object"]


def detect_boxes(frame: Path, prompts: list[str], device: str, points: np.ndarray | None = None,
                 wrist: np.ndarray | None = None, thr: float = 0.25) -> dict[str, list[float] | None]:
    """Grounding DINO tiny → box [x0,y0,x1,y1] cho mỗi prompt.
    prompts[0] = người (lấy score cao nhất). Các prompt sau = vật: trong số ứng viên, chọn box có
    độ sâu (median z từ MoGe point map) gần người nhất — tránh chọn nhầm vật cùng loại ở phía sau."""
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
    img = Image.open(frame).convert("RGB")
    text = " . ".join(p.rstrip(".") for p in prompts) + " ."
    inputs = proc(images=img, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=thr, text_threshold=thr, target_sizes=[img.size[::-1]])[0]
    labels = [str(l) for l in res.get("text_labels", res.get("labels"))]
    cands: dict[str, list[tuple[float, list[float]]]] = {p: [] for p in prompts}
    for box, score, label in zip(res["boxes"], res["scores"], labels):
        for p in prompts:
            words = [w for w in p.lower().rstrip(".").split() if len(w) > 2 and w not in ("the", "and", "with")]
            if any(w in label for w in words) or label in p.lower():
                cands[p].append((float(score), [float(v) for v in box.tolist()]))
    del model
    torch.cuda.empty_cache()

    def box_depth(b):
        if points is None:
            return float("nan")
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        z = points[max(0, y0):y1, max(0, x0):x1, 2]
        z = z[np.isfinite(z)]
        return float(np.median(z)) if z.size else float("nan")

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy; ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    boxes: dict[str, list[float] | None] = {p: None for p in prompts}
    person = max(cands[prompts[0]], key=lambda t: t[0])[1] if cands[prompts[0]] else None
    boxes[prompts[0]] = person
    zp = box_depth(person) if person else float("nan")

    def cost(b, sc):
        # vật đang được thao tác: cùng độ sâu với người + gần cổ tay; score DINO chỉ là phụ
        c = abs(box_depth(b) - zp) if np.isfinite(zp) else 0.0
        if wrist is not None and np.isfinite(wrist).all():
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            c += 0.004 * float(np.hypot(cx - wrist[0], cy - wrist[1]))  # 250 px ≈ 1 m
        return c - 0.2 * sc

    pool = [(p, sc, b) for p in prompts[1:] for sc, b in cands[p] if person is None or iou(b, person) < 0.8]
    if pool:
        pool.sort(key=lambda t: cost(t[2], t[1]))
        p, sc, b = pool[0]
        boxes["object"] = b
        log(f"detect vật: {len(pool)} ứng viên; chọn '{p}' score={sc:.2f} z={box_depth(b):.2f} m "
            f"(người z={zp:.2f} m); kế tiếp: {[(q, round(box_depth(bb), 2)) for q, _, bb in pool[1:4]]}")
    return boxes


def run_sam2(frames_dir: Path, frames: list[Path], out: Path, boxes: dict[str, list[float] | None],
             model_name: str, device: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if all((out / f"{p.stem}.png").exists() for p in frames):
        log("sam2: đã có mask cho mọi frame, bỏ qua")
        return
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    log(f"sam2: load {model_name}")
    predictor = SAM2VideoPredictor.from_pretrained(model_name, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        state = predictor.init_state(video_path=str(frames_dir), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True, async_loading_frames=False)
        obj_ids = {}
        for oid, (name, box) in enumerate(boxes.items(), start=1):
            if box is None:
                log(f"sam2: không có box cho '{name}', bỏ qua")
                continue
            obj_ids[oid] = name
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=np.array(box, dtype=np.float32))
        for fidx, oids, logits in predictor.propagate_in_video(state):
            label = np.zeros(logits.shape[-2:], dtype=np.uint8)
            for k, oid in enumerate(oids):
                m = (logits[k, 0] > 0).cpu().numpy()
                label[m] = oid  # 1 = người, 2 = vật (theo thứ tự boxes)
            cv2.imwrite(str(out / f"{frames[fidx].stem}.png"), label)
    json.dump({"obj_ids": obj_ids, "boxes": boxes}, open(out / "objects.json", "w"), indent=1)
    log(f"sam2: xong, VRAM đỉnh {gpu_mem_mb():.0f} MB")
    del predictor
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stage 4: MediaPipe
MP_POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MP_HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def run_mediapipe(frames: list[Path], out: Path, crop: int = 192) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "keypoints2d.npz").exists():
        log("mediapipe: đã có keypoints2d, bỏ qua")
        return
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    pose_model = download(MP_POSE_URL, MODELS / "mediapipe" / "pose_landmarker_heavy.task")
    hand_model = download(MP_HAND_URL, MODELS / "mediapipe" / "hand_landmarker.task")
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.3))
    body = np.full((len(frames), 33, 3), np.nan, np.float32)   # x,y px, visibility
    hand = np.full((len(frames), 2, 21, 2), np.nan, np.float32)  # [frame, L/R, 21, xy px]
    for i, p in enumerate(frames):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        pr = pose.detect_for_video(mimg, int(i * 1000 / 6))
        if not pr.pose_landmarks:
            continue
        for j, lm in enumerate(pr.pose_landmarks[0]):
            body[i, j] = (lm.x * w, lm.y * h, lm.visibility)
        # tay: người nhỏ trong khung → crop quanh cổ tay (pose idx 15 = trái, 16 = phải) rồi detect
        for side, widx in ((0, 15), (1, 16)):
            wx, wy, vis = body[i, widx]
            if vis < 0.3:
                continue
            x0, y0 = int(max(0, wx - crop // 2)), int(max(0, wy - crop // 2))
            x1, y1 = int(min(w, wx + crop // 2)), int(min(h, wy + crop // 2))
            sub = np.ascontiguousarray(img[y0:y1, x0:x1])
            if sub.size == 0:
                continue
            hr = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=sub))
            if hr.hand_landmarks:
                for j, lm in enumerate(hr.hand_landmarks[0]):
                    hand[i, side, j] = (x0 + lm.x * sub.shape[1], y0 + lm.y * sub.shape[0])
    pose.close(); hands.close()
    np.savez_compressed(out / "keypoints2d.npz", body=body, hand=hand)
    log(f"mediapipe: body {np.isfinite(body[:, :, 0]).any(1).sum()}/{len(frames)} frame, "
        f"hand {np.isfinite(hand[:, :, 0, 0]).any(1).sum()}/{len(frames)} frame")


# ----------------------------------------------------------------------------- stage 5: fuse + outputs
def lift(xy: np.ndarray, points: np.ndarray) -> np.ndarray:
    """xy [N,2] px → 3D từ point map MoGe (lấy điểm gần nhất hợp lệ trong cửa sổ 5px)."""
    H, W = points.shape[:2]
    out = np.full((len(xy), 3), np.nan, np.float32)
    for k, (x, y) in enumerate(xy):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        win = points[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3].reshape(-1, 3)
        win = win[np.isfinite(win).all(1)]
        if len(win):
            out[k] = np.median(win, axis=0)
    return out


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        rec = np.empty(len(xyz), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        rec["xyz"] = xyz; rec["rgb"] = rgb
        f.write(rec.tobytes())


POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]


def fuse(frames: list[Path], run: Path, fps_out: float) -> dict:
    pts_dir = run / "points"; pts_dir.mkdir(exist_ok=True)
    kp = np.load(run / "mediapipe" / "keypoints2d.npz")
    body2d, hand2d = kp["body"], kp["hand"]
    body3d = np.full((len(frames), 33, 3), np.nan, np.float32)
    hand3d = np.full((len(frames), 2, 21, 3), np.nan, np.float32)
    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    writer = cv2.VideoWriter(str(run / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w0, h0))
    stats = {"person_depth_m": [], "object_depth_m": [], "n_person_pts": [], "n_object_pts": []}
    for i, p in enumerate(frames):
        img = cv2.imread(str(p))
        d = np.load(run / "moge" / f"{p.stem}.npz")
        points, valid = d["points"], d["mask"]
        mpath = run / "sam2" / f"{p.stem}.png"
        label = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) if mpath.exists() else np.zeros(valid.shape, np.uint8)
        if label.shape != valid.shape:
            label = cv2.resize(label, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for lid, name in ((1, "person"), (2, "object")):
            m = valid & (label == lid)
            xyz = points[m]; col = rgb[m]
            if len(xyz) > 60000:
                sel = np.random.default_rng(0).choice(len(xyz), 60000, replace=False); xyz, col = xyz[sel], col[sel]
            write_ply(pts_dir / f"{p.stem}_{name}.ply", xyz, col)
            stats[f"n_{name}_pts"].append(int(len(xyz)))
            stats[f"{name}_depth_m"].append(float(np.median(xyz[:, 2])) if len(xyz) else float("nan"))
        body3d[i] = lift(body2d[i, :, :2], points)
        for s in range(2):
            hand3d[i, s] = lift(hand2d[i, s], points)
        # overlay
        ov = img.copy()
        color = np.zeros_like(img); color[label == 1] = (0, 200, 255); color[label == 2] = (255, 120, 0)
        ov = cv2.addWeighted(ov, 1.0, color, 0.45, 0)
        for a, b in POSE_EDGES:
            pa, pb = body2d[i, a], body2d[i, b]
            if np.isfinite(pa[0]) and np.isfinite(pb[0]) and pa[2] > 0.3 and pb[2] > 0.3:
                cv2.line(ov, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 255, 0), 2)
        for s in range(2):
            for j in range(21):
                q = hand2d[i, s, j]
                if np.isfinite(q[0]):
                    cv2.circle(ov, (int(q[0]), int(q[1])), 2, (255, 0, 255), -1)
        dep = stats["object_depth_m"][-1]
        cv2.putText(ov, f"frame {p.stem}  object z={dep:.2f}m", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(ov)
        cv2.imwrite(str(run / "overlay" / f"{p.stem}.jpg"), ov) if (run / "overlay").exists() else None
    writer.release()
    np.savez_compressed(run / "keypoints3d.npz", body=body3d, hand=hand3d,
                        frames=np.array([p.stem for p in frames]))
    summary = {k: (float(np.nanmedian(v)) if "depth" in k else int(np.median(v))) for k, v in stats.items()}
    return summary


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=9)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--moge", default="Ruicheng/moge-2-vitb-normal")
    ap.add_argument("--sam2", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        log("CẢNH BÁO: không có CUDA — chạy CPU sẽ rất chậm (xem M0.1 trong PLAN.md)")

    meta = episode_meta(args.episode)
    prompt = meta.get("object_prompt") or meta.get("target_object", "object")
    video = DATA / "videos" / "chunk-000" / "observation.images.exo_camera" / f"episode_{args.episode:06d}.mp4"
    run = Path(args.out) if args.out else ROOT / "outputs" / f"ep{args.episode:02d}"
    run.mkdir(parents=True, exist_ok=True); (run / "overlay").mkdir(exist_ok=True)
    log(f"episode {args.episode}: '{meta.get('sequence_id')}' prompt='{prompt}' → {run}")
    t0 = time.time()

    frames = extract_frames(video, run / "frames", args.stride, args.max_frames, args.width)
    run_moge(frames, run / "moge", args.moge, device)
    run_mediapipe(frames, run / "mediapipe")
    boxes_path = run / "sam2" / "objects.json"
    if boxes_path.exists():
        boxes = json.load(open(boxes_path))["boxes"]
    else:
        d0 = np.load(run / "moge" / f"{frames[0].stem}.npz")
        body0 = np.load(run / "mediapipe" / "keypoints2d.npz")["body"][0]
        wrist0 = np.nanmean(body0[[15, 16], :2], axis=0)
        obj_name = str(meta.get("object", "")).replace("_", " ")
        obj_prompts = [prompt] + ([obj_name] if obj_name and obj_name not in prompt.lower() else []) + generic_words(obj_name or prompt)
        det = detect_boxes(frames[0], ["a person"] + obj_prompts, device, points=d0["points"], wrist=wrist0)
        boxes = {"a person": det["a person"], prompt: det.get("object")}
        log(f"grounding-dino boxes: {boxes}")
    run_sam2(run / "frames", frames, run / "sam2", boxes, args.sam2, device)
    fps_src = json.load(open(run / "frames" / "frames.json"))["source_fps"]
    summary = fuse(frames, run, max(fps_src / args.stride, 1.0))
    summary.update(episode=args.episode, prompt=prompt, n_frames=len(frames),
                   seconds=round(time.time() - t0, 1), device=device, gpu_peak_mb=round(gpu_mem_mb()))
    json.dump(summary, open(run / "summary.json", "w"), indent=1)
    log(f"XONG {summary}")
    log(f"xem: python demo_track1/viewer.py --run {run}")


if __name__ == "__main__":
    main()
