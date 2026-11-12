#!/usr/bin/env python3
"""M3 — Gộp kết quả fit MHR + vật thành NPZ/mesh đúng chuẩn kit, đóng gói bằng packer CHÍNH THỨC, rồi đọc ngược để kiểm.

  python pack_episode.py --run outputs/ep09 --episode 9 --kit kit/v2d_submission_kit --model checkpoints/mhr/mhr_model.pt \
      [--sample data/track_1_sample_submission.parquet] [--commit https://github.com/<owner>/<repo>/commit/<sha>]

Không có `--sample` (file thật chỉ tải được từ Kaggle): tự sinh sample GIẢ LẬP cho đúng 1 episode, chấm mọi frame,
ngân sách mesh 4096 mặt / 4096 đỉnh (suy từ 740 780 dòng = 9 877 frame × 50 + 30 episode × 8 231 của các đội đã nộp).
File tạo ra khi đó tên `..._LOCAL_CHECK_do_not_upload.parquet` — hợp lệ về format, KHÔNG nộp được.
"""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

FACES_BUDGET = VERTICES_BUDGET = 4096


def synthetic_sample(ms, episode: int, frames: range, out: Path) -> None:
    order = ms.array_order(); rows = []
    per_frame = {"mhr_pose": 46, "object_rotation": 3, "object_translation": 1}
    per_episode = {"mhr_scales": 23, "mhr_shape": 15, "object_scale": 1,
                   "object_mesh_vertices": VERTICES_BUDGET, "object_mesh_faces": FACES_BUDGET}
    for f in frames:
        for name, n in per_frame.items():
            rows += [ms.row_id(1, episode, f, order.index(name), p) for p in range(n)]
    for name, n in per_episode.items():
        rows += [ms.row_id(1, episode, ms.EPISODE_FRAME, order.index(name), p) for p in range(n)]
    pd.DataFrame({"row_id": rows, "x": 0.0, "y": 0.0, "z": 0.0}).to_parquet(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--kit", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--fit", default=None); ap.add_argument("--sample", default=None)
    ap.add_argument("--commit", default="https://github.com/local/check/commit/" + "0" * 40)
    a = ap.parse_args()
    run, kit = Path(a.run), Path(a.kit).resolve()
    sys.path.insert(0, str(kit))
    from v2dlb import mhr_submission as ms

    fit = np.load(a.fit or run / "mhr_fit.npz"); trk = np.load(run / "object_track.npz")
    T = min(len(fit["pose"]), len(trk["object_rotation"]))
    ep_dir = run / "track1_episodes"; ep_dir.mkdir(exist_ok=True)
    stem = f"episode_{a.episode:06d}"
    np.savez(ep_dir / f"{stem}.npz", pose=fit["pose"][:T], scales=fit["scales"], shape=fit["shape"],
             object_rotation=trk["object_rotation"][:T], object_translation=trk["object_translation"][:T],
             object_scale=trk["object_scale"])
    shutil.copy(run / "object.glb", ep_dir / f"{stem}_object.glb")

    local = a.sample is None
    sample = Path(a.sample) if a.sample else run / "track_1_sample_SYNTHETIC.parquet"
    if local:
        synthetic_sample(ms, a.episode, range(T), sample)
    out = run / (f"track1_ep{a.episode:02d}" + ("_LOCAL_CHECK_do_not_upload" if local else "") + ".parquet")
    cmd = [sys.executable, "eval_reconstruction.py", "--episodes", str(ep_dir.resolve()), "--sample", str(sample.resolve()),
           "--commit", a.commit, "--out", str(out.resolve())]
    print("$", " ".join(cmd)); subprocess.run(cmd, cwd=kit, check=True)

    # ---- đọc ngược bằng chính code của kit rồi so với dữ liệu gốc
    df = pd.read_parquet(out)
    assert list(df.columns) == ["row_id", "x", "y", "z", "code_commit_url"], df.columns
    parts = df["row_id"].str.split("_", expand=True); order = ms.array_order()
    df["episode_index"] = parts[1].astype(int); df["frame_index"] = parts[2].astype(int)
    df["array_name"] = parts[3].astype(int).map(lambda i: order[i]); df["point_index"] = parts[4].astype(int)
    raw = ms._to_arrays(df[df["episode_index"] == a.episode], ["x", "y", "z"])
    pose = ms._unpack(raw["mhr_pose"], 136, "mhr_pose"); scales = ms._unpack(raw["mhr_scales"], 68, "mhr_scales")
    shape = ms._unpack(raw["mhr_shape"], 45, "mhr_shape"); rot = ms._rotations(raw["object_rotation"])
    trans = raw["object_translation"][:, 0, :]; scale = ms._object_scale(raw["object_scale"])
    from v2dlb.mesh_common import _mesh_faces
    faces = _mesh_faces(raw["object_mesh_faces"], len(raw["object_mesh_vertices"]))
    frames = np.unique(df.loc[df["frame_index"] != ms.EPISODE_FRAME, "frame_index"])
    checks = {
        "rows": len(df), "scored_frames": len(frames),
        "pose_roundtrip_max_abs": float(np.abs(pose - fit["pose"][frames]).max()),
        "rotation_roundtrip_max_abs": float(np.abs(rot - trk["object_rotation"][frames]).max()),
        "translation_roundtrip_max_abs": float(np.abs(trans - trk["object_translation"][frames]).max()),
        "rotation_det_min": float(np.linalg.det(rot).min()), "object_scale": scale,
        "mesh_vertices_rows": int(len(raw["object_mesh_vertices"])), "mesh_faces_rows": int(len(faces)),
        "mesh_faces_nondegenerate": int((np.ptp(faces, axis=1) > 0).sum()),
        "all_finite": bool(np.isfinite(df[["x", "y", "z"]].to_numpy()).all()),
        "commit_url_unique": df["code_commit_url"].nunique() == 1, "file_mb": round(out.stat().st_size / 1e6, 2),
        "local_check_only": local,
    }

    # ---- dựng lại thân MHR từ tham số ĐÃ ĐÓNG GÓI (như host sẽ làm) + mesh vật, chiếu lên ảnh để kiểm bằng mắt
    import cv2, torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(a.model, map_location=dev).float().eval()
    pick = np.linspace(0, len(frames) - 1, 6).astype(int)
    with torch.no_grad():
        p = torch.tensor(np.concatenate([pose[pick], np.broadcast_to(scales, (len(pick), 68))], 1), dtype=torch.float32, device=dev)
        v, sk = model(torch.tensor(shape, dtype=torch.float32, device=dev)[None].expand(len(pick), -1).contiguous(), p,
                      torch.zeros(len(pick), 72, device=dev), True)
    verts = (v.cpu().numpy() * np.array([1, -1, -1]) / 100.0)             # hệ Track 1 (mét) = hệ camera
    checks["mhr_vertices"] = int(verts.shape[1]); checks["mhr_joints"] = int(sk.shape[1])
    checks["mhr_finite"] = bool(np.isfinite(verts).all())
    fx, fy, cx, cy = fit["intrinsics"]; jpgs = sorted((run / "frames").glob("*.jpg")); tiles = []
    mv = raw["object_mesh_vertices"]
    for k, i in enumerate(pick):
        f = int(frames[i]); im = cv2.imread(str(jpgs[f]))
        obj = scale * mv @ rot[i].T + trans[i]
        for pts, color, stepv in ((obj, (0, 200, 255), 2), (verts[k], (80, 255, 80), 6)):
            q = pts[::stepv]; q = q[q[:, 2] > 0.2]
            u = (fx * q[:, 0] / q[:, 2] + cx).astype(int); w_ = (fy * q[:, 1] / q[:, 2] + cy).astype(int)
            ok = (u >= 0) & (u < im.shape[1]) & (w_ >= 0) & (w_ < im.shape[0])
            im[w_[ok], u[ok]] = color
        cv2.putText(im, f"frame {f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2); tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    png = out.with_suffix(".check.jpg"); cv2.imwrite(str(png), sheet)
    print("checks:", json.dumps(checks, indent=1))
    ok = (checks["pose_roundtrip_max_abs"] < 1e-5 and checks["rotation_roundtrip_max_abs"] < 1e-5 and checks["all_finite"]
          and checks["mhr_finite"] and checks["mesh_faces_nondegenerate"] > 0 and checks["rotation_det_min"] > 0.99)
    json.dump(checks, open(out.with_suffix(".checks.json"), "w"), indent=1)
    print(("FORMAT OK" if ok else "FORMAT FAIL"), "→", out, "| ảnh kiểm:", png)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
