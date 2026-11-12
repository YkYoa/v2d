#!/usr/bin/env python3
"""Xem point cloud 4D (người + vật + keypoint 3D) của một run M1 bằng viser.
    python demo_track1/viewer.py --run outputs/ep09   → mở http://localhost:8080
"""
import argparse, time
from pathlib import Path
import numpy as np
import viser


def read_ply(p: Path):
    with open(p, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        n = int([l for l in hdr.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        rec = np.frombuffer(f.read(), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)], count=n)
    return rec["xyz"].copy(), rec["rgb"].copy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(); run = Path(a.run)
    kp = np.load(run / "keypoints3d.npz"); frames = list(kp["frames"]); body, hand = kp["body"], kp["hand"]
    srv = viser.ViserServer(port=a.port)
    # MoGe camera frame: x phải, y xuống, z tới → đổi sang z lên cho viser
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)
    slider = srv.gui.add_slider("frame", 0, len(frames) - 1, 1, 0)
    play = srv.gui.add_checkbox("play", True)
    nodes = {}

    def show(i):
        stem = frames[i]
        for name, size in (("person", 0.01), ("object", 0.01)):
            p = run / "points" / f"{stem}_{name}.ply"
            if p.exists():
                xyz, rgb = read_ply(p)
                nodes[name] = srv.scene.add_point_cloud(f"/{name}", xyz @ R.T, rgb, point_size=size)
        b = body[i]; ok = np.isfinite(b).all(1)
        if ok.any():
            nodes["body"] = srv.scene.add_point_cloud("/body_kp", b[ok] @ R.T, np.tile([[0, 255, 0]], (ok.sum(), 1)), point_size=0.03)
        h = hand[i].reshape(-1, 3); okh = np.isfinite(h).all(1)
        if okh.any():
            nodes["hand"] = srv.scene.add_point_cloud("/hand_kp", h[okh] @ R.T, np.tile([[255, 0, 255]], (okh.sum(), 1)), point_size=0.02)

    @slider.on_update
    def _(_): show(slider.value)

    show(0); i = 0
    print(f"viser: http://localhost:{a.port}")
    while True:
        time.sleep(0.25)
        if play.value:
            i = (i + 1) % len(frames); slider.value = i


if __name__ == "__main__":
    main()
