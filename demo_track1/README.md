# M1 — Demo perception Track 1 (laptop)

```bash
source .venv/bin/activate
python demo_track1/run_perception.py --episode 9          # ~60 frame, MoGe-2 ViT-B + SAM2.1 small + MediaPipe
python demo_track1/viewer.py --run outputs/ep09            # http://localhost:8080
```
Output `outputs/ep09/`: `frames/`, `moge/*.npz|_depth.png`, `sam2/*.png` (1=người, 2=vật), `mediapipe/keypoints2d.npz`,
`points/*_person.ply|*_object.ply`, `keypoints3d.npz`, `overlay.mp4`, `summary.json`.

Trên server 4090: `--moge Ruicheng/moge-2-vitl-normal --sam2 facebook/sam2.1-hiera-large --max-frames 600 --stride 1`.
