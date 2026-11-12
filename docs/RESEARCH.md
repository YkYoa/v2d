# V2D Challenge — Tài liệu thu thập (cập nhật 2026-11-12)

## 1. Challenge là gì
NVIDIA **Video to Data (V2D) Challenge** (CoRL 2026): đo xem "pipeline từ video người → dữ liệu robot có thể chạy trong sim" gãy ở đâu. 3 track dùng chung bộ task:

| Track | Input → Output | Metric (kit `metric_code/`) | Baseline |
|---|---|---|---|
| 1 Reconstruction | 1 video RGB góc thứ 3 → MHR người (pose 136/frame, scales 68, shape 45) + mesh vật + Sim(3) pose/frame | CD-H, CD-O, ACC-H, ACC-O, PEN (thấp = tốt) | CARI4D |
| 2 Robotic grounding | quỹ đạo 4D (Tier 1 sạch / 2 nhiễu / 3 tự tái dựng) → policy G1+Dex3 (whole-body) | AUC, SP-SR, MP-SR, MPPE | CHORD |
| 3 Egocentric e2e | video đội đầu (1 màu + stereo) → policy (Sharpa Wave mặc định) + mesh vật tự tái dựng | AUC, SP-SR, MP-SR, RPE, MPPE, CD-O | tissue-box e2e |

- Timeline: mở 21/9/2026, **leaderboard đóng 4/11/2026 17:00 EST**, công bố winner 12/11 tại CoRL tutorial. Giải: DGX Spark + GPU credit, tổng $54K.
- Simulation-only, không cần robot thật (FAQ). Pretrained model & dataset ngoài được phép. Code winner phải mở Apache-2.0.
- Xếp hạng = tổng thứ hạng từng metric (thấp nhất thắng).

## 2. Link chính thức
- Trang challenge + leaderboard: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- FAQ: `docs/kit/v2d_challenge_2026_faqs.pdf` (bản gốc: .../assets/v2d_challenge_2026_faqs.pdf)
- Submission kit (đã giải nén `docs/kit/kit/v2d_submission_kit/`): packer `tools/pack_reconstruction.py`, `tools/pack_submission.py`, uploader `submit/v2d_submit.py`, metric code 3 track, evaluator FlashCHORD pin commit `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`, agent skill `skills/v2d-submit/SKILL.md`, converter `tools/track1/mesh_to_mhr_params.py`.
- Repo chính: https://github.com/nvidia-isaac/video_to_data
  - `reconstruction/` 18 module Docker (moge, sam2, sam3d, sam3d_body, grounding_dino, foundation_stereo, foundation_pose, hoi_object_reconstruction, ego_hand_reconstruction, cusfm, bundlesdf, detectron2, unidepth, rosbag, mv_*). Mỗi module: `pip install -e modules/v2d_<name>/docker` + `python -m v2d.<name>.docker.build`.
  - `robotic_grounding/` Isaac Lab 2.3.x + RSL-RL PPO; `docs/SETUP.md` (MANO, TACO/HOT3D/ARCTIC...), `run_example_sequences.sh`.
  - `robotic_grounding/flash_chord/` Newton/Warp/JAX + PPO/FlashSAC, **không cần Isaac Sim**; `scripts/debug/replay.py`, `train_flash_sac.py`, `evaluate_policy.py`.
  - `docs/e2e_workflow.md`: init → reconstruct → retarget → train-expert → collect → finetune GR00T N1.7 → evaluate. Sim-to-real: 4/5 thành công với 10 episode thật.
  - Yêu cầu: GPU 48 GB (A6000/L40/L40S) cho full; driver 580 / CUDA 13; **không hỗ trợ Blackwell** (TensorRT/cuVSLAM).
- Dataset: https://huggingface.co/datasets/nvidia/video_to_data_challenge (CC-BY-4.0, 4.14 GB)
  - `track_1/`: 30 video `videos/chunk-000/observation.images.exo_camera/episode_0000NN.mp4` 1536×1152 30 fps, ~524 MB; `meta/episodes_metadata.jsonl` có `object_prompt`. 10 vật: hula_hoop, big_red_bowl, iron, white_desk, black_pan, foam_grass_block, paint_roller, pink_foam_roll, short_wood_stool, white_laptop_cart. **Không có GT.**
  - `track_2/`: 30 episode LeRobot v2.1, tier1 (SOMA-X sạch) / tier2 (nhiễu); video exo giống hệt nhau.
  - `track_3/public/`: 20 episode, `data/` parquet có `observation.objects` (name, pose [x,y,z,qw,qx,qy,qz], visible) = **GT pose vật → chấm local được**; `mesh/`, `urdf/` 10 vật; cam `ego_cam_a` 2028×1520 màu, `ego_cam_b/c` 1280×800 stereo mono; 20.25 fps.
- HF collection: https://huggingface.co/collections/nvidia/video-to-data (task library + dataset, object assets, form-hoi). Checkpoint mẫu Isaac Lab: `nvidia/video-to-data-robotic-grounding-example`.
- CoRL tutorial: https://nvidia-isaac.github.io/video_to_data/v2d_corl_tutorial/ ; video https://www.youtube.com/watch?v=U8GHIxXqHLQ
- Leaderboard JSON: https://github.com/MVerghese/v2d-leaderboard-data (9/11: T1 59 team, top CD-H 11.4 / CD-O 16.3, baseline CARI4D 15.6 / 23.2; T2-T1 28 team, CHORD baseline AUC 0.51; T2-T3 CHORD(CARI4D) AUC 0.002; T3 21 team, top AUC 0.985).
- Kaggle: 23 competition, xem `docs/kit/kit/v2d_submission_kit/LEADERBOARDS.md`.
- Liên hệ: v2d_challenge@nvidia.com, Discord Omniverse #v2d-challenge, forum https://forums.developer.nvidia.com/t/384050

## 3. Paper / model nền
| Tên | Vai trò | Link |
|---|---|---|
| CARI4D (CVPR 2026) | baseline Track 1, 4D HOI từ video đơn | https://arxiv.org/abs/2512.11988 · https://github.com/NVlabs/CARI4D (Docker `xiexh20/cari4d`, weights HF `nvidia/CARI4D`, cần SMPL-H) |
| CHORD (CoRL 2026) | baseline Track 2/3, contact-wrench reward | https://arxiv.org/abs/2607.00033 · https://nvidia-isaac.github.io/video_to_data/chord/ |
| MoGe-2 | depth metric + intrinsics từ 1 ảnh | https://github.com/microsoft/MoGe (ViT-S/B/L; có ONNX `Ruicheng/moge-2-vit*-normal-onnx`) |
| SAM 2.1 | mask video | https://github.com/facebookresearch/sam2 |
| Grounding DINO | detect theo text prompt | https://github.com/IDEA-Research/GroundingDINO |
| FoundationPose | 6-DoF tracking vật | https://github.com/NVlabs/FoundationPose |
| WiLoR / HaMeR | hand mesh | https://github.com/rolpotamias/WiLoR · https://github.com/geopavlakos/hamer |
| WHAM | người world-frame từ video | https://github.com/yohanshin/WHAM |
| MHR (Meta) | format người Track 1 | https://github.com/facebookresearch/MHR |
| TRELLIS / SAM3D | mesh vật từ ảnh | https://github.com/microsoft/TRELLIS |
| MediaPipe | keypoint 2D nhẹ | https://github.com/google-ai-edge/mediapipe |
| Isaac Lab / Newton | simulator | https://github.com/isaac-sim/IsaacLab · https://github.com/newton-physics/newton |

## 4. Repo cộng đồng (đã đọc README)
| Repo | Track | GPU | Pipeline | Kết quả / bài học |
|---|---|---|---|---|
| skytealang/v2d-track1-reconstruction | 1 | 8×3090 24 GB | MoGe-2 + SAM2.1 + MediaPipe + MHR + FoundationPose + temporal/contact opt | CD-H 33 cm; khởi tạo root MHR từ depth quan trọng |
| Swithinraj/v2d-track1-wham-foundationpose | 1 | ≥24 GB, SM≥8.0, Apptainer | WHAM→MHR, GDINO+SAM2, MoGe-2, TRELLIS, FoundationPose, particle filter cho hula hoop | CD-H 21 / CD-O 29; mô tả rõ format npz |
| priyam-saha-17/nvidia-v2d-challenge | 1 | Kaggle T4 | tương tự, chưa nộp | chạy được trên GPU nhỏ |
| aebatirel/v2d-track3 | 3 | 48 GB (Vast.ai), có Colab | MediaPipe hands + stereo depth → FlashCHORD ref → FlashSAC | perception AUC 0.81 trên 20 ep public |
| the-simsons/ego2sim-nvidia-v2d | 3 | – | FoundationStereo + SAM2 + WiLoR + FoundationPose + FlashSAC, uv py3.11 | 96.8M step/episode |
| Swaguto/NVIDIA-V2D-Challenge | 3 | A100/H100/L40S | Isaac Lab Docker; smoke test `dummy_agent.py --task Sharpa-V2D-v0-Play` synthbox | checklist bootstrap cloud |
| tianny0078/v2d-track3 | 3 | server | WiLoR, TSDF, Sharpa IK, MuJoCo/Newton CPU | local SR = 0, geometry là bottleneck |
| ibrahim-1811/v2d-track2-robotic-grounding, civicW/v2d-challenge-asap, zjifan121-source/NVIDIA_V2D_Egocentric | 2/3 | – | – | tham khảo thêm |

## 5. Format submission (từ kit)
- Track 1: `episode_XXXXXX.npz` (pose [T,136], scales [68], shape [45], object_rotation [T,3,3], object_translation [T,3], object_scale ()) + `episode_XXXXXX_object.glb`; `point_scene = s·R·point_mesh + t`; packer → parquet `row_id,x,y,z,code_commit_url`. Sim(3) align trên human frame đầu.
- Track 2: CSV quỹ đạo vật từ policy cho 30 ref; mỗi tier nộp riêng, chấm so với ref sạch.
- Track 3: `episode_XXXXXX.parquet` quỹ đạo + `episode_XXXXXX/<object>.glb` (metre, quaternion xyzw, ≤4096 face); symmetry: cup quay quanh trục tự do, pot lid 180°.

## 6. Deploy edge (M4) — nguồn
- MoGe-2 ONNX: `Ruicheng/moge-2-vits-normal-onnx`, `-vitb-`, `-vitl-` (opset ≥14). CoreML: `ebowwa/moge-2-vits-normal-coreml`.
- SAM2 export: https://github.com/vietanhdev/samexporter ; RKNN: `happyme531/Segment-Anything-2.1-RKNN2`.
- ViT int8: SmoothQuant alpha 0.5 (NVIDIA TAO/ModelOpt khuyến nghị); `OpenVINO/dino_v2-int8-ov`; `AMD-PAVS-AI/dinov2`, `/sam2`.
- Depth ViT trên edge: Arm `depth-anything-v2-small-int8-xnnpack-executorch`; Hailo Model Zoo v2.19 Depth-Anything-V2-S; Qualcomm AI Hub `qualcomm/Depth-Anything-V2` (~77 ms QCS8550); Depth-Anything-Edge 27–60 FPS Pi5+Hailo.
- Policy export: Isaac Lab `isaaclab_rl.rsl_rl.exporter.export_policy_as_onnx`; https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/
- RK3588: https://github.com/airockchip/rknn-toolkit2 ; https://docs.ultralytics.com/integrations/rockchip-rknn/ ; https://emapco.github.io/rk-transformers/user_guide/export.html
- Hailo: https://community.hailo.ai/t/deploy-state-of-the-art-vit-model/3627
