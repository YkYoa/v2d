# NVIDIA Video-to-Data (V2D) Challenge — status report

Last updated: 13 November 2026. Author: Hoang Pham.

## Summary

We reproduced the core of NVIDIA's Video-to-Data pipeline on our own hardware: an ordinary video of a person handling an object is turned into robot training data, without teleoperation or motion-capture gear. The challenge leaderboard closed on 4 November, so this was a learning and capability exercise, not a competition entry.

Three results are in hand:

1. **A robot policy trained from a human recording.** Two simulated robot hands pick up and put down a tissue box by following a recorded human motion, with a 93.4% success rate.
2. **That policy runs on a plain CPU.** Exported to a 4.6 MB file, it takes 0.1 ms per control step and performs the same in simulation (92.4% success).
3. **A video-to-3D reconstruction in the official submission format.** All 30 challenge videos are converted to a 3D body and object track, packed into one submission file and scored on Kaggle as a late entry. Accuracy is well behind the leaders (body error 34.9 cm against 11.4 cm).

Hardware used: one shared RTX 4090 server (24 GB), plus a laptop for the first experiments. NVIDIA's reference setup asks for a 48 GB GPU; we scaled the runs down to fit.

## What the challenge is

| Track | Task | Our status |
|---|---|---|
| 1. Reconstruction | One third-person video → 3D human body + object shape and motion | All 30 episodes reconstructed, submitted and scored (late entry, not ranked) |
| 2. Robotic grounding | 3D motion → humanoid robot (Unitree G1 + Dex3 hands) policy in simulation | Small trial run only; the policy does not complete the task |
| 3. Egocentric | Head-camera video → robot hand policy in simulation | Policy-training half done on NVIDIA's bundled example; the video-to-motion half not attempted |

## Track 1 — Reconstruction

- **What we built.** A pipeline that uses only freely downloadable models: depth estimation (MoGe-2), segmentation (SAM 2.1), body keypoints (MediaPipe), then a fit of Meta's MHR body model and an object mesh and motion track computed from depth.
- **Why this route.** The official baseline and the stronger community entries rely on models that need manual access approval or a licence registration (SAM 3D Body, SMPL). We avoided those.
- **Result so far.** All 30 episodes are packed with the organisers' tool against the official list of scored frames: 9,877 scored frames, 740,780 rows, matching the official template row for row. Body keypoint error averages 5.7 pixels on a 768-pixel image (worst episode 10.1); the object aligns to the depth data within 0.9 cm on average (median per episode; worst episode 6.6 cm). These are self-consistency checks, not accuracy.
- **Kaggle scores (late submissions, 13 November; lower is better).** The second version corrects depth scale using the static background, fits the body to the depth surface, and filters bad object frames. The third version adds one constraint: the object's depth may differ from the person's by at most 0.3 m, since a handled object is within arm's reach. Part of that gain comes from pulling the object toward the person, not from seeing it more accurately.

  | Metric | First version | Second version | Third version | Reference |
  |---|---|---|---|---|
  | Body shape error (CD-H) | 40.6 cm | 34.9 cm | 34.9 cm | leader 11.4, NVIDIA baseline 15.6, community team on the same route 33 |
  | Object shape error (CD-O) | 80.4 cm | 67.5 cm | 47.7 cm | leader 16.3, NVIDIA baseline 23.2 |
  | Body motion smoothness (ACC-H) | 0.30 cm | 0.22 cm | 0.22 cm | not collected |
  | Object motion smoothness (ACC-O) | 0.60 cm | 0.35 cm | 0.31 cm | not collected |
  | Body–object penetration (PEN) | 0.002 cm | 0.002 cm | 0.001 cm | not collected |

- **Reading the scores.** The pipeline is valid end to end but not accurate. Motion is smooth; position and shape are far off, the object most of all.
- **Known weaknesses.** Body shape and fingers are left at defaults; the object mesh is only the surface visible in the first frame, so parts such as the desk legs drift later in the clip.
- **Compute.** About 5 minutes per episode on the shared 4090, roughly 2.5 hours for all 30.
- **Next.** If we continue: fix the object (full mesh and absolute scale) first, then body shape and hands.

Figure: `docs/figures/m3_ep09_packed_check.jpg` (reconstructed body and desk drawn over the video).

## Track 2 — Robotic grounding

No submission. We ran one scaled-down training trial on NVIDIA's bundled snack-box pick-and-place example for the G1 humanoid: 20 million steps in 80 minutes on the shared 4090, about 6% of NVIDIA's recipe (350 million steps, 2,048 parallel simulations). The setup works end to end, but the policy fails: 0% success over 256 runs, with the box left 58 cm from its target. The robot stays balanced and moves but does not reach the box. A real attempt needs the full training budget and a submission for each of three difficulty tiers.

## Track 3 — Egocentric

- **What we ran.** NVIDIA's FlashCHORD trainer on its bundled tissue-box example: a head-camera recording already converted to hand and object motion, retargeted to two Sharpa robot hands.
- **Training.** 20 million simulation steps in 40 minutes on the shared 4090 (512 parallel simulations, 5.8 GB of GPU memory). NVIDIA's full recipe is 250 million steps on a 48 GB GPU.
- **Result.** 93.4% success over 512 evaluation runs; mean object position error (MPPE) 0.64 cm.
- **Caveat.** This is the bundled example scored by the repository's own evaluator. It is not the challenge's 20 held-out episodes, so it is not comparable to leaderboard numbers (the leader's MPPE was 0.60 cm). The video-to-motion half of Track 3 was not attempted.

Figure: `docs/figures/m2_sharpa_ref_vs_policy_sheet.jpg` (left: recorded motion; right: trained robot policy).

## Deployment to non-NVIDIA hardware

| Model | Result on CPU | vs. original |
|---|---|---|
| Robot policy (from Track 3 work) | 4.6 MB, 0.1 ms per step on one CPU thread | 92.4% success in simulation vs 93.4% |
| Depth model (MoGe-2 small, 8-bit quantised) | 34 MB, 7.5 frames per second on a laptop CPU | 0.5% depth difference vs full precision; 3.6× smaller |

- **Target board:** Renesas R-Car X5H. The board is not available yet, so nothing has been run on it.
- **Open item:** on a real robot the policy needs its 504-value input assembled from sensors; today only the simulator produces it.

## Risks and open items

- Track 1 accuracy is weak (34.9 cm body, 47.7 cm object, about twice NVIDIA's baseline); improving it further likely needs the access-gated models we avoided.
- The 4090 is shared; our runs were sized to fit beside another user's job.
- Track 2 has only a failed small-scale trial; the perception half of Track 3 is untouched.
- The R-Car X5H toolchain is unverified.

## Next steps

1. Decide whether to invest in Track 1 accuracy (object mesh and scale first).
2. Bring up the policy and the depth model on the R-Car X5H when the board arrives.
3. Decide whether Track 2 (humanoid) is worth the setup effort.
