# NVIDIA Video-to-Data (V2D) Challenge — status report

Last updated: 13 November 2026. Author: Hoang Pham.

## Summary

We reproduced the core of NVIDIA's Video-to-Data pipeline on our own hardware: an ordinary video of a person handling an object is turned into robot training data, without teleoperation or motion-capture gear. The challenge leaderboard closed on 4 November, so this was a learning and capability exercise, not a competition entry.

Three results are in hand:

1. **A robot policy trained from a human recording.** Two simulated robot hands pick up and put down a tissue box by following a recorded human motion, with a 93.4% success rate.
2. **That policy runs on a plain CPU.** Exported to a 4.6 MB file, it takes 0.1 ms per control step and performs the same in simulation (92.4% success).
3. **A video-to-3D reconstruction in the official submission format.** One challenge video is converted to a 3D body and object track that passes the organisers' packing tool.

Hardware used: one shared RTX 4090 server (24 GB), plus a laptop for the first experiments. NVIDIA's reference setup asks for a 48 GB GPU; we scaled the runs down to fit.

## What the challenge is

| Track | Task | Our status |
|---|---|---|
| 1. Reconstruction | One third-person video → 3D human body + object shape and motion | Working end to end for one episode; batch over all 30 episodes in progress |
| 2. Robotic grounding | 3D motion → humanoid robot (Unitree G1 + Dex3 hands) policy in simulation | Not attempted |
| 3. Egocentric | Head-camera video → robot hand policy in simulation | Policy-training half done on NVIDIA's bundled example; the video-to-motion half not attempted |

## Track 1 — Reconstruction

- **What we built.** A pipeline that uses only freely downloadable models: depth estimation (MoGe-2), segmentation (SAM 2.1), body keypoints (MediaPipe), then a fit of Meta's MHR body model and an object mesh and motion track computed from depth.
- **Why this route.** The official baseline and the stronger community entries rely on models that need manual access approval or a licence registration (SAM 3D Body, SMPL). We avoided those.
- **Result so far.** Episode 9 (a person tipping a desk upright, 415 frames) is packed with the organisers' tool against the official list of scored frames: 325 scored frames, 24,481 rows, all checks pass. Body keypoint error is about 5 pixels on a 768-pixel image; the object aligns to the depth data within 1.5 cm (median).
- **Accuracy is not yet measured.** The reference data is held by the organisers, so a score needs an upload to Kaggle. A community team using the same approach scored 33 cm body error against 15.6 cm for NVIDIA's baseline and 11.4 cm for the leaderboard leader, so expect a modest score.
- **Known weaknesses.** Body shape and fingers are left at defaults; the object mesh is only the surface visible in the first frame, so parts such as the desk legs drift later in the clip.
- **In progress.** A batch over all 30 episodes is running on the server. A scored submission additionally needs a public code commit link and Kaggle to still accept late entries (not yet checked).

Figure: `docs/figures/m3_ep09_packed_check.jpg` (reconstructed body and desk drawn over the video).

## Track 2 — Robotic grounding

Not attempted. It needs the full humanoid (G1 + Dex3) setup in simulation and a separate submission for each of three difficulty tiers. The training code we ran for Track 3 is the same family (FlashCHORD), so the setup work carries over.

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

- Track 1 has no accuracy number until a Kaggle upload succeeds.
- The 4090 is shared; our runs were sized to fit beside another user's job.
- Track 2 and the perception half of Track 3 are untouched.
- The R-Car X5H toolchain is unverified.

## Next steps

1. Finish the 30-episode Track 1 batch and attempt a late Kaggle submission for a real score.
2. Bring up the policy and the depth model on the R-Car X5H when the board arrives.
3. Decide whether Track 2 (humanoid) is worth the setup effort.
