#!/usr/bin/env python3
"""Demo A — keypoint bàn tay (21 điểm/tay, thứ tự MediaPipe) + quỹ đạo vật → tham chiếu FlashCHORD cho hai tay Sharpa Wave.

KHÔNG dùng MANO: bộ retarget chính thức của NVIDIA (`scripts/retarget/ego_recon_to_sharpa.py`) cần tham số MANO + manotorch,
còn loader của FlashCHORD (`flash_chord/data/mano_sharpa.py`) chỉ đọc mảng số: cổ tay, 22 khớp ngón, vị trí frame robot,
pose vật và điểm tiếp xúc. File này tự tính các mảng đó bằng IK trên model MuJoCo của tay Sharpa (asset Apache-2.0).

Chạy trong venv FlashCHORD trên server (có mujoco, scipy, trimesh, pyarrow):

  # tự kiểm: lấy 21 khớp tay + pose vật trong parquet mẫu của NVIDIA làm đầu vào, so kết quả với bản retarget chính thức
  python retarget_sharpa.py --selftest <.../sequence_id=tissue_box_simple/robot_name=sharpa_wave> --seq selftest_box --out out/selftest

  # dữ liệu tự quay (recon.npz do hands_object_from_video.py ghi)
  python retarget_sharpa.py --recon out/my_clip/recon.npz --seq my_clip --out out/my_clip/task

recon.npz: fps; kp_right, kp_left [T,21,3] m, hệ thế giới z hướng lên (NaN = không thấy tay); obj_pos [T,3]; obj_wxyz [T,4];
           box_size [3] m (hoặc mesh_path: file .obj đã đúng tỉ lệ mét, gốc ở tâm vật); table_z (mặt bàn, m).

Ghi ra (đúng bố cục mà FlashCHORD tự dò mesh và mặt đỡ):
  <out>/processed/sequence_id=<seq>/robot_name=sharpa_wave/data.parquet
  <out>/processed/<seq>.obj, <seq>_visual.stl, <seq>_rigid.urdf
  <out>/reconstructed_stage/<seq>_support.usda
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

SIDES = ("right", "left")
DEFAULT_XML_DIR = ("/data2/huyhoang/v2d/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/"
                   "assets/xmls/sharpawave")
# (frame robot, chỉ số keypoint, trọng số vị trí) — cùng bảng với SHARPA_TO_MANO_MAPPING của NVIDIA; thứ tự 21 khớp MANO = MediaPipe
TASKS = [("hand_C_MC", 0, 0.2), ("thumb_MCP_VL_site", 2, 0.1), ("thumb_tip_site", 4, 1.0),
         ("index_MP_site", 5, 0.1), ("index_tip_site", 8, 1.0), ("middle_MP_site", 9, 0.1), ("middle_tip_site", 12, 1.0),
         ("ring_MP_site", 13, 0.1), ("ring_tip_site", 16, 1.0), ("pinky_MP_site", 17, 0.1), ("pinky_tip_site", 20, 0.5)]
# điểm trên lòng bàn tay dùng để suy ra hướng cổ tay (thay cho hướng xoay cổ tay của MANO)
PALM = [("hand_wrist_site", 0), ("index_MCP_VL_site", 5), ("middle_MCP_VL_site", 9), ("ring_MCP_VL_site", 13), ("pinky_MC_site", 17)]
LINKS = ["palm"] + [f"{d}{k}" for d in ("thumb", "index", "middle", "ring", "pinky") for k in (1, 2, 3)]   # 16 ô tiếp xúc/tay


def kabsch(a: np.ndarray, b: np.ndarray):
    """R, t sao cho R @ a_i + t ≈ b_i."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


class Hand:
    def __init__(self, xml_dir: str, side: str):
        spec = mujoco.MjSpec.from_file(str(Path(xml_dir) / f"{side}_sharpawave.xml"))
        spec.worldbody.first_body().add_freejoint(name="root_joint")
        self.m = m = spec.compile(); self.d = mujoco.MjData(m); self.side = side
        self.finger_names = [m.joint(i).name for i in range(m.njnt) if m.joint(i).name != "root_joint"]
        self.qadr = np.array([m.joint(n).qposadr[0] for n in self.finger_names])
        self.lim = np.array([m.joint(n).range for n in self.finger_names])
        self.palm = m.body(f"{side}_hand_C_MC").id
        self.tasks = [(self._ref(f"{side}_{n}"), k, w) for n, k, w in TASKS]
        self.palm_ref = [(self._ref(f"{side}_{n}"), k) for n, k in PALM]
        # danh sách frame theo đúng quy ước pinocchio của bản chính thức: mỗi body = [khớp, body, các site]; bỏ hand_wrist_site
        self.frames = [("universe", None), ("root_joint", ("body", self.palm))]
        for b in range(1, m.nbody):
            for j in range(m.njnt):
                if m.jnt_bodyid[j] == b and m.joint(j).name != "root_joint":
                    self.frames.append((m.joint(j).name, ("joint", j)))
            self.frames.append((m.body(b).name, ("body", b)))
            for s in range(m.nsite):
                if m.site_bodyid[s] == b and m.site(s).name != f"{side}_hand_wrist_site":
                    self.frames.append((m.site(s).name, ("site", s)))
        q0 = np.zeros(m.nq); q0[3] = 1.0; self.fk(q0)
        self.palm_local = np.stack([self.pos(r) for r, _ in self.palm_ref])     # ở cấu hình 0, gốc = hand_C_MC

    def _ref(self, name):
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return ("site", sid)
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, name
        return ("body", bid)

    def fk(self, q):
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)

    def pos(self, ref):
        kind, i = ref
        return (self.d.site_xpos[i] if kind == "site" else self.d.xanchor[i] if kind == "joint" else self.d.xpos[i]).copy()

    def quat(self, ref):
        kind, i = ref
        if kind == "site":
            q = np.zeros(4); mujoco.mju_mat2Quat(q, self.d.site_xmat[i]); return q
        return self.d.xquat[self.m.jnt_bodyid[i] if kind == "joint" else i].copy()

    def jac(self, ref):
        jp = np.zeros((3, self.m.nv))
        (mujoco.mj_jacSite if ref[0] == "site" else mujoco.mj_jacBody)(self.m, self.d, jp, None, ref[1])
        return jp

    def wrist_guess(self, kp, scale):
        """Pose cổ tay sao cho các điểm lòng bàn tay robot khớp với cổ tay + 4 khớp gốc ngón của người."""
        tgt = np.stack([kp[0] + (kp[k] - kp[0]) * scale for _, k in self.palm_ref])
        return kabsch(self.palm_local, tgt)

    def targets(self, kp, scale):
        """Vị trí đích của 11 frame robot cho 1 chuỗi keypoint [T,21,3] → [T,11,3] (phóng tỉ lệ quanh cổ tay)."""
        return np.stack([kp[:, 0] + (kp[:, k] - kp[:, 0]) * scale for _, k, _ in self.tasks], 1)

    def solve(self, kp, scale, targets, q_prev=None, iters=40, w_rot=0.2, damp=1e-3, w_prev=0.0):
        """IK Levenberg–Marquardt cho 1 frame. Trả q [29] = pos, quat wxyz, 22 khớp."""
        m, d = self.m, self.d
        r_t, t_t = self.wrist_guess(kp, scale)
        if q_prev is None:
            q = np.zeros(m.nq); q[:3] = t_t; q[3:7] = Rot.from_matrix(r_t).as_quat(scalar_first=True)
            q[self.qadr] = np.clip(0.2, self.lim[:, 0], self.lim[:, 1])
        else:
            q = q_prev.copy()
        tg = [(ref, targets[i], w) for i, (ref, _, w) in enumerate(self.tasks)]
        jr = np.zeros((3, m.nv)); jp = np.zeros((3, m.nv)); lam = damp
        qp = None if q_prev is None else q_prev[self.qadr]

        def residual():
            self.fk(q)
            res = [w * (self.pos(ref) - t) for ref, t, w in tg]
            e = Rot.from_matrix(r_t @ d.xmat[self.palm].reshape(3, 3).T).as_rotvec()
            res.append(w_rot * e)
            if qp is not None and w_prev > 0:
                res.append(w_prev * (q[self.qadr] - qp))
            return np.concatenate(res)

        r = residual(); cost = r @ r
        for _ in range(iters):
            rows = [w * self.jac(ref) for ref, _, w in tg]
            mujoco.mj_jacBody(m, d, jp, jr, self.palm); rows.append(-w_rot * jr)
            if qp is not None and w_prev > 0:
                jq = np.zeros((len(self.qadr), m.nv)); jq[:, 6:] = np.eye(len(self.qadr)) * w_prev; rows.append(jq)
            J = np.concatenate(rows)
            dq = np.linalg.solve(J.T @ J + lam * np.eye(m.nv), -J.T @ r)
            q_old = q.copy()
            mujoco.mj_integratePos(m, q, dq, 1.0)
            q[self.qadr] = np.clip(q[self.qadr], self.lim[:, 0], self.lim[:, 1])
            r_new = residual(); c_new = r_new @ r_new
            if c_new < cost:
                done = cost - c_new < 1e-12
                r, cost, lam = r_new, c_new, max(lam * 0.5, 1e-6)
                if done:
                    break
            else:
                q[:] = q_old; lam *= 4.0; self.fk(q)
                if lam > 1e3:
                    break
        self.fk(q)
        err = np.array([np.linalg.norm(self.pos(ref) - t) for ref, t, _ in tg])
        return q.copy(), err

    def frame_poses(self, q):
        self.fk(q); out = np.zeros((len(self.frames), 7)); out[:, 3] = 1.0
        for i, (_, ref) in enumerate(self.frames):
            if ref is not None:
                out[i, :3] = self.pos(ref); out[i, 3:] = self.quat(ref)
        return out


def fill_and_smooth(kp: np.ndarray, window: int):
    """Nội suy khoảng trống (NaN) theo thời gian rồi làm mượt Savitzky–Golay. Trả (kp, tỉ lệ frame có dữ liệu)."""
    T = len(kp); ok = np.isfinite(kp).all((1, 2)); frac = float(ok.mean())
    if ok.sum() < 2:
        return None, frac
    idx = np.flatnonzero(ok); out = kp.copy()
    for j in range(kp.shape[1]):
        for c in range(3):
            out[:, j, c] = np.interp(np.arange(T), idx, kp[idx, j, c])
    w = min(window, T - (1 - T % 2)); w -= 1 - w % 2
    if w >= 5:
        out = savgol_filter(out, w, 2, axis=0, mode="nearest")
    return out, frac


def link_points(kp: np.ndarray) -> np.ndarray:
    """16 điểm đại diện cho các đốt (lòng bàn tay + 3 đốt × 5 ngón) từ 21 keypoint. [T,16,3]."""
    pts = [kp[:, [0, 5, 9, 13, 17]].mean(1)]
    for j0 in (1, 5, 9, 13, 17):
        pts += [0.5 * (kp[:, j0] + kp[:, j0 + 1]), 0.5 * (kp[:, j0 + 1] + kp[:, j0 + 2]), 0.3 * kp[:, j0 + 2] + 0.7 * kp[:, j0 + 3]]
    return np.stack(pts, 1)


def limit_penetration(tg, tip_idx, surf, obj_pos, obj_wxyz, max_pen, window=9):
    """Đầu ngón (đích IK) chui vào vật sâu hơn max_pen → đẩy ra theo pháp tuyến cho còn đúng max_pen.
    Phóng tay ×1.2 quanh cổ tay làm ngón dài thêm nên hay cắm sâu vào vật; frame tham chiếu xuyên sâu khiến solver hất tay ra.
    Giữ lại một ít xuyên (mặc định 1 cm, như bản chính thức ~1.6 cm) vì đó là thứ tạo lực kẹp. Trả (tg mới, số frame đã chỉnh)."""
    tg = tg.copy(); tips = tg[:, tip_idx]
    sd, n_out = surf.signed_distance(tips, obj_pos, obj_wxyz, with_normal=True)
    push = np.clip(-sd - max_pen, 0.0, None)[..., None] * n_out
    w = min(window, len(tg) - (1 - len(tg) % 2)); w -= 1 - w % 2
    if w >= 5:
        push = savgol_filter(push, w, 2, axis=0, mode="nearest")
    tg[:, tip_idx] = tips + push
    return tg, int((np.linalg.norm(push, axis=-1) > 1e-4).any(1).sum())


class Surface:
    """Bề mặt vật lấy mẫu dày + KD-tree (hệ toạ độ của vật) để tìm điểm gần nhất và pháp tuyến hướng ra ngoài."""

    def __init__(self, mesh: trimesh.Trimesh, n=40000, seed=0):
        pts, fid = trimesh.sample.sample_surface(mesh, n, seed=seed)
        self.p = np.asarray(pts); self.n = np.array(mesh.face_normals)[fid]
        ok = np.abs(np.linalg.norm(self.n, axis=1) - 1.0) < 1e-3                         # mặt suy biến → pháp tuyến 0, bỏ
        self.p, self.n = self.p[ok], self.n[ok]; self.n /= np.linalg.norm(self.n, axis=1, keepdims=True)
        out = np.einsum("ij,ij->i", self.n, self.p - mesh.bounding_box.centroid) < 0     # mesh quét có thể lộn mặt → ép hướng ra
        self.n[out] *= -1; self.tree = cKDTree(self.p)

    def signed_distance(self, pts_w, obj_pos, obj_wxyz, with_normal=False):
        """Khoảng cách có dấu tới bề mặt (âm = nằm trong vật; đúng với vật lồi). pts_w [T,K,3] → [T,K] (+ pháp tuyến ra ngoài, hệ thế giới)."""
        T, K, _ = pts_w.shape
        R = Rot.from_quat(obj_wxyz, scalar_first=True).as_matrix()
        loc = np.einsum("tji,tkj->tki", R, pts_w - obj_pos[:, None]).reshape(-1, 3)
        _, i = self.tree.query(loc)
        sd = np.einsum("ij,ij->i", loc - self.p[i], self.n[i]).reshape(T, K)
        return (sd, np.einsum("tij,tkj->tki", R, self.n[i].reshape(T, K, 3))) if with_normal else sd

    def contacts(self, pts_w, obj_pos, obj_wxyz, thr, thr_palm):
        """pts_w [T,K,3] → (vị trí [T,K,3], pháp tuyến [T,K,3], part id [T,K] ∈ {0,1}, khoảng cách [T,K])."""
        T, K, _ = pts_w.shape
        R = Rot.from_quat(obj_wxyz, scalar_first=True).as_matrix()
        loc = np.einsum("tji,tkj->tki", R, pts_w - obj_pos[:, None])
        dist, i = self.tree.query(loc.reshape(-1, 3)); dist = dist.reshape(T, K); i = i.reshape(T, K)
        cp = np.einsum("tij,tkj->tki", R, self.p[i]) + obj_pos[:, None]
        # pháp tuyến hướng VÀO vật = hướng lực ngón tay ép lên vật (objectives/wrench.py dựng nón ma sát quanh véc-tơ này)
        cn = -np.einsum("tij,tkj->tki", R, self.n[i])
        lim = np.full(K, thr); lim[0] = thr_palm
        act = dist < lim[None]
        return cp, cn, act.astype(np.int64), dist


def debounce(act: np.ndarray, k: int = 5) -> np.ndarray:
    """Bỏ các lần bật/tắt tiếp xúc ngắn hơn k frame (nhiễu keypoint)."""
    from scipy.ndimage import binary_closing, binary_opening
    s = np.ones((k, 1), bool)
    return binary_opening(binary_closing(act.astype(bool), s), s)


URDF = """<?xml version="1.0" ?>
<robot name="{name}">
   <mujoco>
      <compiler meshdir="./" balanceinertia="true" discardvisual="false"/>
   </mujoco>
   <link name="object">
      <inertial>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <mass value="{mass}"/>
         <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
      </inertial>
      <visual>
         <origin rpy="0 0 0" xyz="0 0 0"/>
         <geometry><mesh filename="./{name}_visual.stl" scale="1 1 1"/></geometry>
      </visual>
      <collision>
         <origin rpy="0 0 0" xyz="0 0 0"/>
         <geometry><mesh filename="./{name}_visual.stl" scale="1 1 1"/></geometry>
      </collision>
   </link>
</robot>
"""
USDA = """#usda 1.0
(
    defaultPrim = "support_surfaces"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "support_surfaces"
{{
    def Cylinder "{name}_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        double height = 0.01
        double radius = {radius}
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
"""


def load_selftest(path: str):
    f = sorted(glob.glob(str(Path(path) / "*.parquet")))[0] if Path(path).is_dir() else path
    row = pq.read_table(f).to_pylist()[0]
    rec = {"fps": float(row["fps"]), "obj_pos": np.array(row["object_body_position"])[:, 0],
           "obj_wxyz": np.array(row["object_body_wxyz"])[:, 0]}
    for s in SIDES:
        rec[f"kp_{s}"] = np.array(row[f"mano_{s}_joints"])
    assets = Path(f).parents[2]
    return rec, row, assets / f"{row['object_name']}.obj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon"); ap.add_argument("--selftest"); ap.add_argument("--seq", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--xml-dir", default=DEFAULT_XML_DIR)
    ap.add_argument("--scale", type=float, default=1.2, help="tỉ lệ tay người → tay robot quanh cổ tay (NVIDIA: mano_to_robot_scale=1.2)")
    ap.add_argument("--smooth", type=int, default=9); ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--contact-thr", type=float, default=0.02); ap.add_argument("--palm-thr", type=float, default=0.035)
    ap.add_argument("--min-visible", type=float, default=0.3, help="tay thấy ít hơn tỉ lệ này → coi như không dùng, đặt đứng yên bên cạnh")
    ap.add_argument("--max-pen", type=float, default=0.01, help="đầu ngón được phép chui vào vật tối đa (m); âm = không chỉnh")
    ap.add_argument("--w-prev", type=float, default=0.02, help="phạt đổi khớp ngón giữa 2 frame liền nhau (chống rung)")
    a = ap.parse_args()
    out = Path(a.out).resolve(); proc = out / "processed"; proc.mkdir(parents=True, exist_ok=True)

    ref_row = None
    if a.selftest:
        rec, ref_row, mesh_src = load_selftest(a.selftest); smooth = 1
        mesh = trimesh.load(mesh_src, force="mesh", process=False); table_z = None
    else:
        rec = dict(np.load(a.recon, allow_pickle=True)); smooth = a.smooth
        mesh = (trimesh.load(str(rec["mesh_path"]), force="mesh", process=False) if "mesh_path" in rec
                else trimesh.creation.box(extents=np.asarray(rec["box_size"], float)))
        table_z = float(rec["table_z"]) if "table_z" in rec else None
    fps = float(rec["fps"]); obj_pos = np.asarray(rec["obj_pos"], float); obj_wxyz = np.asarray(rec["obj_wxyz"], float)
    obj_wxyz /= np.linalg.norm(obj_wxyz, axis=1, keepdims=True); T = len(obj_pos)

    # ---- tay: lấp khoảng trống + làm mượt; tay không xuất hiện → bản đối xứng gương, đứng yên bên cạnh
    kps, vis = {}, {}
    for s in SIDES:
        kps[s], vis[s] = fill_and_smooth(np.asarray(rec[f"kp_{s}"], float), smooth)
    used = [s for s in SIDES if kps[s] is not None and vis[s] >= a.min_visible]
    assert used, "không tay nào đủ dữ liệu"
    for s in SIDES:
        if s not in used:
            o = used[0]; c = obj_pos[0]
            park = kps[o][:1].copy(); park[..., 0] = 2 * c[0] - park[..., 0]            # gương qua mặt phẳng x = x_vật
            away = np.sign(park[0, 0, 0] - c[0]) or (1.0 if s == "left" else -1.0)
            park[..., 0] += away * 0.25; park[..., 2] += 0.05
            kps[s] = np.repeat(park, T, 0)
            print(f"tay {s}: chỉ thấy {vis[s]:.0%} số frame → không dùng, đặt đứng yên cách vật ~{abs(park[0, 0, 0] - c[0]):.2f} m")

    # ---- IK từng frame (khởi tạo ấm từ frame trước)
    surf = Surface(mesh); cols, stats = {}, {}
    tip_idx = [i for i, (n, _, _) in enumerate(TASKS) if n.endswith("tip_site")]
    for s in SIDES:
        h = Hand(a.xml_dir, s); q = None; Q, E = [], []
        tg = h.targets(kps[s], a.scale); pushed = 0
        if a.max_pen >= 0 and s in used:
            tg, pushed = limit_penetration(tg, tip_idx, surf, obj_pos, obj_wxyz, a.max_pen)
        for t in range(T):
            q, e = h.solve(kps[s][t], a.scale, tg[t], q_prev=q, iters=200 if q is None else 40, w_prev=a.w_prev if s in used else 1.0)
            Q.append(q); E.append(e)
        Q = np.array(Q); E = np.array(E)
        frames = np.stack([h.frame_poses(q) for q in Q])
        cols[f"{s}_robot_finger_joint_names"] = h.finger_names
        cols[f"{s}_robot_frame_names"] = [n for n, _ in h.frames]
        cols[f"{s}_robot_frame_task_names"] = [f"{s}_{n}" for n, _, _ in TASKS]
        cols[f"robot_{s}_wrist_position"] = Q[:, :3]; cols[f"robot_{s}_wrist_wxyz"] = Q[:, 3:7]
        cols[f"robot_{s}_finger_joints"] = Q[:, h.qadr]; cols[f"robot_{s}_frames"] = frames
        cols[f"robot_{s}_frame_task_errors"] = E
        stats[s] = {"used": s in used, "visible": round(vis[s], 3), "task_err_cm_mean": round(float(E.mean() * 100), 2),
                    "task_err_cm_max": round(float(E.max() * 100), 2), "tip_err_cm_mean": round(float(E[:, tip_idx].mean() * 100), 2),
                    "frames_pushed_out_of_object": pushed,
                    "joint_at_limit_frac": round(float((np.isclose(Q[:, h.qadr], h.lim[:, 0]) | np.isclose(Q[:, h.qadr], h.lim[:, 1])).mean()), 3),
                    "finger_step_rad_p99": round(float(np.percentile(np.abs(np.diff(Q[:, h.qadr], axis=0)), 99)), 3)}

    # ---- tiếp xúc tay–vật (hình học): điểm đại diện mỗi đốt → điểm gần nhất trên bề mặt vật
    for s in SIDES:
        lp = link_points(kps[s])
        cp, cn, cid, dist = surf.contacts(lp, obj_pos, obj_wxyz, a.contact_thr, a.palm_thr)
        if s not in used:
            cid[:] = 0
        act = debounce(cid > 0)
        cols[f"mano_{s}_object_contact_positions"] = cp * act[..., None]
        cn = cn / np.linalg.norm(cn, axis=-1, keepdims=True).clip(min=1e-9)               # loader đòi pháp tuyến đơn vị ở ô đang bật
        cols[f"mano_{s}_object_contact_normals"] = cn * act[..., None]
        cols[f"mano_{s}_object_contact_part_ids"] = act.astype(np.int64)
        stats[s]["contact_frames_frac"] = round(float(act.any(1).mean()), 3)
        stats[s]["_dist"] = dist; stats[s]["_act"] = act

    # ---- asset vật + mặt đỡ
    v = np.asarray(mesh.vertices); radius = float(np.linalg.norm(v, axis=1).max())
    ext = mesh.bounding_box.extents; mesh.export(proc / f"{a.seq}.obj"); mesh.export(proc / f"{a.seq}_visual.stl")
    ixx, iyy, izz = (a.mass / 12 * (ext[j] ** 2 + ext[k] ** 2) for j, k in ((1, 2), (0, 2), (0, 1)))
    (proc / f"{a.seq}_rigid.urdf").write_text(URDF.format(name=a.seq, mass=a.mass, ixx=ixx, iyy=iyy, izz=izz))
    R0 = Rot.from_quat(obj_wxyz[0], scalar_first=True).as_matrix()
    bottom = float((v @ R0.T + obj_pos[0])[:, 2].min())
    if table_z is None:
        table_z = bottom
    (out / "reconstructed_stage").mkdir(exist_ok=True)
    (out / "reconstructed_stage" / f"{a.seq}_support.usda").write_text(
        USDA.format(name=a.seq, radius=max(radius, 0.10), x=float(obj_pos[0, 0]), y=float(obj_pos[0, 1]), z=table_z - 0.005))

    cols.update({
        "raw_motion_file": str(a.selftest or Path(a.recon).resolve()), "fps": fps,
        "object_name": a.seq, "safe_object_name": a.seq, "object_body_names": [a.seq], "safe_object_body_names": [a.seq],
        "object_mesh_paths": [str(proc / f"{a.seq}.obj")], "object_urdf_paths": [str(proc / f"{a.seq}_rigid.urdf")],
        "object_mesh_radius": [radius], "object_articulation": np.zeros(T),
        "object_root_position": obj_pos, "object_root_axis_angle": Rot.from_quat(obj_wxyz, scalar_first=True).as_rotvec(),
        "object_body_position": obj_pos[:, None], "object_body_wxyz": obj_wxyz[:, None],
        "source_kind": "keypoints21_no_mano", "hand_to_robot_scale": a.scale,
    })
    pdir = proc / f"sequence_id={a.seq}" / "robot_name=sharpa_wave"; pdir.mkdir(parents=True, exist_ok=True)
    tab = pa.table({k: [v.tolist() if isinstance(v, np.ndarray) else v] for k, v in cols.items()})
    pq.write_table(tab, pdir / "data.parquet")

    # ---- báo cáo
    rep = {"frames": T, "fps": fps, "object_radius_m": round(radius, 4), "object_extents_m": np.round(ext, 3).tolist(),
           "table_z": round(table_z, 4), "object_bottom_minus_table_cm": round((bottom - table_z) * 100, 2),
           "object_travel_m": round(float(np.linalg.norm(obj_pos.max(0) - obj_pos.min(0))), 3)}
    for s in SIDES:
        st = stats[s]; dist, act = st.pop("_dist"), st.pop("_act"); rep[s] = st
        pen = tip_penetration(surf, cols[f"robot_{s}_frames"], cols[f"{s}_robot_frame_names"], obj_pos, obj_wxyz)
        st["tip_inside_gt1cm_frac"], st["tip_inside_gt2cm_frac"], st["tip_inside_max_cm"] = pen
        if ref_row is not None:
            st["official_tip_inside_gt1cm_gt2cm_max"] = tip_penetration(surf, np.array(ref_row[f"robot_{s}_frames"]), ref_row[f"{s}_robot_frame_names"], obj_pos, obj_wxyz)
        if ref_row is not None:                                                   # so với bản retarget chính thức (MANO)
            o_fr = np.array(ref_row[f"robot_{s}_frames"]); names = ref_row[f"{s}_robot_frame_names"]
            st["frame_names_match_official"] = names == cols[f"{s}_robot_frame_names"]
            tip_i = [names.index(f"{s}_{d}_tip_site") for d in ("thumb", "index", "middle", "ring", "pinky")]
            st["vs_official_tip_cm_mean"] = round(float(np.linalg.norm(o_fr[:, tip_i, :3] - cols[f"robot_{s}_frames"][:, tip_i, :3], axis=-1).mean() * 100), 2)
            st["vs_official_wrist_cm_mean"] = round(float(np.linalg.norm(np.array(ref_row[f"robot_{s}_wrist_position"]) - cols[f"robot_{s}_wrist_position"], axis=1).mean() * 100), 2)
            dr = Rot.from_quat(np.array(ref_row[f"robot_{s}_wrist_wxyz"]), scalar_first=True).inv() * Rot.from_quat(cols[f"robot_{s}_wrist_wxyz"], scalar_first=True)
            st["vs_official_wrist_deg_mean"] = round(float(np.degrees(dr.magnitude()).mean()), 1)
            st["vs_official_finger_rad_mean"] = round(float(np.abs(np.array(ref_row[f"robot_{s}_finger_joints"]) - cols[f"robot_{s}_finger_joints"]).mean()), 3)
            st["official_task_err_cm_mean"] = round(float(np.array(ref_row[f"robot_{s}_frame_task_errors"]).mean() * 100), 2)
            o_act = np.array(ref_row[f"mano_{s}_object_contact_part_ids"]) > 0
            st["contact_agree_frac"] = round(float((o_act == act).mean()), 3)
            st["contact_recall"] = round(float((o_act & act).sum() / max(o_act.sum(), 1)), 3)
            st["contact_precision"] = round(float((o_act & act).sum() / max(act.sum(), 1)), 3)
            o_n = np.array(ref_row[f"mano_{s}_object_contact_normals"]); both = o_act & act
            st["contact_normal_cos_mean"] = round(float(np.einsum("tkj,tkj->tk", o_n, cols[f"mano_{s}_object_contact_normals"])[both].mean()), 3)
    (out / "retarget_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)); print("→", pdir / "data.parquet")


def tip_penetration(surf, frames, names, obj_pos, obj_wxyz):
    """Tỉ lệ frame có đầu ngón robot nằm TRONG vật sâu hơn 1 cm / 2 cm, và độ sâu lớn nhất (cm).
    Frame tham chiếu xuyên vật thành reset xuyên vật khi train; mốc của NVIDIA: không frame nào sâu quá 2 cm."""
    tip = [i for i, n in enumerate(names) if n.endswith("tip_site")]
    depth = -surf.signed_distance(frames[:, tip, :3], obj_pos, obj_wxyz)
    return (round(float((depth > 0.01).any(1).mean()), 3), round(float((depth > 0.02).any(1).mean()), 3),
            round(float(max(depth.max(), 0) * 100), 2))


if __name__ == "__main__":
    main()
