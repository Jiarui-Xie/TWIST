"""
CMG Student V2 Sim-to-Sim controller (MuJoCo).

This script runs the CMG Student V2 policy in a MuJoCo sim, using CMGMotionLib
directly (no Redis, no high-level server). The CMG generates the 20-step future
reference in-process, identical to how it works during IsaacGym training.

Architecture:
  CMGMotionLib  →  build priv_mimic_obs (1160)  ─┐
  MuJoCo sim    →  build proprio_obs    (  77)  ─┴─→  Policy  →  actions  →  PD

Usage:
  # 1. Export JIT first:
  cd legged_gym/legged_gym/scripts
  python save_jit_cmg_stu_v2.py --exptid cmg_stu_v4 --device cpu

  # 2. Run sim2sim:
  cd deploy_real
  python sim2sim_cmg_stu_v2.py \\
      --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \\
      --cmd_vx 1.5

Keyboard controls (in MuJoCo viewer):
  The viewer is passive; use the cmd_* flags or modify _commands at runtime.
"""

import os
import sys
import time
import argparse
import math
import numpy as np
import torch
import mujoco
import mujoco.viewer as mjv
from collections import deque
from tqdm import tqdm

# ── path setup ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in [
    os.path.join(REPO_ROOT, "pose"),
    os.path.join(REPO_ROOT, "cmg_workspace"),
    os.path.join(REPO_ROOT, "rsl_rl"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pose.utils.cmg_motion_lib import CMGMotionLib
from data_utils.rot_utils import quatToEuler   # numpy, (w,x,y,z) → (r,p,y)


# ── constants (must match training config) ──────────────────────────────────
NUM_ACTIONS   = 23
SIM_DT        = 0.001          # MuJoCo timestep
DECIMATION    = 20             # policy runs at 50 Hz  (20 × 1 ms)
CMG_DT        = 0.02           # CMG generates at 50 Hz
ACTION_SCALE  = 0.5

# tar_obs_steps used by G1MimicCMGStuV2Cfg  (1 … 20 future frames)
TAR_OBS_STEPS = list(range(1, 21))   # [1,2,...,20]

# obs scaling (must match training)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05

DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left leg  (6)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg (6)
     0.0, 0.0, 0.0,                    # waist     (3)
     0.0, 0.4, 0.0, 1.2,               # left arm  (4)
     0.0,-0.4, 0.0, 1.2,               # right arm (4)
], dtype=np.float32)

# 23 DoF indices in the 25-DoF MuJoCo model (drop wrist-roll joints 19, 24)
BODY_DOF_IDS_25 = [i for i in range(25) if i not in (19, 24)]
ANKLE_IDX       = [4, 5, 10, 11]        # zero out ankle dof-vel in obs


# ── mimic obs builder (mirrors _get_mimic_obs in training) ──────────────────
def build_priv_mimic_obs(cmg_lib: CMGMotionLib,
                         device: str) -> torch.Tensor:
    """
    Build the 1160-dim priv_mimic_obs vector from the CMG trajectory buffer.

    Returns: (1, 1160) float tensor on `device`
    """
    import torch
    dev = torch.device(device)
    num_steps = len(TAR_OBS_STEPS)

    # current time  (env 0 only)
    current_time = cmg_lib._motion_times[0].item()

    # fake tiled query:  motion_ids (20,)  motion_times (20,)
    obs_times = torch.tensor(
        [current_time + s * CMG_DT for s in TAR_OBS_STEPS],
        device=dev, dtype=torch.float32
    )
    motion_ids = torch.zeros(num_steps, dtype=torch.long, device=dev)

    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = \
        cmg_lib.calc_motion_frame(motion_ids, obs_times)
    # shapes: (20, *), root_rot is xyzw

    # euler from xyzw quaternion  (same logic as training)
    def euler_from_xyzw(q):
        # q: (N,4)  xyzw
        x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]
        roll  = torch.atan2(2*(w*x + y*z),  1 - 2*(x*x + y*y))
        sinp  = 2*(w*y - z*x)
        pitch = torch.where(torch.abs(sinp) >= 1,
                            torch.sign(sinp) * math.pi/2,
                            torch.asin(sinp.clamp(-1, 1)))
        yaw   = torch.atan2(2*(w*z + x*y),  1 - 2*(y*y + z*z))
        return roll, pitch, yaw

    roll, pitch, yaw = euler_from_xyzw(root_rot)   # (20,)

    # local root vel: quat_rotate_inverse(root_rot, root_vel) in xyzw convention
    def quat_rotate_inverse_xyzw(q, v):
        # q: (N,4) xyzw → convert to wxyz for formula
        w = q[:,3:4]; xyz = q[:,:3]
        a = v * (2*w**2 - 1)
        b = torch.cross(xyz, v, dim=-1) * (2*w)
        c = xyz * (xyz * v).sum(dim=-1, keepdim=True) * 2
        return a - b + c

    root_vel_local = quat_rotate_inverse_xyzw(root_rot, root_vel)     # (20,3)
    root_ang_vel_z = quat_rotate_inverse_xyzw(root_rot, root_ang_vel)[:,2:3]  # (20,1)

    # key_body_pos: body_pos is (20, 9, 3) local
    key_body_pos_flat = body_pos.reshape(num_steps, -1)   # (20, 27)

    # assemble per-step obs: 1+3+3+1+23+27 = 58 dims
    per_step = torch.cat([
        root_pos[:, 2:3],                          # height  (1)
        roll.unsqueeze(1),                          # roll    (1)
        pitch.unsqueeze(1),                         # pitch   (1)
        yaw.unsqueeze(1),                           # yaw     (1)
        root_vel_local,                             # vel     (3)
        root_ang_vel_z,                             # yaw_rate(1)
        dof_pos,                                    # joints  (23)
        key_body_pos_flat,                          # bodies  (27)
    ], dim=-1)   # (20, 58)

    return per_step.reshape(1, -1)   # (1, 1160)


# ── main controller ──────────────────────────────────────────────────────────
class CMGStuV2SimController:
    def __init__(self, xml_path: str, policy_path: str,
                 cmg_model_path: str, cmg_data_path: str, urdf_path: str,
                 cmd_vx: float = 1.5, cmd_vy: float = 0.0, cmd_yaw: float = 0.0,
                 device: str = "cuda", record_video: bool = False):

        self.device = device
        self.record_video = record_video

        # ── CMG motion lib (1 env) ─────────────────────────────────────────
        print("[CMG] Loading CMGMotionLib...")
        self.cmg = CMGMotionLib(
            cmg_model_path  = cmg_model_path,
            cmg_data_path   = cmg_data_path,
            urdf_path       = urdf_path,
            device          = device,
            num_envs        = 1,
            episode_length_s= 100.0,
            dt              = CMG_DT,
            vx_range        = (cmd_vx, cmd_vx),    # fixed command
            vy_range        = (cmd_vy, cmd_vy),
            yaw_range       = (cmd_yaw, cmd_yaw),
        )
        env_ids = torch.zeros(1, dtype=torch.long, device=device)
        init_cmd = torch.tensor([[cmd_vx, cmd_vy, cmd_yaw]], device=device)
        self.cmg.reset(env_ids, commands=init_cmd)
        # Disable trajectory mirroring for sim2sim (no symmetry augmentation needed)
        self.cmg._mirror_flags[:] = False
        print(f"[CMG] Ready  vx={cmd_vx:.2f}  vy={cmd_vy:.2f}  yaw={cmd_yaw:.2f}")

        # ── JIT policy ────────────────────────────────────────────────────
        print(f"[Policy] Loading {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        # ── MuJoCo sim ────────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = SIM_DT
        self.data  = mujoco.MjData(self.model)

        # ── keyboard command control ───────────────────────────────────────
        self._VX_STEP  = 0.2
        self._VY_STEP  = 0.1
        self._YAW_STEP = 0.2
        self._VX_MIN,  self._VX_MAX  = 0.0, 4.0
        self._VY_MIN,  self._VY_MAX  = -0.8, 0.8
        self._YAW_MIN, self._YAW_MAX = -1.5, 1.5
        self._init_vx  = cmd_vx

        # GLFW key codes
        KEY_UP    = 265
        KEY_DOWN  = 264
        KEY_LEFT  = 263
        KEY_RIGHT = 262
        KEY_Q     = 81
        KEY_E     = 69
        KEY_R     = 82

        def key_callback(keycode):
            cmd = self.cmg._target_commands[0]   # (3,) on device
            changed = True
            if keycode == KEY_UP:
                cmd[0] = (cmd[0] + self._VX_STEP).clamp(self._VX_MIN, self._VX_MAX)
            elif keycode == KEY_DOWN:
                cmd[0] = (cmd[0] - self._VX_STEP).clamp(self._VX_MIN, self._VX_MAX)
            elif keycode == KEY_LEFT:
                cmd[2] = (cmd[2] + self._YAW_STEP).clamp(self._YAW_MIN, self._YAW_MAX)
            elif keycode == KEY_RIGHT:
                cmd[2] = (cmd[2] - self._YAW_STEP).clamp(self._YAW_MIN, self._YAW_MAX)
            elif keycode == KEY_Q:
                cmd[1] = (cmd[1] + self._VY_STEP).clamp(self._VY_MIN, self._VY_MAX)
            elif keycode == KEY_E:
                cmd[1] = (cmd[1] - self._VY_STEP).clamp(self._VY_MIN, self._VY_MAX)
            elif keycode == KEY_R:
                cmd[0] = torch.tensor(self._init_vx, device=self.device)
                cmd[1] = torch.tensor(0.0, device=self.device)
                cmd[2] = torch.tensor(0.0, device=self.device)
            else:
                changed = False
            if changed:
                # Sync _commands so CMG obs/rewards see the new value immediately
                self.cmg._commands[0] = cmd.clone()
                # Force trajectory regeneration with new command
                env_ids = torch.zeros(1, dtype=torch.long, device=self.device)
                frame_idx = min(self.cmg._buffer_frame_idx[0].item(),
                                self.cmg.TRAJECTORY_BUFFER_FRAMES - 1)
                self.cmg._current_motion_norm[0] = self.cmg._trajectory_buffer[0, frame_idx]
                self.cmg._generate_trajectory(env_ids)
                print(f"[CMG cmd]  vx={cmd[0]:.2f}  vy={cmd[1]:.2f}  yaw={cmd[2]:.2f}")

        # ── MuJoCo viewer (key_callback passed at construction time) ──────
        self.viewer = mjv.launch_passive(
            self.model, self.data,
            key_callback=key_callback,
            show_left_ui=False, show_right_ui=False,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # 25-DoF mujoco default  (wrist-roll included, zero)
        self.mujoco_default_qpos = np.concatenate([
            [0, 0, 1.0], [1, 0, 0, 0],
            [-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # L leg
             -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # R leg
              0.0, 0.0, 0.0,                    # waist
              0.0, 0.4, 0.0, 1.2, 0.0,          # L arm + wrist-roll
              0.0,-0.4, 0.0, 1.2, 0.0]          # R arm + wrist-roll
        ], dtype=np.float32)

        self.stiffness = np.array([
            100,100,100,150, 40, 40,
            100,100,100,150, 40, 40,
            150,150,150,
             40, 40, 40, 40, 20,
             40, 40, 40, 40, 20,
        ], dtype=np.float32)
        self.damping = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            4, 4, 4,
            5, 5, 5, 5, 1,
            5, 5, 5, 5, 1,
        ], dtype=np.float32)
        self.torque_limits = np.array([
             88,139, 88,139, 50, 50,
             88,139, 88,139, 50, 50,
             88, 50, 50,
             25, 25, 25, 25, 25,
             25, 25, 25, 25, 25,
        ], dtype=np.float32)

    # ── obs builder ─────────────────────────────────────────────────────────
    def _get_obs(self) -> torch.Tensor:
        """Build the 1237-dim student obs: priv_mimic(1160) + proprio(77)."""
        # ── priv_mimic_obs from CMG ──────────────────────────────────────
        priv_mimic = build_priv_mimic_obs(self.cmg, self.device)   # (1,1160)

        # ── proprio from MuJoCo ──────────────────────────────────────────
        qpos = self.data.qpos.astype(np.float32)   # 32 = 7 (root) + 25 (dof)
        qvel = self.data.qvel.astype(np.float32)   # 31 = 6 (root) + 25 (dof)

        body_dof_pos = qpos[7:][BODY_DOF_IDS_25]   # (23,)
        body_dof_vel = qvel[6:][BODY_DOF_IDS_25]   # (23,)

        quat_wxyz = self.data.sensor("orientation").data.astype(np.float32)  # w,x,y,z
        ang_vel   = self.data.sensor("angular-velocity").data.astype(np.float32)

        rpy = quatToEuler(quat_wxyz)   # (3,)  roll, pitch, yaw

        # zero ankle dof-vel (matches training)
        body_dof_vel_obs = body_dof_vel.copy()
        body_dof_vel_obs[ANKLE_IDX] = 0.0

        # current velocity command (vx, vy, yaw) - mirrors training's get_user_commands()
        cmd = self.cmg.get_user_commands()[0].cpu().numpy()   # (3,)

        proprio = np.concatenate([
            ang_vel * ANG_VEL_SCALE,                          # 3
            rpy[:2],                                          # 2 (roll, pitch only)
            (body_dof_pos - DEFAULT_DOF_POS) * DOF_POS_SCALE,# 23
            body_dof_vel_obs * DOF_VEL_SCALE,                 # 23
            self.last_action,                                 # 23
            cmd,                                              # 3
        ], dtype=np.float32)   # total 77

        proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)   # (1,77)
        obs = torch.cat([priv_mimic, proprio_t], dim=-1)   # (1,1237)
        return obs

    # ── run ─────────────────────────────────────────────────────────────────
    def run(self, duration_s: float = 30.0):
        if self.record_video:
            import imageio
            mp4_writer = imageio.get_writer("cmg_stu_v2_sim2sim.mp4", fps=50)
        else:
            mp4_writer = None

        # reset sim
        self.data.qpos[:] = self.mujoco_default_qpos
        mujoco.mj_forward(self.model, self.data)

        steps = int(duration_s / SIM_DT)
        print(f"[Sim2Sim] Running for {duration_s:.0f}s  ({steps} sim steps)")
        print("[Sim2Sim] Keyboard: ↑/↓=vx  ←/→=yaw  Q/E=vy  R=reset cmd")

        try:
            for i in tqdm(range(steps)):
                if i % DECIMATION == 0:
                    # ── step CMG ──────────────────────────────────────────
                    all_envs = torch.zeros(1, dtype=torch.long, device=self.device)
                    self.cmg.step(all_envs)

                    # ── build obs & run policy ────────────────────────────
                    obs = self._get_obs()
                    with torch.no_grad():
                        raw_action = self.policy(obs).cpu().numpy().squeeze()   # (23,)

                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10.0, 10.0)
                    pd_target_23 = raw_action * ACTION_SCALE + DEFAULT_DOF_POS

                    # reindex 23 → 25 (insert wrist-roll zeros)
                    pd_target_25 = np.zeros(25, dtype=np.float32)
                    pd_target_25[BODY_DOF_IDS_25] = pd_target_23

                # ── PD control ────────────────────────────────────────────
                qpos_all = self.data.qpos[7:].astype(np.float32)   # 25
                qvel_all = self.data.qvel[6:].astype(np.float32)   # 25
                torque = (pd_target_25 - qpos_all) * self.stiffness \
                       - qvel_all * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)

                # ── viewer ────────────────────────────────────────────────
                if i % DECIMATION == 0:
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()
                    if mp4_writer is not None:
                        mp4_writer.append_data(self.viewer.read_pixels())

        except KeyboardInterrupt:
            print("\n[Sim2Sim] Interrupted.")
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("[Sim2Sim] Video saved → cmg_stu_v2_sim2sim.mp4")
            self.viewer.close()


# ── entry point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CMG Student V2 Sim2Sim")

    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument("--policy_path", required=True,
                        help="Path to the JIT-exported student policy (.pt)")
    parser.add_argument("--xml_file",
                        default=os.path.join(REPO, "assets/g1/g1_sim2sim_with_wrist_roll.xml"))
    parser.add_argument("--cmg_model",
                        default=os.path.join(REPO, "cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"))
    parser.add_argument("--cmg_data",
                        default=os.path.join(REPO, "cmg_workspace/dataloader/cmg_training_data.pt"))
    parser.add_argument("--urdf",
                        default=os.path.join(REPO, "assets/g1/g1_custom_collision_with_fixed_hand.urdf"))
    parser.add_argument("--cmd_vx",  type=float, default=2.0)
    parser.add_argument("--cmd_vy",  type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--duration",type=float, default=30.0, help="Sim duration (s)")
    parser.add_argument("--device",  type=str,   default="cuda")
    parser.add_argument("--record_video", action="store_true")
    args = parser.parse_args()

    ctrl = CMGStuV2SimController(
        xml_path       = args.xml_file,
        policy_path    = args.policy_path,
        cmg_model_path = args.cmg_model,
        cmg_data_path  = args.cmg_data,
        urdf_path      = args.urdf,
        cmd_vx         = args.cmd_vx,
        cmd_vy         = args.cmd_vy,
        cmd_yaw        = args.cmd_yaw,
        device         = args.device,
        record_video   = args.record_video,
    )
    ctrl.run(args.duration)


if __name__ == "__main__":
    main()
