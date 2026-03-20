"""
CMG Student V2 Real-Robot Deployment (G1).

Architecture:
  CMGMotionLib (in-process)  →  build priv_mimic_obs (1160)  ─┐
  G1RealWorldEnv             →  build proprio_obs    (  77)  ─┴─→  Policy  →  send_robot_action

No Redis, no high-level server needed.

Startup sequence:
  1. Power on G1, launch this script.
  2. Press [START] on remote → robot enters zero-torque (support by hand).
  3. Robot moves to default pose over 2s.
  4. Press [A]  on remote → enters hold state (verify pose is correct).
  5. Press [A]  again     → main policy loop begins.
  6. Press [Select]       → exits and cuts torque.

Keyboard commands (remote controller):
  L-stick up/down  → not used (CMG cmd set via --cmd_vx at launch or modified live)

Usage:
  cd deploy_real
  python deploy_real_cmg_stu_v2.py \\
      --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \\
      --cmd_vx 1.0 --cmd_vy 0.0 --cmd_yaw 0.0 \\
      --net eno1
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

# ── path setup ──────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in [
    os.path.join(REPO_ROOT, "pose"),
    os.path.join(REPO_ROOT, "cmg_workspace"),
    os.path.join(REPO_ROOT, "rsl_rl"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pose.utils.cmg_motion_lib import CMGMotionLib
from data_utils.rot_utils import quatToEuler          # (w,x,y,z) → (roll,pitch,yaw)
from robot_control.g1_wrapper import G1RealWorldEnv
from robot_control.config import Config
from robot_control.common.remote_controller import KeyMap


# ── constants (must match training config G1MimicCMGStuV2Cfg) ───────────────
NUM_ACTIONS   = 23
CONTROL_DT    = 0.02           # 50 Hz — matches CMG_DT
ACTION_SCALE  = 0.5
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANKLE_IDX     = [4, 5, 10, 11]  # zero dof_vel for ankles in obs

TAR_OBS_STEPS = list(range(1, 21))   # [1 … 20]  future frames queried from CMG

DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left  leg  (6)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg  (6)
     0.0, 0.0, 0.0,                    # waist      (3)
     0.0, 0.4, 0.0, 1.2,               # left  arm  (4)
     0.0,-0.4, 0.0, 1.2,               # right arm  (4)
], dtype=np.float32)


# ── priv_mimic_obs builder (mirrors training _get_mimic_obs) ────────────────
def build_priv_mimic_obs(cmg_lib: CMGMotionLib, device: str) -> torch.Tensor:
    """Return (1, 1160) priv_mimic tensor from the CMG trajectory buffer."""
    import math
    dev = torch.device(device)
    current_time = cmg_lib._motion_times[0].item()

    obs_times = torch.tensor(
        [current_time + s * CONTROL_DT for s in TAR_OBS_STEPS],
        device=dev, dtype=torch.float32,
    )
    motion_ids = torch.zeros(len(TAR_OBS_STEPS), dtype=torch.long, device=dev)

    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = \
        cmg_lib.calc_motion_frame(motion_ids, obs_times)

    # euler from xyzw quaternion
    def euler_from_xyzw(q):
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        roll  = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        sinp  = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1,
                            torch.sign(sinp) * math.pi / 2,
                            torch.asin(sinp.clamp(-1, 1)))
        yaw   = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    roll, pitch, yaw = euler_from_xyzw(root_rot)

    def quat_rotate_inverse_xyzw(q, v):
        w = q[:, 3:4]; xyz = q[:, :3]
        a = v * (2 * w ** 2 - 1)
        b = torch.cross(xyz, v, dim=-1) * (2 * w)
        c = xyz * (xyz * v).sum(dim=-1, keepdim=True) * 2
        return a - b + c

    root_vel_local     = quat_rotate_inverse_xyzw(root_rot, root_vel)          # (20,3)
    root_ang_vel_z     = quat_rotate_inverse_xyzw(root_rot, root_ang_vel)[:, 2:3]  # (20,1)
    key_body_pos_flat  = body_pos.reshape(len(TAR_OBS_STEPS), -1)              # (20,27)

    per_step = torch.cat([
        root_pos[:, 2:3],       # height   (1)
        roll.unsqueeze(1),      # roll     (1)
        pitch.unsqueeze(1),     # pitch    (1)
        yaw.unsqueeze(1),       # yaw      (1)
        root_vel_local,         # vel      (3)
        root_ang_vel_z,         # yaw_rate (1)
        dof_pos,                # joints  (23)
        key_body_pos_flat,      # bodies  (27)
    ], dim=-1)   # (20, 58)

    return per_step.reshape(1, -1)   # (1, 1160)


# ── main controller ──────────────────────────────────────────────────────────
class CMGStuV2RealController:
    def __init__(
        self,
        policy_path: str,
        config_path: str,
        cmg_model_path: str,
        cmg_data_path: str,
        urdf_path: str,
        cmd_vx: float = 1.0,
        cmd_vy: float = 0.0,
        cmd_yaw: float = 0.0,
        device: str = "cuda",
        net: str = "eno1",
    ):
        self.device   = device
        self._init_vx = cmd_vx
        self._init_vy = cmd_vy
        self._init_yaw = cmd_yaw

        # ── robot hardware ────────────────────────────────────────────────
        print("[Robot] Initializing G1RealWorldEnv...")
        self.config = Config(config_path)
        self.env    = G1RealWorldEnv(net=net, config=self.config)

        # ── CMG motion lib ────────────────────────────────────────────────
        print("[CMG] Loading CMGMotionLib...")
        self.cmg = CMGMotionLib(
            cmg_model_path  = cmg_model_path,
            cmg_data_path   = cmg_data_path,
            urdf_path       = urdf_path,
            device          = device,
            num_envs        = 1,
            episode_length_s= 300.0,
            dt              = CONTROL_DT,
            vx_range        = (cmd_vx,  cmd_vx),
            vy_range        = (cmd_vy,  cmd_vy),
            yaw_range       = (cmd_yaw, cmd_yaw),
        )
        env_ids  = torch.zeros(1, dtype=torch.long, device=device)
        init_cmd = torch.tensor([[cmd_vx, cmd_vy, cmd_yaw]], device=device)
        self.cmg.reset(env_ids, commands=init_cmd)
        self.cmg._mirror_flags[:] = False   # no symmetry augmentation on real robot
        print(f"[CMG] Ready  vx={cmd_vx:.2f}  vy={cmd_vy:.2f}  yaw={cmd_yaw:.2f}")

        # ── JIT policy ────────────────────────────────────────────────────
        print(f"[Policy] Loading {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # ── obs builder ─────────────────────────────────────────────────────────
    def _get_obs(self, dof_pos, dof_vel, quat_wxyz, ang_vel) -> torch.Tensor:
        """Build 1237-dim student obs: priv_mimic(1160) + proprio(77)."""
        priv_mimic = build_priv_mimic_obs(self.cmg, self.device)   # (1,1160)

        rpy = quatToEuler(quat_wxyz)   # (roll, pitch, yaw)

        dof_vel_obs = dof_vel.copy()
        dof_vel_obs[ANKLE_IDX] = 0.0

        cmd = self.cmg.get_user_commands()[0].cpu().numpy()   # (3,)

        proprio = np.concatenate([
            ang_vel * ANG_VEL_SCALE,                           # 3
            rpy[:2],                                           # 2  (roll, pitch)
            (dof_pos - DEFAULT_DOF_POS) * DOF_POS_SCALE,      # 23
            dof_vel_obs * DOF_VEL_SCALE,                       # 23
            self.last_action,                                  # 23
            cmd,                                               # 3
        ], dtype=np.float32)   # 77

        proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
        return torch.cat([priv_mimic, proprio_t], dim=-1)   # (1,1237)

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self):
        # ── startup sequence ──────────────────────────────────────────────
        self.env.zero_torque_state()         # waits for [START]
        self.env.move_to_default_pos()       # 2s interpolation to default pose
        self.env.default_pos_state()         # hold & wait for [A]

        print("[Deploy] Press [A] again to start policy loop.")
        while self.env.remote_controller.button[KeyMap.A] != 1:
            dof_pos, dof_vel, quat, ang_vel = self.env.get_robot_state()
            self.env.send_robot_action(DEFAULT_DOF_POS)
            time.sleep(CONTROL_DT)

        print("[Deploy] Policy loop started! Press [Select] to exit.")
        print(f"[Deploy] Initial cmd: vx={self._init_vx:.2f}  vy={self._init_vy:.2f}  yaw={self._init_yaw:.2f}")

        all_envs = torch.zeros(1, dtype=torch.long, device=self.device)

        try:
            while True:
                t_start = time.time()

                # ── exit condition ─────────────────────────────────────────
                if self.env.remote_controller.button[KeyMap.select] == 1:
                    print("[Deploy] [Select] pressed, exiting.")
                    break

                # ── step CMG ──────────────────────────────────────────────
                self.cmg.step(all_envs)

                # ── get robot state ───────────────────────────────────────
                dof_pos, dof_vel, quat, ang_vel = self.env.get_robot_state()

                # ── build obs & infer ─────────────────────────────────────
                obs = self._get_obs(dof_pos, dof_vel, quat, ang_vel)
                with torch.no_grad():
                    raw_action = self.policy(obs).cpu().numpy().squeeze()   # (23,)

                self.last_action = raw_action.copy()
                raw_action = np.clip(raw_action, -10.0, 10.0)
                target_dof_pos = DEFAULT_DOF_POS + raw_action * ACTION_SCALE

                # ── send action ───────────────────────────────────────────
                self.env.send_robot_action(target_dof_pos)

                # ── timing ────────────────────────────────────────────────
                elapsed = time.time() - t_start
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)
                else:
                    print(f"[Deploy] WARNING: loop overrun {elapsed*1000:.1f}ms > {CONTROL_DT*1000:.0f}ms")

        except KeyboardInterrupt:
            print("\n[Deploy] KeyboardInterrupt.")
        finally:
            print("[Deploy] Entering zero torque state.")
            self.env.zero_torque_state()


# ── entry point ─────────────────────────────────────────────────────────────
def main():
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(description="CMG Student V2 Real-Robot Deploy")
    parser.add_argument("--policy_path", required=True,
                        help="Path to JIT-exported student policy (.pt)")
    parser.add_argument("--config_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "robot_control/configs/g1.yaml"))
    parser.add_argument("--cmg_model",
                        default=os.path.join(REPO, "cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"))
    parser.add_argument("--cmg_data",
                        default=os.path.join(REPO, "cmg_workspace/dataloader/cmg_training_data.pt"))
    parser.add_argument("--urdf",
                        default=os.path.join(REPO, "assets/g1/g1_custom_collision_with_fixed_hand.urdf"))
    parser.add_argument("--cmd_vx",  type=float, default=1.0,
                        help="Initial forward velocity command (m/s)")
    parser.add_argument("--cmd_vy",  type=float, default=0.0,
                        help="Initial lateral velocity command (m/s)")
    parser.add_argument("--cmd_yaw", type=float, default=0.0,
                        help="Initial yaw rate command (rad/s)")
    parser.add_argument("--device",  type=str,   default="cuda")
    parser.add_argument("--net",     type=str,   default="eno1",
                        help="Network interface for DDS (e.g. eno1, eth0)")
    args = parser.parse_args()

    ctrl = CMGStuV2RealController(
        policy_path    = args.policy_path,
        config_path    = args.config_path,
        cmg_model_path = args.cmg_model,
        cmg_data_path  = args.cmg_data,
        urdf_path      = args.urdf,
        cmd_vx         = args.cmd_vx,
        cmd_vy         = args.cmd_vy,
        cmd_yaw        = args.cmd_yaw,
        device         = args.device,
        net            = args.net,
    )
    ctrl.run()


if __name__ == "__main__":
    main()
