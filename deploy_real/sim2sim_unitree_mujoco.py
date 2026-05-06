"""
CMG Student V2 Sim-to-Sim via unitree_mujoco (DDS interface).

Uses the same DDS protocol as the real robot (unitree_sdk2py), talking to
unitree_mujoco which runs as a separate process. The controller code is
nearly identical to deploy_real_cmg_stu_v2.py — only domain_id and interface differ.

Architecture:
  [This process]                         [unitree_mujoco process]
  CMGMotionLib → obs → Policy → LowCmd  ──DDS──→  MuJoCo sim
                                LowState ←─DDS──  MuJoCo sim

Setup:
  1. Install: pip install mujoco pygame unitree_sdk2py
  2. Clone unitree_mujoco: git clone https://github.com/unitreerobotics/unitree_mujoco.git
  3. In unitree_mujoco/simulate_python/config.py, set:
       ROBOT = "g1"
       ROBOT_SCENE = "../unitree_robots/g1/scene_23dof.xml"  (or scene.xml for 29dof)
       DOMAIN_ID = 1
       INTERFACE = "lo"

Run:
  Terminal 1 (simulator):
    cd unitree_mujoco/simulate_python
    python unitree_mujoco.py

  Terminal 2 (this controller):
    cd deploy_real
    python sim2sim_unitree_mujoco.py \\
        --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \\
        --cmd_vx 1.5

  Keyboard controls (in unitree_mujoco viewer, NOT this terminal):
    Viewer has its own camera controls. Use --cmd_vx/vy/yaw to set initial speed.
    Press Ctrl+C in this terminal to stop.
"""

import os
import sys
import time
import argparse
import math
import threading
import numpy as np
import torch

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
from data_utils.rot_utils import quatToEuler   # (w,x,y,z) → (roll,pitch,yaw)

# DDS communication (same as real robot)
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC


# ── constants (must match training config G1MimicCMGStuV2Cfg) ───────────────
NUM_ACTIONS   = 23
CONTROL_DT    = 0.02           # 50 Hz — matches CMG_DT
CMG_DT        = 0.02
ACTION_SCALE  = 0.5

TAR_OBS_STEPS = list(range(1, 21))   # [1 … 20] future frames

# obs scaling
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANKLE_IDX     = [4, 5, 10, 11]

DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left  leg  (6)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg  (6)
     0.0, 0.0, 0.0,                    # waist      (3)
     0.0, 0.4, 0.0, 1.2,               # left  arm  (4)
     0.0,-0.4, 0.0, 1.2,               # right arm  (4)
], dtype=np.float32)

# TWIST 23-DOF → G1 DDS 29-motor index mapping
# Legs (0-11) and waist (12-14) are identity.
# Left arm TWIST[15-18] → DDS motor[15-18]
# Right arm TWIST[19-22] → DDS motor[22-25]
LEG_MOTOR_IDX       = list(range(12))             # TWIST 0-11 → DDS 0-11
WAIST_MOTOR_IDX     = [12, 13, 14]                # TWIST 12-14 → DDS 12-14
LEFT_ARM_MOTOR_IDX  = [15, 16, 17, 18]            # TWIST 15-18 → DDS 15-18
RIGHT_ARM_MOTOR_IDX = [22, 23, 24, 25]            # TWIST 19-22 → DDS 22-25

# Combined: TWIST index i → DDS motor index
TWIST_TO_DDS_MOTOR = (
    LEG_MOTOR_IDX + WAIST_MOTOR_IDX +
    LEFT_ARM_MOTOR_IDX + RIGHT_ARM_MOTOR_IDX
)
assert len(TWIST_TO_DDS_MOTOR) == 23

# PD gains per TWIST DOF (must match g1.yaml)
KPS = np.array([
    100, 100, 100, 150, 40, 40,   # left leg
    100, 100, 100, 150, 40, 40,   # right leg
    150, 150, 150,                 # waist
     40,  40,  40, 40,            # left arm
     40,  40,  40, 40,            # right arm
], dtype=np.float32)

KDS = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    4, 4, 4,
    5, 5, 5, 5,
    5, 5, 5, 5,
], dtype=np.float32)

# Wrist motor indices and gains (not controlled by policy)
WRIST_MOTOR_IDX = [19, 20, 21, 26, 27, 28]
WRIST_KP = 20.0
WRIST_KD = 1.0


# ── priv_mimic_obs builder (same as deploy_real_cmg_stu_v2.py) ──────────────
def build_priv_mimic_obs(cmg_lib: CMGMotionLib, device: str) -> torch.Tensor:
    """Return (1, 1160) priv_mimic tensor from the CMG trajectory buffer."""
    dev = torch.device(device)
    num_steps = len(TAR_OBS_STEPS)

    current_time = cmg_lib._motion_times[0].item()
    obs_times = torch.tensor(
        [current_time + s * CMG_DT for s in TAR_OBS_STEPS],
        device=dev, dtype=torch.float32,
    )
    motion_ids = torch.zeros(num_steps, dtype=torch.long, device=dev)

    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = \
        cmg_lib.calc_motion_frame(motion_ids, obs_times)

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

    root_vel_local    = quat_rotate_inverse_xyzw(root_rot, root_vel)
    root_ang_vel_z    = quat_rotate_inverse_xyzw(root_rot, root_ang_vel)[:, 2:3]
    key_body_pos_flat = body_pos.reshape(num_steps, -1)

    per_step = torch.cat([
        root_pos[:, 2:3],
        roll.unsqueeze(1),
        pitch.unsqueeze(1),
        yaw.unsqueeze(1),
        root_vel_local,
        root_ang_vel_z,
        dof_pos,
        key_body_pos_flat,
    ], dim=-1)

    return per_step.reshape(1, -1)


# ── DDS-based sim environment ───────────────────────────────────────────────
class G1DDSSimEnv:
    """Talks to unitree_mujoco via DDS, same protocol as real robot.

    Difference from real robot:
      - domain_id = 1  (sim), not 0 (real)
      - interface = "lo" (loopback), not physical NIC
    """

    def __init__(self, domain_id: int = 1, interface: str = "lo"):
        ChannelFactoryInitialize(domain_id, interface)

        # Publisher for motor commands
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_pub.Init()

        # Subscriber for robot state
        self._low_state = unitree_hg_msg_dds__LowState_()
        self._state_lock = threading.Lock()
        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_sub.Init(self._state_callback, 10)

        self.crc = CRC()

        # Initialize command message
        self.low_cmd.mode_pr = 0   # PR mode for ankle
        self.low_cmd.mode_machine = 0
        for i in range(35):
            self.low_cmd.motor_cmd[i].mode = 1  # enable
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].qd = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

        self._connected = False

    def _state_callback(self, msg: LowStateHG):
        with self._state_lock:
            self._low_state = msg
            self.low_cmd.mode_machine = msg.mode_machine
            if not self._connected:
                self._connected = True

    def wait_for_connection(self, timeout: float = 30.0):
        """Block until unitree_mujoco starts publishing LowState."""
        print("[DDS] Waiting for unitree_mujoco to connect...")
        t0 = time.time()
        while not self._connected:
            time.sleep(0.1)
            if time.time() - t0 > timeout:
                raise TimeoutError(
                    f"No LowState received within {timeout}s. "
                    "Is unitree_mujoco running with DOMAIN_ID=1?"
                )
        print("[DDS] Connected to unitree_mujoco.")

    def get_robot_state(self):
        """Returns (dof_pos_23, dof_vel_23, quat_wxyz, ang_vel) like G1RealWorldEnv."""
        with self._state_lock:
            state = self._low_state

        dof_pos = np.zeros(NUM_ACTIONS, dtype=np.float32)
        dof_vel = np.zeros(NUM_ACTIONS, dtype=np.float32)

        for twist_idx in range(NUM_ACTIONS):
            motor_idx = TWIST_TO_DDS_MOTOR[twist_idx]
            dof_pos[twist_idx] = state.motor_state[motor_idx].q
            dof_vel[twist_idx] = state.motor_state[motor_idx].dq

        # IMU: quaternion (w,x,y,z), gyroscope (3,)
        quat_wxyz = np.array(state.imu_state.quaternion, dtype=np.float32)
        ang_vel   = np.array(state.imu_state.gyroscope, dtype=np.float32)

        return dof_pos, dof_vel, quat_wxyz, ang_vel

    def send_action(self, target_dof_pos_23: np.ndarray):
        """Send 23-DOF position targets via DDS LowCmd."""
        # Policy-controlled joints
        for twist_idx in range(NUM_ACTIONS):
            motor_idx = TWIST_TO_DDS_MOTOR[twist_idx]
            self.low_cmd.motor_cmd[motor_idx].q   = float(target_dof_pos_23[twist_idx])
            self.low_cmd.motor_cmd[motor_idx].qd  = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp   = float(KPS[twist_idx])
            self.low_cmd.motor_cmd[motor_idx].kd   = float(KDS[twist_idx])
            self.low_cmd.motor_cmd[motor_idx].tau  = 0.0

        # Wrist joints: hold at zero
        for motor_idx in WRIST_MOTOR_IDX:
            self.low_cmd.motor_cmd[motor_idx].q   = 0.0
            self.low_cmd.motor_cmd[motor_idx].qd  = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp   = WRIST_KP
            self.low_cmd.motor_cmd[motor_idx].kd   = WRIST_KD
            self.low_cmd.motor_cmd[motor_idx].tau  = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_pub.Write(self.low_cmd)

    def move_to_default_pos(self, duration: float = 2.0):
        """Slowly interpolate to default pose."""
        dof_pos, _, _, _ = self.get_robot_state()
        num_steps = int(duration / CONTROL_DT)

        for i in range(num_steps):
            alpha = i / num_steps
            target = dof_pos * (1 - alpha) + DEFAULT_DOF_POS * alpha
            self.send_action(target)
            time.sleep(CONTROL_DT)

        print(f"[Sim] Moved to default pose over {duration}s.")


# ── main controller ──────────────────────────────────────────────────────────
class CMGStuV2DDSController:
    def __init__(
        self,
        policy_path: str,
        cmg_model_path: str,
        cmg_data_path: str,
        urdf_path: str,
        cmd_vx: float = 1.5,
        cmd_vy: float = 0.0,
        cmd_yaw: float = 0.0,
        device: str = "cpu",
        domain_id: int = 1,
        interface: str = "lo",
    ):
        self.device = device

        # ── DDS sim environment ──────────────────────────────────────────
        print("[DDS] Initializing G1DDSSimEnv...")
        self.env = G1DDSSimEnv(domain_id=domain_id, interface=interface)
        self.env.wait_for_connection()

        # ── CMG motion lib ───────────────────────────────────────────────
        print("[CMG] Loading CMGMotionLib...")
        self.cmg = CMGMotionLib(
            cmg_model_path  = cmg_model_path,
            cmg_data_path   = cmg_data_path,
            urdf_path       = urdf_path,
            device          = device,
            num_envs        = 1,
            episode_length_s= 300.0,
            dt              = CMG_DT,
            vx_range        = (cmd_vx, cmd_vx),
            vy_range        = (cmd_vy, cmd_vy),
            yaw_range       = (cmd_yaw, cmd_yaw),
        )
        env_ids  = torch.zeros(1, dtype=torch.long, device=device)
        init_cmd = torch.tensor([[cmd_vx, cmd_vy, cmd_yaw]], device=device)
        self.cmg.reset(env_ids, commands=init_cmd)
        self.cmg._mirror_flags[:] = False
        print(f"[CMG] Ready  vx={cmd_vx:.2f}  vy={cmd_vy:.2f}  yaw={cmd_yaw:.2f}")

        # ── JIT policy ───────────────────────────────────────────────────
        print(f"[Policy] Loading {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def _get_obs(self, dof_pos, dof_vel, quat_wxyz, ang_vel) -> torch.Tensor:
        """Build 1237-dim student obs: priv_mimic(1160) + proprio(77)."""
        priv_mimic = build_priv_mimic_obs(self.cmg, self.device)

        rpy = quatToEuler(quat_wxyz)
        dof_vel_obs = dof_vel.copy()
        dof_vel_obs[ANKLE_IDX] = 0.0

        cmd = self.cmg.get_user_commands()[0].cpu().numpy()

        proprio = np.concatenate([
            ang_vel * ANG_VEL_SCALE,
            rpy[:2],
            (dof_pos - DEFAULT_DOF_POS) * DOF_POS_SCALE,
            dof_vel_obs * DOF_VEL_SCALE,
            self.last_action,
            cmd,
        ], dtype=np.float32)

        proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
        return torch.cat([priv_mimic, proprio_t], dim=-1)

    def run(self, duration: float = 30.0):
        """Main control loop."""
        # Move to default pose first
        self.env.move_to_default_pos(duration=2.0)

        # Hold default pose for 1 second
        print("[Sim] Holding default pose for 1s...")
        for _ in range(int(1.0 / CONTROL_DT)):
            self.env.send_action(DEFAULT_DOF_POS)
            time.sleep(CONTROL_DT)

        print(f"[Sim] Starting policy loop for {duration}s. Press Ctrl+C to stop.")
        all_envs = torch.zeros(1, dtype=torch.long, device=self.device)
        num_steps = int(duration / CONTROL_DT)

        try:
            for step in range(num_steps):
                t_start = time.time()

                # Step CMG
                self.cmg.step(all_envs)

                # Get state from unitree_mujoco via DDS
                dof_pos, dof_vel, quat, ang_vel = self.env.get_robot_state()

                # Build obs & infer
                obs = self._get_obs(dof_pos, dof_vel, quat, ang_vel)
                with torch.no_grad():
                    raw_action = self.policy(obs).cpu().numpy().squeeze()

                self.last_action = raw_action.copy()
                raw_action = np.clip(raw_action, -10.0, 10.0)
                target_dof_pos = DEFAULT_DOF_POS + raw_action * ACTION_SCALE

                # Send via DDS
                self.env.send_action(target_dof_pos)

                # Timing
                elapsed = time.time() - t_start
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)

                if step % 50 == 0:
                    cmd = self.cmg.get_user_commands()[0].cpu().numpy()
                    print(f"  step {step}/{num_steps}  "
                          f"cmd=[{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]  "
                          f"dt={elapsed*1000:.1f}ms")

        except KeyboardInterrupt:
            print("\n[Sim] KeyboardInterrupt.")
        finally:
            # Send zero-torque to stop
            for _ in range(10):
                self.env.send_action(DEFAULT_DOF_POS)
                time.sleep(CONTROL_DT)
            print("[Sim] Done.")


# ── entry point ──────────────────────────────────────────────────────────────
def main():
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(
        description="CMG Student V2 Sim2Sim via unitree_mujoco (DDS)")
    parser.add_argument("--policy_path", required=True,
                        help="Path to JIT-exported student policy (.pt)")
    parser.add_argument("--cmg_model",
                        default=os.path.join(REPO, "cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"))
    parser.add_argument("--cmg_data",
                        default=os.path.join(REPO, "cmg_workspace/dataloader/cmg_training_data.pt"))
    parser.add_argument("--urdf",
                        default=os.path.join(REPO, "assets/g1/g1_custom_collision_with_fixed_hand.urdf"))
    parser.add_argument("--cmd_vx",  type=float, default=1.5)
    parser.add_argument("--cmd_vy",  type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Sim duration (seconds)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--domain_id", type=int, default=1,
                        help="DDS domain ID (1=sim, 0=real)")
    parser.add_argument("--interface", type=str, default="lo",
                        help="Network interface (lo=sim loopback, eno1=real robot)")
    args = parser.parse_args()

    ctrl = CMGStuV2DDSController(
        policy_path    = args.policy_path,
        cmg_model_path = args.cmg_model,
        cmg_data_path  = args.cmg_data,
        urdf_path      = args.urdf,
        cmd_vx         = args.cmd_vx,
        cmd_vy         = args.cmd_vy,
        cmd_yaw        = args.cmd_yaw,
        device         = args.device,
        domain_id      = args.domain_id,
        interface      = args.interface,
    )
    ctrl.run(duration=args.duration)


if __name__ == "__main__":
    main()
