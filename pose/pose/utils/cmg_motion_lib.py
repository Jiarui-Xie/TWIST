"""
CMG Motion Library - Uses a Conditional Motion Generator to produce motion references.

This module provides a MotionLib-compatible interface that generates motion references
using a trained CMG model instead of loading pre-recorded motion clips.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple

import sys
# 添加 cmg_workspace 到 Python 路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_cmg_workspace = os.path.abspath(os.path.join(_current_dir, '..', '..', '..', 'cmg_workspace'))
if _cmg_workspace not in sys.path:
    sys.path.insert(0, _cmg_workspace)

from module.cmg import CMG
from pose.utils.forward_kinematics import ForwardKinematics


# CMG uses 29 DOF, G1 training uses 23 DOF
# Mapping: skip wrist joints (19-21 for left, 26-28 for right)
CMG_TO_G1_INDICES = [
    0, 1, 2, 3, 4, 5,       # Left leg (6)
    6, 7, 8, 9, 10, 11,     # Right leg (6)
    12, 13, 14,             # Waist (3)
    15, 16, 17, 18,         # Left arm (4)
    22, 23, 24, 25,         # Right arm (4) - skip left wrist 19-21
]

# Mirror indices: swap left and right for 23 DOF
# Used to prevent left-right bias in CMG training
DOF_MIRROR_INDICES_23 = [
    6, 7, 8, 9, 10, 11,     # right leg -> left leg position
    0, 1, 2, 3, 4, 5,       # left leg -> right leg position
    12, 13, 14,              # waist stays
    19, 20, 21, 22,          # right arm -> left arm position
    15, 16, 17, 18,          # left arm -> right arm position
]

# Sign flips: roll and yaw joints flip sign when mirrored
# Joint order per group: pitch, roll, yaw, knee, ankle_pitch, ankle_roll (legs)
#                        shoulder_pitch, shoulder_roll, shoulder_yaw, elbow (arms)
#                        yaw, roll, pitch (waist)
DOF_MIRROR_SIGNS_23 = [
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,   # left leg (from right)
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,   # right leg (from left)
    -1.0, -1.0, 1.0,                     # waist: yaw, roll, pitch
    1.0, -1.0, -1.0, 1.0,               # left arm (from right)
    1.0, -1.0, -1.0, 1.0,               # right arm (from left)
]

# Key body mirror: swap left and right
# Order: [left_hand, right_hand, left_ankle, right_ankle, left_knee, right_knee,
#          left_elbow, right_elbow, head]
KEYBODY_MIRROR_INDICES = [1, 0, 3, 2, 5, 4, 7, 6, 8]


class CMGMotionLib:
    """
    Motion library that uses CMG (Conditional Motion Generator) to generate
    motion references in real-time based on velocity commands.

    Provides the same interface as MotionLib for compatibility with HumanoidMimic.

    Key Design:
    - Maintains a trajectory buffer for each environment to support future frame queries
    - The trajectory buffer stores pre-generated motion states for ~2 seconds ahead
    - When queried for future timesteps (by _get_mimic_obs), interpolates from buffer
    """

    # Number of frames to pre-generate in the trajectory buffer (at CMG's 50 Hz)
    TRAJECTORY_BUFFER_FRAMES = 100  # 2 seconds at 50 Hz

    def __init__(
        self,
        cmg_model_path: str,
        cmg_data_path: str,
        urdf_path: str,
        device: str,
        num_envs: int,
        episode_length_s: float = 10.0,
        dt: float = 0.02,  # 50 Hz, matches CMG training
        vx_range: Tuple[float, float] = (0.5, 1.5),
        vy_range: Tuple[float, float] = (-0.3, 0.3),
        yaw_range: Tuple[float, float] = (-0.5, 0.5),
        root_height: float = 0.75,
        ramp_enabled: bool = False,
        ramp_up_range: Tuple[float, float] = (1.5, 1.5),
        ramp_down_range: Tuple[float, float] = (3.0, 3.0),
        ramp_stand_duration: float = 1.0,
        ramp_crawl_range: Tuple[float, float] = (1.0, 1.0),
        ramp_crawl_ratio: float = 0.01,
        ramp_probability: float = 0.5,
        ramp_floor_ratio: float = 0.0,
        ramp_min_steady: float = 3.0,
    ):
        """
        Initialize CMG motion library.

        Args:
            cmg_model_path: Path to trained CMG model checkpoint
            cmg_data_path: Path to CMG training data (for normalization stats)
            urdf_path: Path to robot URDF for forward kinematics
            device: Compute device ('cuda' or 'cpu')
            num_envs: Number of parallel environments
            episode_length_s: Episode length in seconds
            dt: Time step for CMG inference (should match training, typically 0.02s)
            vx_range: Range for forward velocity commands (m/s)
            vy_range: Range for lateral velocity commands (m/s)
            yaw_range: Range for yaw rate commands (rad/s)
            root_height: Default root height (m)
        """
        self._device = device
        self._num_envs = num_envs
        self._episode_length_s = episode_length_s
        self._dt = dt
        self._vx_range = vx_range
        self._vy_range = vy_range
        self._yaw_range = yaw_range
        self._root_height = root_height

        # Ramp velocity profile parameters
        # Profile: [stand → ramp_up → steady → ramp_down → crawl → stand]
        # Durations are per-env random (sampled at each reset)
        self._ramp_enabled = ramp_enabled
        self._ramp_up_range = ramp_up_range
        self._ramp_down_range = ramp_down_range
        self._ramp_stand_duration = ramp_stand_duration  # fixed stand duration
        self._ramp_crawl_range = ramp_crawl_range
        self._ramp_crawl_ratio = ramp_crawl_ratio
        self._ramp_probability = ramp_probability
        self._ramp_floor_ratio = ramp_floor_ratio
        self._ramp_min_steady = ramp_min_steady

        # Load CMG model and stats
        self._load_cmg_model(cmg_model_path, cmg_data_path)

        # Initialize forward kinematics for key body position computation
        self._fk = ForwardKinematics(urdf_path, device)

        # Key body names (same order as G1 config key_bodies)
        self._body_link_list = [
            "left_rubber_hand", "right_rubber_hand",
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_knee_link", "right_knee_link",
            "left_elbow_link", "right_elbow_link",
            "head_mocap"
        ]

        # Initialize motion state buffers
        self._init_buffers()

        # Build standing pose for ramp stand phases
        self._build_standing_pose()

        # Build velocity estimator from training data
        self._build_velocity_estimator()

        print(f"[CMGMotionLib] Initialized with {num_envs} envs, "
              f"vx=[{vx_range[0]:.1f}, {vx_range[1]:.1f}], "
              f"vy=[{vy_range[0]:.1f}, {vy_range[1]:.1f}], "
              f"yaw=[{yaw_range[0]:.1f}, {yaw_range[1]:.1f}]")
        if self._ramp_enabled:
            print(f"[CMGMotionLib] Ramp: stand={ramp_stand_duration:.1f}s(fixed), "
                  f"ramp_up={ramp_up_range}, ramp_down={ramp_down_range}, "
                  f"crawl={ramp_crawl_range}(ratio={ramp_crawl_ratio}), "
                  f"floor={ramp_floor_ratio:.2f}, min_steady={ramp_min_steady:.1f}s")

    def _load_cmg_model(self, model_path: str, data_path: str):
        """Load CMG model and normalization statistics."""
        # Load training data stats
        data = torch.load(data_path, weights_only=False, map_location=self._device)
        self._stats = data["stats"]

        # Get initial motion states from samples for reset
        self._init_samples = data["samples"]

        # Create model with same architecture as training
        self._cmg_model = CMG(
            motion_dim=self._stats["motion_dim"],
            command_dim=self._stats["command_dim"],
            hidden_dim=512,
            num_experts=4,
            num_layers=3,
        )

        # Load trained weights
        checkpoint = torch.load(model_path, weights_only=False, map_location=self._device)
        self._cmg_model.load_state_dict(checkpoint["model_state_dict"])
        self._cmg_model = self._cmg_model.to(self._device)
        self._cmg_model.eval()

        # Pre-compute normalization tensors
        self._motion_mean = torch.from_numpy(self._stats["motion_mean"]).to(self._device)
        self._motion_std = torch.from_numpy(self._stats["motion_std"]).to(self._device)
        self._cmd_min = torch.from_numpy(self._stats["command_min"]).to(self._device)
        self._cmd_max = torch.from_numpy(self._stats["command_max"]).to(self._device)

        print(f"[CMGMotionLib] Loaded CMG model from {model_path}")
        print(f"[CMGMotionLib] Motion dim: {self._stats['motion_dim']}, Command dim: {self._stats['command_dim']}")

    def _init_buffers(self):
        """Initialize state buffers for all environments."""
        # Current motion state (normalized): [pos_29, vel_29] = 58 dims
        self._current_motion_norm = torch.zeros(
            self._num_envs, self._stats["motion_dim"],
            device=self._device
        )

        # Trajectory buffer: stores pre-generated future motion states
        # Shape: (num_envs, buffer_frames, motion_dim)
        self._trajectory_buffer = torch.zeros(
            self._num_envs, self.TRAJECTORY_BUFFER_FRAMES, self._stats["motion_dim"],
            device=self._device
        )
        # Current frame index in buffer for each env
        self._buffer_frame_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # Root position trajectory buffer (for future queries)
        self._root_pos_buffer = torch.zeros(
            self._num_envs, self.TRAJECTORY_BUFFER_FRAMES, 3, device=self._device
        )
        self._root_rot_buffer = torch.zeros(
            self._num_envs, self.TRAJECTORY_BUFFER_FRAMES, 4, device=self._device
        )
        self._root_rot_buffer[:, :, 0] = 1.0  # Unit quaternion

        # Current velocity commands (instantaneous, updated by ramp schedule)
        self._commands = torch.zeros(self._num_envs, 3, device=self._device)

        # Target velocity commands (sampled at reset, constant during episode)
        self._target_commands = torch.zeros(self._num_envs, 3, device=self._device)

        # Per-env flag: whether this episode uses ramp profile
        self._ramp_enabled_flags = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # Per-env ramp durations (sampled at each reset)
        self._env_ramp_up = torch.zeros(self._num_envs, device=self._device)
        self._env_ramp_down = torch.zeros(self._num_envs, device=self._device)
        self._env_crawl = torch.zeros(self._num_envs, device=self._device)

        # Root state tracking
        self._root_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._root_pos[:, 2] = self._root_height

        self._root_rot = torch.zeros(self._num_envs, 4, device=self._device)
        self._root_rot[:, 0] = 1.0  # Unit quaternion [w, x, y, z]

        self._root_yaw = torch.zeros(self._num_envs, device=self._device)

        # Episode time tracking
        self._motion_times = torch.zeros(self._num_envs, device=self._device)

        # Motion IDs (used for interface compatibility, maps to command sets)
        self._motion_ids = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # DEPRECATED: _actual_commands was a linear-regression estimate of kinematic velocity.
        # All reward targets, observations, and root integration now use _commands directly.
        # Kept for backward compatibility only; not updated or read anywhere active.
        self._actual_commands = torch.zeros(self._num_envs, 3, device=self._device)

    def _build_standing_pose(self):
        """Build a normalized standing pose for ramp stand phases.

        During stand phases (v=0), we freeze the CMG and output this static pose
        instead of feeding v=0 to the CMG (which is out-of-distribution).

        The standing pose uses G1 default joint angles with zero velocities.
        """
        # G1 default joint angles (23 DOF)
        default_23 = torch.tensor([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left leg
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg
             0.0, 0.0, 0.0,                    # waist
             0.0, 0.4, 0.0, 1.2,               # left arm
             0.0,-0.4, 0.0, 1.2,               # right arm
        ], device=self._device, dtype=torch.float32)

        # Map 23 DOF back to 29 DOF (insert zeros for wrist joints)
        default_29 = torch.zeros(29, device=self._device)
        default_29[CMG_TO_G1_INDICES] = default_23

        # Standing motion state: [pos_29, vel_29] = 58 dims, vel=0
        standing_raw = torch.zeros(self._stats["motion_dim"], device=self._device)
        standing_raw[:29] = default_29

        # Normalize
        self._standing_pose_norm = self._normalize_motion(standing_raw.unsqueeze(0)).squeeze(0)

    # Mirror flags for left-right symmetry training
        self._mirror_flags = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._dof_mirror_indices = torch.tensor(DOF_MIRROR_INDICES_23, device=self._device, dtype=torch.long)
        self._dof_mirror_signs = torch.tensor(DOF_MIRROR_SIGNS_23, device=self._device, dtype=torch.float)
        self._keybody_mirror_indices = torch.tensor(KEYBODY_MIRROR_INDICES, device=self._device, dtype=torch.long)

    def _build_velocity_estimator(self):
        """Build a linear regression from 29 DOF velocities to [vx, vy, yaw].

        Uses CMG training data where commands correspond to actual root velocities
        from simulation. Fits least-squares: vel_cmd = dof_vel @ W + b
        """
        # Collect all (dof_vel, command) pairs from training samples
        all_dof_vels = []
        all_commands = []

        for sample in self._init_samples:
            motion = sample["motion"]  # (seq_len+1, 58)
            command = sample["command"]  # (seq_len, 3)
            seq_len = command.shape[0]

            # Extract DOF velocities (dims 29-57) for frames that have commands
            dof_vel = motion[:seq_len, 29:]  # (seq_len, 29)
            all_dof_vels.append(dof_vel)
            all_commands.append(command)

        # Stack into matrices
        X = np.concatenate(all_dof_vels, axis=0)  # (N, 29)
        Y = np.concatenate(all_commands, axis=0)   # (N, 3)

        # Add bias column
        X_bias = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # (N, 30)

        # Least-squares solve: X_bias @ [W; b] = Y
        # Using numpy lstsq for numerical stability
        result, residuals, rank, sv = np.linalg.lstsq(X_bias, Y, rcond=None)

        W = result[:29, :]  # (29, 3)
        b = result[29, :]   # (3,)

        # Store as torch tensors
        self._vel_est_W = torch.from_numpy(W).float().to(self._device)
        self._vel_est_b = torch.from_numpy(b).float().to(self._device)

        # Compute and print RMSE for diagnostics
        Y_pred = X @ W + b
        rmse = np.sqrt(np.mean((Y_pred - Y) ** 2, axis=0))
        print(f"[CMGMotionLib] Velocity estimator built from {X.shape[0]} frames")
        print(f"[CMGMotionLib] RMSE: vx={rmse[0]:.4f} m/s, vy={rmse[1]:.4f} m/s, yaw={rmse[2]:.4f} rad/s")

    @torch.no_grad()
    def _estimate_actual_velocity(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Estimate actual gait velocity from trajectory buffer DOF velocities.

        Applies linear regression per-frame, averages over frames 20-100 (skip warmup).

        Args:
            env_ids: Environment indices to estimate for

        Returns:
            Estimated [vx, vy, yaw] per env, shape (len(env_ids), 3)
        """
        # Get trajectory buffer for these envs
        traj = self._trajectory_buffer[env_ids]  # (n, 100, 58) normalized

        # Denormalize to get raw motion
        traj_raw = traj * self._motion_std + self._motion_mean  # (n, 100, 58)

        # Extract DOF velocities (dims 29-57)
        dof_vel = traj_raw[:, :, 29:]  # (n, 100, 29)

        # Apply linear regression: vel = dof_vel @ W + b
        vel_est = torch.matmul(dof_vel, self._vel_est_W) + self._vel_est_b  # (n, 100, 3)

        # Average over frames 20-100 (skip warmup)
        vel_avg = vel_est[:, 20:, :].mean(dim=1)  # (n, 3)

        return vel_avg

    def _sample_uniform(self, n: int, range: Tuple[float, float]) -> torch.Tensor:
        """Sample n values uniformly from [range[0], range[1]]."""
        return torch.rand(n, device=self._device) * (range[1] - range[0]) + range[0]

    def _sample_commands(self, n: int) -> torch.Tensor:
        """Sample random velocity commands within configured ranges."""
        vx = torch.rand(n, device=self._device) * (self._vx_range[1] - self._vx_range[0]) + self._vx_range[0]
        vy = torch.rand(n, device=self._device) * (self._vy_range[1] - self._vy_range[0]) + self._vy_range[0]
        yaw = torch.rand(n, device=self._device) * (self._yaw_range[1] - self._yaw_range[0]) + self._yaw_range[0]
        return torch.stack([vx, vy, yaw], dim=-1)

    def _compute_ramp_scale(self, env_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Compute velocity scale factor for 6-phase ramp profile with per-env durations.

        Profile: [stand → ramp_up → steady → ramp_down → crawl → stand]

        Stand duration is fixed (1s). ramp_up, ramp_down, crawl are per-env random.
        Steady phase fills remaining time.

        Args:
            env_ids: Environment indices, shape (n,)
            times: Episode-local times for each env, shape (n,)

        Returns:
            Scale factors in [min(floor, crawl), 1.0], shape (n,)
        """
        n = len(env_ids)
        scale = torch.ones(n, device=self._device)

        if not self._ramp_enabled:
            return scale

        ramp_mask = self._ramp_enabled_flags[env_ids]
        if not ramp_mask.any():
            return scale

        t_stand = self._ramp_stand_duration
        t_ep = self._episode_length_s
        floor = self._ramp_floor_ratio
        crawl = self._ramp_crawl_ratio

        # Per-env durations (vectorized)
        t_ramp_up = self._env_ramp_up[env_ids[ramp_mask]]   # (m,)
        t_ramp_down = self._env_ramp_down[env_ids[ramp_mask]]  # (m,)
        t_crawl = self._env_crawl[env_ids[ramp_mask]]       # (m,)

        t = times[ramp_mask]  # (m,)

        # Phase boundaries per env
        t1 = t_stand                                # end of initial stand (scalar)
        t2 = t1 + t_ramp_up                         # end of ramp-up (m,)
        t6 = t_ep                                   # episode end (scalar)
        t5 = t6 - t_stand                           # start of final stand (scalar)
        t4 = t5 - t_crawl                           # start of crawl → end of ramp-down (m,)
        t3 = t4 - t_ramp_down                       # start of ramp-down (m,)

        # Build scale per phase (vectorized with per-env boundaries)
        s = torch.ones_like(t)

        # Phase 1: initial stand [0, t1)
        mask_stand1 = t < t1
        s[mask_stand1] = floor

        # Phase 2: ramp-up [t1, t2) — floor → 1.0
        mask_ramp_up = (t >= t1) & (t < t2)
        if mask_ramp_up.any():
            progress = (t[mask_ramp_up] - t1) / t_ramp_up[mask_ramp_up]
            s[mask_ramp_up] = floor + (1.0 - floor) * progress

        # Phase 3: steady [t2, t3) — 1.0
        # (already 1.0)

        # Phase 4: ramp-down [t3, t4) — 1.0 → crawl
        mask_ramp_down = (t >= t3) & (t < t4)
        if mask_ramp_down.any():
            progress = (t4[mask_ramp_down] - t[mask_ramp_down]) / t_ramp_down[mask_ramp_down]
            s[mask_ramp_down] = crawl + (1.0 - crawl) * progress

        # Phase 5: crawl [t4, t5) — crawl_ratio
        mask_crawl = (t >= t4) & (t < t5)
        s[mask_crawl] = crawl

        # Phase 6: final stand [t5, t6] — floor
        mask_stand2 = t >= t5
        s[mask_stand2] = floor

        scale[ramp_mask] = s

        return scale

    def _get_init_motion(self, n: int) -> torch.Tensor:
        """Get initial motion states from training data samples."""
        # Randomly select initial states from training samples
        indices = np.random.randint(0, len(self._init_samples), size=n)
        # Vectorized extraction using list comprehension and stack
        init_motions = np.stack([self._init_samples[idx]["motion"][0] for idx in indices], axis=0)
        return torch.from_numpy(init_motions).float().to(self._device)

    def _normalize_motion(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalize motion state."""
        return (motion - self._motion_mean) / self._motion_std

    def _denormalize_motion(self, motion_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize motion state."""
        return motion_norm * self._motion_std + self._motion_mean

    def _normalize_command(self, command: torch.Tensor) -> torch.Tensor:
        """Normalize command to [-1, 1]."""
        return (command - self._cmd_min) / (self._cmd_max - self._cmd_min) * 2 - 1

    def _map_29_to_23(self, motion_29: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map CMG's 29 DOF to G1's 23 DOF."""
        if motion_29.dim() == 2:
            pos_29 = motion_29[:, :29]
            vel_29 = motion_29[:, 29:]
            pos_23 = pos_29[:, CMG_TO_G1_INDICES]
            vel_23 = vel_29[:, CMG_TO_G1_INDICES]
        else:
            # Handle 3D tensor (batch, time, features)
            pos_29 = motion_29[..., :29]
            vel_29 = motion_29[..., 29:]
            pos_23 = pos_29[..., CMG_TO_G1_INDICES]
            vel_23 = vel_29[..., CMG_TO_G1_INDICES]
        return pos_23, vel_23

    def _compute_root_state_at_time(self, env_idx: int, time_offset: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute root position and rotation at a given time offset from current."""
        vx_local = self._commands[env_idx, 0]
        vy_local = self._commands[env_idx, 1]
        yaw_rate = self._commands[env_idx, 2]

        # Current state
        base_yaw = self._root_yaw[env_idx]
        base_pos = self._root_pos[env_idx].clone()

        # Integrate yaw
        new_yaw = base_yaw + yaw_rate * time_offset

        # Average yaw for position integration (trapezoidal approximation)
        avg_yaw = base_yaw + yaw_rate * time_offset * 0.5
        cos_yaw = torch.cos(avg_yaw)
        sin_yaw = torch.sin(avg_yaw)

        # Integrate position
        vx_world = vx_local * cos_yaw - vy_local * sin_yaw
        vy_world = vx_local * sin_yaw + vy_local * cos_yaw

        new_pos = base_pos.clone()
        new_pos[0] += vx_world * time_offset
        new_pos[1] += vy_world * time_offset

        # Compute quaternion from yaw
        half_yaw = new_yaw * 0.5
        new_rot = torch.zeros(4, device=self._device)
        new_rot[0] = torch.cos(half_yaw)  # w
        new_rot[3] = torch.sin(half_yaw)  # z

        return new_pos, new_rot

    @torch.no_grad()
    def _generate_trajectory(self, env_ids: torch.Tensor):
        """Pre-generate trajectory buffer for specified environments (vectorized).

        When ramp is enabled, each frame uses a time-varying command scaled by the
        trapezoidal ramp profile. Root position/rotation are integrated frame-by-frame
        (Euler) instead of using closed-form time_offset * velocity.
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # Get current state
        current_norm = self._current_motion_norm[env_ids].clone()

        # Store initial position/rotation
        root_pos = self._root_pos[env_ids].clone()  # (n, 3)
        root_yaw = self._root_yaw[env_ids].clone()  # (n,)

        # Base time for ramp scale computation
        base_time = self._motion_times[env_ids]  # (n,)

        # Generate trajectory with per-frame ramp-scaled commands
        for frame in range(self.TRAJECTORY_BUFFER_FRAMES):
            self._trajectory_buffer[env_ids, frame] = current_norm

            frame_time = base_time + frame * self._dt
            scale = self._compute_ramp_scale(env_ids, frame_time)  # (n,)
            frame_cmd = self._target_commands[env_ids] * scale.unsqueeze(-1)  # (n, 3)
            frame_cmd_norm = self._normalize_command(frame_cmd)

            current_norm = self._cmg_model(current_norm, frame_cmd_norm)

        # Reset buffer frame index
        self._buffer_frame_idx[env_ids] = 0

        # Build root position/rotation buffers with per-frame Euler integration
        pos = root_pos.clone()  # (n, 3)
        yaw = root_yaw.clone()  # (n,)

        for frame in range(self.TRAJECTORY_BUFFER_FRAMES):
            frame_time = base_time + frame * self._dt
            scale = self._compute_ramp_scale(env_ids, frame_time)  # (n,)
            frame_cmd = self._target_commands[env_ids] * scale.unsqueeze(-1)  # (n, 3)

            # Store current frame position/rotation
            self._root_pos_buffer[env_ids, frame, 0] = pos[:, 0]
            self._root_pos_buffer[env_ids, frame, 1] = pos[:, 1]
            self._root_pos_buffer[env_ids, frame, 2] = self._root_height

            half_yaw = yaw * 0.5
            self._root_rot_buffer[env_ids, frame, 0] = torch.cos(half_yaw)
            self._root_rot_buffer[env_ids, frame, 1] = 0.0
            self._root_rot_buffer[env_ids, frame, 2] = 0.0
            self._root_rot_buffer[env_ids, frame, 3] = torch.sin(half_yaw)

            # Euler integrate to next frame
            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            pos[:, 0] += (frame_cmd[:, 0] * cos_y - frame_cmd[:, 1] * sin_y) * self._dt
            pos[:, 1] += (frame_cmd[:, 0] * sin_y + frame_cmd[:, 1] * cos_y) * self._dt
            yaw = yaw + frame_cmd[:, 2] * self._dt

    def _update_root_state(self, dt: float):
        """Update root position and orientation based on user commands."""
        vx_local = self._commands[:, 0]
        vy_local = self._commands[:, 1]
        yaw_rate = self._commands[:, 2]

        # Convert local velocity to world frame
        cos_yaw = torch.cos(self._root_yaw)
        sin_yaw = torch.sin(self._root_yaw)

        vx_world = vx_local * cos_yaw - vy_local * sin_yaw
        vy_world = vx_local * sin_yaw + vy_local * cos_yaw

        # Integrate position
        self._root_pos[:, 0] += vx_world * dt
        self._root_pos[:, 1] += vy_world * dt
        # Z position stays constant (managed by motion state)

        # Integrate yaw
        self._root_yaw += yaw_rate * dt

        # Update quaternion from yaw
        half_yaw = self._root_yaw * 0.5
        self._root_rot[:, 0] = torch.cos(half_yaw)  # w
        self._root_rot[:, 1] = 0.0  # x
        self._root_rot[:, 2] = 0.0  # y
        self._root_rot[:, 3] = torch.sin(half_yaw)  # z

    @torch.no_grad()
    def step(self, env_ids: Optional[torch.Tensor] = None):
        """
        Advance CMG one step autoregressively.

        Args:
            env_ids: Optional subset of environments to step. If None, step all.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        if len(env_ids) == 0:
            return

        # Advance buffer frame index
        self._buffer_frame_idx[env_ids] += 1

        # Check if any env needs trajectory regeneration
        needs_regen = self._buffer_frame_idx[env_ids] >= (self.TRAJECTORY_BUFFER_FRAMES - 10)
        if needs_regen.any():
            regen_ids = env_ids[needs_regen]
            # Copy current state from buffer
            for env_id in regen_ids:
                frame_idx = min(self._buffer_frame_idx[env_id].item(), self.TRAJECTORY_BUFFER_FRAMES - 1)
                self._current_motion_norm[env_id] = self._trajectory_buffer[env_id, frame_idx]
            # Regenerate trajectory
            self._generate_trajectory(regen_ids)

        # Update motion time
        self._motion_times[env_ids] += self._dt

        # Update instantaneous commands from ramp schedule
        if self._ramp_enabled:
            all_ids = torch.arange(self._num_envs, device=self._device)
            scale = self._compute_ramp_scale(all_ids, self._motion_times)
            self._commands[:] = self._target_commands * scale.unsqueeze(-1)

        # Debug: periodic status print (every 50 steps ~1s for env 0)
        self._step_counter = getattr(self, '_step_counter', 0) + 1
        if self._step_counter % 50 == 0 and self._num_envs <= 4:
            env0 = 0
            fi = self._buffer_frame_idx[env0].item()
            mt = self._motion_times[env0].item()
            cmd = self._commands[env0].cpu().numpy()
            tcmd = self._target_commands[env0].cpu().numpy()
            ramp_flag = self._ramp_enabled_flags[env0].item() if self._ramp_enabled else False
            norm_std = self._current_motion_norm[env0].std().item()
            # Sample a few DOF positions from current buffer frame
            frame_idx = min(fi, self.TRAJECTORY_BUFFER_FRAMES - 1)
            motion_norm = self._trajectory_buffer[env0, frame_idx]
            motion = self._denormalize_motion(motion_norm.unsqueeze(0))
            knee_l = motion[0, CMG_TO_G1_INDICES[3]].item()
            knee_r = motion[0, CMG_TO_G1_INDICES[9]].item()
            print(f"[CMG dbg] step={self._step_counter} t={mt:.2f}s frame={fi} "
                  f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] "
                  f"target=[{tcmd[0]:.2f},{tcmd[1]:.2f},{tcmd[2]:.2f}] "
                  f"ramp={ramp_flag} norm_std={norm_std:.3f} "
                  f"knee=[{knee_l:.3f},{knee_r:.3f}]")

    def reset(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor] = None):
        """
        Reset specified environments.

        Args:
            env_ids: Environment indices to reset
            commands: Optional velocity commands. If None, sample randomly.
        """
        n = len(env_ids)
        if n == 0:
            return

        # Reset motion times
        self._motion_times[env_ids] = 0.0

        # Sample or set target commands
        if commands is None:
            self._target_commands[env_ids] = self._sample_commands(n)
        else:
            self._target_commands[env_ids] = commands

        # All episodes use ramp profile: stand → ramp_up → steady → ramp_down → stand
        if self._ramp_enabled:
            self._ramp_enabled_flags[env_ids] = True

            # Sample per-env random durations
            self._env_ramp_up[env_ids] = self._sample_uniform(n, self._ramp_up_range)
            self._env_ramp_down[env_ids] = self._sample_uniform(n, self._ramp_down_range)
            self._env_crawl[env_ids] = self._sample_uniform(n, self._ramp_crawl_range)

            # Clamp to ensure minimum steady phase
            max_variable = self._episode_length_s - 2 * self._ramp_stand_duration - self._ramp_min_steady
            total_variable = self._env_ramp_up[env_ids] + self._env_ramp_down[env_ids] + self._env_crawl[env_ids]
            over = (total_variable - max_variable).clamp(min=0)
            if over.any().item():
                # Proportionally shrink to fit
                ratio = max_variable / total_variable.clamp(min=1e-6)
                ratio = ratio.clamp(max=1.0)
                self._env_ramp_up[env_ids] *= ratio
                self._env_ramp_down[env_ids] *= ratio
                self._env_crawl[env_ids] *= ratio

            # Initial commands = target * scale(t=0), near-zero during stand phase
            scale = self._compute_ramp_scale(env_ids, torch.zeros(n, device=self._device))
            self._commands[env_ids] = self._target_commands[env_ids] * scale.unsqueeze(-1)
        else:
            self._ramp_enabled_flags[env_ids] = False
            self._commands[env_ids] = self._target_commands[env_ids]

        # Initialize motion states: all envs start from standing pose
        # (matches the stand phase at t=0 where v ≈ 0)
        self._current_motion_norm[env_ids] = self._standing_pose_norm.unsqueeze(0)

        # Reset root state
        self._root_pos[env_ids] = 0.0
        self._root_pos[env_ids, 2] = self._root_height
        self._root_yaw[env_ids] = 0.0
        self._root_rot[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        # Reset buffer frame index
        self._buffer_frame_idx[env_ids] = 0

        # Randomly mirror 50% of environments for symmetry training
        mirror_mask = torch.rand(n, device=self._device) < 0.5
        self._mirror_flags[env_ids] = mirror_mask

        # Generate trajectory for reset envs
        self._generate_trajectory(env_ids)

        # Debug: log reset events
        if self._num_envs <= 4:
            for eid in env_ids:
                eid_val = eid.item() if isinstance(eid, torch.Tensor) else eid
                tcmd = self._target_commands[eid_val].cpu().numpy()
                ramp_flag = self._ramp_enabled_flags[eid_val].item() if self._ramp_enabled else False
                mirror = self._mirror_flags[eid_val].item()
                print(f"[CMG reset] env={eid_val} target=[{tcmd[0]:.2f},{tcmd[1]:.2f},{tcmd[2]:.2f}] "
                      f"ramp={ramp_flag} mirror={mirror}")

    # ==================== MotionLib Interface ====================

    def num_motions(self) -> int:
        """Return number of 'motions' - for CMG this is essentially infinite."""
        return 1000  # Return a large number for compatibility

    def get_motion_length(self, motion_ids) -> torch.Tensor:
        """Return episode length for all motion IDs.

        Args:
            motion_ids: Can be int, scalar, or tensor
        Returns:
            Episode length as tensor or scalar
        """
        if isinstance(motion_ids, int):
            return torch.tensor(self._episode_length_s, device=self._device)
        elif isinstance(motion_ids, torch.Tensor):
            return torch.full_like(motion_ids, self._episode_length_s, dtype=torch.float)
        else:
            return torch.tensor(self._episode_length_s, device=self._device)

    def get_total_length(self) -> float:
        """Return total motion length."""
        return self._episode_length_s * self.num_motions()

    def sample_motions(self, n: int, motion_difficulty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample motion IDs. For CMG, this just returns indices and triggers command sampling.
        Motion difficulty is ignored for CMG.
        """
        motion_ids = torch.randint(0, self.num_motions(), (n,), device=self._device)
        return motion_ids

    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """
        Sample time within motions. For CMG, always return 0 since we start fresh.
        """
        return torch.zeros(motion_ids.shape, device=self._device)

    def calc_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate motion frame for given IDs and times.

        This method handles:
        1. Simple queries (batch_size == num_envs): returns current state for all envs
        2. Partial queries (env_ids provided): returns current state for specified envs
        3. Tiled queries (batch_size > num_envs): used by _get_mimic_obs for future frames

        Args:
            motion_ids: Motion IDs (unused for CMG, kept for API compatibility)
            motion_times: Time offsets for each query
            env_ids: Optional environment indices for partial queries (e.g., during reset)

        Returns:
            root_pos: (batch_size, 3)
            root_rot: (batch_size, 4) - quaternion [x, y, z, w] (Isaac Gym convention)
            root_vel: (batch_size, 3)
            root_ang_vel: (batch_size, 3)
            dof_pos: (batch_size, 23)
            dof_vel: (batch_size, 23)
            local_key_body_pos: (batch_size, num_key_bodies, 3)
        """
        batch_size = motion_ids.shape[0]

        if batch_size == self._num_envs:
            # Simple case: return current frame for all envs
            return self._calc_current_frame()
        elif env_ids is not None and batch_size == len(env_ids) and batch_size < self._num_envs:
            # Partial query case: return current frame for specified envs only
            return self._calc_partial_frame(env_ids)
        else:
            # Tiled case: used by _get_mimic_obs for future frames (vectorized)
            # motion_times contains offsets from the start of episode

            # Determine how many timesteps per environment
            num_steps = batch_size // self._num_envs

            # Compute env indices for each batch element
            # Flatten order is (num_envs, num_steps) -> so batch index i maps to env i // num_steps
            batch_indices = torch.arange(batch_size, device=self._device)
            env_indices = batch_indices // num_steps  # (batch_size,)

            # Get current times and buffer frame indices for all envs in batch
            current_times = self._motion_times[env_indices]  # (batch_size,)
            current_frames = self._buffer_frame_idx[env_indices]  # (batch_size,)

            # Calculate time offsets and target frames
            time_offsets = motion_times - current_times  # (batch_size,)
            frame_offsets = (time_offsets / self._dt).long()  # (batch_size,)
            target_frames = (current_frames + frame_offsets).clamp(0, self.TRAJECTORY_BUFFER_FRAMES - 1)  # (batch_size,)

            # Get motion states from buffer using advanced indexing
            motion_norm = self._trajectory_buffer[env_indices, target_frames]  # (batch_size, motion_dim)
            motion = self._denormalize_motion(motion_norm)  # (batch_size, motion_dim)
            dof_pos, dof_vel = self._map_29_to_23(motion)  # (batch_size, 23) each

            # Get root states from buffer
            root_pos = self._root_pos_buffer[env_indices, target_frames]  # (batch_size, 3)
            root_rot = self._root_rot_buffer[env_indices, target_frames]  # (batch_size, 4)

            # Compute time-varying commands at query times (for ramp support)
            scale = self._compute_ramp_scale(env_indices, motion_times)  # (batch_size,)
            commands = self._target_commands[env_indices] * scale.unsqueeze(-1)  # (batch_size, 3)
            vx_local = commands[:, 0]
            vy_local = commands[:, 1]
            yaw_rate = commands[:, 2]

            # Compute velocities from commands (vectorized)
            cos_yaw = root_rot[:, 0]**2 - root_rot[:, 3]**2  # cos(yaw) from quaternion
            sin_yaw = 2 * root_rot[:, 0] * root_rot[:, 3]    # sin(yaw) from quaternion

            root_vel = torch.zeros(batch_size, 3, device=self._device)
            root_vel[:, 0] = vx_local * cos_yaw - vy_local * sin_yaw
            root_vel[:, 1] = vx_local * sin_yaw + vy_local * cos_yaw

            root_ang_vel = torch.zeros(batch_size, 3, device=self._device)
            root_ang_vel[:, 2] = yaw_rate

            # Compute key body positions using forward kinematics
            local_key_body_pos = self._fk.compute_body_positions(root_pos, root_rot, dof_pos)

            # Convert quaternion from wxyz (CMG internal) to xyzw (Isaac Gym convention)
            root_rot_xyzw = torch.cat([root_rot[:, 1:], root_rot[:, :1]], dim=-1)

            # Apply left-right mirror for flagged environments
            root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos = \
                self._apply_mirror(env_indices, root_pos, root_rot_xyzw, root_vel, root_ang_vel,
                                   dof_pos, dof_vel, local_key_body_pos)

            return root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos

    def _calc_current_frame(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate motion frame for current state of all environments."""
        # Get current frame from buffer
        batch_indices = torch.arange(self._num_envs, device=self._device)
        frame_indices = self._buffer_frame_idx.clamp(0, self.TRAJECTORY_BUFFER_FRAMES - 1)

        motion_norm = self._trajectory_buffer[batch_indices, frame_indices]
        motion = self._denormalize_motion(motion_norm)

        # Map 29 DOF to 23 DOF
        dof_pos, dof_vel = self._map_29_to_23(motion)

        # Get root state from buffer
        root_pos = self._root_pos_buffer[batch_indices, frame_indices]
        root_rot = self._root_rot_buffer[batch_indices, frame_indices]

        # Compute root velocities from user commands (in world frame)
        cos_yaw = torch.cos(self._root_yaw)
        sin_yaw = torch.sin(self._root_yaw)

        vx_local = self._commands[:, 0]
        vy_local = self._commands[:, 1]

        root_vel = torch.zeros(self._num_envs, 3, device=self._device)
        root_vel[:, 0] = vx_local * cos_yaw - vy_local * sin_yaw
        root_vel[:, 1] = vx_local * sin_yaw + vy_local * cos_yaw
        root_vel[:, 2] = 0.0

        root_ang_vel = torch.zeros(self._num_envs, 3, device=self._device)
        root_ang_vel[:, 2] = self._commands[:, 2]  # yaw rate

        # Compute key body positions using forward kinematics
        local_key_body_pos = self._fk.compute_body_positions(root_pos, root_rot, dof_pos)

        # Convert quaternion from wxyz (CMG internal) to xyzw (Isaac Gym convention)
        root_rot_xyzw = torch.cat([root_rot[:, 1:], root_rot[:, :1]], dim=-1)

        # Apply left-right mirror for flagged environments
        env_indices = torch.arange(self._num_envs, device=self._device)
        root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos = \
            self._apply_mirror(env_indices, root_pos, root_rot_xyzw, root_vel, root_ang_vel,
                               dof_pos, dof_vel, local_key_body_pos)

        return root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos

    def _calc_partial_frame(
        self,
        env_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate motion frame for a subset of environments (used during partial reset)."""
        n = len(env_ids)
        frame_indices = self._buffer_frame_idx[env_ids].clamp(0, self.TRAJECTORY_BUFFER_FRAMES - 1)

        motion_norm = self._trajectory_buffer[env_ids, frame_indices]
        motion = self._denormalize_motion(motion_norm)

        # Map 29 DOF to 23 DOF
        dof_pos, dof_vel = self._map_29_to_23(motion)

        # Get root state from buffer
        root_pos = self._root_pos_buffer[env_ids, frame_indices]
        root_rot = self._root_rot_buffer[env_ids, frame_indices]

        # Compute root velocities from user commands
        cos_yaw = torch.cos(self._root_yaw[env_ids])
        sin_yaw = torch.sin(self._root_yaw[env_ids])

        vx_local = self._commands[env_ids, 0]
        vy_local = self._commands[env_ids, 1]

        root_vel = torch.zeros(n, 3, device=self._device)
        root_vel[:, 0] = vx_local * cos_yaw - vy_local * sin_yaw
        root_vel[:, 1] = vx_local * sin_yaw + vy_local * cos_yaw
        root_vel[:, 2] = 0.0

        root_ang_vel = torch.zeros(n, 3, device=self._device)
        root_ang_vel[:, 2] = self._commands[env_ids, 2]  # yaw rate

        # Compute key body positions using forward kinematics
        local_key_body_pos = self._fk.compute_body_positions(root_pos, root_rot, dof_pos)

        # Convert quaternion from wxyz (CMG internal) to xyzw (Isaac Gym convention)
        root_rot_xyzw = torch.cat([root_rot[:, 1:], root_rot[:, :1]], dim=-1)

        # Apply left-right mirror for flagged environments
        root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos = \
            self._apply_mirror(env_ids, root_pos, root_rot_xyzw, root_vel, root_ang_vel,
                               dof_pos, dof_vel, local_key_body_pos)

        return root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos

    def _apply_mirror(self, env_indices, root_pos, root_rot_xyzw, root_vel, root_ang_vel,
                       dof_pos, dof_vel, local_key_body_pos):
        """Apply left-right mirror transformation for flagged environments.

        Mirrors DOFs (swap L/R, flip roll/yaw signs), key body positions (swap L/R, flip y),
        root rotation (negate roll/yaw), root velocity (flip vy), root angular velocity (flip yaw).

        Args:
            env_indices: Environment index for each batch element (used to look up mirror flags)
            Others: Motion frame outputs to mirror in-place for flagged envs
        Returns:
            Tuple of mirrored outputs
        """
        mirror = self._mirror_flags[env_indices]
        if not mirror.any():
            return root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos

        # Mirror DOFs: swap left/right, flip roll/yaw signs
        dof_pos = dof_pos.clone()
        dof_vel = dof_vel.clone()
        dof_pos[mirror] = dof_pos[mirror][:, self._dof_mirror_indices] * self._dof_mirror_signs
        dof_vel[mirror] = dof_vel[mirror][:, self._dof_mirror_indices] * self._dof_mirror_signs

        # Mirror root position: flip y
        root_pos = root_pos.clone()
        root_pos[mirror, 1] *= -1

        # Mirror root rotation: negate x and z components (xyzw format)
        # This negates roll and yaw while preserving pitch
        root_rot_xyzw = root_rot_xyzw.clone()
        root_rot_xyzw[mirror, 0] *= -1  # x (roll)
        root_rot_xyzw[mirror, 2] *= -1  # z (yaw)

        # Mirror root velocity: flip vy
        root_vel = root_vel.clone()
        root_vel[mirror, 1] *= -1

        # Mirror root angular velocity: flip roll rate and yaw rate
        root_ang_vel = root_ang_vel.clone()
        root_ang_vel[mirror, 0] *= -1  # roll rate
        root_ang_vel[mirror, 2] *= -1  # yaw rate

        # Mirror key body positions: swap left/right, flip y
        local_key_body_pos = local_key_body_pos.clone()
        local_key_body_pos[mirror] = local_key_body_pos[mirror][:, self._keybody_mirror_indices, :]
        local_key_body_pos[mirror, :, 1] *= -1

        return root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos

    def get_key_body_idx(self, key_body_names: List[str]) -> List[int]:
        """Get indices of key bodies by name."""
        key_body_idx = []
        for name in key_body_names:
            if name in self._body_link_list:
                key_body_idx.append(self._body_link_list.index(name))
            else:
                # Map to FK body list
                key_body_idx.append(self._fk.get_body_idx(name))
        return key_body_idx

    def get_motion_names(self) -> List[str]:
        """Return motion names. For CMG, return command descriptions."""
        return [f"cmg_vx{self._vx_range}_vy{self._vy_range}_yaw{self._yaw_range}"]

    def get_commands(self) -> torch.Tensor:
        """Get raw user velocity commands for all environments.
        Returns mirrored commands (vy, yaw negated) for mirrored environments.
        DEPRECATED alias for get_user_commands(); use get_user_commands() directly."""
        return self.get_user_commands()

    def get_user_commands(self) -> torch.Tensor:
        """Get raw user-specified velocity commands for all environments.
        Applies mirror correction (vy, yaw negated) for mirrored environments
        so the target matches the physically executed mirrored trajectory.
        Use this as the reward target for velocity tracking."""
        commands = self._commands.clone()
        mirror = self._mirror_flags
        if mirror.any():
            commands[mirror, 1] *= -1  # flip vy
            commands[mirror, 2] *= -1  # flip yaw_rate
        return commands

    def set_commands(self, env_ids: torch.Tensor, commands: torch.Tensor):
        """Set velocity commands for specified environments."""
        self._target_commands[env_ids] = commands
        self._commands[env_ids] = commands
