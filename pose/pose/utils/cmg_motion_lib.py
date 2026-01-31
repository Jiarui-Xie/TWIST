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

        print(f"[CMGMotionLib] Initialized with {num_envs} envs, "
              f"vx=[{vx_range[0]:.1f}, {vx_range[1]:.1f}], "
              f"vy=[{vy_range[0]:.1f}, {vy_range[1]:.1f}], "
              f"yaw=[{yaw_range[0]:.1f}, {yaw_range[1]:.1f}]")

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

        # Current velocity commands
        self._commands = torch.zeros(self._num_envs, 3, device=self._device)

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

    def _sample_commands(self, n: int) -> torch.Tensor:
        """Sample random velocity commands within configured ranges."""
        vx = torch.rand(n, device=self._device) * (self._vx_range[1] - self._vx_range[0]) + self._vx_range[0]
        vy = torch.rand(n, device=self._device) * (self._vy_range[1] - self._vy_range[0]) + self._vy_range[0]
        yaw = torch.rand(n, device=self._device) * (self._yaw_range[1] - self._yaw_range[0]) + self._yaw_range[0]
        return torch.stack([vx, vy, yaw], dim=-1)

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
        # Get command for this env
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
        """Pre-generate trajectory buffer for specified environments (vectorized)."""
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # Get current state and commands
        current_norm = self._current_motion_norm[env_ids].clone()
        commands = self._commands[env_ids]
        commands_norm = self._normalize_command(commands)

        # Store initial position/rotation
        root_pos = self._root_pos[env_ids].clone()  # (n, 3)
        root_yaw = self._root_yaw[env_ids].clone()  # (n,)

        # Extract velocity components
        vx_local = commands[:, 0]  # (n,)
        vy_local = commands[:, 1]  # (n,)
        yaw_rate = commands[:, 2]  # (n,)

        # Generate trajectory - vectorized over environments
        for frame in range(self.TRAJECTORY_BUFFER_FRAMES):
            # Store current state
            self._trajectory_buffer[env_ids, frame] = current_norm

            # Compute root state for this frame (vectorized)
            time_offset = frame * self._dt

            # Compute yaw angles
            avg_yaw = root_yaw + yaw_rate * time_offset * 0.5  # (n,)
            new_yaw = root_yaw + yaw_rate * time_offset  # (n,)

            cos_yaw = torch.cos(avg_yaw)  # (n,)
            sin_yaw = torch.sin(avg_yaw)  # (n,)

            # Update root position buffer (vectorized)
            self._root_pos_buffer[env_ids, frame, 0] = root_pos[:, 0] + (vx_local * cos_yaw - vy_local * sin_yaw) * time_offset
            self._root_pos_buffer[env_ids, frame, 1] = root_pos[:, 1] + (vx_local * sin_yaw + vy_local * cos_yaw) * time_offset
            self._root_pos_buffer[env_ids, frame, 2] = self._root_height

            # Update root rotation buffer (vectorized)
            half_yaw = new_yaw * 0.5  # (n,)
            self._root_rot_buffer[env_ids, frame, 0] = torch.cos(half_yaw)
            self._root_rot_buffer[env_ids, frame, 1] = 0.0
            self._root_rot_buffer[env_ids, frame, 2] = 0.0
            self._root_rot_buffer[env_ids, frame, 3] = torch.sin(half_yaw)

            # CMG forward pass for next state
            next_motion_norm = self._cmg_model(current_norm, commands_norm)
            current_norm = next_motion_norm

        # Reset buffer frame index
        self._buffer_frame_idx[env_ids] = 0

    def _update_root_state(self, dt: float):
        """Update root position and orientation based on velocity commands."""
        # Get local velocities
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

        # Sample or set commands
        if commands is None:
            self._commands[env_ids] = self._sample_commands(n)
        else:
            self._commands[env_ids] = commands

        # Get initial motion states
        init_motion = self._get_init_motion(n)
        self._current_motion_norm[env_ids] = self._normalize_motion(init_motion)

        # Reset root state
        self._root_pos[env_ids] = 0.0
        self._root_pos[env_ids, 2] = self._root_height
        self._root_yaw[env_ids] = 0.0
        self._root_rot[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        # Reset buffer frame index
        self._buffer_frame_idx[env_ids] = 0

        # Generate trajectory for reset envs
        self._generate_trajectory(env_ids)

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

            # Get commands for each batch element
            commands = self._commands[env_indices]  # (batch_size, 3)
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

        # Compute root velocities from commands (in world frame)
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

        # Compute root velocities from commands
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
        """Get current velocity commands for all environments."""
        return self._commands.clone()

    def set_commands(self, env_ids: torch.Tensor, commands: torch.Tensor):
        """Set velocity commands for specified environments."""
        self._commands[env_ids] = commands
