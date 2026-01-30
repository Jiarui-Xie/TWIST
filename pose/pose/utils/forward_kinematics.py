"""
Forward Kinematics Module for Computing Key Body Positions.

Provides a simplified FK implementation that doesn't require full URDF parsing.
For CMG training, we use approximate body positions based on joint angles.
"""

import torch
from typing import List, Optional


class ForwardKinematics:
    """
    Simplified forward kinematics calculator for humanoid robot.

    This implementation provides approximate key body positions without
    requiring complex URDF parsing. For RL training, the exact positions
    are less critical than consistent relative positioning.
    """

    # Key body names for G1 robot tracking
    KEY_BODIES = [
        "left_rubber_hand",
        "right_rubber_hand",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_elbow_link",
        "right_elbow_link",
        "head_mocap",
    ]

    # Approximate link lengths for G1 (in meters)
    # These are used for simplified FK calculations
    LINK_LENGTHS = {
        "thigh": 0.35,      # Hip to knee
        "shank": 0.35,      # Knee to ankle
        "upper_arm": 0.25,  # Shoulder to elbow
        "forearm": 0.25,    # Elbow to hand
        "torso": 0.45,      # Pelvis to shoulder
        "head": 0.15,       # Shoulder to head
    }

    def __init__(self, urdf_path: str, device: str):
        """
        Initialize forward kinematics.

        Args:
            urdf_path: Path to robot URDF file (not used in simplified version)
            device: Compute device ('cuda' or 'cpu')
        """
        self._device = device
        self._urdf_path = urdf_path

        # Pre-compute index tensors for vectorized joint mapping
        self._g1_valid_indices = None
        self._urdf_valid_indices = None

        print(f"[ForwardKinematics] Initialized (simplified mode)")
        print(f"[ForwardKinematics] Key bodies: {self.KEY_BODIES}")

    def compute_body_positions(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        dof_pos: torch.Tensor,
        key_bodies: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute approximate 3D positions of key bodies.

        This simplified implementation uses geometric approximations based on
        joint angles rather than full kinematic chain computation.

        Args:
            root_pos: Root position (batch, 3)
            root_rot: Root rotation quaternion [w, x, y, z] (batch, 4)
            dof_pos: Joint positions in G1 training order (batch, 23)
            key_bodies: List of body names to compute. If None, use KEY_BODIES.

        Returns:
            Local body positions relative to root (batch, num_bodies, 3)
        """
        if key_bodies is None:
            key_bodies = self.KEY_BODIES

        batch_size = dof_pos.shape[0]
        num_bodies = len(key_bodies)

        # Initialize output tensor
        body_positions = torch.zeros(batch_size, num_bodies, 3, device=self._device)

        # G1 DOF ordering:
        # 0-5: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        # 6-11: right leg
        # 12-14: waist (yaw, roll, pitch)
        # 15-18: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
        # 19-22: right arm

        # Compute approximate positions for each key body
        for i, body_name in enumerate(key_bodies):
            if body_name == "left_ankle_roll_link":
                # Left ankle: depends on left leg joints (0-5)
                body_positions[:, i, :] = self._compute_ankle_pos(
                    dof_pos[:, 0:6], side="left"
                )
            elif body_name == "right_ankle_roll_link":
                # Right ankle: depends on right leg joints (6-11)
                body_positions[:, i, :] = self._compute_ankle_pos(
                    dof_pos[:, 6:12], side="right"
                )
            elif body_name == "left_knee_link":
                # Left knee: depends on hip joints (0-3)
                body_positions[:, i, :] = self._compute_knee_pos(
                    dof_pos[:, 0:4], side="left"
                )
            elif body_name == "right_knee_link":
                # Right knee: depends on hip joints (6-9)
                body_positions[:, i, :] = self._compute_knee_pos(
                    dof_pos[:, 6:10], side="right"
                )
            elif body_name == "left_rubber_hand":
                # Left hand: depends on waist (12-14) and left arm (15-18)
                body_positions[:, i, :] = self._compute_hand_pos(
                    dof_pos[:, 12:15], dof_pos[:, 15:19], side="left"
                )
            elif body_name == "right_rubber_hand":
                # Right hand: depends on waist (12-14) and right arm (19-22)
                body_positions[:, i, :] = self._compute_hand_pos(
                    dof_pos[:, 12:15], dof_pos[:, 19:23], side="right"
                )
            elif body_name == "left_elbow_link":
                # Left elbow
                body_positions[:, i, :] = self._compute_elbow_pos(
                    dof_pos[:, 12:15], dof_pos[:, 15:18], side="left"
                )
            elif body_name == "right_elbow_link":
                # Right elbow
                body_positions[:, i, :] = self._compute_elbow_pos(
                    dof_pos[:, 12:15], dof_pos[:, 19:22], side="right"
                )
            elif body_name == "head_mocap":
                # Head position: above torso
                body_positions[:, i, :] = self._compute_head_pos(dof_pos[:, 12:15])

        return body_positions

    def _compute_ankle_pos(self, leg_dof: torch.Tensor, side: str) -> torch.Tensor:
        """Compute approximate ankle position from leg DOFs."""
        batch_size = leg_dof.shape[0]

        # Simplified: ankle is below pelvis, offset by thigh + shank length
        # Adjusted by hip and knee angles
        hip_pitch = leg_dof[:, 0]
        knee = leg_dof[:, 3]

        thigh_len = self.LINK_LENGTHS["thigh"]
        shank_len = self.LINK_LENGTHS["shank"]

        # Approximate x, y, z offset
        y_offset = 0.1 if side == "left" else -0.1  # Hip width

        # Z is negative (below pelvis), affected by leg extension
        z = -(thigh_len * torch.cos(hip_pitch) + shank_len * torch.cos(hip_pitch + knee))
        x = thigh_len * torch.sin(hip_pitch) + shank_len * torch.sin(hip_pitch + knee)
        y = torch.full((batch_size,), y_offset, device=self._device)

        return torch.stack([x, y, z], dim=-1)

    def _compute_knee_pos(self, hip_dof: torch.Tensor, side: str) -> torch.Tensor:
        """Compute approximate knee position from hip DOFs."""
        batch_size = hip_dof.shape[0]

        hip_pitch = hip_dof[:, 0]
        thigh_len = self.LINK_LENGTHS["thigh"]

        y_offset = 0.1 if side == "left" else -0.1

        z = -thigh_len * torch.cos(hip_pitch)
        x = thigh_len * torch.sin(hip_pitch)
        y = torch.full((batch_size,), y_offset, device=self._device)

        return torch.stack([x, y, z], dim=-1)

    def _compute_hand_pos(
        self, waist_dof: torch.Tensor, arm_dof: torch.Tensor, side: str
    ) -> torch.Tensor:
        """Compute approximate hand position from waist and arm DOFs."""
        batch_size = waist_dof.shape[0]

        shoulder_pitch = arm_dof[:, 0]
        elbow = arm_dof[:, 3] if arm_dof.shape[1] > 3 else torch.zeros(batch_size, device=self._device)

        upper_arm_len = self.LINK_LENGTHS["upper_arm"]
        forearm_len = self.LINK_LENGTHS["forearm"]
        torso_height = self.LINK_LENGTHS["torso"]

        y_offset = 0.2 if side == "left" else -0.2  # Shoulder width

        # Hand position relative to shoulder
        arm_x = upper_arm_len * torch.sin(shoulder_pitch) + forearm_len * torch.sin(shoulder_pitch + elbow)
        arm_z = -upper_arm_len * torch.cos(shoulder_pitch) - forearm_len * torch.cos(shoulder_pitch + elbow)

        x = arm_x
        y = torch.full((batch_size,), y_offset, device=self._device)
        z = torso_height + arm_z  # Shoulder is at torso height

        return torch.stack([x, y, z], dim=-1)

    def _compute_elbow_pos(
        self, waist_dof: torch.Tensor, arm_dof: torch.Tensor, side: str
    ) -> torch.Tensor:
        """Compute approximate elbow position from waist and arm DOFs."""
        batch_size = waist_dof.shape[0]

        shoulder_pitch = arm_dof[:, 0]
        upper_arm_len = self.LINK_LENGTHS["upper_arm"]
        torso_height = self.LINK_LENGTHS["torso"]

        y_offset = 0.2 if side == "left" else -0.2

        x = upper_arm_len * torch.sin(shoulder_pitch)
        y = torch.full((batch_size,), y_offset, device=self._device)
        z = torso_height - upper_arm_len * torch.cos(shoulder_pitch)

        return torch.stack([x, y, z], dim=-1)

    def _compute_head_pos(self, waist_dof: torch.Tensor) -> torch.Tensor:
        """Compute approximate head position."""
        batch_size = waist_dof.shape[0]

        torso_height = self.LINK_LENGTHS["torso"]
        head_height = self.LINK_LENGTHS["head"]

        waist_pitch = waist_dof[:, 2] if waist_dof.shape[1] > 2 else torch.zeros(batch_size, device=self._device)

        x = (torso_height + head_height) * torch.sin(waist_pitch)
        y = torch.zeros(batch_size, device=self._device)
        z = (torso_height + head_height) * torch.cos(waist_pitch)

        return torch.stack([x, y, z], dim=-1)

    def get_body_idx(self, body_name: str) -> int:
        """Get index of a body in the KEY_BODIES list."""
        if body_name in self.KEY_BODIES:
            return self.KEY_BODIES.index(body_name)
        return -1

    def get_all_body_positions(self, dof_pos: torch.Tensor) -> dict:
        """
        Compute positions for all key bodies.

        Args:
            dof_pos: Joint positions in G1 training order (batch, 23)

        Returns:
            Dictionary mapping body names to positions (batch, 3)
        """
        root_pos = torch.zeros(dof_pos.shape[0], 3, device=self._device)
        root_rot = torch.zeros(dof_pos.shape[0], 4, device=self._device)
        root_rot[:, 0] = 1.0  # Unit quaternion

        body_positions = self.compute_body_positions(root_pos, root_rot, dof_pos)

        positions = {}
        for i, name in enumerate(self.KEY_BODIES):
            positions[name] = body_positions[:, i, :]

        return positions
