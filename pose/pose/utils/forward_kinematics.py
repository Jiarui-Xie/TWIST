"""
Forward Kinematics Module for Computing Key Body Positions.

Uses pytorch_kinematics to compute the 3D positions of key body links
from joint angles and root state.
"""

import torch
import pytorch_kinematics as pk
from typing import List, Optional


class ForwardKinematics:
    """
    Forward kinematics calculator for humanoid robot.

    Computes 3D positions of key body links from joint angles using
    the robot URDF and pytorch_kinematics.
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

    # Joint ordering in G1 training (23 DOF)
    # The URDF uses a different ordering, so we need to reindex
    # G1 training order:
    # 0-5: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    # 6-11: right leg
    # 12-14: waist (yaw, roll, pitch)
    # 15-18: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
    # 19-22: right arm

    # pytorch_kinematics joint order from URDF:
    # The chain.get_joint_parameter_names() will tell us the exact order
    # We need to map from G1 training order to URDF order

    def __init__(self, urdf_path: str, device: str):
        """
        Initialize forward kinematics from URDF.

        Args:
            urdf_path: Path to robot URDF file
            device: Compute device ('cuda' or 'cpu')
        """
        self._device = device

        # Build kinematic chain from URDF
        with open(urdf_path, "rb") as f:
            urdf_content = f.read()
        self._chain = pk.build_chain_from_urdf(urdf_content)
        self._chain = self._chain.to(device=device)

        # Get joint names from the chain
        self._joint_names = self._chain.get_joint_parameter_names()
        self._num_joints = len(self._joint_names)

        # Build mapping from G1 training DOF order to URDF joint order
        self._build_joint_mapping()

        # Get all link names for body index lookup
        self._link_names = self._chain.get_link_names()

        print(f"[ForwardKinematics] Loaded URDF with {self._num_joints} joints")
        print(f"[ForwardKinematics] Key bodies: {self.KEY_BODIES}")

    def _build_joint_mapping(self):
        """Build mapping from G1 training DOF order (23) to URDF joint order."""
        # G1 training joint names in order
        g1_joint_names = [
            # Left leg (0-5)
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right leg (6-11)
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # Waist (12-14)
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            # Left arm (15-18)
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            # Right arm (19-22)
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]

        # Find mapping from G1 order to URDF order
        self._g1_to_urdf = []
        for g1_name in g1_joint_names:
            if g1_name in self._joint_names:
                self._g1_to_urdf.append(self._joint_names.index(g1_name))
            else:
                # Joint not in URDF (e.g., wrist joints might be fixed)
                self._g1_to_urdf.append(-1)

        # Count valid mappings
        valid_count = sum(1 for idx in self._g1_to_urdf if idx >= 0)
        print(f"[ForwardKinematics] Mapped {valid_count}/{len(g1_joint_names)} joints to URDF")

        # If the URDF has more joints than G1 training, we need to provide zeros for them
        self._urdf_has_extra_joints = self._num_joints > 23

    def _reindex_joints(self, dof_pos_g1: torch.Tensor) -> torch.Tensor:
        """
        Reindex joint positions from G1 training order to URDF order.

        Args:
            dof_pos_g1: Joint positions in G1 training order (batch, 23)

        Returns:
            Joint positions in URDF order (batch, num_urdf_joints)
        """
        batch_size = dof_pos_g1.shape[0]

        # Create full joint array for URDF
        dof_pos_urdf = torch.zeros(batch_size, self._num_joints, device=self._device)

        # Map G1 joints to URDF positions
        for g1_idx, urdf_idx in enumerate(self._g1_to_urdf):
            if urdf_idx >= 0 and g1_idx < dof_pos_g1.shape[1]:
                dof_pos_urdf[:, urdf_idx] = dof_pos_g1[:, g1_idx]

        return dof_pos_urdf

    def compute_body_positions(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        dof_pos: torch.Tensor,
        key_bodies: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute 3D positions of key bodies.

        Args:
            root_pos: Root position (batch, 3) - used as offset
            root_rot: Root rotation quaternion [w, x, y, z] (batch, 4)
            dof_pos: Joint positions in G1 training order (batch, 23)
            key_bodies: List of body names to compute. If None, use KEY_BODIES.

        Returns:
            Local body positions relative to root (batch, num_bodies, 3)
        """
        if key_bodies is None:
            key_bodies = self.KEY_BODIES

        batch_size = dof_pos.shape[0]

        # Reindex joints to URDF order
        dof_pos_urdf = self._reindex_joints(dof_pos)

        # Compute forward kinematics
        fk_result = self._chain.forward_kinematics(dof_pos_urdf)

        # Extract positions for key bodies
        body_positions = torch.zeros(batch_size, len(key_bodies), 3, device=self._device)

        for i, body_name in enumerate(key_bodies):
            if body_name in fk_result:
                transform = fk_result[body_name]
                matrix = transform.get_matrix()  # (batch, 4, 4)
                pos = matrix[:, :3, 3]  # Extract translation
                body_positions[:, i, :] = pos
            else:
                # Body not found, try alternate names or use zero
                alt_name = self._get_alternate_name(body_name)
                if alt_name and alt_name in fk_result:
                    transform = fk_result[alt_name]
                    matrix = transform.get_matrix()
                    pos = matrix[:, :3, 3]
                    body_positions[:, i, :] = pos

        # Rotate body positions by root rotation to get local coordinates
        # The FK gives positions in the root frame, so we need to transform them
        # For local body positions (relative to root), we don't need additional rotation
        # since FK already outputs in the pelvis frame

        return body_positions

    def _get_alternate_name(self, body_name: str) -> Optional[str]:
        """Get alternate link name for bodies that might have different names."""
        alternates = {
            "head_mocap": ["head_link", "imu_in_torso", "torso_link"],
            "left_rubber_hand": ["left_palm_link", "left_wrist_yaw_link"],
            "right_rubber_hand": ["right_palm_link", "right_wrist_yaw_link"],
        }
        if body_name in alternates:
            for alt in alternates[body_name]:
                if alt in self._link_names:
                    return alt
        return None

    def get_body_idx(self, body_name: str) -> int:
        """Get index of a body in the KEY_BODIES list."""
        if body_name in self.KEY_BODIES:
            return self.KEY_BODIES.index(body_name)
        # Try alternate names
        alt_name = self._get_alternate_name(body_name)
        if alt_name and alt_name in self.KEY_BODIES:
            return self.KEY_BODIES.index(alt_name)
        return -1

    def get_all_body_positions(
        self,
        dof_pos: torch.Tensor
    ) -> dict:
        """
        Compute positions for all bodies in the kinematic chain.

        Args:
            dof_pos: Joint positions in G1 training order (batch, 23)

        Returns:
            Dictionary mapping body names to positions (batch, 3)
        """
        dof_pos_urdf = self._reindex_joints(dof_pos)
        fk_result = self._chain.forward_kinematics(dof_pos_urdf)

        positions = {}
        for name, transform in fk_result.items():
            matrix = transform.get_matrix()
            positions[name] = matrix[:, :3, 3]

        return positions
