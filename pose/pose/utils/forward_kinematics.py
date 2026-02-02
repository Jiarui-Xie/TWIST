"""
Forward Kinematics Module for Computing Key Body Positions.

Uses pytorch_kinematics for accurate FK computation from URDF.
"""

import torch
import pytorch_kinematics as pk
from typing import List, Optional


class ForwardKinematics:
    """
    Accurate forward kinematics calculator using pytorch_kinematics.

    Computes key body positions from joint angles using the actual URDF model.
    """

    # Key body names for G1 robot tracking (must match config)
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

    def __init__(self, urdf_path: str, device: str):
        """
        Initialize forward kinematics from URDF.

        Args:
            urdf_path: Path to robot URDF file
            device: Compute device ('cuda' or 'cpu')
        """
        self._device = device
        self._urdf_path = urdf_path

        # Build kinematic chain from URDF
        with open(urdf_path, 'rb') as f:
            urdf_content = f.read()
        self._chain = pk.build_chain_from_urdf(urdf_content)
        self._chain = self._chain.to(device=device)

        # Verify joint count matches G1 (23 DOF)
        joint_names = self._chain.get_joint_parameter_names()
        self._num_joints = len(joint_names)
        assert self._num_joints == 23, f"Expected 23 joints, got {self._num_joints}"

        # G1 training order matches URDF order, no reindexing needed
        # 0-5: left leg, 6-11: right leg, 12-14: waist, 15-18: left arm, 19-22: right arm

        print(f"[ForwardKinematics] Loaded URDF from {urdf_path}")
        print(f"[ForwardKinematics] {self._num_joints} joints, key bodies: {self.KEY_BODIES}")

    def compute_body_positions(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        dof_pos: torch.Tensor,
        key_bodies: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute 3D positions of key bodies using forward kinematics.

        Args:
            root_pos: Root position (batch, 3) - not used, positions are local
            root_rot: Root rotation quaternion [w, x, y, z] (batch, 4) - not used
            dof_pos: Joint positions in G1 training order (batch, 23)
            key_bodies: List of body names to compute. If None, use KEY_BODIES.

        Returns:
            Local body positions relative to root/pelvis (batch, num_bodies, 3)
        """
        if key_bodies is None:
            key_bodies = self.KEY_BODIES

        batch_size = dof_pos.shape[0]
        num_bodies = len(key_bodies)

        # Ensure dof_pos is on correct device and has right shape
        if dof_pos.device != self._device:
            dof_pos = dof_pos.to(self._device)

        assert dof_pos.shape[1] == self._num_joints, \
            f"Expected {self._num_joints} joints, got {dof_pos.shape[1]}"

        # Run forward kinematics
        fk_result = self._chain.forward_kinematics(dof_pos)

        # Extract positions for key bodies
        body_positions = torch.zeros(batch_size, num_bodies, 3, device=self._device)

        for i, body_name in enumerate(key_bodies):
            if body_name in fk_result:
                transform = fk_result[body_name]
                # Get translation from transformation matrix
                matrix = transform.get_matrix()  # (batch, 4, 4)
                body_positions[:, i, :] = matrix[:, :3, 3]
            else:
                print(f"[ForwardKinematics] Warning: body '{body_name}' not found in FK result")

        return body_positions

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
        root_rot[:, 0] = 1.0  # Unit quaternion [w, x, y, z]

        body_positions = self.compute_body_positions(root_pos, root_rot, dof_pos)

        positions = {}
        for i, name in enumerate(self.KEY_BODIES):
            positions[name] = body_positions[:, i, :]

        return positions
