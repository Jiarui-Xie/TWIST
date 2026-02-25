#!/usr/bin/env python3
"""
Test script to visualize CMG mirrored sequences.

Shows two G1 robots side by side in mujoco:
  - LEFT robot (y>0): original CMG output
  - RIGHT robot (y<0): mirrored CMG output (left-right swapped)

Usage:
    conda activate Main
    python test_cmg_mirror.py [--cmd_vx 1.5] [--cmd_vy 0.3] [--cmd_yaw 0.2] [--speed 1.0]
"""

import sys
import os
import argparse
import time
import tempfile
import copy
import xml.etree.ElementTree as ET

import torch
import numpy as np

# Add project paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, 'pose'))
sys.path.insert(0, os.path.join(_script_dir, 'cmg_workspace'))

import mujoco
import mujoco_viewer

from pose.utils.cmg_motion_lib import (
    CMGMotionLib, DOF_MIRROR_INDICES_23, DOF_MIRROR_SIGNS_23, KEYBODY_MIRROR_INDICES
)


# ==================== XML Duplication ====================

def create_dual_robot_xml(base_xml_path: str, offset_y: float = 1.5) -> str:
    """
    Create a temporary XML with two G1 robots side by side.
    Robot 1 (original) at y=+offset_y/2, Robot 2 (mirrored) at y=-offset_y/2.
    """
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    # Fix meshdir to absolute path so temp file can find meshes
    compiler = root.find('compiler')
    if compiler is not None:
        meshdir = compiler.get('meshdir', '')
        if meshdir and not os.path.isabs(meshdir):
            abs_meshdir = os.path.join(os.path.dirname(os.path.abspath(base_xml_path)), meshdir)
            compiler.set('meshdir', abs_meshdir)

    # Remove keyframe (qpos size won't match dual robot)
    keyframe = root.find('keyframe')
    if keyframe is not None:
        root.remove(keyframe)

    worldbody = root.find('worldbody')

    pelvis = worldbody.find("body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("Cannot find pelvis body in XML")

    orig_pos = pelvis.get('pos', '0 0 0.793').split()
    orig_x, orig_y, orig_z = float(orig_pos[0]), float(orig_pos[1]), float(orig_pos[2])

    pelvis2 = copy.deepcopy(pelvis)

    def rename_elements(elem, suffix):
        for attr in ['name']:
            if attr in elem.attrib:
                elem.set(attr, elem.get(attr) + suffix)
        if 'joint' in elem.attrib:
            elem.set('joint', elem.get('joint') + suffix)
        for child in elem:
            rename_elements(child, suffix)

    rename_elements(pelvis2, '_mirror')

    pelvis.set('pos', f'{orig_x} {offset_y / 2} {orig_z}')
    pelvis2.set('pos', f'{orig_x} {-offset_y / 2} {orig_z}')
    worldbody.append(pelvis2)

    actuator = root.find('actuator')
    if actuator is not None:
        new_actuators = []
        for act in actuator:
            act2 = copy.deepcopy(act)
            for attr in ['name', 'joint']:
                if attr in act2.attrib:
                    act2.set(attr, act2.get(attr) + '_mirror')
            new_actuators.append(act2)
        for act2 in new_actuators:
            actuator.append(act2)

    tmp_file = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, dir=_script_dir)
    tree.write(tmp_file.name)
    return tmp_file.name


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Visualize CMG mirrored sequences')
    parser.add_argument('--cmd_vx', type=float, default=1.5, help='Forward velocity command (m/s)')
    parser.add_argument('--cmd_vy', type=float, default=0.3, help='Lateral velocity command (m/s)')
    parser.add_argument('--cmd_yaw', type=float, default=0.2, help='Yaw rate command (rad/s)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--duration', type=float, default=4.0, help='Duration in seconds')
    parser.add_argument('--no_viewer', action='store_true', help='Print DOF comparison only, no viewer')
    args = parser.parse_args()

    device = 'cpu'
    dt = 0.02  # CMG timestep (50 Hz)
    num_frames = int(args.duration / dt)

    # ==================== Load CMG ====================
    print("[1/4] Loading CMG model...")
    cmg_model_path = os.path.join(_script_dir, 'cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt')
    cmg_data_path = os.path.join(_script_dir, 'cmg_workspace/dataloader/cmg_training_data.pt')
    urdf_path = os.path.join(_script_dir, 'assets/g1/g1_custom_collision_with_fixed_hand.urdf')

    # Use 1 env - we test mirroring by toggling the flag on the SAME trajectory
    cmg = CMGMotionLib(
        cmg_model_path=cmg_model_path,
        cmg_data_path=cmg_data_path,
        urdf_path=urdf_path,
        device=device,
        num_envs=1,
        episode_length_s=args.duration + 2.0,
        dt=dt,
        vx_range=(args.cmd_vx, args.cmd_vx),
        vy_range=(args.cmd_vy, args.cmd_vy),
        yaw_range=(args.cmd_yaw, args.cmd_yaw),
    )

    # ==================== Generate sequences ====================
    print("[2/4] Generating trajectories...")
    print(f"       Command: vx={args.cmd_vx:.2f}, vy={args.cmd_vy:.2f}, yaw={args.cmd_yaw:.2f}")

    env_ids = torch.tensor([0], device=device)
    cmg.reset(env_ids)

    # Ensure no mirroring first - collect original frames
    cmg._mirror_flags[0] = False

    frames_orig_dof = []
    frames_orig_root = []
    frames_mirror_dof = []
    frames_mirror_root = []

    motion_ids = torch.zeros(1, dtype=torch.long, device=device)

    for i in range(num_frames):
        motion_times = cmg._motion_times.clone()

        # --- Original (no mirror) ---
        cmg._mirror_flags[0] = False
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = \
            cmg.calc_motion_frame(motion_ids, motion_times)

        rot_np = root_rot[0].numpy()
        frames_orig_dof.append(dof_pos[0].numpy().copy())
        frames_orig_root.append((root_pos[0].numpy().copy(),
                                 np.array([rot_np[3], rot_np[0], rot_np[1], rot_np[2]])))  # xyzw -> wxyz

        # --- Mirrored (same trajectory buffer, just apply mirror) ---
        cmg._mirror_flags[0] = True
        root_pos_m, root_rot_m, root_vel_m, root_ang_vel_m, dof_pos_m, dof_vel_m, body_pos_m = \
            cmg.calc_motion_frame(motion_ids, motion_times)

        rot_m_np = root_rot_m[0].numpy()
        frames_mirror_dof.append(dof_pos_m[0].numpy().copy())
        frames_mirror_root.append((root_pos_m[0].numpy().copy(),
                                   np.array([rot_m_np[3], rot_m_np[0], rot_m_np[1], rot_m_np[2]])))

        # Step with mirror off (advance the underlying state normally)
        cmg._mirror_flags[0] = False
        cmg.step()
        cmg._update_root_state(dt)

    # ==================== Print DOF comparison ====================
    print("\n[3/4] DOF comparison at frame 50 (t=1.0s):")
    print("       Testing: mirrored_output == manual_mirror(original_output)")
    joint_names = [
        'L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ankle_p', 'L_ankle_r',
        'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ankle_p', 'R_ankle_r',
        'waist_yaw', 'waist_roll', 'waist_pitch',
        'L_sh_pitch', 'L_sh_roll', 'L_sh_yaw', 'L_elbow',
        'R_sh_pitch', 'R_sh_roll', 'R_sh_yaw', 'R_elbow',
    ]

    mirror_signs = np.array(DOF_MIRROR_SIGNS_23)
    mirror_indices = DOF_MIRROR_INDICES_23

    frame_idx = min(50, num_frames - 1)
    orig = frames_orig_dof[frame_idx]
    mirr = frames_mirror_dof[frame_idx]

    # Manual mirror of original for verification
    manual_mirror = orig[mirror_indices] * mirror_signs

    print(f"\n{'Joint':<16} {'Original':>10} {'Mirror Output':>14} {'Manual Mirror':>14} {'Diff':>10} {'Match':>6}")
    print("-" * 74)
    all_match = True
    for j in range(23):
        diff = abs(mirr[j] - manual_mirror[j])
        match = diff < 1e-5
        if not match:
            all_match = False
        print(f"{joint_names[j]:<16} {orig[j]:>10.4f} {mirr[j]:>14.4f} {manual_mirror[j]:>14.4f} {diff:>10.2e} {'OK' if match else 'FAIL':>6}")

    print(f"\n>>> Mirror DOF verification: {'ALL PASSED' if all_match else 'SOME FAILED'}")

    # Root state comparison
    orig_pos, orig_rot = frames_orig_root[frame_idx]
    mirr_pos, mirr_rot = frames_mirror_root[frame_idx]
    print(f"\nRoot position:  original={np.array2string(orig_pos, precision=4)}")
    print(f"                mirrored={np.array2string(mirr_pos, precision=4)}")
    y_match = abs(orig_pos[1] + mirr_pos[1]) < 1e-4
    print(f"  -> y flipped: {y_match} (orig_y={orig_pos[1]:.4f}, mirr_y={mirr_pos[1]:.4f})")

    print(f"\nRoot rotation (wxyz):  original={np.array2string(orig_rot, precision=4)}")
    print(f"                       mirrored={np.array2string(mirr_rot, precision=4)}")

    # Command comparison
    cmg._mirror_flags[0] = False
    cmd_orig = cmg.get_commands()[0].numpy()
    cmg._mirror_flags[0] = True
    cmd_mirr = cmg.get_commands()[0].numpy()
    print(f"\nCommands:  original={cmd_orig}  mirrored={cmd_mirr}")
    vy_ok = abs(cmd_orig[1] + cmd_mirr[1]) < 1e-4
    yaw_ok = abs(cmd_orig[2] + cmd_mirr[2]) < 1e-4
    print(f"  -> vy flipped: {vy_ok}, yaw flipped: {yaw_ok}")

    # Body pos comparison
    orig_body = body_pos  # last frame's body_pos from the loop
    # Re-check a specific frame
    cmg._mirror_flags[0] = False
    r1 = cmg.calc_motion_frame(motion_ids, cmg._motion_times)
    cmg._mirror_flags[0] = True
    r2 = cmg.calc_motion_frame(motion_ids, cmg._motion_times)
    body_orig = r1[6][0].numpy()  # (9, 3)
    body_mirr = r2[6][0].numpy()  # (9, 3)
    body_manual = body_orig[KEYBODY_MIRROR_INDICES].copy()
    body_manual[:, 1] *= -1
    body_match = np.allclose(body_mirr, body_manual, atol=1e-4)

    body_names = ['L_hand', 'R_hand', 'L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'L_elbow', 'R_elbow', 'head']
    print(f"\nKey body positions (last frame):")
    print(f"{'Body':<10} {'Orig x':>8} {'Orig y':>8} {'Orig z':>8}  |  {'Mirr x':>8} {'Mirr y':>8} {'Mirr z':>8}")
    print("-" * 72)
    for b in range(9):
        print(f"{body_names[b]:<10} {body_orig[b,0]:>8.4f} {body_orig[b,1]:>8.4f} {body_orig[b,2]:>8.4f}  |  "
              f"{body_mirr[b,0]:>8.4f} {body_mirr[b,1]:>8.4f} {body_mirr[b,2]:>8.4f}")
    print(f"\n>>> Body position mirror verification: {'PASSED' if body_match else 'FAILED'}")

    if args.no_viewer:
        print("\n[4/4] Skipping viewer (--no_viewer)")
        return

    # ==================== Mujoco Visualization ====================
    print("\n[4/4] Launching mujoco viewer...")
    print("       LEFT (y>0) = original,  RIGHT (y<0) = mirrored")
    print("       Close window to exit.")

    xml_path = os.path.join(_script_dir, 'assets/g1/g1_sim2sim.xml')
    dual_xml_path = create_dual_robot_xml(xml_path, offset_y=2.0)

    try:
        model = mujoco.MjModel.from_xml_path(dual_xml_path)
        data = mujoco.MjData(model)

        nq_per_robot = 30  # 7 (freejoint) + 23 (hinge)
        robot1_start = 0
        robot2_start = nq_per_robot

        viewer = mujoco_viewer.MujocoViewer(model, data)

        frame = 0
        while viewer.is_alive:
            fi = frame % num_frames

            # Robot 1 (original) at y > 0
            pos1, rot1 = frames_orig_root[fi]
            pos1_v = pos1.copy()
            pos1_v[1] += 1.0
            data.qpos[robot1_start:robot1_start + 3] = pos1_v
            data.qpos[robot1_start + 3:robot1_start + 7] = rot1
            data.qpos[robot1_start + 7:robot1_start + 30] = frames_orig_dof[fi]

            # Robot 2 (mirrored) at y < 0
            pos2, rot2 = frames_mirror_root[fi]
            pos2_v = pos2.copy()
            pos2_v[1] -= 1.0
            data.qpos[robot2_start:robot2_start + 3] = pos2_v
            data.qpos[robot2_start + 3:robot2_start + 7] = rot2
            data.qpos[robot2_start + 7:robot2_start + 30] = frames_mirror_dof[fi]

            mujoco.mj_forward(model, data)
            viewer.render()

            time.sleep(dt / args.speed)
            frame += 1

        viewer.close()
    finally:
        if os.path.exists(dual_xml_path):
            os.unlink(dual_xml_path)


if __name__ == '__main__':
    main()
