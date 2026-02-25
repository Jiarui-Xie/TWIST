#!/usr/bin/env python3
"""Diagnostic script for CMG velocity calibration quality.

Runs CMG at a grid of input velocities and compares commanded vs estimated
actual velocity using the built-in linear regression estimator.

Usage:
    conda activate twist
    python tools/test_cmg_velocity.py
    python tools/test_cmg_velocity.py --vx_min 0.5 --vx_max 3.0 --vx_step 0.25
"""

import os
import sys
import argparse
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)

from pose.utils.cmg_motion_lib import CMGMotionLib


def main():
    parser = argparse.ArgumentParser(description="CMG Velocity Calibration Diagnostic")
    parser.add_argument("--vx_min", type=float, default=0.5, help="Min forward velocity")
    parser.add_argument("--vx_max", type=float, default=3.0, help="Max forward velocity")
    parser.add_argument("--vx_step", type=float, default=0.25, help="Forward velocity step")
    parser.add_argument("--vy", type=float, default=0.0, help="Fixed lateral velocity")
    parser.add_argument("--yaw", type=float, default=0.0, help="Fixed yaw rate")
    parser.add_argument("--n_trials", type=int, default=5, help="Trials per velocity for averaging")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:N)")
    args = parser.parse_args()

    # Paths
    cmg_model_path = os.path.join(_project_dir, 'cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt')
    cmg_data_path = os.path.join(_project_dir, 'cmg_workspace/dataloader/cmg_training_data.pt')
    urdf_path = os.path.join(_project_dir, 'assets/g1/g1_custom_collision_with_fixed_hand.urdf')

    # Generate velocity grid
    vx_values = []
    vx = args.vx_min
    while vx <= args.vx_max + 1e-6:
        vx_values.append(vx)
        vx += args.vx_step

    n_envs = args.n_trials

    print(f"Testing {len(vx_values)} velocities x {n_envs} trials")
    print(f"Fixed: vy={args.vy:.2f}, yaw={args.yaw:.2f}")
    print()

    # Header
    print(f"{'cmd_vx':>8s}  {'est_vx':>8s}  {'ratio':>7s}  {'est_vy':>8s}  {'est_yaw':>8s}")
    print("-" * 50)

    for cmd_vx in vx_values:
        # Create CMG with fixed velocity range
        cmg = CMGMotionLib(
            cmg_model_path=cmg_model_path,
            cmg_data_path=cmg_data_path,
            urdf_path=urdf_path,
            device=args.device,
            num_envs=n_envs,
            episode_length_s=4.0,
            dt=0.02,
            vx_range=(cmd_vx, cmd_vx),
            vy_range=(args.vy, args.vy),
            yaw_range=(args.yaw, args.yaw),
        )

        # Reset all envs (triggers trajectory generation + velocity estimation)
        env_ids = torch.arange(n_envs, device=args.device)
        cmg.reset(env_ids)

        # Read estimated actual commands (averaged over trials)
        actual = cmg._actual_commands.mean(dim=0)  # (3,)
        est_vx = actual[0].item()
        est_vy = actual[1].item()
        est_yaw = actual[2].item()
        ratio = est_vx / cmd_vx if abs(cmd_vx) > 1e-6 else float('nan')

        print(f"{cmd_vx:8.3f}  {est_vx:8.3f}  {ratio:7.3f}  {est_vy:8.3f}  {est_yaw:8.3f}")

    print()

    # Also test lateral velocity grid
    if abs(args.vy) < 1e-6:
        print("\n--- Lateral velocity sweep (vx=1.0) ---")
        print(f"{'cmd_vy':>8s}  {'est_vx':>8s}  {'est_vy':>8s}  {'est_yaw':>8s}")
        print("-" * 42)

        for cmd_vy in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
            cmg = CMGMotionLib(
                cmg_model_path=cmg_model_path,
                cmg_data_path=cmg_data_path,
                urdf_path=urdf_path,
                device=args.device,
                num_envs=n_envs,
                episode_length_s=4.0,
                dt=0.02,
                vx_range=(1.0, 1.0),
                vy_range=(cmd_vy, cmd_vy),
                yaw_range=(0.0, 0.0),
            )
            env_ids = torch.arange(n_envs, device=args.device)
            cmg.reset(env_ids)
            actual = cmg._actual_commands.mean(dim=0)
            print(f"{cmd_vy:8.3f}  {actual[0].item():8.3f}  {actual[1].item():8.3f}  {actual[2].item():8.3f}")

    # Also test yaw rate grid
    print("\n--- Yaw rate sweep (vx=1.0, vy=0.0) ---")
    print(f"{'cmd_yaw':>8s}  {'est_vx':>8s}  {'est_vy':>8s}  {'est_yaw':>8s}")
    print("-" * 42)

    for cmd_yaw in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
        cmg = CMGMotionLib(
            cmg_model_path=cmg_model_path,
            cmg_data_path=cmg_data_path,
            urdf_path=urdf_path,
            device=args.device,
            num_envs=n_envs,
            episode_length_s=4.0,
            dt=0.02,
            vx_range=(1.0, 1.0),
            vy_range=(0.0, 0.0),
            yaw_range=(cmd_yaw, cmd_yaw),
        )
        env_ids = torch.arange(n_envs, device=args.device)
        cmg.reset(env_ids)
        actual = cmg._actual_commands.mean(dim=0)
        print(f"{cmd_yaw:8.3f}  {actual[0].item():8.3f}  {actual[1].item():8.3f}  {actual[2].item():8.3f}")


if __name__ == "__main__":
    main()
