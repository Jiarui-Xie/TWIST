#!/usr/bin/env python3
"""Test CMG motion quality under ramp (accel/decel) velocity profiles.

Generates a 10s episode with trapezoidal velocity profile:
  [0, T_ramp]: ramp up from 0 to v_target
  [T_ramp, T_ep - T_ramp]: steady at v_target
  [T_ep - T_ramp, T_ep]: ramp down from v_target to 0

Prints per-frame DOF position/velocity stats to check whether CMG
produces stable motions at low speeds (especially near v=0).

Usage:
    conda activate twist
    python tools/test_cmg_ramp.py
    python tools/test_cmg_ramp.py --cmd_vx 2.0 --ramp_duration 2.0
    python tools/test_cmg_ramp.py --cmd_vx 1.5 --plot   # save plot to file
"""

import os
import sys
import argparse
import torch
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)

from pose.utils.cmg_motion_lib import CMGMotionLib, CMG_TO_G1_INDICES


def main():
    parser = argparse.ArgumentParser(description="CMG Ramp Profile Test")
    parser.add_argument("--cmd_vx", type=float, default=1.5, help="Target forward velocity")
    parser.add_argument("--cmd_vy", type=float, default=0.0, help="Target lateral velocity")
    parser.add_argument("--cmd_yaw", type=float, default=0.0, help="Target yaw rate")
    parser.add_argument("--ramp_duration", type=float, default=1.5, help="Ramp duration (seconds)")
    parser.add_argument("--stand_duration", type=float, default=1.0, help="Standing duration before/after ramp (v=0)")
    parser.add_argument("--episode_length", type=float, default=14.0, help="Episode length (seconds)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", action="store_true", help="Save plot to tools/cmg_ramp_test.png")
    args = parser.parse_args()

    cmg_model_path = os.path.join(_project_dir, 'cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt')
    cmg_data_path = os.path.join(_project_dir, 'cmg_workspace/dataloader/cmg_training_data.pt')
    urdf_path = os.path.join(_project_dir, 'assets/g1/g1_custom_collision_with_fixed_hand.urdf')

    dt = 0.02
    n_envs = 1
    t_ramp = args.ramp_duration
    t_stand = args.stand_duration
    t_ep = args.episode_length
    n_steps = int(t_ep / dt)

    print(f"Target velocity: vx={args.cmd_vx:.2f}, vy={args.cmd_vy:.2f}, yaw={args.cmd_yaw:.2f}")
    print(f"Stand: {t_stand:.1f}s, Ramp: {t_ramp:.1f}s, Episode: {t_ep:.1f}s, Steps: {n_steps}")
    print()

    # Create CMG (ramp disabled — we'll manually apply ramp to test raw CMG output)
    cmg = CMGMotionLib(
        cmg_model_path=cmg_model_path,
        cmg_data_path=cmg_data_path,
        urdf_path=urdf_path,
        device=args.device,
        num_envs=n_envs,
        episode_length_s=t_ep,
        dt=dt,
        vx_range=(args.cmd_vx, args.cmd_vx),
        vy_range=(args.cmd_vy, args.cmd_vy),
        yaw_range=(args.cmd_yaw, args.cmd_yaw),
    )

    env_ids = torch.arange(n_envs, device=args.device)
    cmg.reset(env_ids)

    # Storage for analysis
    times = []
    cmd_vx_profile = []
    dof_pos_std_list = []
    dof_vel_rms_list = []
    dof_pos_list = []  # (n_steps, 23)

    # Manually step CMG with time-varying commands
    current_norm = cmg._current_motion_norm.clone()

    for step_i in range(n_steps):
        t = step_i * dt

        # 5-phase ramp scale: [stand → ramp_up → steady → ramp_down → stand]
        t1 = t_stand                     # end of initial stand
        t2 = t_stand + t_ramp            # end of ramp-up
        t3 = t_ep - t_stand - t_ramp     # start of ramp-down
        t4 = t_ep - t_stand              # end of ramp-down
        if t < t1:
            scale = 0.0
        elif t < t2:
            scale = (t - t1) / t_ramp
        elif t < t3:
            scale = 1.0
        elif t < t4:
            scale = (t4 - t) / t_ramp
        else:
            scale = 0.0
        scale = max(scale, 0.0)

        # Scaled command
        vx = args.cmd_vx * scale
        vy = args.cmd_vy * scale
        yaw = args.cmd_yaw * scale
        cmd = torch.tensor([[vx, vy, yaw]], device=args.device)
        cmd_norm = cmg._normalize_command(cmd)

        # Step CMG
        next_norm = cmg._cmg_model(current_norm, cmd_norm)

        # Denormalize to get raw DOF
        motion_raw = next_norm * cmg._motion_std + cmg._motion_mean  # (1, 58)
        pos_29 = motion_raw[0, :29]
        vel_29 = motion_raw[0, 29:]

        pos_23 = pos_29[CMG_TO_G1_INDICES]
        vel_23 = vel_29[CMG_TO_G1_INDICES]

        times.append(t)
        cmd_vx_profile.append(vx)
        dof_pos_std_list.append(pos_23.std().item())
        dof_vel_rms_list.append(vel_23.pow(2).mean().sqrt().item())
        dof_pos_list.append(pos_23.detach().cpu().numpy())

        current_norm = next_norm

    dof_pos_arr = np.stack(dof_pos_list)  # (n_steps, 23)
    times_arr = np.array(times)
    cmd_vx_arr = np.array(cmd_vx_profile)

    # Print summary table
    # Sample at key time points
    sample_times = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 8.0, 8.5, 9.0, 9.5, 10.0 - dt]
    sample_times = [t for t in sample_times if t < t_ep]

    print(f"{'time':>6s}  {'cmd_vx':>7s}  {'dof_std':>8s}  {'vel_rms':>8s}  {'knee_L':>7s}  {'knee_R':>7s}  {'hip_p_L':>8s}  {'hip_p_R':>8s}")
    print("-" * 75)

    for t_sample in sample_times:
        idx = min(int(t_sample / dt), n_steps - 1)
        print(f"{times_arr[idx]:6.2f}  {cmd_vx_arr[idx]:7.3f}  "
              f"{dof_pos_std_list[idx]:8.4f}  {dof_vel_rms_list[idx]:8.4f}  "
              f"{dof_pos_arr[idx, 3]:7.3f}  {dof_pos_arr[idx, 9]:7.3f}  "
              f"{dof_pos_arr[idx, 0]:8.3f}  {dof_pos_arr[idx, 6]:8.3f}")

    # Check stability: look for divergence (NaN, large values)
    max_pos = np.abs(dof_pos_arr).max()
    has_nan = np.isnan(dof_pos_arr).any()
    print(f"\nMax |DOF pos|: {max_pos:.4f}  NaN: {has_nan}")

    if max_pos > 5.0:
        print("WARNING: DOF positions diverged! CMG may be unstable at low speeds.")
    elif has_nan:
        print("WARNING: NaN detected in DOF positions!")
    else:
        print("OK: DOF positions stable throughout ramp profile.")

    # Compare steady vs ramp phases
    steady_start = int(t_ramp / dt) + 10
    steady_end = int((t_ep - t_ramp) / dt) - 10
    ramp_up_end = int(t_ramp * 0.3 / dt)  # early ramp (low speed)
    ramp_down_start = int((t_ep - t_ramp * 0.3) / dt)  # late ramp (low speed)

    if steady_start < steady_end:
        steady_vel_rms = np.mean(dof_vel_rms_list[steady_start:steady_end])
        early_vel_rms = np.mean(dof_vel_rms_list[:max(ramp_up_end, 1)])
        late_vel_rms = np.mean(dof_vel_rms_list[min(ramp_down_start, n_steps-1):])

        print(f"\nPhase velocity RMS comparison:")
        print(f"  Early ramp (v≈0):  {early_vel_rms:.4f}")
        print(f"  Steady (v=target): {steady_vel_rms:.4f}")
        print(f"  Late ramp (v≈0):   {late_vel_rms:.4f}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

            # 1. Command velocity profile
            axes[0].plot(times_arr, cmd_vx_arr, 'b-', linewidth=2)
            axes[0].set_ylabel('cmd_vx (m/s)')
            axes[0].set_title(f'CMG Ramp Test: vx_target={args.cmd_vx}, ramp={t_ramp}s')
            axes[0].axvline(t_ramp, color='gray', linestyle='--', alpha=0.5, label='ramp boundary')
            axes[0].axvline(t_ep - t_ramp, color='gray', linestyle='--', alpha=0.5)
            axes[0].legend()

            # 2. DOF velocity RMS
            axes[1].plot(times_arr, dof_vel_rms_list, 'r-', linewidth=1)
            axes[1].set_ylabel('DOF vel RMS')
            axes[1].axvline(t_ramp, color='gray', linestyle='--', alpha=0.5)
            axes[1].axvline(t_ep - t_ramp, color='gray', linestyle='--', alpha=0.5)

            # 3. Key joint positions (knees + hip pitch)
            axes[2].plot(times_arr, dof_pos_arr[:, 0], label='hip_pitch_L')
            axes[2].plot(times_arr, dof_pos_arr[:, 6], label='hip_pitch_R')
            axes[2].plot(times_arr, dof_pos_arr[:, 3], label='knee_L')
            axes[2].plot(times_arr, dof_pos_arr[:, 9], label='knee_R')
            axes[2].set_ylabel('Joint pos (rad)')
            axes[2].legend(fontsize=8)
            axes[2].axvline(t_ramp, color='gray', linestyle='--', alpha=0.5)
            axes[2].axvline(t_ep - t_ramp, color='gray', linestyle='--', alpha=0.5)

            # 4. Ankle positions
            axes[3].plot(times_arr, dof_pos_arr[:, 4], label='ankle_pitch_L')
            axes[3].plot(times_arr, dof_pos_arr[:, 10], label='ankle_pitch_R')
            axes[3].plot(times_arr, dof_pos_arr[:, 5], label='ankle_roll_L')
            axes[3].plot(times_arr, dof_pos_arr[:, 11], label='ankle_roll_R')
            axes[3].set_ylabel('Joint pos (rad)')
            axes[3].set_xlabel('Time (s)')
            axes[3].legend(fontsize=8)
            axes[3].axvline(t_ramp, color='gray', linestyle='--', alpha=0.5)
            axes[3].axvline(t_ep - t_ramp, color='gray', linestyle='--', alpha=0.5)

            plt.tight_layout()
            out_path = os.path.join(_script_dir, 'cmg_ramp_test.png')
            plt.savefig(out_path, dpi=150)
            print(f"\nPlot saved to {out_path}")
        except ImportError:
            print("\nmatplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
