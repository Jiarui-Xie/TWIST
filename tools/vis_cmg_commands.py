#!/usr/bin/env python3
"""Real-time visualization of CMG velocity commands and joint output.

Runs CMGMotionLib step-by-step with ramp profile and displays:
  - Top: velocity command profile (target vs instantaneous vx/vy/yaw)
  - Bottom: key joint positions (knees, hip pitch) to verify CMG output quality

No robot rendering — pure command/output monitoring.

Usage:
    conda activate Main
    cd /home/lumi/TWIST
    python tools/vis_cmg_commands.py
    python tools/vis_cmg_commands.py --cmd_vx 2.0 --episode_length 14
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in [os.path.join(REPO_ROOT, "pose"), os.path.join(REPO_ROOT, "cmg_workspace")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pose.utils.cmg_motion_lib import CMGMotionLib, CMG_TO_G1_INDICES


def main():
    parser = argparse.ArgumentParser(description="Real-time CMG command visualization")
    parser.add_argument("--cmd_vx", type=float, default=1.5)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--ramp_up", type=float, default=1.5)
    parser.add_argument("--ramp_down", type=float, default=3.0)
    parser.add_argument("--stand_duration", type=float, default=1.0)
    parser.add_argument("--crawl_duration", type=float, default=1.0)
    parser.add_argument("--crawl_ratio", type=float, default=0.01)
    parser.add_argument("--episode_length", type=float, default=16.5)
    parser.add_argument("--floor_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (2.0 = 2x fast)")
    args = parser.parse_args()

    dt = 0.02
    t_ep = args.episode_length
    n_steps = int(t_ep / dt)

    # ── CMG ──
    cmg = CMGMotionLib(
        cmg_model_path=os.path.join(REPO_ROOT, "cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"),
        cmg_data_path=os.path.join(REPO_ROOT, "cmg_workspace/dataloader/cmg_training_data.pt"),
        urdf_path=os.path.join(REPO_ROOT, "assets/g1/g1_custom_collision_with_fixed_hand.urdf"),
        device=args.device, num_envs=1,
        episode_length_s=t_ep, dt=dt,
        vx_range=(args.cmd_vx, args.cmd_vx),
        vy_range=(args.cmd_vy, args.cmd_vy),
        yaw_range=(args.cmd_yaw, args.cmd_yaw),
        ramp_enabled=True,
        ramp_up_range=(args.ramp_up, args.ramp_up),
        ramp_down_range=(args.ramp_down, args.ramp_down),
        ramp_stand_duration=args.stand_duration,
        ramp_crawl_range=(args.crawl_duration, args.crawl_duration),
        ramp_crawl_ratio=args.crawl_ratio,
        ramp_probability=1.0,
        ramp_floor_ratio=args.floor_ratio,
    )

    env_ids = torch.zeros(1, dtype=torch.long, device=args.device)
    init_cmd = torch.tensor([[args.cmd_vx, args.cmd_vy, args.cmd_yaw]], device=args.device)
    cmg.reset(env_ids, commands=init_cmd)
    cmg._mirror_flags[:] = False

    # ── Storage ──
    max_display = n_steps
    times = np.zeros(max_display)
    cmd_vx = np.zeros(max_display)
    cmd_vy = np.zeros(max_display)
    cmd_yaw = np.zeros(max_display)
    target_vx = np.zeros(max_display)
    knee_l = np.zeros(max_display)
    knee_r = np.zeros(max_display)
    hip_l = np.zeros(max_display)
    hip_r = np.zeros(max_display)

    step_idx = [0]  # mutable counter for animation

    # ── Plot setup ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"CMG Ramp Profile: vx={args.cmd_vx}, ramp_up={args.ramp_up}s, "
                 f"ramp_down={args.ramp_down}s, stand={args.stand_duration}s, floor={args.floor_ratio}")

    # Subplot 1: vx command
    ax1 = axes[0]
    line_target, = ax1.plot([], [], 'b--', lw=1.5, label='target_vx')
    line_cmd_vx, = ax1.plot([], [], 'r-', lw=2, label='cmd_vx (instantaneous)')
    ax1.set_ylabel('vx (m/s)')
    ax1.set_ylim(-0.1, args.cmd_vx * 1.3 + 0.1)
    ax1.legend(loc='upper right')
    ax1.axhline(0, color='gray', lw=0.5)
    # Phase boundaries: stand | ramp_up | steady | ramp_down | crawl | stand
    t_stand = args.stand_duration
    t_ramp_up = args.ramp_up
    t_ramp_down = args.ramp_down
    t_crawl = args.crawl_duration
    b1 = t_stand                                        # end stand1
    b2 = b1 + t_ramp_up                                 # end ramp_up
    b5 = t_ep - t_stand                                  # start stand2
    b4 = b5 - t_crawl                                   # start crawl (end ramp_down)
    b3 = b4 - t_ramp_down                               # start ramp_down
    boundaries = [b1, b2, b3, b4, b5]
    for tb in boundaries:
        ax1.axvline(tb, color='gray', ls='--', alpha=0.3)

    # Subplot 2: vy + yaw
    ax2 = axes[1]
    line_cmd_vy, = ax2.plot([], [], 'g-', lw=2, label='cmd_vy')
    line_cmd_yaw, = ax2.plot([], [], 'm-', lw=2, label='cmd_yaw')
    ax2.set_ylabel('vy / yaw')
    vy_max = max(abs(args.cmd_vy), abs(args.cmd_yaw), 0.3) * 1.3
    ax2.set_ylim(-vy_max, vy_max)
    ax2.legend(loc='upper right')
    ax2.axhline(0, color='gray', lw=0.5)
    for tb in boundaries:
        ax2.axvline(tb, color='gray', ls='--', alpha=0.3)

    # Subplot 3: joint positions
    ax3 = axes[2]
    line_kl, = ax3.plot([], [], 'b-', lw=1.5, label='knee_L')
    line_kr, = ax3.plot([], [], 'r-', lw=1.5, label='knee_R')
    line_hl, = ax3.plot([], [], 'b--', lw=1, label='hip_pitch_L')
    line_hr, = ax3.plot([], [], 'r--', lw=1, label='hip_pitch_R')
    ax3.set_ylabel('Joint pos (rad)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylim(-1.0, 1.5)
    ax3.legend(loc='upper right', fontsize=8)
    for tb in boundaries:
        ax3.axvline(tb, color='gray', ls='--', alpha=0.3)

    for ax in axes:
        ax.set_xlim(0, t_ep)
        ax.grid(True, alpha=0.3)

    # Phase labels
    y_label = args.cmd_vx * 1.2
    ax1.text(b1 / 2, y_label, 'STAND', ha='center', fontsize=7, color='gray')
    ax1.text((b1 + b2) / 2, y_label, 'RAMP↑', ha='center', fontsize=7, color='gray')
    ax1.text((b2 + b3) / 2, y_label, 'STEADY', ha='center', fontsize=7, color='gray')
    ax1.text((b3 + b4) / 2, y_label, 'RAMP↓', ha='center', fontsize=7, color='gray')
    ax1.text((b4 + b5) / 2, y_label, 'CRAWL', ha='center', fontsize=7, color='orange')
    ax1.text((b5 + t_ep) / 2, y_label, 'STAND', ha='center', fontsize=7, color='gray')

    # Current time indicator
    vline1 = ax1.axvline(0, color='red', lw=1, alpha=0.5)
    vline2 = ax2.axvline(0, color='red', lw=1, alpha=0.5)
    vline3 = ax3.axvline(0, color='red', lw=1, alpha=0.5)

    def update(frame_num):
        i = step_idx[0]
        if i >= n_steps:
            # Reset episode
            cmg.reset(env_ids, commands=init_cmd)
            cmg._mirror_flags[:] = False
            step_idx[0] = 0
            i = 0
            # Clear arrays
            times[:] = 0; cmd_vx[:] = 0; cmd_vy[:] = 0; cmd_yaw[:] = 0
            target_vx[:] = 0; knee_l[:] = 0; knee_r[:] = 0; hip_l[:] = 0; hip_r[:] = 0

        # Step CMG
        cmg.step(env_ids)
        t = cmg._motion_times[0].item()

        # Read commands
        c = cmg._commands[0].cpu().numpy()
        tc = cmg._target_commands[0].cpu().numpy()

        # Read joint positions from current frame
        motion_ids = torch.zeros(1, dtype=torch.long, device=args.device)
        motion_times = cmg._motion_times[:1]
        _, _, _, _, dof_pos, _, _ = cmg.calc_motion_frame(motion_ids, motion_times)
        dp = dof_pos[0].cpu().numpy()

        # Store
        times[i] = t
        cmd_vx[i] = c[0]
        cmd_vy[i] = c[1]
        cmd_yaw[i] = c[2]
        target_vx[i] = tc[0]
        knee_l[i] = dp[3]   # left knee
        knee_r[i] = dp[9]   # right knee
        hip_l[i] = dp[0]    # left hip pitch
        hip_r[i] = dp[6]    # right hip pitch

        # Update plots
        sl = slice(0, i + 1)
        line_target.set_data(times[sl], target_vx[sl])
        line_cmd_vx.set_data(times[sl], cmd_vx[sl])
        line_cmd_vy.set_data(times[sl], cmd_vy[sl])
        line_cmd_yaw.set_data(times[sl], cmd_yaw[sl])
        line_kl.set_data(times[sl], knee_l[sl])
        line_kr.set_data(times[sl], knee_r[sl])
        line_hl.set_data(times[sl], hip_l[sl])
        line_hr.set_data(times[sl], hip_r[sl])

        vline1.set_xdata([t, t])
        vline2.set_xdata([t, t])
        vline3.set_xdata([t, t])

        step_idx[0] += 1
        return (line_target, line_cmd_vx, line_cmd_vy, line_cmd_yaw,
                line_kl, line_kr, line_hl, line_hr, vline1, vline2, vline3)

    # Real-time animation: dt=20ms per CMG step, adjusted by speed
    interval_ms = max(1, int(dt * 1000 / args.speed))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=True, cache_frame_data=False)

    plt.tight_layout()
    print(f"[vis_cmg_commands] Episode: {t_ep}s, dt={dt}s, steps={n_steps}")
    print(f"[vis_cmg_commands] Profile: stand({args.stand_duration}s) → ramp_up({args.ramp_up}s) → "
          f"steady → ramp_down({args.ramp_down}s) → crawl({args.crawl_duration}s) → stand({args.stand_duration}s)")
    print(f"[vis_cmg_commands] Playback speed: {args.speed}x (interval={interval_ms}ms)")
    print(f"[vis_cmg_commands] Close window to exit. Episode auto-resets at {t_ep}s.")
    plt.show()


if __name__ == "__main__":
    main()
