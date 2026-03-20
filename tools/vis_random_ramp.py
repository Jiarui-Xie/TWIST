"""Visualize multiple random ramp resets for a single env + CMG joint output.

Usage:
    conda activate Main
    python tools/vis_random_ramp.py
    python tools/vis_random_ramp.py --n_resets 8 --cmd_vx 2.0
"""
import os, sys, argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_dir, 'pose'))
sys.path.insert(0, os.path.join(_project_dir, 'cmg_workspace'))

from pose.utils.cmg_motion_lib import CMGMotionLib


def main():
    parser = argparse.ArgumentParser(description="Visualize random ramp profiles (single env, multiple resets)")
    parser.add_argument("--cmd_vx", type=float, default=1.5)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--n_resets", type=int, default=6, help="Number of random resets to show")
    parser.add_argument("--episode_length", type=float, default=16.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="tools/vis_random_ramp.png")
    args = parser.parse_args()

    dt = 0.02
    t_ep = args.episode_length
    n_steps = int(t_ep / dt)
    n_resets = args.n_resets

    cmg = CMGMotionLib(
        cmg_model_path=os.path.join(_project_dir, "cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"),
        cmg_data_path=os.path.join(_project_dir, "cmg_workspace/dataloader/cmg_training_data.pt"),
        urdf_path=os.path.join(_project_dir, "assets/g1/g1_custom_collision_with_fixed_hand.urdf"),
        device=args.device, num_envs=1,
        episode_length_s=t_ep, dt=dt,
        vx_range=(args.cmd_vx, args.cmd_vx),
        vy_range=(args.cmd_vy, args.cmd_vy),
        yaw_range=(args.cmd_yaw, args.cmd_yaw),
        ramp_enabled=True,
        ramp_up_range=(0.5, 2.5),
        ramp_down_range=(1.0, 4.0),
        ramp_stand_duration=1.0,
        ramp_crawl_range=(0.5, 2.0),
        ramp_crawl_ratio=0.01,
        ramp_probability=1.0,
        ramp_floor_ratio=0.1,
        ramp_min_steady=3.0,
    )

    env_ids = torch.zeros(1, dtype=torch.long, device=args.device)
    cmd = torch.tensor([[args.cmd_vx, args.cmd_vy, args.cmd_yaw]], device=args.device)

    # Storage
    all_times = np.zeros(n_steps)
    all_cmd_vx = np.zeros((n_resets, n_steps))
    all_dof_std = np.zeros((n_resets, n_steps))
    all_root_vx = np.zeros((n_resets, n_steps))
    labels = []

    print(f"\n{'reset':>6}  {'ramp_up':>8}  {'ramp_down':>10}  {'crawl':>6}  {'steady':>7}")
    print("-" * 47)

    for r in range(n_resets):
        cmg.reset(env_ids, commands=cmd)
        cmg._mirror_flags[:] = False

        ru = cmg._env_ramp_up[0].item()
        rd = cmg._env_ramp_down[0].item()
        cr = cmg._env_crawl[0].item()
        steady = t_ep - 2 * 1.0 - ru - rd - cr
        print(f"{r:>6}  {ru:>8.2f}  {rd:>10.2f}  {cr:>6.2f}  {steady:>7.2f}")
        labels.append(f"up={ru:.1f} down={rd:.1f} crawl={cr:.1f}")

        for step in range(n_steps):
            t = step * dt
            if r == 0:
                all_times[step] = t

            scale = cmg._compute_ramp_scale(env_ids, torch.tensor([t], device=args.device))
            inst_cmd = cmg._target_commands * scale.unsqueeze(-1)
            all_cmd_vx[r, step] = inst_cmd[0, 0].item()

            cmg.step(env_ids)

            motion_ids = torch.zeros(1, dtype=torch.long, device=args.device)
            motion_times = cmg._motion_times.clone()
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, kbp = \
                cmg.calc_motion_frame(motion_ids, motion_times)

            all_dof_std[r, step] = dof_pos.std(dim=-1).item()
            all_root_vx[r, step] = root_vel[0, 0].item()

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n_resets))

    ax1 = axes[0]
    for r in range(n_resets):
        ax1.plot(all_times, all_cmd_vx[r], color=colors[r], lw=1.5, alpha=0.8, label=labels[r])
    ax1.set_ylabel("cmd_vx (m/s)")
    ax1.set_title(f"Random Ramp Profiles (1 env, {n_resets} resets) — target vx={args.cmd_vx}")
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.axhline(args.cmd_vx, color='gray', ls='--', lw=0.8)
    ax1.axhline(0, color='gray', lw=0.5)

    ax2 = axes[1]
    for r in range(n_resets):
        ax2.plot(all_times, all_dof_std[r], color=colors[r], lw=1, alpha=0.7)
    ax2.set_ylabel("DOF pos std (rad)")
    ax2.set_title("Joint Activity (DOF std across joints)")

    ax3 = axes[2]
    for r in range(n_resets):
        ax3.plot(all_times, all_root_vx[r], color=colors[r], lw=1, alpha=0.7)
    ax3.set_ylabel("root_vx (m/s)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Root Forward Velocity (CMG integration)")
    ax3.axhline(args.cmd_vx, color='gray', ls='--', lw=0.8)

    plt.tight_layout()
    save_path = os.path.join(_project_dir, args.save)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
