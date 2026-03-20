"""Render CMG random ramp playback to image strips (offscreen, no display needed).

Usage:
    conda activate Main
    python tools/render_random_ramp.py
    python tools/render_random_ramp.py --n_resets 4 --cmd_vx 2.0
"""
import os, sys, argparse
import torch
import numpy as np
import mujoco
import matplotlib.pyplot as plt

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_dir, 'pose'))
sys.path.insert(0, os.path.join(_project_dir, 'cmg_workspace'))

from pose.utils.cmg_motion_lib import CMGMotionLib

# Constants matching vis_cmg_ramp.py
CMG_DT = 0.02
SIM_DT = 0.002
PLAYBACK_DECIMATION = int(CMG_DT / SIM_DT)

DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left leg
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg
    0.0, 0.0, 0.0,                      # waist
    0.0, 0.4, 0.0, 1.2,                 # left arm
    0.0, -0.4, 0.0, 1.2,                # right arm
], dtype=np.float32)

BODY_DOF_IDS_25 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,22,23,24]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd_vx", type=float, default=1.5)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--n_resets", type=int, default=3, help="Number of different ramp profiles")
    parser.add_argument("--n_snapshots", type=int, default=8, help="Snapshots per episode")
    parser.add_argument("--episode_length", type=float, default=16.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="tools/render_random_ramp.png")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    device = args.device
    t_ep = args.episode_length

    cmg = CMGMotionLib(
        cmg_model_path=os.path.join(_project_dir, "cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"),
        cmg_data_path=os.path.join(_project_dir, "cmg_workspace/dataloader/cmg_training_data.pt"),
        urdf_path=os.path.join(_project_dir, "assets/g1/g1_custom_collision_with_fixed_hand.urdf"),
        device=device, num_envs=1,
        episode_length_s=t_ep, dt=CMG_DT,
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

    # MuJoCo setup
    xml_path = os.path.join(_project_dir, "assets/g1/g1_mocap_with_wrist_roll.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = SIM_DT
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, args.height, args.width)

    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    cmd = torch.tensor([[args.cmd_vx, args.cmd_vy, args.cmd_yaw]], device=device)

    # Snapshot times evenly spaced across episode
    snap_times = np.linspace(0, t_ep - 0.5, args.n_snapshots)

    n_resets = args.n_resets
    n_snaps = args.n_snapshots

    all_frames = np.zeros((n_resets, n_snaps, args.height, args.width, 3), dtype=np.uint8)
    all_cmd_vx = np.zeros((n_resets, n_snaps))
    labels = []

    n_steps = int(t_ep / CMG_DT)

    for r in range(n_resets):
        cmg.reset(env_ids, commands=cmd)
        cmg._mirror_flags[:] = False

        ru = cmg._env_ramp_up[0].item()
        rd = cmg._env_ramp_down[0].item()
        cr = cmg._env_crawl[0].item()
        steady = t_ep - 2.0 - ru - rd - cr
        labels.append(f"up={ru:.1f}s  down={rd:.1f}s  crawl={cr:.1f}s  steady={steady:.1f}s")
        print(f"Reset {r}: {labels[-1]}")

        # Initial pose
        dof_25 = np.zeros(25, dtype=np.float32)
        dof_25[BODY_DOF_IDS_25] = DEFAULT_DOF_POS

        snap_idx = 0
        for step in range(n_steps):
            t = step * CMG_DT
            cmg.step(env_ids)

            # Check if we should capture a snapshot
            if snap_idx < n_snaps and t >= snap_times[snap_idx] - CMG_DT * 0.5:
                motion_ids = torch.zeros(1, dtype=torch.long, device=device)
                motion_times = cmg._motion_times[:1]
                root_pos, root_rot_xyzw, _, _, dof_pos_23, _, _ = \
                    cmg.calc_motion_frame(motion_ids, motion_times)

                rp = root_pos[0].cpu().numpy()
                rr = root_rot_xyzw[0].cpu().numpy()
                quat_wxyz = np.array([rr[3], rr[0], rr[1], rr[2]])
                dp = dof_pos_23[0].cpu().numpy()

                data.qpos[0:3] = rp
                data.qpos[3:7] = quat_wxyz
                dof_full = np.zeros(25, dtype=np.float32)
                dof_full[BODY_DOF_IDS_25] = dp
                data.qpos[7:] = dof_full
                mujoco.mj_forward(model, data)

                # Render
                renderer.update_scene(data)
                frame = renderer.render()
                all_frames[r, snap_idx] = frame

                # Get cmd at this time
                scale = cmg._compute_ramp_scale(env_ids, torch.tensor([t], device=device))
                inst_cmd = cmg._target_commands * scale.unsqueeze(-1)
                all_cmd_vx[r, snap_idx] = inst_cmd[0, 0].item()

                snap_idx += 1

    # ── Plot: grid of snapshots ──
    fig, axes = plt.subplots(n_resets, n_snaps, figsize=(n_snaps * 2.5, n_resets * 2.5 + 1))
    if n_resets == 1:
        axes = axes[np.newaxis, :]

    for r in range(n_resets):
        for s in range(n_snaps):
            ax = axes[r, s]
            ax.imshow(all_frames[r, s])
            ax.set_xticks([])
            ax.set_yticks([])
            t = snap_times[s]
            vx = all_cmd_vx[r, s]
            ax.set_title(f"t={t:.1f}s  vx={vx:.2f}", fontsize=7)
        # Row label
        axes[r, 0].set_ylabel(f"Reset {r}\n{labels[r]}", fontsize=7, rotation=0,
                               labelpad=100, va='center')

    fig.suptitle(f"CMG Random Ramp — MuJoCo Kinematic Playback  (target vx={args.cmd_vx})",
                 fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(_project_dir, args.save)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
