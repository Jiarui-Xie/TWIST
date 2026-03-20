#!/usr/bin/env python3
"""Visualize CMG reference motion with ramp velocity profile in MuJoCo.

Plays back the CMG-generated joint trajectory directly on the MuJoCo model
(no policy, pure kinematic playback). Shows how the reference motion looks
when velocity ramps from 0 → v_target → 0 (trapezoidal profile).

Usage:
    conda activate Main
    cd /home/lumi/TWIST
    python tools/vis_cmg_ramp.py --cmd_vx 1.5
    python tools/vis_cmg_ramp.py --cmd_vx 2.0 --ramp_up 2.0
    python tools/vis_cmg_ramp.py --cmd_vx 1.5 --no_ramp   # constant speed for comparison

Keyboard controls (in MuJoCo viewer):
    ↑/↓   = adjust vx  (+/- 0.2)
    ←/→   = adjust yaw  (+/- 0.2)
    A/D   = adjust vy  (+/- 0.1)
    R     = reset episode
    SPACE = toggle pause
"""

import os
import sys
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer as mjv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in [
    os.path.join(REPO_ROOT, "pose"),
    os.path.join(REPO_ROOT, "cmg_workspace"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pose.utils.cmg_motion_lib import CMGMotionLib, CMG_TO_G1_INDICES

# MuJoCo sim constants
SIM_DT = 0.001
CMG_DT = 0.02          # 50 Hz
PLAYBACK_DECIMATION = 20  # update pose every 20ms (50 Hz)

# 23 DOF indices in 25-DOF MuJoCo model (skip wrist-roll at 19, 24)
BODY_DOF_IDS_25 = [i for i in range(25) if i not in (19, 24)]

DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left leg
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg
     0.0, 0.0, 0.0,                    # waist
     0.0, 0.4, 0.0, 1.2,               # left arm
     0.0,-0.4, 0.0, 1.2,               # right arm
], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Visualize CMG ramp in MuJoCo")
    parser.add_argument("--cmd_vx", type=float, default=1.5)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--ramp_up", type=float, default=1.5,
                        help="Ramp-up duration in seconds (fixed for vis)")
    parser.add_argument("--ramp_down", type=float, default=3.0,
                        help="Ramp-down duration in seconds (fixed for vis)")
    parser.add_argument("--stand_duration", type=float, default=1.0,
                        help="Standing duration before/after ramp")
    parser.add_argument("--crawl_duration", type=float, default=1.0,
                        help="Near-zero speed phase before final stand")
    parser.add_argument("--crawl_ratio", type=float, default=0.01,
                        help="Crawl speed = target * ratio")
    parser.add_argument("--episode_length", type=float, default=16.5)
    parser.add_argument("--no_ramp", action="store_true",
                        help="Disable ramp (constant speed) for comparison")
    parser.add_argument("--random", action="store_true",
                        help="Use random duration ranges (like training) instead of fixed values")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--xml_file", default=os.path.join(
        REPO_ROOT, "assets/g1/g1_mocap_with_wrist_roll.xml"))
    parser.add_argument("--cmg_model", default=os.path.join(
        REPO_ROOT, "cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"))
    parser.add_argument("--cmg_data", default=os.path.join(
        REPO_ROOT, "cmg_workspace/dataloader/cmg_training_data.pt"))
    parser.add_argument("--urdf", default=os.path.join(
        REPO_ROOT, "assets/g1/g1_custom_collision_with_fixed_hand.urdf"))
    args = parser.parse_args()

    device = args.device
    t_ramp = args.ramp_up
    t_ep = args.episode_length

    # Duration ranges: --random uses training ranges, otherwise fixed values
    if args.random:
        up_range = (0.5, 2.5)
        down_range = (1.0, 4.0)
        crawl_range = (0.5, 2.0)
    else:
        up_range = (t_ramp, t_ramp)
        down_range = (args.ramp_down, args.ramp_down)
        crawl_range = (args.crawl_duration, args.crawl_duration)

    # ── CMG motion lib ──────────────────────────────────────────────────────
    ramp_on = not args.no_ramp
    cmg = CMGMotionLib(
        cmg_model_path=args.cmg_model,
        cmg_data_path=args.cmg_data,
        urdf_path=args.urdf,
        device=device,
        num_envs=1,
        episode_length_s=t_ep,
        dt=CMG_DT,
        vx_range=(args.cmd_vx, args.cmd_vx),
        vy_range=(args.cmd_vy, args.cmd_vy),
        yaw_range=(args.cmd_yaw, args.cmd_yaw),
        ramp_enabled=ramp_on,
        ramp_up_range=up_range,
        ramp_down_range=down_range,
        ramp_stand_duration=args.stand_duration,
        ramp_crawl_range=crawl_range,
        ramp_crawl_ratio=getattr(args, 'crawl_ratio', 0.01),
        ramp_probability=1.0,   # always ramp for visualization
        ramp_floor_ratio=0.1,   # near-stop (CMG stays in valid range)
    )

    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    init_cmd = torch.tensor([[args.cmd_vx, args.cmd_vy, args.cmd_yaw]], device=device)
    cmg.reset(env_ids, commands=init_cmd)
    cmg._mirror_flags[:] = False  # no mirroring for visualization

    # ── MuJoCo ──────────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(args.xml_file)
    model.opt.timestep = SIM_DT
    # Disable gravity so the kinematic playback doesn't fall
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)

    # Initial pose
    data.qpos[0:3] = [0, 0, 1.0]
    data.qpos[3:7] = [1, 0, 0, 0]  # wxyz
    dof_25 = np.zeros(25, dtype=np.float32)
    dof_25[BODY_DOF_IDS_25] = DEFAULT_DOF_POS
    data.qpos[7:] = dof_25
    mujoco.mj_forward(model, data)

    # State
    paused = False
    step_counter = 0
    episode_time = 0.0

    def reset_episode():
        nonlocal step_counter, episode_time
        step_counter = 0
        episode_time = 0.0
        cmd = torch.tensor([[cmg._target_commands[0, 0].item(),
                             cmg._target_commands[0, 1].item(),
                             cmg._target_commands[0, 2].item()]], device=device)
        cmg.reset(env_ids, commands=cmd)
        cmg._mirror_flags[:] = False
        data.qpos[0:3] = [0, 0, 1.0]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:] = dof_25
        mujoco.mj_forward(model, data)
        ru = cmg._env_ramp_up[0].item()
        rd = cmg._env_ramp_down[0].item()
        cr = cmg._env_crawl[0].item()
        steady = t_ep - 2 * args.stand_duration - ru - rd - cr
        print(f"[Reset] vx={cmd[0,0]:.2f} vy={cmd[0,1]:.2f} yaw={cmd[0,2]:.2f}  "
              f"up={ru:.1f}s down={rd:.1f}s crawl={cr:.1f}s steady={steady:.1f}s")

    # GLFW key codes
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    KEY_A, KEY_D, KEY_R, KEY_SPACE = 65, 68, 82, 32

    def key_callback(keycode):
        nonlocal paused
        target = cmg._target_commands[0]
        changed = True
        if keycode == KEY_UP:
            target[0] = (target[0] + 0.2).clamp(0.0, 4.0)
        elif keycode == KEY_DOWN:
            target[0] = (target[0] - 0.2).clamp(0.0, 4.0)
        elif keycode == KEY_LEFT:
            target[2] = (target[2] + 0.2).clamp(-1.5, 1.5)
        elif keycode == KEY_RIGHT:
            target[2] = (target[2] - 0.2).clamp(-1.5, 1.5)
        elif keycode == KEY_A:
            target[1] = (target[1] + 0.1).clamp(-0.8, 0.8)
        elif keycode == KEY_D:
            target[1] = (target[1] - 0.1).clamp(-0.8, 0.8)
        elif keycode == KEY_R:
            reset_episode()
            return
        elif keycode == KEY_SPACE:
            paused = not paused
            print(f"[{'Paused' if paused else 'Running'}]")
            return
        else:
            changed = False
        if changed:
            # Update commands and reset episode with new target
            reset_episode()

    viewer = mjv.launch_passive(
        model, data,
        key_callback=key_callback,
        show_left_ui=False, show_right_ui=False,
    )
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15.0

    mode_str = "RAMP" if ramp_on else "CONSTANT"
    print(f"\n[Vis] Mode: {mode_str}  vx={args.cmd_vx:.2f}  "
          f"ramp_up={t_ramp:.1f}s  episode={t_ep:.1f}s")
    print("[Vis] Keyboard: ↑/↓=vx  ←/→=yaw  A/D=vy  R=reset  SPACE=pause")
    print("[Vis] This is kinematic playback (no physics, no policy)")
    print()

    # ── Main loop ───────────────────────────────────────────────────────────
    try:
        while viewer.is_running():
            if paused:
                viewer.sync()
                import time; time.sleep(0.02)
                continue

            if step_counter % PLAYBACK_DECIMATION == 0:
                # Step CMG
                cmg.step(env_ids)
                episode_time = cmg._motion_times[0].item()

                # Get current frame
                motion_ids = torch.zeros(1, dtype=torch.long, device=device)
                motion_times = cmg._motion_times[:1]
                root_pos, root_rot_xyzw, _, _, dof_pos_23, _, _ = \
                    cmg.calc_motion_frame(motion_ids, motion_times)

                # Convert to numpy
                rp = root_pos[0].cpu().numpy()
                # xyzw → wxyz for MuJoCo
                rr = root_rot_xyzw[0].cpu().numpy()
                quat_wxyz = np.array([rr[3], rr[0], rr[1], rr[2]])
                dp = dof_pos_23[0].cpu().numpy()

                # Set MuJoCo qpos
                data.qpos[0:3] = rp
                data.qpos[3:7] = quat_wxyz
                dof_full = np.zeros(25, dtype=np.float32)
                dof_full[BODY_DOF_IDS_25] = dp
                data.qpos[7:] = dof_full
                mujoco.mj_forward(model, data)

                # Print status periodically
                cmd_now = cmg._commands[0].cpu().numpy()
                cmd_target = cmg._target_commands[0].cpu().numpy()
                if step_counter % (PLAYBACK_DECIMATION * 25) == 0:  # every 0.5s
                    print(f"  t={episode_time:5.2f}s  "
                          f"cmd=[{cmd_now[0]:5.2f}, {cmd_now[1]:5.2f}, {cmd_now[2]:5.2f}]  "
                          f"target=[{cmd_target[0]:5.2f}, {cmd_target[1]:5.2f}, {cmd_target[2]:5.2f}]")

                # Auto-reset at episode end
                if episode_time >= t_ep - CMG_DT:
                    print(f"[Episode end at t={episode_time:.2f}s, resetting]")
                    reset_episode()

                # Camera follow
                pelvis_pos = data.qpos[0:3]
                viewer.cam.lookat[:] = pelvis_pos
                viewer.sync()

            step_counter += 1

            # Real-time pacing
            import time
            time.sleep(SIM_DT)

    except KeyboardInterrupt:
        print("\n[Vis] Interrupted.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
