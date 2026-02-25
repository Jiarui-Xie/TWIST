#!/usr/bin/env python3
"""
诊断脚本: 查看不同速度命令下 CMG 的原始输出

分析内容:
1. DOF position/velocity 的统计量 (mean, std, range)
2. 不同 vx 命令下输出的差异
3. 步频分析 (通过 hip pitch 周期检测)
4. 步幅分析 (通过 FK 计算脚踝位移)
5. 训练数据中 command 与 DOF velocity 的相关性

Usage:
    conda activate Main
    python tools/test_cmg_output.py
    python tools/test_cmg_output.py --vx_list 0.5 1.0 1.5 2.0 2.5 3.0
"""

import os
import sys
import argparse
import torch
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)
sys.path.insert(0, os.path.join(_project_dir, 'cmg_workspace'))

from module.cmg import CMG
from pose.utils.cmg_motion_lib import CMG_TO_G1_INDICES
from pose.utils.forward_kinematics import ForwardKinematics


def load_cmg(device='cpu'):
    """加载 CMG 模型和训练数据"""
    model_path = os.path.join(_project_dir, 'cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt')
    data_path = os.path.join(_project_dir, 'cmg_workspace/dataloader/cmg_training_data.pt')

    data = torch.load(data_path, weights_only=False, map_location=device)
    stats = data["stats"]
    samples = data["samples"]

    model = CMG(
        motion_dim=stats["motion_dim"],
        command_dim=stats["command_dim"],
        hidden_dim=512,
        num_experts=4,
        num_layers=3,
    )
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, stats, samples


def generate_trajectory(model, stats, samples, vx, vy, yaw, n_frames=200, n_trials=5, device='cpu'):
    """生成多条轨迹, 返回 raw motion (n_trials, n_frames+1, 58)"""
    motion_mean = torch.from_numpy(stats["motion_mean"]).to(device)
    motion_std = torch.from_numpy(stats["motion_std"]).to(device)
    cmd_min = torch.from_numpy(stats["command_min"]).to(device)
    cmd_max = torch.from_numpy(stats["command_max"]).to(device)

    cmd = torch.tensor([[vx, vy, yaw]], dtype=torch.float32, device=device)
    cmd_norm = (cmd - cmd_min) / (cmd_max - cmd_min) * 2 - 1  # (1, 3)
    cmd_norm_batch = cmd_norm.expand(n_trials, -1)  # (n_trials, 3)

    # 随机选初始帧
    indices = np.random.randint(0, len(samples), size=n_trials)
    init_motions = np.stack([samples[idx]["motion"][0] for idx in indices], axis=0)
    current = torch.from_numpy(init_motions).float().to(device)
    current = (current - motion_mean) / motion_std

    all_frames = [current.clone()]
    with torch.no_grad():
        for _ in range(n_frames):
            current = model(current, cmd_norm_batch)
            all_frames.append(current.clone())

    traj = torch.stack(all_frames, dim=1)  # (n_trials, n_frames+1, 58)
    traj = traj * motion_std + motion_mean  # denormalize
    return traj.cpu().numpy()


def detect_step_frequency(traj_29pos, dt=0.02):
    """通过左 hip pitch (joint 0) 检测步频

    Returns: freq_hz (mean step frequency across trials)
    """
    # Joint 0 = left hip pitch
    hip_pitch = traj_29pos[:, :, 0]  # (n_trials, n_frames)

    freqs = []
    for trial in range(hip_pitch.shape[0]):
        signal = hip_pitch[trial]
        # 去均值
        signal = signal - signal.mean()
        # 找零交叉 (上升沿)
        crossings = []
        for i in range(1, len(signal)):
            if signal[i-1] < 0 and signal[i] >= 0:
                crossings.append(i)
        if len(crossings) >= 2:
            periods = np.diff(crossings) * dt
            freq = 1.0 / np.mean(periods)
            freqs.append(freq)

    return np.mean(freqs) if freqs else 0.0


def compute_ankle_stride(traj_23pos, fk, dt=0.02, device='cpu'):
    """通过 FK 计算脚踝在 pelvis frame 下的前后摆动幅度 (stride proxy)

    Returns: mean stride amplitude in meters (x-direction swing of left ankle)
    """
    dof_pos = torch.from_numpy(traj_23pos).float().to(device)
    n_trials, n_frames, _ = dof_pos.shape

    # Flatten for batch FK
    dof_flat = dof_pos.reshape(-1, 23)
    root_pos = torch.zeros(dof_flat.shape[0], 3, device=device)
    root_rot = torch.zeros(dof_flat.shape[0], 4, device=device)
    root_rot[:, 0] = 1.0

    body_pos = fk.compute_body_positions(root_pos, root_rot, dof_flat)
    # body order: [left_hand, right_hand, left_ankle, right_ankle, left_knee, right_knee, ...]
    # left_ankle = index 2
    left_ankle_x = body_pos[:, 2, 0].cpu().numpy().reshape(n_trials, n_frames)

    strides = []
    for trial in range(n_trials):
        x = left_ankle_x[trial]
        # 去均值后取 peak-to-peak
        x_centered = x - x.mean()
        stride = x_centered.max() - x_centered.min()
        strides.append(stride)

    return np.mean(strides)


def analyze_training_data(samples, max_samples=5000):
    """分析训练数据中 command 与 DOF velocity 的关系"""
    print("\n" + "="*70)
    print("训练数据分析: command vs DOF velocity 相关性")
    print("="*70)

    all_cmds = []
    all_vel_stats = []

    n = min(len(samples), max_samples)
    for i in range(n):
        s = samples[i]
        cmd = s["command"]       # (seq_len, 3)
        motion = s["motion"]     # (seq_len+1, 58)
        vel = motion[:cmd.shape[0], 29:]  # (seq_len, 29) DOF velocities

        # 每个 sample 取均值
        all_cmds.append(cmd.mean(axis=0))
        all_vel_stats.append(vel.mean(axis=0))

    cmds = np.stack(all_cmds)       # (n, 3)
    vels = np.stack(all_vel_stats)  # (n, 29)

    print(f"\n样本数: {n}")
    print(f"Command 统计:")
    print(f"  vx:  mean={cmds[:,0].mean():.3f}, std={cmds[:,0].std():.3f}, range=[{cmds[:,0].min():.2f}, {cmds[:,0].max():.2f}]")
    print(f"  vy:  mean={cmds[:,1].mean():.3f}, std={cmds[:,1].std():.3f}, range=[{cmds[:,1].min():.2f}, {cmds[:,1].max():.2f}]")
    print(f"  yaw: mean={cmds[:,2].mean():.3f}, std={cmds[:,2].std():.3f}, range=[{cmds[:,2].min():.2f}, {cmds[:,2].max():.2f}]")

    # 计算每个 DOF velocity 与 vx 的相关系数
    joint_names_29 = [
        "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
        "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "L_s_pitch", "L_s_roll", "L_s_yaw", "L_elbow",
        "L_wrist_r", "L_wrist_p", "L_wrist_y",
        "R_s_pitch", "R_s_roll", "R_s_yaw", "R_elbow",
        "R_wrist_r", "R_wrist_p", "R_wrist_y",
    ]

    print(f"\nDOF velocity 与 vx 的 Pearson 相关系数 (|r| > 0.1 才显示):")
    print(f"  {'Joint':<18s} {'corr(vx)':>8s} {'corr(vy)':>8s} {'corr(yaw)':>8s}  {'mean_vel':>8s}  {'std_vel':>8s}")
    print("  " + "-" * 70)

    for j in range(29):
        r_vx = np.corrcoef(cmds[:, 0], vels[:, j])[0, 1]
        r_vy = np.corrcoef(cmds[:, 1], vels[:, j])[0, 1]
        r_yaw = np.corrcoef(cmds[:, 2], vels[:, j])[0, 1]

        if abs(r_vx) > 0.1 or abs(r_vy) > 0.1 or abs(r_yaw) > 0.1:
            print(f"  {joint_names_29[j]:<18s} {r_vx:>8.3f} {r_vy:>8.3f} {r_yaw:>8.3f}  {vels[:,j].mean():>8.3f}  {vels[:,j].std():>8.3f}")

    # 按速度分桶, 看不同速度下 DOF velocity 的变化
    print(f"\n按 vx 分桶, 查看 DOF velocity magnitude 变化:")
    print(f"  {'vx_range':<14s} {'n':>5s} {'mean|vel|':>10s} {'std|vel|':>10s} {'hip_pitch_vel':>14s}")
    print("  " + "-" * 60)

    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0)]:
        mask = (cmds[:, 0] >= lo) & (cmds[:, 0] < hi)
        if mask.sum() == 0:
            continue
        subset = vels[mask]
        vel_mag = np.abs(subset).mean(axis=1)
        hip_pitch = subset[:, 0]  # L_hip_pitch velocity
        print(f"  [{lo:.1f}, {hi:.1f})     {mask.sum():>5d} {vel_mag.mean():>10.3f} {vel_mag.std():>10.3f} {hip_pitch.mean():>14.3f}")


def main():
    parser = argparse.ArgumentParser(description="CMG 输出诊断")
    parser.add_argument("--vx_list", type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--n_frames", type=int, default=200, help="每条轨迹帧数 (50Hz, 200=4s)")
    parser.add_argument("--n_trials", type=int, default=10, help="每个速度的试验次数")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--skip_training_data", action='store_true', help="跳过训练数据分析")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    model, stats, samples = load_cmg(device)
    print(f"训练样本数: {len(samples)}")
    print(f"Motion dim: {stats['motion_dim']}, Command dim: {stats['command_dim']}")
    print(f"Command range: vx=[{stats['command_min'][0]:.2f}, {stats['command_max'][0]:.2f}], "
          f"vy=[{stats['command_min'][1]:.2f}, {stats['command_max'][1]:.2f}], "
          f"yaw=[{stats['command_min'][2]:.2f}, {stats['command_max'][2]:.2f}]")

    # FK for ankle position
    urdf_path = os.path.join(_project_dir, 'assets/g1/g1_custom_collision_with_fixed_hand.urdf')
    fk = ForwardKinematics(urdf_path, device)

    # ==================== 1. 不同速度下 CMG 输出对比 ====================
    print("\n" + "="*70)
    print(f"CMG 输出分析: vx sweep, vy={args.vy}, yaw={args.yaw}")
    print(f"每个速度: {args.n_trials} trials x {args.n_frames} frames ({args.n_frames*0.02:.1f}s)")
    print("="*70)

    header = (f"  {'cmd_vx':>7s}  {'pos_mean':>8s} {'pos_std':>8s}  "
              f"{'vel_mean':>8s} {'vel_std':>8s}  "
              f"{'|vel|':>8s}  "
              f"{'step_Hz':>7s}  {'stride_m':>8s}")
    print(header)
    print("  " + "-" * 80)

    all_results = {}

    for vx in args.vx_list:
        traj = generate_trajectory(model, stats, samples, vx, args.vy, args.yaw,
                                   n_frames=args.n_frames, n_trials=args.n_trials, device=device)
        # traj: (n_trials, n_frames+1, 58)

        pos_29 = traj[:, :, :29]   # (n_trials, n_frames+1, 29)
        vel_29 = traj[:, :, 29:]   # (n_trials, n_frames+1, 29)

        # 跳过前 20 帧 warmup
        pos_29_steady = pos_29[:, 20:, :]
        vel_29_steady = vel_29[:, 20:, :]

        pos_mean = pos_29_steady.mean()
        pos_std = pos_29_steady.std()
        vel_mean = vel_29_steady.mean()
        vel_std = vel_29_steady.std()
        vel_abs_mean = np.abs(vel_29_steady).mean()

        # 步频
        step_freq = detect_step_frequency(pos_29, dt=0.02)

        # 步幅 (用 23 DOF)
        pos_23 = pos_29[:, :, CMG_TO_G1_INDICES]
        stride = compute_ankle_stride(pos_23, fk, dt=0.02, device=device)

        print(f"  {vx:>7.2f}  {pos_mean:>8.4f} {pos_std:>8.4f}  "
              f"{vel_mean:>8.4f} {vel_std:>8.4f}  "
              f"{vel_abs_mean:>8.4f}  "
              f"{step_freq:>7.2f}  {stride:>8.4f}")

        all_results[vx] = {
            'traj': traj,
            'step_freq': step_freq,
            'stride': stride,
            'vel_abs_mean': vel_abs_mean,
        }

    # ==================== 2. 详细关节对比 ====================
    print("\n" + "="*70)
    print("详细关节对比 (mean |velocity| per joint, 跳过前20帧)")
    print("="*70)

    joint_names_23 = [
        "L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r",
        "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r",
        "w_yaw", "w_roll", "w_pitch",
        "L_sh_p", "L_sh_r", "L_sh_y", "L_elbow",
        "R_sh_p", "R_sh_r", "R_sh_y", "R_elbow",
    ]

    # Header
    vx_strs = [f"vx={v:.1f}" for v in args.vx_list]
    print(f"  {'joint':<10s}  " + "  ".join(f"{s:>8s}" for s in vx_strs))
    print("  " + "-" * (12 + 10 * len(args.vx_list)))

    for j, jname in enumerate(joint_names_23):
        row = f"  {jname:<10s}  "
        for vx in args.vx_list:
            traj = all_results[vx]['traj']
            vel_23 = traj[:, 20:, 29:][:, :, CMG_TO_G1_INDICES]  # (n, frames, 23)
            mean_abs = np.abs(vel_23[:, :, j]).mean()
            row += f"{mean_abs:>8.3f}  "
        print(row)

    # ==================== 3. 关节 position 对比 ====================
    print("\n" + "="*70)
    print("关节 position range (max - min) per joint, 跳过前20帧")
    print("="*70)

    print(f"  {'joint':<10s}  " + "  ".join(f"{s:>8s}" for s in vx_strs))
    print("  " + "-" * (12 + 10 * len(args.vx_list)))

    for j, jname in enumerate(joint_names_23):
        row = f"  {jname:<10s}  "
        for vx in args.vx_list:
            traj = all_results[vx]['traj']
            pos_23 = traj[:, 20:, :29][:, :, CMG_TO_G1_INDICES]  # (n, frames, 23)
            # Mean of per-trial range
            ranges = pos_23[:, :, j].max(axis=1) - pos_23[:, :, j].min(axis=1)
            row += f"{ranges.mean():>8.3f}  "
        print(row)

    # ==================== 4. 脚踝 FK 位置 ====================
    print("\n" + "="*70)
    print("脚踝 FK 位置 (pelvis frame, 跳过前20帧)")
    print("="*70)

    print(f"  {'cmd_vx':>7s}  {'L_ankle_x':>10s} {'L_ankle_z':>10s}  {'R_ankle_x':>10s} {'R_ankle_z':>10s}  {'x_swing':>8s}")
    print("  " + "-" * 65)

    for vx in args.vx_list:
        traj = all_results[vx]['traj']
        pos_23 = traj[:, 20:, :29][:, :, CMG_TO_G1_INDICES]

        dof_flat = torch.from_numpy(pos_23.reshape(-1, 23)).float().to(device)
        root_pos = torch.zeros(dof_flat.shape[0], 3, device=device)
        root_rot = torch.zeros(dof_flat.shape[0], 4, device=device)
        root_rot[:, 0] = 1.0
        body_pos = fk.compute_body_positions(root_pos, root_rot, dof_flat).cpu().numpy()

        n_trials = traj.shape[0]
        n_frames = traj.shape[1] - 20
        body_pos = body_pos.reshape(n_trials, n_frames, -1, 3)

        # left_ankle=2, right_ankle=3
        la_x = body_pos[:, :, 2, 0]  # (n_trials, n_frames)
        la_z = body_pos[:, :, 2, 2]
        ra_x = body_pos[:, :, 3, 0]
        ra_z = body_pos[:, :, 3, 2]

        x_swing = (la_x.max(axis=1) - la_x.min(axis=1)).mean()

        print(f"  {vx:>7.2f}  {la_x.mean():>10.4f} {la_z.mean():>10.4f}  "
              f"{ra_x.mean():>10.4f} {ra_z.mean():>10.4f}  {x_swing:>8.4f}")

    # ==================== 5. 训练数据分析 ====================
    if not args.skip_training_data:
        analyze_training_data(samples)

    print("\n" + "="*70)
    print("完成")
    print("="*70)


if __name__ == "__main__":
    main()
