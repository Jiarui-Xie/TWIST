"""
绘制数据集速度分布和Motion数据分布
包含归一化前后的对比
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data_path = "cmg_training_data.pt"
data = torch.load(data_path, weights_only=False)
samples = data["samples"]
stats = data["stats"]

print(f"(Preprocessed) Loaded {len(samples)} samples")
print(f"Stats keys: {stats.keys()}")

# 提取所有原始数据
all_vx = [] 
all_motions_raw = []

for s in samples:
    motion = s["motion"]  # [seq_len+1, 58] - 原始数据
    cmd = s["command"]    # [seq_len, 3] - 原始数据
    
    all_vx.append(cmd[:, 0])  # vx
    all_motions_raw.append(motion)

all_vx = np.concatenate(all_vx)
all_motions_raw = np.concatenate(all_motions_raw, axis=0)  # [total_frames, 58]

print(f"Total frames: {len(all_vx)}")
print(f"vx range: [{all_vx.min():.2f}, {all_vx.max():.2f}] m/s")
print(f"Motion shape: {all_motions_raw.shape}")

# 归一化 motion
motion_mean = stats["motion_mean"]
motion_std = stats["motion_std"]
motion_std_clipped = np.maximum(motion_std, 0.1)  # 和你的 dataloader 一致

all_motions_norm = (all_motions_raw - motion_mean) / motion_std_clipped

# ========== 图1: 速度分布 (保持原样) ==========
fig, ax = plt.subplots(figsize=(10, 6))

counts, bins, patches = ax.hist(all_vx, bins=50, edgecolor='black', alpha=0.7)
percentages = counts / len(all_vx) * 100
ax.clear()
ax.bar(bins[:-1], percentages, width=np.diff(bins), edgecolor='black', alpha=0.7, align='edge')

ax.set_xlabel('Body Frame Longitudinal Speed vx (m/s)', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Speed Distribution in Training Data', fontsize=14)
ax.grid(True, alpha=0.3)

mean_vx = all_vx.mean()
std_vx = all_vx.std()
ax.axvline(mean_vx, color='r', linestyle='--', label=f'Mean: {mean_vx:.2f} m/s')
ax.axvline(mean_vx + std_vx, color='orange', linestyle=':', label=f'Std: {std_vx:.2f} m/s')
ax.axvline(mean_vx - std_vx, color='orange', linestyle=':')
ax.legend()

plt.tight_layout()
plt.savefig('speed_distribution.png', dpi=150)
print("Saved to speed_distribution.png")

# 打印分段统计
print("\n=== Speed Distribution ===")
bins_custom = [-4, -2, -1, 0, 1, 2, 4]
for i in range(len(bins_custom)-1):
    mask = (all_vx >= bins_custom[i]) & (all_vx < bins_custom[i+1])
    pct = mask.sum() / len(all_vx) * 100
    print(f"  [{bins_custom[i]:+.0f}, {bins_custom[i+1]:+.0f}) m/s: {pct:.1f}%")


# ========== 图2: Motion 原始数据分布 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Motion Data Distribution (Raw)', fontsize=16)

dof_pos_raw = all_motions_raw[:, :29]
dof_vel_raw = all_motions_raw[:, 29:]

# 2.1 关节位置分布
ax1 = axes[0, 0]
ax1.hist(dof_pos_raw.flatten(), bins=100, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Joint Position (rad)', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Joint Position Distribution', fontsize=12)
ax1.axvline(dof_pos_raw.mean(), color='r', linestyle='--', 
            label=f'Mean: {dof_pos_raw.mean():.3f}\nStd: {dof_pos_raw.std():.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2.2 关节速度分布
ax2 = axes[0, 1]
ax2.hist(dof_vel_raw.flatten(), bins=100, edgecolor='black', alpha=0.7, color='coral')
ax2.set_xlabel('Joint Velocity (rad/s)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Joint Velocity Distribution', fontsize=12)
ax2.axvline(dof_vel_raw.mean(), color='r', linestyle='--', 
            label=f'Mean: {dof_vel_raw.mean():.3f}\nStd: {dof_vel_raw.std():.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 2.3 整体 motion 分布
ax3 = axes[0, 2]
ax3.hist(all_motions_raw.flatten(), bins=100, edgecolor='black', alpha=0.7, color='green')
ax3.set_xlabel('Motion Value', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Overall Motion Distribution', fontsize=12)
ax3.axvline(all_motions_raw.mean(), color='r', linestyle='--', 
            label=f'Mean: {all_motions_raw.mean():.3f}\nStd: {all_motions_raw.std():.3f}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 2.4 每个关节的平均位置
ax4 = axes[1, 0]
joint_pos_mean = dof_pos_raw.mean(axis=0)
joint_pos_std = dof_pos_raw.std(axis=0)
x_joints = np.arange(29)
ax4.bar(x_joints, joint_pos_mean, yerr=joint_pos_std, capsize=2, alpha=0.7, color='steelblue')
ax4.set_xlabel('Joint Index', fontsize=11)
ax4.set_ylabel('Mean Position (rad)', fontsize=11)
ax4.set_title('Mean Joint Positions', fontsize=12)
ax4.grid(True, alpha=0.3)

# 2.5 每个关节的平均速度
ax5 = axes[1, 1]
joint_vel_mean = dof_vel_raw.mean(axis=0)
joint_vel_std = dof_vel_raw.std(axis=0)
ax5.bar(x_joints, joint_vel_mean, yerr=joint_vel_std, capsize=2, alpha=0.7, color='coral')
ax5.set_xlabel('Joint Index', fontsize=11)
ax5.set_ylabel('Mean Velocity (rad/s)', fontsize=11)
ax5.set_title('Mean Joint Velocities', fontsize=12)
ax5.grid(True, alpha=0.3)

# 2.6 motion_std 可视化 (显示哪些关节 std 太小)
ax6 = axes[1, 2]
colors = ['red' if s < 0.1 else 'steelblue' for s in motion_std]
ax6.bar(np.arange(58), motion_std, color=colors, alpha=0.7)
ax6.axhline(0.1, color='orange', linestyle='--', label='Clipping threshold (0.1)')
ax6.set_xlabel('Motion Dimension', fontsize=11)
ax6.set_ylabel('Std Value', fontsize=11)
ax6.set_title('Motion Std (Red = clipped)', fontsize=12)
ax6.set_yscale('log')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('motion_distribution_raw.png', dpi=150)
print("Saved to motion_distribution_raw.png")


# ========== 图3: Motion 归一化后分布 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Motion Data Distribution (Normalized)', fontsize=16)

dof_pos_norm = all_motions_norm[:, :29]
dof_vel_norm = all_motions_norm[:, 29:]

# 3.1 关节位置分布 (归一化后)
ax1 = axes[0, 0]
ax1.hist(dof_pos_norm.flatten(), bins=100, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Normalized Joint Position', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Joint Position Distribution (Normalized)', fontsize=12)
ax1.axvline(dof_pos_norm.mean(), color='r', linestyle='--', 
            label=f'Mean: {dof_pos_norm.mean():.3f}\nStd: {dof_pos_norm.std():.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3.2 关节速度分布 (归一化后)
ax2 = axes[0, 1]
ax2.hist(dof_vel_norm.flatten(), bins=100, edgecolor='black', alpha=0.7, color='coral')
ax2.set_xlabel('Normalized Joint Velocity', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Joint Velocity Distribution (Normalized)', fontsize=12)
ax2.axvline(dof_vel_norm.mean(), color='r', linestyle='--', 
            label=f'Mean: {dof_vel_norm.mean():.3f}\nStd: {dof_vel_norm.std():.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3.3 整体 motion 分布 (归一化后)
ax3 = axes[0, 2]
ax3.hist(all_motions_norm.flatten(), bins=100, edgecolor='black', alpha=0.7, color='green')
ax3.set_xlabel('Normalized Motion Value', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Overall Motion Distribution (Normalized)', fontsize=12)
ax3.axvline(all_motions_norm.mean(), color='r', linestyle='--', 
            label=f'Mean: {all_motions_norm.mean():.3f}\nStd: {all_motions_norm.std():.3f}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 3.4 归一化后值域分布
ax4 = axes[1, 0]
ax4.hist(all_motions_norm.flatten(), bins=50, range=(-10, 10), 
         edgecolor='black', alpha=0.7, color='purple')
ax4.set_xlabel('Normalized Value', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Value Range Distribution (Normalized)', fontsize=12)
ax4.axvline(-5, color='orange', linestyle='--', alpha=0.5, label='±5 range')
ax4.axvline(5, color='orange', linestyle='--', alpha=0.5)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 3.5 每个维度的归一化统计
ax5 = axes[1, 1]
dim_means = all_motions_norm.mean(axis=0)
dim_stds = all_motions_norm.std(axis=0)
ax5.scatter(np.arange(58), dim_means, alpha=0.7, label='Mean', s=30)
ax5.scatter(np.arange(58), dim_stds, alpha=0.7, label='Std', s=30)
ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax5.axhline(1, color='gray', linestyle=':', alpha=0.5)
ax5.set_xlabel('Motion Dimension', fontsize=11)
ax5.set_ylabel('Normalized Value', fontsize=11)
ax5.set_title('Per-dimension Statistics (Normalized)', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 3.6 异常值统计
ax6 = axes[1, 2]
outlier_counts = []
thresholds = [5, 10, 15, 20]
for thresh in thresholds:
    count = (np.abs(all_motions_norm) > thresh).sum()
    pct = count / all_motions_norm.size * 100
    outlier_counts.append(pct)

ax6.bar(range(len(thresholds)), outlier_counts, alpha=0.7, color='red')
ax6.set_xticks(range(len(thresholds)))
ax6.set_xticklabels([f'>{t}' for t in thresholds])
ax6.set_xlabel('Threshold', fontsize=11)
ax6.set_ylabel('Percentage of Values (%)', fontsize=11)
ax6.set_title('Outlier Statistics (Normalized)', fontsize=12)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('motion_distribution_normalized.png', dpi=150)
print("Saved to motion_distribution_normalized.png")


# ========== 打印统计信息 ==========
print("\n=== Motion Statistics (Raw) ===")
print(f"Joint Position - Mean: {dof_pos_raw.mean():.4f}, Std: {dof_pos_raw.std():.4f}")
print(f"               - Range: [{dof_pos_raw.min():.2f}, {dof_pos_raw.max():.2f}]")
print(f"Joint Velocity - Mean: {dof_vel_raw.mean():.4f}, Std: {dof_vel_raw.std():.4f}")
print(f"               - Range: [{dof_vel_raw.min():.2f}, {dof_vel_raw.max():.2f}]")

print("\n=== Motion Statistics (Normalized) ===")
print(f"Overall - Mean: {all_motions_norm.mean():.4f}, Std: {all_motions_norm.std():.4f}")
print(f"        - Range: [{all_motions_norm.min():.2f}, {all_motions_norm.max():.2f}]")
print(f"Values > 5:  {(np.abs(all_motions_norm) > 5).sum() / all_motions_norm.size * 100:.2f}%")
print(f"Values > 10: {(np.abs(all_motions_norm) > 10).sum() / all_motions_norm.size * 100:.2f}%")

# 找出 std 太小的维度
print("\n=== Dimensions with Small Std (< 0.1) ===")
small_std_dims = np.where(motion_std < 0.1)[0]
print(f"Found {len(small_std_dims)} dimensions:")
for dim in small_std_dims[:10]:  # 只显示前10个
    print(f"  Dim {dim}: std = {motion_std[dim]:.6f}")

plt.show()