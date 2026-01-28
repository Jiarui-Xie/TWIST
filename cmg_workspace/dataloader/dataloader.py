import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


class CMGDataset(Dataset):
    """CMG训练数据集"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        data = torch.load(data_path, weights_only=False)
        self.samples = data["samples"]
        self.stats = data["stats"]
        self.normalize = normalize
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        motion = torch.from_numpy(sample["motion"])  # [seq_len+1, 58]
        command = torch.from_numpy(sample["command"])  # [seq_len, 3]
        
        if self.normalize:
            motion = (motion - self.stats["motion_mean"]) / self.stats["motion_std"]
            cmd_min = self.stats["command_min"]
            cmd_max = self.stats["command_max"]
            command = (command - cmd_min) / (cmd_max - cmd_min) * 2 - 1  # [-1, 1]
        
        return {
            "motion": motion,
            "command": command,
        }


def get_dataloader(data_path: str, batch_size: int=256, num_workers: int=4, shuffle: bool = True):
    """创建DataLoader"""
    dataset = CMGDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    ), dataset.stats


def world_to_local_velocity(root_rot_wxyz, vel_world):
    """将世界坐标系速度转换到机器人坐标系"""
    T = vel_world.shape[0]
    vel_local = np.zeros_like(vel_world)
    
    for t in range(T):
        q = root_rot_wxyz[t]  # [w, x, y, z]
        rot = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy 要 xyzw
        vel_local[t] = rot.inv().apply(vel_world[t])
    
    return vel_local


def is_locomotion(root_lin_vel_local, root_ang_vel_local, 
                  min_speed: float, max_speed: float, 
                  max_lateral: float, max_yaw_rate: float,
                  percentile: int):
    """
    用百分位数判断是否是正常的行走/跑步动作
    允许少量帧有异常值
    """ 
    vx = root_lin_vel_local[:, 0]
    vy = root_lin_vel_local[:, 1]
    yaw = root_ang_vel_local[:, 2]
    
    speed = np.sqrt(vx**2 + vy**2)
    
    # 百分位数检查
    if np.percentile(speed, 100 - percentile) > max_speed:
        return False, "speed"
    if np.percentile(speed, percentile) < min_speed:
        return False, "speed"
    
    if np.percentile(np.abs(vy), 100 - percentile) > max_lateral:
        return False, "lateral"
    
    if np.percentile(np.abs(yaw), 100 - percentile) > max_yaw_rate:
        return False, "yaw"
    
    if np.percentile(vx, percentile) < 0:
        return False, "backward"
    
    return True, "ok"

def mirror_motion(motion, command):
    """
    镜像动作
    - 左右交换: 左腿(0-5) <-> 右腿(6-11), 左臂(15-21) <-> 右臂(22-28)
    - roll/yaw 取反: 左右对称的关节
    """
    motion_mirror = motion.copy()
    command_mirror = command.copy()
    
    pos = motion[:, :29].copy()
    vel = motion[:, 29:].copy()
    
    pos_m = pos.copy()
    vel_m = vel.copy()
    
    # === 1. 交换左右 ===
    # 左腿 <-> 右腿
    pos_m[:, 0:6] = pos[:, 6:12]
    pos_m[:, 6:12] = pos[:, 0:6]
    vel_m[:, 0:6] = vel[:, 6:12]
    vel_m[:, 6:12] = vel[:, 0:6]
    
    # 左臂 <-> 右臂
    pos_m[:, 15:22] = pos[:, 22:29]
    pos_m[:, 22:29] = pos[:, 15:22]
    vel_m[:, 15:22] = vel[:, 22:29]
    vel_m[:, 22:29] = vel[:, 15:22]
    
    # === 2. roll/yaw 关节取反 ===
    # 这些关节在镜像后需要取反符号
    # roll: 1,5,7,11,13,16,19,23,26
    # yaw:  2,8,12,17,21,24,28
    negate_indices = [
        1, 2, 5,      # 左腿 roll, yaw, ankle_roll
        7, 8, 11,     # 右腿 roll, yaw, ankle_roll
        12, 13,       # 腰 yaw, roll
        16, 17, 19, 21,  # 左臂 roll, yaw, wrist_roll, wrist_yaw
        23, 24, 26, 28,  # 右臂 roll, yaw, wrist_roll, wrist_yaw
    ]
    pos_m[:, negate_indices] *= -1
    vel_m[:, negate_indices] *= -1
    
    motion_mirror[:, :29] = pos_m
    motion_mirror[:, 29:] = vel_m
    
    # === 3. Command 取反 ===
    command_mirror[:, 1] *= -1  # vy
    command_mirror[:, 2] *= -1  # yaw_rate
    
    return motion_mirror.astype(np.float32), command_mirror.astype(np.float32)

def preprocess_amass_for_cmg(
    data_root: str,
    save_path: str,
    seq_len: int = 20,
    root_body_idx: int = 0,
    filter_locomotion: bool = True,
    # locomotion 过滤参数
    min_speed: float = 0.3,
    max_speed: float = 4.0,
    max_lateral: float = 1.5,
    max_yaw_rate: float = 3.0,
    percentile: int = 5,
):
    """把AMASS G1数据转成CMG训练格式 (使用local frame速度)"""
    all_samples = []
    skipped_too_short = 0
    skipped_bad_format = 0
    skipped_reasons = {"speed": 0, "lateral": 0, "yaw": 0, "backward": 0}
    
    
    npz_files = []
    for dirpath, _, filenames in os.walk(data_root):
        for filename in filenames:
            if filename.endswith(".npz"):
                npz_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(npz_files)} files")
    
    for npz_path in tqdm(npz_files, desc="Processing"):
        # 跳过非locomotion数据集
            
        try:
            data = np.load(npz_path)
            
            # 检查必要字段
            required_keys = ['dof_positions', 'dof_velocities', 'body_rotations', 
                           'body_linear_velocities', 'body_angular_velocities']
            if not all(k in data.files for k in required_keys):
                skipped_bad_format += 1
                continue
            
            dof_pos = data["dof_positions"]
            dof_vel = data["dof_velocities"]
            root_rot = data["body_rotations"][:, root_body_idx, :]
            body_lin_vel_world = data["body_linear_velocities"][:, root_body_idx, :]
            body_ang_vel_world = data["body_angular_velocities"][:, root_body_idx, :]
            
            T = dof_pos.shape[0]
            
            if T < seq_len + 1:
                skipped_too_short += 1
                continue
            
            # 转换到 local frame
            root_lin_vel_local = world_to_local_velocity(root_rot, body_lin_vel_world)
            root_ang_vel_local = world_to_local_velocity(root_rot, body_ang_vel_world)
            
            # 过滤非locomotion动作
            if filter_locomotion:
                is_loco, reason = is_locomotion(
                    root_lin_vel_local, root_ang_vel_local,
                    min_speed, max_speed, max_lateral, max_yaw_rate, percentile
                )
                if not is_loco:
                    skipped_reasons[reason] += 1
                    continue
            
            # motion state: [pos, vel]
            dof_pos = np.clip(dof_pos, -2.5, 2.5)
            dof_vel = np.clip(dof_vel, -15, 15) 
            motion = np.concatenate([dof_pos, dof_vel], axis=-1)

            # slight gaussian noise 
            motion += np.random.normal(loc=0.0, scale=0.02, size=motion.shape)
            
            # command: [vx, vy, yaw_rate] in local frame (clipped)
            command = np.stack([
                root_lin_vel_local[:, 0],   # vx
                root_lin_vel_local[:, 1],   # vy
                root_ang_vel_local[:, 2],   # yaw
            ], axis=-1)
            
            # 滑窗切片，50%重叠
            for start in range(0, T - seq_len, seq_len // 2):
                end = start + seq_len
                
                segment_speed = np.sqrt(
                    root_lin_vel_local[start:end, 0]**2 + 
                    root_lin_vel_local[start:end, 1]**2
                ).mean()

                
                motion_seg = motion[start:end+1].astype(np.float32)  # [seq_len+1, 58]
                command_seg = command[start:end].astype(np.float32)  # [seq_len, 3]

                # Up-sampling: 高速动作重复添加
                if segment_speed > 2.0:
                    repeat = 8 
                elif segment_speed > 1.0 :
                    repeat = 4
                else:  # 走路
                    repeat = 1

                for _ in range(repeat):
                    all_samples.append({
                        "motion": motion_seg,
                        "command": command_seg,
                    })

                    if repeat != 1:
                        motion_mirrored, command_mirrored = mirror_motion(motion_seg, command_seg)
                        all_samples.append({
                            "motion": motion_mirrored,
                            "command": command_mirrored,
                        })

                
                
        except Exception as e:
            # 静默跳过格式有问题的文件
            skipped_bad_format += 1
            continue
    
    print(f"\n=== Statistics ===")
    filtered_count = (17714 - skipped_too_short - skipped_bad_format 
                    - skipped_reasons['speed'] - skipped_reasons['lateral']
                    - skipped_reasons['yaw'] - skipped_reasons['backward'])
    print(f"  (Before Augmentation, filtered) Number of Samples: {filtered_count}")
    print(f"  (After Augmentation) Number of Samples: {len(all_samples)}")
    print(f"  Skipped (too short): {skipped_too_short}")
    print(f"  Skipped (bad format): {skipped_bad_format}")
    if filter_locomotion:
        print(f"  Skipped (speed): {skipped_reasons['speed']}")
        print(f"  Skipped (lateral): {skipped_reasons['lateral']}")
        print(f"  Skipped (yaw): {skipped_reasons['yaw']}")
        print(f"  Skipped (backward): {skipped_reasons['backward']}")
    
    if len(all_samples) == 0:
        print("ERROR: No samples! Try relaxing filters.")
        return None, None
    
    all_motion = np.concatenate([s["motion"] for s in all_samples], axis=0)
    all_command = np.concatenate([s["command"] for s in all_samples], axis=0)
    
    stats = {
        "motion_mean": all_motion.mean(axis=0).astype(np.float32),
        "motion_std": (all_motion.std(axis=0) + 1e-8).astype(np.float32),
        "command_min": all_command.min(axis=0).astype(np.float32),
        "command_max": (all_command.max(axis=0) + 1e-8).astype(np.float32),
        "num_joints": 29,
        "motion_dim": 58,
        "command_dim": 3,
    }
    
    print(f"\n=== Data Statistics (LOCAL FRAME) ===")
    print(f"  Filtered # Samples: {len(all_samples)}")    
    print(f"  Motion dim: {stats['motion_dim']}")
    print(f"  Command range: vx=[{all_command[:,0].min():.2f}, {all_command[:,0].max():.2f}] m/s")
    print(f"                 vy=[{all_command[:,1].min():.2f}, {all_command[:,1].max():.2f}] m/s")
    print(f"                 yaw=[{all_command[:,2].min():.2f}, {all_command[:,2].max():.2f}] rad/s")
    
    torch.save({
        "samples": all_samples,
        "stats": stats,
    }, save_path)
    
    print(f"\nSaved to {save_path}")
    return all_samples, stats


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = "/root/gpufree-data/AMASS_Retargeted_for_G1/g1"
    save_path = os.path.join(current_dir, "cmg_training_data.pt")

    
    preprocess_amass_for_cmg(
        data_root=data_root,
        save_path=save_path,
        seq_len=30,
        root_body_idx=0,
        filter_locomotion=True,
        min_speed=0.3,
        max_speed=4.0,
        max_lateral=1.5,
        max_yaw_rate=2.0,
        percentile=5,
    )


    # print(data["body_names"]) 

    # ['pelvis' 'left_hip_pitch_link' 'left_hip_roll_link' 'left_hip_yaw_link'
    #  'left_knee_link' 'left_ankle_pitch_link' 'left_ankle_roll_link'
    #  'right_hip_pitch_link' 'right_hip_roll_link' 'right_hip_yaw_link'
    #  'right_knee_link' 'right_ankle_pitch_link' 'right_ankle_roll_link'
    #  'waist_yaw_link' 'waist_roll_link' 'torso_link'
    #  'left_shoulder_pitch_link' 'left_shoulder_roll_link'
    #  'left_shoulder_yaw_link' 'left_elbow_link' 'left_wrist_roll_link'
    #  'left_wrist_pitch_link' 'left_wrist_yaw_link' 'right_shoulder_pitch_link'
    #  'right_shoulder_roll_link' 'right_shoulder_yaw_link' 'right_elbow_link'
    #  'right_wrist_roll_link' 'right_wrist_pitch_link' 'right_wrist_yaw_link']


    # fps: shape=(1,), dtype=float32
    # dof_names: shape=(29,), dtype=<U26
    # body_names: shape=(30,), dtype=<U25
    # dof_positions: shape=(85, 29), dtype=float32
    # dof_velocities: shape=(85, 29), dtype=float32
    # body_positions: shape=(85, 30, 3), dtype=float32
    # body_rotations: shape=(85, 30, 4), dtype=float32
    # body_linear_velocities: shape=(85, 30, 3), dtype=float32
    # body_angular_velocities: shape=(85, 30, 3), dtype=float32


    # fps: shape=(1,), dtype=float32
    # dof_names: shape=(29,), dtype=<U26
    # body_names: shape=(30,), dtype=<U25
    # dof_positions: shape=(653, 29), dtype=float32
    # dof_velocities: shape=(653, 29), dtype=float32
    # body_positions: shape=(653, 30, 3), dtype=float32
    # body_rotations: shape=(653, 30, 4), dtype=float32
    # body_linear_velocities: shape=(653, 30, 3), dtype=float32
    # body_angular_velocities: shape=(653, 30, 3), dtype=float32