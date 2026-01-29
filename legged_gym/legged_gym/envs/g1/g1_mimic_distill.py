from isaacgym.torch_utils import *

import torch
import numpy as np
import sys
import os

from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .g1_mimic_distill_config import G1MimicPrivCfg, G1MimicStuCfg
from legged_gym.gym_utils.math import *
from pose.utils import torch_utils
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos

# Import CMG module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../cmg_workspace'))
from module.cmg import CMG

def g1_body_from_38_to_52(body_pos_38: torch.Tensor) -> torch.Tensor:
    """
    将形状 (N, 38, 3) 的关节坐标转换为形状 (N, 52, 3)。
    多出来的手指等关节在输出中将填充为 (0, 0, 0)。

    参数:
    -------
        body_pos_38 : torch.Tensor
            大小为 (N, 38, 3) 的关节坐标，N 为批量大小

    返回:
    -------
        body_pos_52 : torch.Tensor
            大小为 (N, 52, 3) 的关节坐标
    """

    # 构建一个大小为 52 的整型索引张量，用于说明：
    # “52-link 的每个关节，对应到 38-link 中的哪一个下标？”
    # 若对应不到（例如手指关节），则为 -1。
    idx_map_52_list = [-1] * 52
    # 直接在列表中指定 38->52 的映射：
    # 0~29 不变, 30->37, 31->38, 32->39, 33->40, 34->41, 35->42, 36->43, 37->44
    # ---------------------------------------------------------------------
    idx_map_52_list[0]  = 0   # pelvis
    idx_map_52_list[1]  = 1
    idx_map_52_list[2]  = 2
    idx_map_52_list[3]  = 3
    idx_map_52_list[4]  = 4
    idx_map_52_list[5]  = 5
    idx_map_52_list[6]  = 6
    idx_map_52_list[7]  = 7
    idx_map_52_list[8]  = 8
    idx_map_52_list[9]  = 9
    idx_map_52_list[10] = 10
    idx_map_52_list[11] = 11
    idx_map_52_list[12] = 12
    idx_map_52_list[13] = 13
    idx_map_52_list[14] = 14
    idx_map_52_list[15] = 15
    idx_map_52_list[16] = 16
    idx_map_52_list[17] = 17
    idx_map_52_list[18] = 18
    idx_map_52_list[19] = 19
    idx_map_52_list[20] = 20
    idx_map_52_list[21] = 21
    idx_map_52_list[22] = 22
    idx_map_52_list[23] = 23
    idx_map_52_list[24] = 24
    idx_map_52_list[25] = 25
    idx_map_52_list[26] = 26
    idx_map_52_list[27] = 27
    idx_map_52_list[28] = 28
    idx_map_52_list[29] = 29
    idx_map_52_list[37] = 30
    idx_map_52_list[38] = 31
    idx_map_52_list[39] = 32
    idx_map_52_list[40] = 33
    idx_map_52_list[41] = 34
    idx_map_52_list[42] = 35
    idx_map_52_list[43] = 36
    idx_map_52_list[44] = 37
    # 其余下标(手指相关)依旧保持 -1

    # 转换成 PyTorch 张量，放到和输入相同的 device 上
    idx_map_52 = torch.tensor(idx_map_52_list, 
                              dtype=torch.long, 
                              device=body_pos_38.device)

    # 创建输出张量，大小 (N, 52, 3)，默认填零
    N = body_pos_38.shape[0]
    body_pos_52 = torch.zeros((N, 52, 3), 
                              dtype=body_pos_38.dtype, 
                              device=body_pos_38.device)

    # 构建一个布尔掩码，筛选出 idx_map_52 >= 0 的关节
    valid_mask = (idx_map_52 >= 0)

    # 对有效关节通过高级索引直接复制，无需对 N 进行循环
    body_pos_52[:, valid_mask, :] = body_pos_38[:, idx_map_52[valid_mask], :]

    return body_pos_52



class G1MimicDistill(HumanoidMimic):
    def __init__(self, cfg: G1MimicPrivCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.obs_type = cfg.env.obs_type
        self.use_cmg = getattr(cfg.motion, 'use_cmg', False)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        if self.obs_type == 'student':
            self.total_env_steps_counter = 24 * 100000
            self.global_counter = 24 * 100000
            # self.motion_difficulty = torch.ones_like(self.motion_difficulty)

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        
        if self.use_cmg:
            # Use CMG to generate reference trajectory
            self._reset_ref_motion_cmg(env_ids)
            return
        
        # Original motion library based reset
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)
        
        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)
        
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        if body_pos.shape[1] != self._ref_body_pos[env_ids].shape[1]:
            body_pos = g1_body_from_38_to_52(body_pos)
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
    
    def _reset_ref_motion_cmg(self, env_ids):
        """Reset reference motion using CMG generated trajectory"""
        n = len(env_ids)
        
        # Generate new trajectory for these environments
        trajectory = self._generate_cmg_trajectory(env_ids)
        
        # Reset time offset to 0
        self._motion_time_offsets[env_ids] = 0
        
        # Get initial frame (frame 0) for robot reset
        init_dof_pos = trajectory[:, 0, :29]  # First 29 dims are dof_pos
        init_dof_vel = trajectory[:, 0, 29:]  # Last 29 dims are dof_vel
        
        # Set reference states from CMG (simplified - no body_pos tracking)
        self._ref_dof_pos[env_ids] = init_dof_pos[:, :self.num_dof]  # CMG has 29 dof, we use 23
        self._ref_dof_vel[env_ids] = init_dof_vel[:, :self.num_dof]
        
        # Set default root states (standing pose, no tracking)
        self._ref_root_pos[env_ids] = torch.tensor([0.0, 0.0, 0.75], device=self.device, dtype=torch.float).expand(n, -1)
        self._ref_root_rot[env_ids] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float).expand(n, -1)  # Identity quaternion
        self._ref_root_vel[env_ids] = torch.tensor([self.cfg.motion.cmg_velocity[0], 0.0, 0.0], device=self.device, dtype=torch.float).expand(n, -1)
        self._ref_root_ang_vel[env_ids] = torch.tensor([0.0, 0.0, self.cfg.motion.cmg_velocity[2]], device=self.device, dtype=torch.float).expand(n, -1)
    
    def _update_ref_motion(self):
        if self.use_cmg:
            self._update_ref_motion_cmg()
            return
        
        # Original motion library based update
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        if body_pos.shape[1] != self._ref_body_pos.shape[1]:
            body_pos = g1_body_from_38_to_52(body_pos)
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
    
    def _update_ref_motion_cmg(self):
        """Update reference motion from CMG trajectory buffer"""
        # Calculate current frame index in CMG trajectory
        # CMG fps is 50, simulation dt is typically 0.005 (200Hz), decimation is 10
        # So each policy step is 0.05s, matching CMG fps
        frame_idx = (self.episode_length_buf * self.dt * self.cmg_fps).long()
        frame_idx = torch.clamp(frame_idx, max=self.cmg_horizon_frames)
        
        # Index into trajectory buffer
        batch_indices = torch.arange(self.num_envs, device=self.device)
        current_frame = self.cmg_trajectories[batch_indices, frame_idx]  # [num_envs, 58]
        
        # Extract dof_pos and dof_vel (CMG outputs 29 dof, we use first 23)
        self._ref_dof_pos[:] = current_frame[:, :self.num_dof]
        self._ref_dof_vel[:] = current_frame[:, 29:29+self.num_dof]
        
        # Keep root states constant (no root position tracking for CMG)
        # Root velocity is the commanded velocity
        self._ref_root_vel[:, 0] = self.cfg.motion.cmg_velocity[0]  # vx
        self._ref_root_vel[:, 1] = self.cfg.motion.cmg_velocity[1]  # vy
        self._ref_root_ang_vel[:, 2] = self.cfg.motion.cmg_velocity[2]  # yaw rate
    
    def check_termination(self):
        """Override to handle CMG trajectory end"""
        if self.use_cmg:
            # CMG mode: simplified termination check
            contact_force_termination = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1
            )
            self.reset_buf = contact_force_termination
            
            # Height check (absolute, not relative to ref)
            height_cutoff = self.root_states[:, 2] < 0.4  # Too low
            
            roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
            pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
            self.reset_buf |= roll_cut
            self.reset_buf |= pitch_cut
            self.reset_buf |= height_cutoff
            
            # CMG trajectory end
            cmg_motion_end = self.episode_length_buf * self.dt >= self.cmg_horizon
            if self.viewer is None:
                self.reset_buf |= cmg_motion_end
            
            self.time_out_buf = self.episode_length_buf > self.max_episode_length
            if self.viewer is None:
                self.time_out_buf |= cmg_motion_end
            
            self.reset_buf |= self.time_out_buf
            
            vel_too_large = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.
            self.reset_buf |= vel_too_large
        else:
            super().check_termination()
    
    def reset_idx(self, env_ids, motion_ids=None):
        """Override to handle CMG mode episode stats"""
        if len(env_ids) == 0:
            return
        
        # Fill extras with episode stats
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            if self.use_cmg:
                # CMG mode: use fixed horizon length for normalization
                self.extras["episode"]['metric_' + key] = torch.mean(
                    self.episode_sums[key][env_ids] / self.cmg_horizon
                )
                self.extras["episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][env_ids] * self.reward_scales[key] / self.cmg_horizon
                )
            else:
                # Original: use motion library length
                self.extras["episode"]['metric_' + key] = torch.mean(
                    self.episode_sums[key][env_ids] / self._motion_lib.get_motion_length(self._motion_ids[env_ids])
                )
                self.extras["episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][env_ids] * self.reward_scales[key] / self._motion_lib.get_motion_length(self._motion_ids[env_ids])
                )
            self.episode_sums[key][env_ids] = 0.
        
        if self.cfg.motion.motion_curriculum:
            self._update_motion_difficulty(env_ids)
        self._reset_ref_motion(env_ids=env_ids, motion_ids=motion_ids)
        
        vel_factor = 0.8

        # RSI - Reference State Initialization
        self._reset_dofs(env_ids, self._ref_dof_pos, self._ref_dof_vel * vel_factor)
        self._reset_root_states(
            env_ids=env_ids, 
            root_vel=self._ref_root_vel * vel_factor, 
            root_quat=self._ref_root_rot,
            root_pos=self._ref_root_pos, 
            root_ang_vel=self._ref_root_ang_vel * vel_factor
        )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.feet_land_time[env_ids] = 0.
        self.deviate_tracking_frames[env_ids] = 0.
        self.deviate_vel_tracking_frames[env_ids] = 0.
        self._reset_buffers_extra(env_ids)

        self.episode_length_buf[env_ids] = 0
        
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
        if self.cfg.motion.motion_curriculum and not self.use_cmg:
            self.mean_motion_difficulty = torch.mean(self.motion_difficulty)
            
        _, _, y = euler_from_quaternion(self.root_states[:, 3:7])
        self.init_yaw[env_ids] = y[env_ids]
        return
        
    def _update_motion_difficulty(self, env_ids):
        if self.use_cmg:
            return  # No difficulty curriculum for CMG
        if self.obs_type == 'priv':
            super()._update_motion_difficulty(env_ids)
        elif self.obs_type == 'student':
            super()._update_motion_difficulty(env_ids) # currently we use the same strategy for student
        else:
            return

    def _get_body_indices(self):
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                                 requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                  torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                                requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
    
    def _load_motions(self):
        """Override to conditionally load motion library"""
        if self.use_cmg:
            # CMG mode: create a dummy motion lib for compatibility
            # We still need it for some dimension calculations
            print("[G1MimicDistill] CMG mode: Loading motion library for compatibility...")
        # Always load motion library (needed for key_body_ids_motion even in CMG mode)
        super()._load_motions()
    
    def _init_motion_buffers(self):
        """Override to handle CMG mode"""
        super()._init_motion_buffers()
        
        if self.use_cmg:
            # Set max episode length based on CMG horizon
            self.max_episode_length_s = self.cmg_horizon
            self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))
            print(f"[G1MimicDistill] CMG mode: max_episode_length = {self.max_episode_length} steps ({self.max_episode_length_s}s)")
    
    def _init_buffers(self):
        # Initialize CMG before calling super()._init_buffers() since it calls _load_motions()
        if self.use_cmg:
            self._init_cmg()
        super()._init_buffers()
        self.obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_obs_single), device=self.device)
        self.privileged_obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_priv_obs_single), device=self.device)
    
    def _init_cmg(self):
        """Initialize CMG model and related buffers"""
        print("[G1MimicDistill] Initializing CMG motion generator...")
        
        # Load CMG data (stats and initial samples)
        cmg_data = torch.load(self.cfg.motion.cmg_data_path, weights_only=False)
        self.cmg_stats = cmg_data["stats"]
        self.cmg_samples = cmg_data["samples"]
        
        # Initialize CMG model
        self.cmg_model = CMG(
            motion_dim=self.cmg_stats["motion_dim"],
            command_dim=self.cmg_stats["command_dim"],
            hidden_dim=512,
            num_experts=4,
            num_layers=3,
        )
        
        # Load model weights
        checkpoint = torch.load(self.cfg.motion.cmg_model_path, weights_only=False)
        self.cmg_model.load_state_dict(checkpoint["model_state_dict"])
        self.cmg_model = self.cmg_model.to(self.device)
        self.cmg_model.eval()
        
        # Freeze CMG model
        for param in self.cmg_model.parameters():
            param.requires_grad = False
        
        # Precompute normalization tensors
        self.cmg_motion_mean = torch.from_numpy(self.cmg_stats["motion_mean"]).float().to(self.device)
        self.cmg_motion_std = torch.from_numpy(self.cmg_stats["motion_std"]).float().to(self.device)
        self.cmg_cmd_min = torch.from_numpy(self.cmg_stats["command_min"]).float().to(self.device)
        self.cmg_cmd_max = torch.from_numpy(self.cmg_stats["command_max"]).float().to(self.device)
        
        # CMG parameters
        self.cmg_fps = self.cfg.motion.cmg_fps
        self.cmg_horizon = self.cfg.motion.cmg_horizon
        self.cmg_horizon_frames = int(self.cmg_horizon * self.cmg_fps)
        self.cmg_velocity = torch.tensor(self.cfg.motion.cmg_velocity, device=self.device, dtype=torch.float)
        
        # Buffer for CMG trajectories: [num_envs, horizon_frames+1, motion_dim]
        # motion_dim = 58 (29 dof_pos + 29 dof_vel)
        self.cmg_trajectories = torch.zeros(
            (self.num_envs, self.cmg_horizon_frames + 1, self.cmg_stats["motion_dim"]),
            device=self.device, dtype=torch.float
        )
        
        print(f"[G1MimicDistill] CMG initialized: velocity={self.cfg.motion.cmg_velocity}, horizon={self.cmg_horizon}s ({self.cmg_horizon_frames} frames)")
    
    @torch.no_grad()
    def _generate_cmg_trajectory(self, env_ids):
        """Generate reference trajectory using CMG for specified environments"""
        n_envs = len(env_ids)
        
        # Get initial motion from CMG training samples (random sample)
        sample_indices = torch.randint(0, len(self.cmg_samples), (n_envs,))
        init_motions = torch.stack([
            torch.from_numpy(self.cmg_samples[idx.item()]["motion"][0]).float()
            for idx in sample_indices
        ]).to(self.device)  # [n_envs, 58]
        
        # Normalize initial motion
        current = (init_motions - self.cmg_motion_mean) / self.cmg_motion_std
        
        # Prepare fixed velocity command and normalize
        command = self.cmg_velocity.unsqueeze(0).expand(n_envs, -1)  # [n_envs, 3]
        command_norm = (command - self.cmg_cmd_min) / (self.cmg_cmd_max - self.cmg_cmd_min) * 2 - 1
        
        # Generate trajectory autoregressively
        trajectory = [current.clone()]
        for t in range(self.cmg_horizon_frames):
            pred = self.cmg_model(current, command_norm)
            trajectory.append(pred.clone())
            current = pred
        
        # Stack and denormalize
        trajectory = torch.stack(trajectory, dim=1)  # [n_envs, horizon+1, 58]
        trajectory = trajectory * self.cmg_motion_std + self.cmg_motion_mean
        
        # Store in buffer
        self.cmg_trajectories[env_ids] = trajectory
        
        return trajectory

    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        ang_vel_dim = 3
        imu_dim = 2
        
        noise_scale_vec[:, 0:ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, ang_vel_dim:ang_vel_dim+imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, ang_vel_dim+imu_dim:ang_vel_dim+imu_dim+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, ang_vel_dim+imu_dim+self.num_dof:ang_vel_dim+imu_dim+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        
        return noise_scale_vec
            
    def _get_mimic_obs(self):
        if self.use_cmg:
            return self._get_mimic_obs_cmg()
        
        # Original motion library based observation
        num_steps = self._tar_obs_steps.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_obs_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
      
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
     
        # teacher v0
        priv_mimic_obs_buf = torch.cat((
            root_pos[..., 2:3], # 1 dim
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
            whole_key_body_pos, # num_bodies * 3 dims
        ), dim=-1) # shape: (num_envs, num_steps, 7 + num_dof + num_key_bodies * 3)
        
        
        # v6, align mocap
        mimic_obs_buf = torch.cat((
            root_pos[..., 2:3], # 1 dim
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
        ), dim=-1)[:, 0:1] # shape: (num_envs, 1, 7 + num_dof)
        
        
        return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)
    
    def _get_mimic_obs_cmg(self):
        """Get mimic observations from CMG trajectory buffer"""
        num_steps = self._tar_obs_steps.shape[0]
        
        # Calculate current frame and future frame indices
        current_frame = (self.episode_length_buf * self.dt * self.cmg_fps).long()
        
        # tar_obs_steps is [1, 5, 10, ...] - steps into the future
        # Convert to CMG frame indices
        # Policy dt = sim_dt * decimation = 0.002 * 10 = 0.02s, CMG dt = 1/50 = 0.02s, so they match
        future_frames = current_frame.unsqueeze(-1) + self._tar_obs_steps.unsqueeze(0)  # [num_envs, num_steps]
        future_frames = torch.clamp(future_frames, max=self.cmg_horizon_frames)
        
        # Index into CMG trajectory buffer
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(-1).expand(-1, num_steps)
        future_motion = self.cmg_trajectories[batch_indices, future_frames]  # [num_envs, num_steps, 58]
        
        # Extract dof_pos from CMG output (first 29 dims, we use first num_dof)
        dof_pos = future_motion[..., :self.num_dof]  # [num_envs, num_steps, num_dof]
        
        # For CMG, we don't have root pose tracking, so use constant values
        # root_height = 0.75 (standing height)
        root_height = torch.ones((self.num_envs, num_steps, 1), device=self.device) * 0.75
        
        # Orientation: standing upright (roll=0, pitch=0, yaw=0)
        roll = torch.zeros((self.num_envs, num_steps, 1), device=self.device)
        pitch = torch.zeros((self.num_envs, num_steps, 1), device=self.device)
        yaw = torch.zeros((self.num_envs, num_steps, 1), device=self.device)
        
        # Root velocity: from CMG config (1.5, 0, 0)
        root_vel = torch.zeros((self.num_envs, num_steps, 3), device=self.device)
        root_vel[..., 0] = self.cfg.motion.cmg_velocity[0]
        root_vel[..., 1] = self.cfg.motion.cmg_velocity[1]
        
        # Root angular velocity: yaw rate from config
        root_ang_vel_yaw = torch.ones((self.num_envs, num_steps, 1), device=self.device) * self.cfg.motion.cmg_velocity[2]
        
        # For key body positions, use zeros (simplified - no keybody tracking)
        # Original: 9 key bodies * 3 dims = 27 dims
        num_key_bodies = len(self.cfg.motion.key_bodies)
        whole_key_body_pos = torch.zeros((self.num_envs, num_steps, num_key_bodies * 3), device=self.device)
        
        # teacher v0 format
        priv_mimic_obs_buf = torch.cat((
            root_height,  # 1 dim
            roll, pitch, yaw,  # 3 dims
            root_vel,  # 3 dims
            root_ang_vel_yaw,  # 1 dim
            dof_pos,  # num_dof dims (23)
            whole_key_body_pos,  # num_key_bodies * 3 dims (27)
        ), dim=-1)  # shape: (num_envs, num_steps, 8 + 23 + 27 = 58)
        
        # v6 format for student
        mimic_obs_buf = torch.cat((
            root_height,  # 1 dim
            roll, pitch, yaw,  # 3 dims
            root_vel,  # 3 dims
            root_ang_vel_yaw,  # 1 dim
            dof_pos,  # num_dof dims
        ), dim=-1)[:, 0:1]  # Only first step for student
        
        return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)

    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        priv_mimic_obs, mimic_obs = self._get_mimic_obs()
        
        proprio_obs_buf = torch.cat((
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        
        if self.cfg.noise.add_noise and self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec
        else:
            proprio_obs_buf += 0.
        dof_vel_start_dim = 5 + self.dof_pos.shape[1]

        # disable ankle dof
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.
        
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1) # shape: (num_envs, num_key_bodies * 3)
        
        if self.cfg.domain_rand.domain_rand_general:
            priv_info = torch.cat((
                self.base_lin_vel, # 3 dims
                self.root_states[:, 2:3], # 1 dim
                key_body_pos, # num_bodies * 3 dims
                self.contact_forces[:, self.feet_indices, 2] > 5., # 2 dims, foot contact
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1,
            ), dim=-1)
        else:
            priv_info = torch.zeros((self.num_envs, self.cfg.env.n_priv_info), device=self.device)
        
        obs_buf = torch.cat((
            mimic_obs,
            proprio_obs_buf,
        ), dim=-1)
        
        priv_obs_buf = torch.cat((
            priv_mimic_obs,
            proprio_obs_buf,
            priv_info,
        ), dim=-1)
        
        self.privileged_obs_buf = priv_obs_buf
        
        if self.obs_type == 'priv':
            self.obs_buf = priv_obs_buf
        elif self.obs_type == 'student':
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        if self.cfg.env.history_len > 0:
            self.privileged_obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([priv_obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.privileged_obs_history_buf[:, 1:],
                    priv_obs_buf.unsqueeze(1)
                ], dim=1)
            )
            if self.obs_type == 'priv':
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            elif self.obs_type == 'student':
                self.obs_history_buf = torch.where(
                    (self.episode_length_buf <= 1)[:, None, None], 
                    torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                    torch.cat([
                        self.obs_history_buf[:, 1:],
                        obs_buf.unsqueeze(1)
                    ], dim=1)
                )


############################################################################################################
##################################### Extra Reward Functions################################################
############################################################################################################

    def _reward_waist_dof_acc(self):
        waist_dof_idx = [13, 14]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, waist_dof_idx], dim=1)
    
    def _reward_waist_dof_vel(self):
        waist_dof_idx = [13, 14]
        return torch.sum(torch.square(self.dof_vel[:, waist_dof_idx]), dim=1)
    
    def _reward_ankle_dof_acc(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)
    
    def _reward_ankle_dof_vel(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)
    
    def _reward_ankle_action(self):
        return torch.norm(self.action_history_buf[:, -1, [4, 5, 10, 11]], dim=1)
