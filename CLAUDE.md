# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TWIST (Teleoperated Whole-Body Imitation System) is a humanoid robot motion imitation system for Unitree G1. It trains RL policies to track reference motions using a two-stage teacher-student learning approach.

## Common Commands

### Environment Setup
```bash
conda activate twist
redis-server --daemonize yes  # Required for deployment
```

### Training
```bash
# Teacher policy (with privileged info)
bash train_teacher.sh <exptid> <cuda_device>
# Example: bash train_teacher.sh my_teacher cuda:0

# Student policy (DAgger: RL+BC from teacher)
bash train_student.sh <student_id> <teacher_id> <cuda_device>
# Example: bash train_student.sh my_student my_teacher cuda:0

# CMG-based training (uses neural motion generator instead of mocap)
bash train_teacher_cmg.sh <speed_mode> <exptid> <cuda_device>
# speed_mode: slow (1m/s) | medium (2m/s) | fast (3m/s)
# Example: bash train_teacher_cmg.sh medium cmg_v1 cuda:0
```

### Export & Deployment
```bash
# Export to JIT model
bash to_jit.sh <student_exptid>

# Sim2sim verification (run in separate terminals)
cd deploy_real
python server_high_level_motion_lib.py --motion_file PATH/TO/MOTION.pkl --vis
python server_low_level_g1_sim.py --policy_path PATH/TO/model.pt

# Sim2real (requires robot connection at 192.168.123.164)
python server_low_level_g1_real.py --policy_path PATH/TO/model.pt --net <interface>
```

### Visualization/Playback
```bash
cd legged_gym/legged_gym/scripts
python play.py --task g1_priv_mimic --exptid <exptid>
```

## Architecture

### Package Structure
```
legged_gym/    # Isaac Gym environment wrapper, configs, training scripts
rsl_rl/        # RL algorithms (PPO, DAgger), runners, policy networks
pose/          # Motion libraries (MotionLib, CMGMotionLib), kinematics
cmg_workspace/ # Conditional Motion Generator neural network
deploy_real/   # Sim2sim and sim2real deployment servers
assets/        # Robot URDFs, pretrained checkpoints
```

### Two-Stage Training Pipeline
1. **Teacher Policy** (`g1_priv_mimic`): Trains with privileged information (full state access) using PPO
2. **Student Policy** (`g1_stu_rl`): Distills teacher knowledge via DAgger, uses only proprioceptive observations

### Motion Reference Sources
- **MotionLib**: Loads mocap data from pickle files (`twist_dataset.yaml`)
- **CMGMotionLib**: Neural motion generator conditioned on velocity commands (vx, vy, yaw)

### Key Configuration Classes
Located in `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`:
- `G1MimicPrivCfg` / `G1MimicPrivCfgPPO` - Teacher training (mocap)
- `G1MimicStuRLCfg` / `G1MimicStuRLCfgDAgger` - Student training (mocap)
- `G1MimicCMG{Slow,Medium,Fast}Cfg` - CMG teacher training
- `G1MimicCMGStuRLCfg` / `G1MimicCMGStuRLCfgDAgger` - CMG student DAgger distillation

### Policy Architecture
`ActorCriticMimic` (in `rsl_rl/modules/actor_critic_mimic.py`):
- **Motion Encoder**: Conv1D on multi-step future references → 128D latent
- **Actor**: MLP [512,512,256,128] → 23 DOF actions
- **Critic**: MLP [512,512,256,128] → value

### Deployment Architecture
Uses Redis for decoupling high-level motion commands from low-level policy control:
- High-level server: Reads motion file, publishes reference poses
- Low-level server: Runs policy inference, sends joint commands to sim/real robot

## Key Files

| File | Purpose |
|------|---------|
| `legged_gym/legged_gym/scripts/train.py` | Training entry point |
| `legged_gym/legged_gym/envs/base/humanoid_mimic.py` | Main environment class |
| `rsl_rl/rsl_rl/runners/on_policy_runner_mimic.py` | Training loop |
| `pose/pose/utils/cmg_motion_lib.py` | CMG motion interface |
| `pose/pose/utils/motion_lib_pkl.py` | Mocap motion loader |

## Training Parameters

| Parameter | Location | Default |
|-----------|----------|---------|
| `max_iterations` | `g1_mimic_distill_config.py` | 30002 |
| `save_interval` | `g1_mimic_distill_config.py` | 500 |
| `num_envs` | `g1_mimic_distill_config.py` | 4096 |
| `episode_length_s` | `g1_mimic_distill_config.py` | 10s |

## DOF Mapping

G1 uses 23 DOF (body joints only):
- Left leg: 0-5 (hip pitch/roll/yaw, knee, ankle pitch/roll)
- Right leg: 6-11
- Waist: 12-14 (yaw, roll, pitch)
- Left arm: 15-18 (shoulder pitch/roll/yaw, elbow)
- Right arm: 19-22

CMG outputs 29 DOF → skip wrist joints (19-21 left, 26-28 right)

## Reward Functions

Reward implementations in `humanoid_mimic.py`, scales in `g1_mimic_distill_config.py`.

### Base rewards (all modes):
- `tracking_joint_dof`: Joint angle tracking (scale: 0.6)
- `tracking_joint_vel`: Joint velocity tracking (scale: 0.2)
- `tracking_root_pose`: Root position/orientation (base: 0.6, CMG: 0.2)
- `tracking_root_vel`: Root linear/angular velocity (base: 1.0, CMG: 0.8)
- `tracking_keybody_pos`: Key body positions (scale: 2.0, CMG mode: lower body only)

### CMG-specific rewards:
- `tracking_keybody_pos_upper`: Weak upper body tracking — hands, elbows, head (scale: 0.3, exp_scale=5.0 vs lower body 10.0)
- `tracking_cmd_vel`: Track vx/vy commands (scale: 1.5)
- `tracking_cmd_yaw`: Track yaw rate command (scale: 1.0)
- `action_symmetry`: Weak left-right action symmetry (scale: 0.1)

### CMG Left-Right Mirroring

50% of environments randomly mirror the CMG output at each reset to prevent left-right bias. Implemented in `cmg_motion_lib.py:_apply_mirror()`:
- DOFs: swap left↔right indices, flip roll/yaw signs (`DOF_MIRROR_INDICES_23`, `DOF_MIRROR_SIGNS_23`)
- Key body positions: swap left↔right, flip y (`KEYBODY_MIRROR_INDICES`)
- Root: flip y position, negate yaw/roll rotation, flip vy and yaw_rate
- Commands via `get_commands()`: return flipped vy/yaw for mirrored envs
- Applied at output level in all 3 `calc_motion_frame` paths (current, partial, tiled)

### CMG Velocity Command Observation

CMG 模式下速度指令 `[vx, vy, yaw_rate]` 被加入观测，使策略能显式感知速度目标。

- `use_cmd_obs = True` 在 `G1MimicCMGBaseCfg.env` 中启用
- 指令追加在 `proprio_obs_buf` 中、噪声添加之前（命令本身不加噪）
- 指令来源：`CMGMotionLib.get_commands()` → 返回经速度估计校准的 `_actual_commands`（含镜像修正）
- **Teacher** 可同时观测指令速度（`proprio`）和实际速度（`base_lin_vel` 在 `priv_info`）
- **Student** 可观测指令速度，实际速度通过 DAgger 蒸馏隐式学习

#### CMG 观测维度（含速度指令）

| 配置 | `n_proprio` | `n_obs_single` | `num_observations` |
|------|------------|----------------|-------------------|
| 非 CMG teacher | 74 | 1318 | 1318 |
| CMG teacher (`G1MimicCMGBaseCfg`) | 77 (+3 cmd) | 1321 | 1321 |
| CMG student (`G1MimicCMGStuRLCfg`) | 77 (+3 cmd) | 108 | 1188 (×11 history) |

### CMG Velocity Calibration

CMG generates joint trajectories conditioned on velocity commands, but the actual gait velocity may differ from the commanded velocity. A linear regression velocity estimator (`_build_velocity_estimator`) maps 29 DOF velocities → [vx, vy, yaw] using training data. After each trajectory generation:
- `_estimate_actual_velocity()` averages per-frame predictions over frames 20-100
- `_actual_commands` stores estimated velocities, used for root state integration and RL rewards
- `get_commands()` returns `_actual_commands` (not raw commands), so `tracking_cmd_vel` tracks achievable velocity
- Raw `_commands` are still used to condition CMG input

### CMG Upper Body Relaxation
- `tracking_keybody_pos` only tracks lower body (ankles+knees) in CMG mode
- `tracking_keybody_pos_upper` separately tracks upper body with weak scale
- Arm `dof_err_w` reduced from 0.8/1.0 to 0.3/0.4 in CMG config

### Mirror DOF Mapping (23 DOF)
```
Swap: left_leg(0-5) ↔ right_leg(6-11), left_arm(15-18) ↔ right_arm(19-22)
Sign flip: roll joints (-1), yaw joints (-1), pitch joints (+1)
Waist: yaw(-1), roll(-1), pitch(+1)
```

## Testing

```bash
# CMG mirror visualization (conda env: Main)
python test_cmg_mirror.py --cmd_vx 1.5 --cmd_vy 0.3 --cmd_yaw 0.2
python test_cmg_mirror.py --no_viewer  # numerical verification only

# CMG velocity calibration diagnostic (conda env: twist)
python tools/test_cmg_velocity.py
python tools/test_cmg_velocity.py --vx_min 0.5 --vx_max 3.0 --vx_step 0.25
```

## Registered Tasks

| Task name | Config | Description |
|-----------|--------|-------------|
| `g1_priv_mimic` | `G1MimicPrivCfg` | Mocap teacher (PPO) |
| `g1_stu_rl` | `G1MimicStuRLCfg` | Mocap student (DAgger) |
| `g1_cmg_slow` | `G1MimicCMGSlowCfg` | CMG teacher, ~1 m/s |
| `g1_cmg_medium` | `G1MimicCMGMediumCfg` | CMG teacher, ~2 m/s |
| `g1_cmg_fast` | `G1MimicCMGFastCfg` | CMG teacher, ~3 m/s |
| `g1_cmg_stu_rl` | `G1MimicCMGStuRLCfg` | CMG student (DAgger, sees velocity cmd) |

## Changelog

All CMG-related changes: [`docs/CHANGELOG_CMG_Integration.md`](docs/CHANGELOG_CMG_Integration.md)
