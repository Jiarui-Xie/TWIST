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
# Export to JIT model (old teacher/student mocap pipeline)
bash to_jit.sh <student_exptid>

# Export CMG Student V2 to JIT (bakes normalizer in)
cd legged_gym/legged_gym/scripts
python save_jit_cmg_stu_v2.py --exptid cmg_stu_v4 --device cpu
# Output: legged_gym/logs/g1_cmg_stu_v2/<exptid>/traced/<exptid>-<step>-jit.pt

# Sim2sim (CMG student, no Redis needed)
cd deploy_real
python sim2sim_cmg_stu_v2.py \
    --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \
    --cmd_vx 1.5 --duration 30.0 --device cuda
# Keyboard (in MuJoCo window): ↑/↓=vx  ←/→=yaw  A/D=vy  R=reset

# Sim2real (CMG student, no Redis needed)
cd deploy_real
python deploy_real_cmg_stu_v2.py \
    --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \
    --cmd_vx 1.0 --net eno1
# Remote: [START]→zero torque → auto default pose → [A]→hold → [A]→policy → [Select]→exit

# Sim2sim verification (old teacher pipeline, requires Redis)
cd deploy_real
python server_high_level_motion_lib.py --motion_file PATH/TO/MOTION.pkl --vis
python server_low_level_g1_sim.py --policy_path PATH/TO/model.pt

# Sim2real (old teacher pipeline)
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

### CMG Student V2 Pipeline (`g1_cmg_stu_v2`)
- Task: `g1_cmg_stu_v2`, config: `G1MimicCMGStuV2Cfg`
- obs: 1237 dims = priv_mimic(1160) + proprio(77), no priv_info(84)
- Resume: `bash train_student_cmg_v2.sh cmg_stu_v4 global_obs_v4 cuda:0 -1 --resumeid cmg_stu_v4 --max_iterations 20001`
- Latest checkpoint: `legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/model_21000.pt`

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
| `legged_gym/legged_gym/scripts/save_jit_cmg_stu_v2.py` | Export CMG Stu V2 → TorchScript JIT |
| `deploy_real/sim2sim_cmg_stu_v2.py` | MuJoCo sim2sim for CMG Stu V2 |
| `deploy_real/deploy_real_cmg_stu_v2.py` | Real-robot deploy for CMG Stu V2 |

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
- `tracking_cmd_vel`: Track vx/vy commands (scale: 1.5) — target is raw user `_commands`, not kinematic estimate
- `tracking_cmd_yaw`: Track yaw rate command (scale: 1.0) — same, uses raw `_commands`
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
- 指令来源：`CMGMotionLib.get_user_commands()` → 返回原始 `_commands`（含镜像修正），与奖励目标一致
- **Teacher** 可同时观测指令速度（`proprio`）和实际速度（`base_lin_vel` 在 `priv_info`）
- **Student** 可观测指令速度，实际速度通过 DAgger 蒸馏隐式学习

#### CMG 观测维度（含速度指令）

| 配置 | `n_proprio` | `n_obs_single` | `num_observations` |
|------|------------|----------------|-------------------|
| 非 CMG teacher | 74 | 1318 | 1318 |
| CMG teacher (`G1MimicCMGBaseCfg`) | 77 (+3 cmd) | 1321 | 1321 |
| CMG student (`G1MimicCMGStuRLCfg`) | 77 (+3 cmd) | 108 | 1188 (×11 history) |

### CMG Velocity Calibration (DEPRECATED)

~~`_actual_commands`~~ was a linear-regression kinematic velocity estimate. It is now **deprecated** — declared but never read. All usages replaced by raw `_commands`.

`_commands` (raw user input) is used uniformly for:
- CMG trajectory conditioning
- Root position/velocity integration
- `cmd_obs` (policy observation)
- Velocity tracking reward targets (`tracking_cmd_vel`, `tracking_cmd_yaw`)
- `get_commands()` is a deprecated alias for `get_user_commands()`

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
| `g1_cmg_stu_v2` | `G1MimicCMGStuV2Cfg` | CMG student V2 (1237 obs, no priv_info) |

## Deployment Files

### CMG Student V2 (current, no Redis)

| File | Purpose |
|------|---------|
| `save_jit_cmg_stu_v2.py` | Export: actor + baked Normalizer → TorchScript |
| `sim2sim_cmg_stu_v2.py` | MuJoCo sim2sim: CMGMotionLib + JIT policy + keyboard |
| `deploy_real_cmg_stu_v2.py` | Real-robot: G1RealWorldEnv + CMGMotionLib + JIT policy |

**Key constants (must match `G1MimicCMGStuV2Cfg`):**
- `NUM_OBS = 1237` (1160 priv_mimic + 77 proprio)
- `ACTION_SCALE = 0.5`
- `DECIMATION = 20`, `CMG_DT = 0.02`
- `DEFAULT_DOF_POS`: leg[-0.2,0,0,0.4,-0.2,0]×2, waist[0,0,0], arms[0,±0.4,0,1.2]
- URDF: `assets/g1/g1_custom_collision_with_fixed_hand.urdf` (23 DOF)

**Normalizer baking**: `Normalizer._mean/_std/_eps/_clip` (not `.mean/.var`)

### Old Teacher Pipeline (mocap, requires Redis)
`server_high_level_motion_lib.py` + `server_low_level_g1_{sim,real}.py`

## Changelog

All CMG-related changes: [`docs/CHANGELOG_CMG_Integration.md`](docs/CHANGELOG_CMG_Integration.md)

### 2026-03-19 — CMG Velocity Ramp (Start/Stop Training)

**目标**：让 policy 学会自然起步和停止。每个 episode 使用 6 段速度曲线。

**Profile**: `[stand → ramp_up → steady → ramp_down → crawl → stand]`

**改动：**
- `cmg_motion_lib.py`：新增 `_target_commands`（采样的目标速度）、`_ramp_enabled_flags`、`_compute_ramp_scale()` 6 段 profile
- `cmg_motion_lib.py`：`_commands` 从恒定变为时变（每步由 ramp schedule 更新）
- `cmg_motion_lib.py`：`_generate_trajectory()` 逐帧计算时变 command + Euler 积分 root 位置
- `cmg_motion_lib.py`：`calc_motion_frame()` tiled case 根据查询时间计算瞬时速度
- `cmg_motion_lib.py`：所有 env 从 standing pose 起始（`_standing_pose_norm`）
- `g1_mimic_distill_config.py`：`G1MimicCMGBaseCfg.motion` 新增 ramp 全部参数
- `g1_mimic_distill_config.py`：`G1MimicCMGBaseCfg.terrain` 改为 `plane`（无 trimesh）
- `g1_mimic_distill_config.py`：`G1MimicCMGBaseCfg.domain_rand` 重力随机化增大到 ±5°
- `humanoid_mimic.py`：构造 `CMGMotionLib` 时传入 ramp 参数
- `terrain.py`：`curiculum()` 支持 `max_difficulty` 为 float 值；`add_terrain_to_map` 保护无 `goals` 的 terrain

**参数（当前值）：**
- `cmg_ramp_enabled = True`
- `cmg_ramp_up_range = [1.0, 3.0]`（ramp-up 时长范围，per-env 随机采样）
- `cmg_ramp_down_range = [1.5, 4.0]`（ramp-down 时长范围）
- `cmg_ramp_stand_duration = 5.0s`（首尾站立时长，固定）
- `cmg_ramp_crawl_range = [0.5, 1.5]`（近零速阶段时长范围）
- `cmg_ramp_crawl_ratio = 0.01`（crawl 速度 = target × 0.01）
- `cmg_ramp_probability = 1.0`（100% episode 使用 ramp）
- `cmg_ramp_floor_ratio = 0.1`（stand 阶段速度 = target × 0.1）
- `cmg_ramp_min_steady = 3.0`（最小稳态时长保证）
- `episode_length_s = 16.5`
- `num_envs = 1024`
- `terrain: plane` + `gravity_range = (-0.86, 0.86)` ≈ ±5° 等效坡度

ramp_up/ramp_down/crawl 时长在每个 env 每次 reset 时独立随机采样，steady 自动填充剩余时间。如果采样总和超过可用时长，按比例等比缩小。

**无需修改**：奖励函数、观测、mirror 逻辑均自动读取时变 `_commands`。

**可视化工具**：
- `tools/vis_cmg_ramp.py`：MuJoCo kinematic playback（无 policy），支持 `--random` 使用训练随机范围
- `tools/vis_cmg_commands.py`：matplotlib 实时 command + joint 曲线
- `tools/vis_random_ramp.py`：单 env 多次 reset 对比随机 ramp profile
- `tools/render_random_ramp.py`：MuJoCo offscreen 渲染随机 ramp 快照

### 2026-03-20 — Random Ramp Durations & Sim2Sim Fixes

**目标**：让 ramp 各阶段时长随机化，增加训练多样性；修复 sim2sim 键盘控制和物理参数。

**改动：**

#### 随机 Ramp 时长
- `g1_mimic_distill_config.py`：`cmg_ramp_duration` → `cmg_ramp_up_range = [1.0, 3.0]`，`cmg_ramp_down_duration` → `cmg_ramp_down_range = [1.5, 4.0]`，`cmg_ramp_crawl_duration` → `cmg_ramp_crawl_range = [0.5, 1.5]`，新增 `cmg_ramp_min_steady = 3.0`
- `cmg_motion_lib.py`：构造函数改为接收 range 元组；新增 per-env buffer `_env_ramp_up/_env_ramp_down/_env_crawl`；reset 时随机采样并裁剪；`_compute_ramp_scale` 用 per-env 张量计算相界
- `humanoid_mimic.py`：传参适配新接口

#### Sim2Sim 修复
- `sim2sim_cmg_stu_v2.py`：键盘回调改写 `_target_commands`（而非 `_commands`），并立即重新生成轨迹 buffer，解决键盘改速度后 CMG 轨迹不更新的 bug
- `sim2sim_cmg_stu_v2.py`：vy 控制键从 A/D 改为 Q/E，避免与 MuJoCo viewer 相机控制冲突
- `g1_sim2sim_with_wrist_roll.xml`：地面摩擦 0.6 → 1.0，`condim` 3 → 4（加入扭转摩擦），减少高速行走时左右甩动

### 2026-03-03 — CMG Student V2 Sim2Sim & Sim2Real

**新增文件：**
- `legged_gym/legged_gym/scripts/save_jit_cmg_stu_v2.py`：导出 CMG Stu V2 的 actor + Normalizer 为 TorchScript JIT
- `deploy_real/sim2sim_cmg_stu_v2.py`：MuJoCo sim2sim，内嵌 CMGMotionLib，无需 Redis
- `deploy_real/deploy_real_cmg_stu_v2.py`：实机部署，G1RealWorldEnv + CMGMotionLib

**修复：**
- `save_jit_cmg_stu_v2.py`：`Normalizer` 属性名 `_mean/_std/_eps/_clip`（非 `mean/var`）
- `sim2sim_cmg_stu_v2.py`：URDF 改为 23-DOF `g1_custom_collision_with_fixed_hand.urdf`
- `sim2sim_cmg_stu_v2.py`：`viewer._impl.set_key_callback` → `launch_passive(key_callback=...)`
- `sim2sim_cmg_stu_v2.py`：`ACTION_SCALE = 0.5`（训练值，非 0.25）
- `sim2sim_cmg_stu_v2.py`：初始 root height 1.0m，shoulder_roll ±0.4（对齐训练 default_joint_angles）
- `sim2sim_cmg_stu_v2.py`：reset 后 `_mirror_flags[:] = False`，cmd 用 `get_user_commands()`
- `legged_gym/legged_gym/envs/base/base_task.py`：IsaacGym play 中添加方向键键盘控制 CMG 速度指令

### 2026-02-26 — Velocity Reward Target Fix
**Problem**: `tracking_cmd_vel` / `tracking_cmd_yaw` were comparing `robot.base_lin_vel` (physics) against `_actual_commands` (linear regression estimate from CMG kinematics). These live in different worlds — the kinematic estimate is neither the user's intent nor the physics-achievable velocity, making the reward signal noisy.

**Fix**:
- Added `CMGMotionLib.get_user_commands()` in `cmg_motion_lib.py` — returns raw `_commands` with mirror correction (vy/yaw flipped for mirrored envs)
- Updated `_reward_tracking_cmd_vel` and `_reward_tracking_cmd_yaw` in `humanoid_mimic.py` to use `get_user_commands()` instead of `get_commands()` (`_actual_commands`)

**Result**: Policy is now directly incentivized to achieve the user-specified velocity in physics simulation. CMG body tracking rewards handle motion style; velocity rewards handle speed.
