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
- `G1MimicPrivCfg` / `G1MimicPrivCfgPPO` - Teacher training
- `G1MimicStuRLCfg` / `G1MimicStuRLCfgDAgger` - Student training
- `G1MimicCMG{Slow,Medium,Fast}Cfg` - CMG-based training

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

Main tracking rewards (in `humanoid_mimic.py`):
- `tracking_joint_dof`: Joint angle tracking (scale: 0.6)
- `tracking_joint_vel`: Joint velocity tracking (scale: 0.2)
- `tracking_root_pose`: Root position/orientation (scale: 0.6)
- `tracking_root_vel`: Root linear/angular velocity (scale: 1.0)
- `tracking_keybody_pos`: Key body positions (scale: 2.0)
