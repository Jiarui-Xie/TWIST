# TWIST + CMG Extension

**Personal fork of [TWIST (CoRL 2025)](https://github.com/YanjieZe/TWIST)** — I extended the original MoCap-based motion imitation system with a velocity-command-driven locomotion pipeline using a Conditional Motion Generator (CMG), removing the dependency on pre-recorded motion capture data.

> Full contribution details: [`docs/MY_CONTRIBUTIONS.md`](docs/MY_CONTRIBUTIONS.md) | Original changelog: [`docs/CHANGELOG_CMG_Integration.md`](docs/CHANGELOG_CMG_Integration.md)

---

## My Extensions

### What I Built

The original TWIST pipeline requires pre-recorded MoCap data as motion references. I replaced that with a **Conditional Motion Generator (CMG)** — a neural network that generates motion references conditioned on velocity commands `[vx, vy, yaw_rate]` in real time, enabling:

- **No MoCap data required** for training or deployment
- **Continuous velocity command control** (speed up, slow down, turn)
- **Natural start/stop behavior** via ramp-profile training
- **Direct sim-to-real transfer** via a standalone MuJoCo sim2sim + real-robot deployment stack

### Architecture

```
                    ┌─────────────────────────┐
  [vx, vy, yaw] ──▶│  CMG (neural network)   │──▶ 29-DOF motion reference
                    └─────────────────────────┘
                                │  (29→23 DOF mapping, root integration)
                                ▼
                    ┌─────────────────────────┐
                    │      CMGMotionLib        │──▶ 100-frame trajectory buffer
                    │  (drop-in MotionLib API) │    (future-frame queries)
                    └─────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                  ▼
   ┌─────────────────────┐           ┌─────────────────────────┐
   │  CMG Teacher (PPO)  │           │  CMG Student V2 (DAgger)│
   │  obs: 1321 dims     │  ──────▶  │  obs: 1237 dims         │
   │  (+ privileged info)│  distill  │  (no privileged info)   │
   └─────────────────────┘           └─────────────────────────┘
                                                  │
                              ┌───────────────────┴──────────────┐
                              ▼                                   ▼
                   ┌─────────────────┐               ┌─────────────────────┐
                   │   Sim2Sim       │               │   Sim2Real (G1)     │
                   │  (MuJoCo, 50Hz) │               │  (50Hz, no Redis)   │
                   └─────────────────┘               └─────────────────────┘
```

**Key design choices:**
- `CMGMotionLib` implements the same API as `MotionLib` — the training env needed minimal changes
- **Velocity ramp training**: each episode uses a 6-phase profile (`stand → ramp_up → steady → ramp_down → crawl → stand`) with per-env random durations, teaching the policy to accelerate and decelerate naturally
- **Left-right mirroring**: 50% of envs mirror CMG output per reset to prevent left/right bias
- **Student V2**: drops the 84-dim privileged info from observations, making the policy directly deployable on hardware

### Sim2Sim Demo

https://github.com/user-attachments/assets/53e1b473-35b6-4ff2-925c-bd6d8dd3fe88

### CMG Deployment (Quick Start)

```bash
conda activate twist

# 1. Export trained student to JIT
cd legged_gym/legged_gym/scripts
python save_jit_cmg_stu_v2.py --exptid cmg_stu_v4 --device cpu

# 2. Sim2sim verification (keyboard: ↑/↓=vx  Q/E=vy  ←/→=yaw  R=reset)
cd deploy_real
python sim2sim_cmg_stu_v2.py \
    --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \
    --cmd_vx 1.5 --duration 30.0

# 3. Sim2real on Unitree G1
python deploy_real_cmg_stu_v2.py \
    --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \
    --cmd_vx 1.0 --net eno1
```

### CMG Training Pipeline

```bash
# Teacher (PPO) — pick a speed mode
bash train_teacher_cmg.sh medium cmg_teacher_v1 cuda:0

# Student V2 (DAgger distillation)
bash train_student_cmg_v2.sh cmg_stu_v4 global_obs_v4 cuda:0

# Export
cd legged_gym/legged_gym/scripts
python save_jit_cmg_stu_v2.py --exptid cmg_stu_v4 --device cpu
```

### Registered CMG Tasks

| Task | Speed | Description |
|------|-------|-------------|
| `g1_cmg_slow` | ~1 m/s | CMG teacher |
| `g1_cmg_medium` | ~2 m/s | CMG teacher |
| `g1_cmg_fast` | ~3 m/s | CMG teacher |
| `g1_cmg_stu_v2` | all | Student V2 — deployable, no privileged info |

### Visualization & Debug Tools

```bash
# Visualize velocity ramp profiles (MuJoCo kinematic playback)
python tools/vis_cmg_ramp.py --random

# Plot real-time command + joint curves
python tools/vis_cmg_commands.py

# Verify left-right mirror symmetry
python test_cmg_mirror.py --cmd_vx 1.5 --cmd_vy 0.3 --cmd_yaw 0.2
```

---

# Original TWIST Project

[CoRL 2025] | [[Website]](https://humanoid-teleop.github.io/) | [[Arxiv]](https://arxiv.org/abs/2505.02833) | [[Video]](https://www.youtube.com/watch?v=QgA7jNoiIZo)

```bibtex
@article{ze2025twist,
  title={TWIST: Teleoperated Whole-Body Imitation System},
  author={Yanjie Ze and Zixuan Chen and João Pedro Araújo and Zi-ang Cao and Xue Bin Peng and Jiajun Wu and C. Karen Liu},
  year={2025},
  journal={arXiv preprint arXiv:2505.02833}
}
```

Demo 1: diverse loco-manipulation skills by TWIST.

https://github.com/user-attachments/assets/7c2b874e-e713-47e1-8e84-0efb93c419b5

Demo 2: low-level controller and high-level motion sender in TWIST (fully reproduced in this repo).

https://github.com/user-attachments/assets/4953b6de-5c84-4a4b-9391-75818903a654

## News
- [2025.09.29] TWIST is fully open-sourced, including training datasets, teacher & student training code, sim2sim & sim2real scripts, and a model checkpoint (`assets/twist_general_motion_tracker.pt`).
- [2025.08.04] Real-time retargeting released — see [GMR](https://github.com/YanjieZe/GMR).
- [2025.08.01] TWIST accepted to CoRL 2025.

## Installation

Training runs on a single Nvidia RTX 4090 (24 GB) in 1–2 days.

**1. Create conda environment:**
```bash
conda create -n twist python=3.8
conda activate twist
```

**2. Install Isaac Gym** (download from [official link](https://developer.nvidia.com/isaac-gym)):
```bash
cd isaacgym/python && pip install -e .
```

**3. Install packages:**
```bash
cd rsl_rl && pip install -e . && cd ..
cd legged_gym && pip install -e . && cd ..
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco pytorch-kinematics rich termcolor
pip install "redis[hiredis]"
pip install pyttsx3
cd pose && pip install -e . && cd ..
```

Start Redis (required for the original mocap deployment pipeline):
```bash
redis-server --daemonize yes
```

For sim2real, also install [unitree_sdk2py](https://github.com/unitreerobotics/unitree_sdk2_python):
```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
```

**4. Download TWIST dataset** from [Google Drive](https://drive.google.com/file/d/1bRAGwRAJ3qZV94IBIyuu4cySqZM95XBi/view?usp=sharing). Unzip and set `root_path` in `legged_gym/motion_data_configs/twist_dataset.yaml`.

## Usage (Original Mocap Pipeline)

**Train teacher:**
```bash
bash train_teacher.sh 0927_twist_teacher cuda:0
```

**Train student:**
```bash
bash train_student.sh 0927_twist_rlbcstu 0927_twist_teacher cuda:0
```

**Export to JIT:**
```bash
bash to_jit.sh 0927_twist_rlbcstu
```

**Sim2sim verification:**
```bash
cd deploy_real
python server_high_level_motion_lib.py --motion_file PATH/TO/MOTION.pkl --vis
python server_low_level_g1_sim.py --policy_path PATH/TO/model.pt
```

**Sim2real:**
```bash
cd deploy_real
python server_low_level_g1_real.py --policy_path PATH/TO/model.pt --net YOUR_NET_INTERFACE
```

See [unitree_g1.md](./unitree_g1.md) or [unitree_g1.zh.md](./unitree_g1.zh.md) for full sim2real setup instructions.

## Q & A

**Q: NumPy version mismatch between TWIST (1.23.0) and GMR (2.2.6) causes pkl compatibility issues.**

A: See [issue #10](https://github.com/YanjieZe/TWIST/issues/10).

## Contact

For questions about the original TWIST system: `yanjieze@stanford.edu`
