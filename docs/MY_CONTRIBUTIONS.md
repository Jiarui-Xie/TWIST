# CMG Extension — Contributions by Jiarui Xie

This document consolidates all changes I made on top of the original [TWIST](https://github.com/YanjieZe/TWIST) codebase (CoRL 2025).

## Overview

The original TWIST pipeline requires **pre-recorded MoCap data** as motion references for both teacher and student training. My extension replaces that data dependency with a **Conditional Motion Generator (CMG)** — a neural network conditioned on velocity commands — enabling velocity-commanded locomotion without any MoCap capture.

---

## New Files

| File | Description |
|------|-------------|
| `pose/pose/utils/cmg_motion_lib.py` | Core CMG motion interface: autoregressive trajectory generation, 29→23 DOF mapping, root state integration, left-right mirroring, velocity ramp scheduling |
| `pose/pose/utils/forward_kinematics.py` | FK calculator: joint angles → 9 key-body 3D positions using `pytorch_kinematics` |
| `legged_gym/legged_gym/scripts/save_jit_cmg_stu_v2.py` | Export CMG Student V2 actor + baked Normalizer to TorchScript JIT |
| `deploy_real/sim2sim_cmg_stu_v2.py` | MuJoCo sim2sim: embedded CMGMotionLib + JIT policy + keyboard control (no Redis) |
| `deploy_real/deploy_real_cmg_stu_v2.py` | Real-robot deployment: G1RealWorldEnv + CMGMotionLib + JIT policy |
| `train_teacher_cmg.sh` | CMG teacher training script (slow / medium / fast speed modes) |
| `train_teacher_cmg_viz.sh` | Same but with MuJoCo visualization (fewer envs) |
| `train_student_cmg_v2.sh` | CMG Student V2 DAgger training script |
| `test_cmg_mirror.py` | Mirror symmetry verification (MuJoCo playback + numerical checks) |
| `setup_local.sh` | Local environment setup without cloud instance assumptions |
| `tools/vis_cmg_ramp.py` | MuJoCo kinematic playback of velocity ramp profiles |
| `tools/vis_cmg_commands.py` | Matplotlib real-time command + joint angle visualization |
| `tools/vis_random_ramp.py` | Multi-env ramp profile comparison across resets |
| `tools/render_random_ramp.py` | Offscreen MuJoCo rendering of ramp snapshots |
| `tools/test_cmg_output.py` | CMG output correctness tests |
| `tools/test_cmg_ramp.py` | Ramp schedule unit tests |
| `tools/test_cmg_velocity.py` | Velocity calibration diagnostic |
| `tests/smoke_test_cmg_integration.py` | Smoke test for CMG integration |
| `docs/CHANGELOG_CMG_Integration.md` | Detailed per-commit changelog |

---

## Modified Files

### Core Environment

**`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`**
- Added `G1MimicCMGBaseCfg` and speed-specific variants (`Slow`, `Medium`, `Fast`)
- Added `G1MimicCMGStuRLCfg` / `G1MimicCMGStuRLCfgDAgger` for DAgger distillation
- Added `G1MimicCMGStuV2Cfg` — Student V2 (1237 obs, no privileged info)
- Added velocity ramp parameters: `cmg_ramp_enabled`, `cmg_ramp_up_range`, `cmg_ramp_down_range`, `cmg_ramp_crawl_range`, `cmg_ramp_min_steady`, etc.
- Added `use_cmd_obs = True` for velocity command observations

**`legged_gym/legged_gym/envs/base/humanoid_mimic.py`**
- Wired `CMGMotionLib` as a drop-in replacement for `MotionLib`
- Added `_reward_tracking_cmd_vel` / `_reward_tracking_cmd_yaw` using raw user commands (fixed reward target bug)
- Added `_reward_tracking_keybody_pos_upper` for relaxed upper-body tracking
- Added `_reward_action_symmetry` for left-right action regularization
- Added `_reward_foot_contact_plane` for flat foot contact
- Added velocity command obs (`use_cmd_obs`) appended to proprioception buffer

**`legged_gym/legged_gym/envs/__init__.py`**
- Registered 7 new tasks: `g1_cmg_slow`, `g1_cmg_medium`, `g1_cmg_fast`, `g1_cmg_stu_rl`, `g1_cmg_stu_v2`, etc.

**`legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`**
- CMG task initialization and environment reset hooks

**`legged_gym/legged_gym/gym_utils/terrain.py`**
- `curriculum()` supports float `max_difficulty`
- `add_terrain_to_map` guards against terrains without `goals`

**`legged_gym/legged_gym/scripts/play.py`**
- Keyboard arrow-key control for CMG velocity commands during playback

**`legged_gym/legged_gym/scripts/train.py`**
- CMG-specific argument forwarding to training runner

### Deployment

**`deploy_real/sim2sim_cmg_stu_v2.py`** (new, detailed above)
**`deploy_real/deploy_real_cmg_stu_v2.py`** (new, detailed above)

**`assets/g1/g1_sim2sim_with_wrist_roll.xml`**
- Ground friction 0.6 → 1.0, `condim` 3 → 4 (torsional friction) to reduce lateral sway at high speeds

### Training Infrastructure

**`rsl_rl/rsl_rl/runners/on_policy_runner_mimic.py`**
**`rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py`**
- CMG-compatible runner hooks, logging improvements

---

## Key Design Decisions

### 1. CMG as Drop-in MotionLib Replacement
`CMGMotionLib` exposes the same interface as `MotionLib` (`calc_motion_frame`, `get_motion_state`, etc.), so the training environment requires minimal modification. The CMG model runs autoregressively to fill a 100-frame (2s) trajectory buffer, which the env queries for future reference frames.

### 2. Velocity Ramp — Natural Start/Stop
Each episode uses a 6-phase profile: `stand → ramp_up → steady → ramp_down → crawl → stand`. Phase durations are randomly sampled per-env per-reset (from configurable ranges), creating a curriculum where the policy learns to accelerate, maintain speed, and decelerate smoothly.

### 3. Left-Right Mirroring
50% of environments randomly mirror the CMG output at each reset (joint angles, key-body positions, root orientation, and velocity commands). This prevents the policy from developing a left/right bias — critical for symmetric locomotion.

### 4. Velocity Tracking Reward Fix
Original reward compared robot physics velocity against a kinematic regression estimate (`_actual_commands`). I fixed this to compare against the raw user velocity command, making the reward signal clean and consistent with the policy's observation.

### 5. Student V2 — No Privileged Info
`G1MimicCMGStuV2Cfg` drops the 84-dim privileged information from observations (1321 → 1237 dims). The student policy is deployable directly on hardware without any estimation of privileged states.

---

## Registered Tasks

| Task | Config | Description |
|------|--------|-------------|
| `g1_cmg_slow` | `G1MimicCMGSlowCfg` | CMG teacher, ~1 m/s |
| `g1_cmg_medium` | `G1MimicCMGMediumCfg` | CMG teacher, ~2 m/s |
| `g1_cmg_fast` | `G1MimicCMGFastCfg` | CMG teacher, ~3 m/s |
| `g1_cmg_stu_rl` | `G1MimicCMGStuRLCfg` | CMG student DAgger |
| `g1_cmg_stu_v2` | `G1MimicCMGStuV2Cfg` | CMG student V2 (deployable) |
