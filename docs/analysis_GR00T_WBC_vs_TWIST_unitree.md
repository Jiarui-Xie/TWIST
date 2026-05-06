# GR00T-WBC (SONIC) 核心技术分析 & 对 TWIST / unitree_rl_lab 的启发

> 分析日期：2026-03-28
> 源码路径：`/home/lubuntu/GR00T-WholeBodyControl/`（NVIDIA SONIC + Decoupled WBC）

---

## 一、SONIC 是什么？

**SONIC = Supersizing Motion Tracking for Natural Humanoid Whole-Body Control**

核心思路：用 **142K+ 人类动作**（BONES-SEED 数据集，288 小时，522 位演员）大规模训练一个**统一的全身运动跟踪策略**，作为人形机器人的「运动基础模型」。单一策略支持 27 种运动风格（行走/跑步/蹲/跪/爬/拳击/舞蹈/风格化行走等）。

---

## 二、SONIC 核心架构 Tricks

### 1. Encoder-Decoder 分离架构（最重要的创新）

```
Motion Reference (10帧 joint_pos/vel + anchor_orn + root_z, ~650D)
    |
    v
Encoder (ONNX) --> 64D Token（压缩运动意图）
    |
    v
Policy/Decoder (ONNX): token(64) + proprio(~90D) --> 29 DOF actions
```

- **Encoder** 只看运动参考（10帧 × step5 = 0.9s 前瞻窗口），输出 **64D 紧凑 token**
- **Decoder/Policy** 看 token + 本体感知（ang_vel, joint_pos/vel, last_actions, gravity）
- 比 TWIST 的 Conv1D motion encoder（1160D→128D）更激进地压缩，且**明确分离运动意图和本体反馈**
- Episode attention masking 暗示 encoder 内部使用了 **Transformer/Attention 机制**

**Encoder 输入（obs_config.yaml）：**

| 观测 | 维度 | 说明 |
|------|------|------|
| `motion_joint_positions_10frame_step5` | 290 | 10帧关节角度，每帧隔0.1s |
| `motion_joint_velocities_10frame_step5` | 290 | 10帧关节角速度 |
| `motion_anchor_orientation_10frame_step5` | 60 | 10帧 6D rotation（heading-corrected） |
| `motion_root_z_position_10frame_step5` | 10 | 10帧根高度 |
| **合计** | **650** | 压缩至 **64D token** |

**Policy 输入：**

| 观测 | 维度 |
|------|------|
| `token_state`（encoder 输出） | 64 |
| `base_angular_velocity` | 3 |
| `body_joint_positions` | 29 |
| `body_joint_velocities` | 29 |
| `last_actions` | 29 |
| **合计** | **154** |

**对比 TWIST**：TWIST actor 输入 1321D（1160 motion ref + 77 proprio + 84 priv_info），SONIC policy 仅 154D。encoder 大幅降低了 policy 的学习难度。

---

### 2. 多模式支持（27 种运动风格）

SONIC 不是只做行走，一个策略通过 `encoder_mode` 切换覆盖：

| 类别 | 模式 |
|------|------|
| **基础移动** | Idle, SlowWalk, Walk, Run |
| **蹲/地面** | Squat, KneelTwoLeg, KneelOneLeg, LyingFacedown, HandCrawling, ElbowCrawling |
| **格斗** | IdleBoxing, WalkBoxing, LeftJab, RightJab, RandomPunches, LeftHook, RightHook |
| **风格行走** | Happy, Stealth, Injured, Careful, ObjectCarrying, Crouch, HappyDance, Zombie, Point, Scared |

**对比**：TWIST 分 slow/medium/fast 三个独立训练；unitree_rl_lab 用速度门控切换 Walk/Run。SONIC 证明 **encoder 压缩 + 大数据** 可以让单一策略覆盖极多样的行为。

---

### 3. Kinematic Planner + Policy 两层解耦

```
Planner (10Hz): style + speed + direction + height --> 参考轨迹序列（qpos）
    |  （8帧 cross-fade blending 保证连续性）
    v  （30Hz→50Hz 线性插值 + slerp 重采样）
Policy (50Hz): 跟踪参考轨迹 --> 29 DOF 关节力矩
```

- Planner 是独立 ONNX 模型，输入高层指令，输出 MuJoCo qpos 序列
- 输入：`context_mujoco_qpos[1,4,36]` + `mode` + `target_vel` + `movement_direction[1,3]` + `facing_direction[1,3]` + `height`
- 输出：`mujoco_qpos[1,N,36]`（预测的运动序列）
- Replan 决策逻辑：模式/朝向/高度变化时立即重规划；非静态模式下速度/方向变化或定时器触发重规划

**vs CMG**：类似 TWIST 的 CMG 运动生成器，但更强大——
- CMG 只能条件于速度 (vx, vy, yaw)，Planner 可条件于 **27 种风格 + 速度 + 方向 + 高度**
- Planner 从大量 mocap 数据学出（而非 CMG 的自回归生成），更稳定
- CMG 纯开环自回归有漂移风险，Planner 输入最近 4 帧实际 qpos（闭环）

---

### 4. Decoupled WBC：上下体解耦控制

```
G1DecoupledWholeBodyPolicy
├── Upper Body (17 DOF): IK + 平滑插值（开环，确定性）
│   └── 3 waist + 7 left arm + 7 right arm
└── Lower Body (12 DOF): RL Policy（闭环，反应式）
    └── 6 left leg + 6 right leg
```

- 上半身用 IK + 插值跟踪遥操作/规划目标
- 下半身 RL 策略保持平衡和行走
- 关键：下半身 RL 观测包含 `ref_upper_dof_pos`（17D）—— **下半身能感知上半身在做什么，主动配合**
- 安全超时（1s）：遥操作通信断开时注入安全目标

**对比**：TWIST 和 unitree_rl_lab 都是全身统一控制。解耦方案在做 loco-manipulation 时更有优势。

---

### 5. 大规模数据 + 多样性

| | TWIST | unitree_rl_lab | SONIC |
|---|---|---|---|
| **运动数据** | CMG 自回归生成 | CMG 自回归生成 | **142K mocap clips (288h)** |
| **运动多样性** | 行走（3 速度档） | 行走（自适应速度） | **8 类 20 子类** |
| **演员多样性** | N/A | N/A | **522 人（身高 145-199cm, 体重 38-145kg）** |
| **数据格式** | CMG neural network 输出 | CMG neural network 输出 | **SMPL-X retarget → G1 29-DOF CSV** |

BONES-SEED 数据集分类：

| 类别 | 动作数量 |
|------|----------|
| Locomotion | 74,488 |
| Communication | 21,493 |
| Interactions | 14,643 |
| Dances | 11,006 |
| Gaming | 8,700 |
| Everyday | 5,816 |
| Sport | 3,993 |
| Other | 2,081 |

这是 SONIC 最大的优势——不是算法有多新，而是**数据量和多样性碾压**。

---

### 6. Phase Encoding（步态相位编码）

```yaml
obs_dims:
  ref_motion_phase: 1   # 参考运动的归一化相位 [0,1]
  sin_phase: 1           # sin(2π × phase)
  cos_phase: 1           # cos(2π × phase)
```

- 显式告诉策略「当前处于步态周期的哪个位置」
- 配合 `GAIT_PERIOD = 0.9s`，policy 精确知道何时该抬脚/落脚
- sin/cos 编码保证周期边界连续性

**对比**：TWIST 和 unitree_rl_lab 都没有显式步态相位。unitree_rl_lab 通过 LSTM 隐式学习步态节奏。

---

### 7. 6D Rotation Representation（Anchor Orientation）

```
motion_anchor_orientation: 6D (rotation matrix 前两列)
= heading-corrected relative rotation（机器人当前朝向 → 参考运动朝向的相对旋转）
```

- 用 **6D rotation representation**（Zhou et al. 2019），比 quaternion 更连续，比 euler angle 无万向锁
- Heading-corrected：去除全局朝向，只保留相对偏差

**对比**：TWIST 用 euler angles 表示 root 姿态。

---

### 8. History-based 观测（4 步时序上下文）

```yaml
history_config:
  base_ang_vel: 4
  projected_gravity: 4
  command_lin_vel: 4
  command_ang_vel: 4
  command_base_height: 4
  command_stand: 4
  ref_upper_dof_pos: 4
  dof_pos: 4
  dof_vel: 4
  actions: 4
  ref_motion_phase: 4
  sin_phase: 4
  cos_phase: 4
```

- 4 帧历史堆叠，比 TWIST student 的 11 帧少，但配合 encoder 的 10 帧前瞻窗口，时序信息更高效
- 有三种历史配置：`history_config`（全量）、`history_loco_config`（行走）、`history_mimic_config`（模仿）

---

### 9. PD 控制参数对比

| 关节 | TWIST Kp/Kd | SONIC MOTOR_KP/KD | SONIC JOINT_KP/KD |
|------|-------------|--------------------|--------------------|
| Hip | 100/2 | 150/2 | 100/2.5 |
| Knee | 150/4 | 200/4 | 200/5 |
| Ankle | 40/2 | 40/2 | 20/0.1-0.2 |
| Waist | 150/4 | 250/5 | 400/5 |
| Shoulder | 40/5 | 100/5 | 90/2 |
| Elbow | 40/5 | 40/2 | 60/1 |
| Wrist | N/A (23 DOF) | 20/2 | 4/0.2 |

SONIC 有两套 KP/KD（MOTOR 和 JOINT），可能分别用于不同场景。SONIC waist 刚度显著更高（250-400 vs 150）。

---

### 10. 控制频率

| | TWIST | SONIC |
|---|---|---|
| 仿真 dt | 0.005s (200Hz) | 0.005s (200Hz) |
| 控制 dt | 0.02s (50Hz, dec=4) | 0.02s (50Hz, dec=4) |
| Planner | N/A | 10Hz |
| 遥操作 | N/A | 20Hz |

---

### 11. 部署架构对比

| | TWIST | unitree_rl_lab | SONIC |
|---|---|---|---|
| **推理框架** | Python + TorchScript JIT | C++ + ONNX | **C++ + ONNX + TensorRT** |
| **GPU 加速** | CUDA (torch) | CPU/GPU | **TensorRT + CUDA Graph** |
| **延迟** | ~ms 级 | ~ms 级 | **<1ms（CUDA Graph capture）** |
| **通信** | Redis | 直接调用 | **ZMQ + ROS2** |

SONIC 部署栈高度工程化：ONNX→TensorRT 编译，CUDA Graph 固化推理图，确保确定性低延迟。

---

## 三、三者完整对比

| 维度 | TWIST | unitree_rl_lab | SONIC |
|---|---|---|---|
| **核心范式** | 全量模仿 (actor 直接输出) | 残差学习 (CMG ref + RL 修正) | **运动跟踪基础模型** (planner + tracker) |
| **运动来源** | CMG 自回归 | CMG 自回归 | **142K mocap + learned planner** |
| **Actor 网络** | MLP [512,512,256,128] + Conv1D | LSTM 2×256 | **Encoder(→64D) + Decoder** |
| **Actor 看参考?** | 是 (1160D 直接拼接) | 否 (只给 critic) | **是，但 encoder 压缩到 64D** |
| **Policy 输入维度** | 1321D (teacher) | 475D (95×5) | **154D (64 token + 90 proprio)** |
| **行为多样性** | 行走 3 档 | 行走全速域 | **27 种运动风格** |
| **上下体** | 统一控制 | 统一控制 | **可解耦 (RL 下体 + IK 上体)** |
| **步态相位** | 无 | 无 | **显式 sin/cos phase** |
| **旋转表示** | Euler angles | Euler angles | **6D rotation matrix** |
| **DOF** | 23 (无手腕) | 29 (含手腕) | **29 (含手腕)** |
| **训练数据量** | CMG 生成 ~数十条 | CMG 生成 ~数十条 | **142K clips, 288h** |
| **CMG AR 方式** | 纯开环 | Leaky AR (5%真实) | **Planner 输入真实 qpos (闭环)** |
| **部署** | Python JIT | C++ | **C++ + TensorRT** |
| **num_envs** | 1024 | 4096 | 未公开 |
| **训练代码** | 开源 | 开源 | **未开源（仅推理）** |

---

## 四、可借鉴的改进方向（按优先级）

### P0：影响最大、实现可行

1. **Encoder 压缩运动参考**
   - 将 1160D CMG 参考用独立 encoder 压缩成 64-128D token
   - 降低 policy 学习难度，加速收敛
   - 可以先用 TWIST 现有的 Conv1D encoder 单独预训练

2. **加入 Phase Encoding**
   - 在 obs 中加入 `sin(2π × gait_phase)` 和 `cos(2π × gait_phase)`（+2D）
   - 步态相位可从 CMG 参考的步态周期计算
   - 低成本高收益，提升步态周期精确性和稳定性

3. **6D Rotation 替代 Euler Angles**
   - 参考轨迹中的姿态用 rotation matrix 前两列（6D）表示
   - 消除万向锁和不连续性问题

### P1：中等难度、显著收益

4. **CMG 闭环修正（Leaky AR）**
   - 借鉴 unitree_rl_lab 的 leaky AR：`next_input = 0.95 × CMG_output + 0.05 × robot_actual`
   - 或借鉴 SONIC planner 的做法：输入最近 4 帧真实 qpos
   - 解决纯开环 CMG AR 链的漂移/卡死问题

5. **上下体解耦控制**（如果要做操作任务）
   - 下半身 RL 保持行走平衡
   - 上半身 IK/插值跟踪操作目标
   - 下半身 obs 加入 `ref_upper_dof_pos`（17D）感知上半身状态

### P2：长期方向

6. **接入 BONES-SEED 数据集**
   - 用 mocap 数据做 motion tracking 预训练
   - 然后 fine-tune 到 CMG 条件化行走
   - 数据多样性是 SONIC 效果好的根本原因

7. **学习 Kinematic Planner 替代 CMG**
   - 从大量 retarget 后的 mocap 数据训练 planner
   - 输入：运动风格 + 速度 + 方向 → 输出：参考轨迹
   - 比 CMG 自回归更稳定，支持更多行为

---

## 五、关键文件路径

### GR00T-WholeBodyControl

| 文件 | 用途 |
|------|------|
| `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml` | 完整观测/动作/PD 配置 |
| `gear_sonic_deploy/policy/release/observation_config.yaml` | 部署观测配置（encoder + policy） |
| `gear_sonic/trl/utils/rl.py` | Episode attention masking |
| `decoupled_wbc/control/policy/g1_decoupled_whole_body_policy.py` | 上下体解耦策略 |
| `decoupled_wbc/control/policy/g1_gear_wbc_policy.py` | 下半身 RL 策略（ONNX 推理） |
| `decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml` | 下半身策略配置（86D obs × 6 history = 516D） |
| `docs/source/references/planner_onnx.md` | Kinematic Planner 完整规格 |
| `docs/source/references/observation_config.md` | 观测系统完整参考 |
| `docs/source/references/motion_reference.md` | 运动参考数据格式 |
| `docs/source/user_guide/training_data.md` | BONES-SEED 数据集说明 |

### TWIST (对比)

| 文件 | 用途 |
|------|------|
| `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py` | 所有配置 |
| `legged_gym/legged_gym/envs/base/humanoid_mimic.py` | 主环境 + 奖励 |
| `rsl_rl/rsl_rl/modules/actor_critic_mimic.py` | 网络架构（MLP + Conv1D） |
| `pose/pose/utils/cmg_motion_lib.py` | CMG 运动库 |

### unitree_rl_lab (对比)

| 文件 | 用途 |
|------|------|
| `source/.../tasks/locomotion/robots/g1/29dof/RuN_env_cfg.py` | CMG 残差训练配置 |
| `source/.../tasks/locomotion/mdp/rewards.py` | 奖励函数（含门控） |
| `deploy/include/FSM/State_RLResidual.h` | 残差部署：CMG ref + residual |

---

## 六、注意事项

- SONIC **训练代码未开源**（README 标注 TODO），以上分析基于推理代码、配置文件和文档推断
- SONIC 的核心优势很大程度来自**数据规模**（142K clips），纯算法层面的改进可能不如数据量的影响大
- BONES-SEED 数据集已开源（HuggingFace: `bones-studio/seed`），可以直接下载使用
