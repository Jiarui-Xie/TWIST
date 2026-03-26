# TWIST 开发者完全指南

> **TWIST** (Teleoperated Whole-Body Imitation System) — 面向 Unitree G1 人形机器人的全身动作模仿系统。
> 采用两阶段 Teacher-Student 学习框架，基于强化学习训练策略来跟踪参考动作。

**论文**：[arXiv:2505.02833](https://arxiv.org/abs/2505.02833) | **主页**：[humanoid-teleop.github.io](https://humanoid-teleop.github.io/) | **视频**：[YouTube](https://www.youtube.com/watch?v=QgA7jNoiIZo) | **会议**：CoRL 2025

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 仓库目录结构](#2-仓库目录结构)
- [3. 环境安装](#3-环境安装)
- [4. 核心概念](#4-核心概念)
- [5. 训练流程详解](#5-训练流程详解)
- [6. 模型导出与部署](#6-模型导出与部署)
- [7. 配置系统详解](#7-配置系统详解)
- [8. 奖励函数详解](#8-奖励函数详解)
- [9. 网络架构详解](#9-网络架构详解)
- [10. 运动参考源详解](#10-运动参考源详解)
- [11. 机器人模型详解](#11-机器人模型详解)
- [12. 可视化与调试工具](#12-可视化与调试工具)
- [13. 开发指南：如何修改和扩展](#13-开发指南如何修改和扩展)
- [14. 常见问题](#14-常见问题)

---

## 1. 项目概述

### 核心思路

TWIST 的目标是让人形机器人模仿参考动作（来自 MoCap 数据或神经网络生成器），实现自然的行走、转弯、起步和停止等行为。

整体流程：

```
参考动作（MoCap / CMG）──→ Teacher 策略（PPO，有特权信息）──→ Student 策略（DAgger 蒸馏，仅本体感知）──→ 导出 JIT ──→ Sim2Sim / Sim2Real
```

### 两条训练管线

| 管线 | 动作来源 | 适用场景 | 是否需要 MoCap 数据 |
|------|----------|----------|-------------------|
| **Mocap 管线** | 预录制的动捕 PKL 文件 | 特定动作模仿（挥手、走路等） | 是 |
| **CMG 管线** | 神经网络运动生成器 (Conditional Motion Generator) | 速度指令控制的行走 | 否（使用预训练 CMG 模型） |

### 训练阶段

1. **Teacher 训练**：策略能看到"特权信息"（真实全局速度、地形高度等），用 PPO 优化。相当于一个"作弊"的全知策略。
2. **Student 蒸馏**：策略只能看到"本体感知信息"（关节角度、角速度、IMU 等），通过 DAgger（模仿学习+强化学习混合）从 Teacher 学习。这是最终要部署的策略。

---

## 2. 仓库目录结构

```
TWIST/
│
├── legged_gym/                    # 🏗️ Isaac Gym 环境（训练核心）
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── base/
│   │   │   │   ├── humanoid_mimic.py      # ⭐ 主环境类（奖励函数在这里）
│   │   │   │   ├── humanoid_mimic_config.py # 环境基础配置
│   │   │   │   ├── humanoid_char.py       # 角色环境（观测、重置逻辑）
│   │   │   │   ├── base_task.py           # 任务基类（Isaac Gym 接口）
│   │   │   │   └── legged_robot.py        # 四足基类（共享工具方法）
│   │   │   └── g1/
│   │   │       ├── g1_mimic_distill_config.py  # ⭐ 所有 G1 训练配置（核心文件）
│   │   │       └── g1_mimic_distill.py    # G1 任务实现
│   │   ├── scripts/
│   │   │   ├── train.py                   # ⭐ 训练入口脚本
│   │   │   ├── play.py                    # 回放/评估脚本
│   │   │   ├── save_jit_stu_rlbc.py       # Mocap Student → JIT 导出
│   │   │   └── save_jit_cmg_stu_v2.py     # CMG Student V2 → JIT 导出
│   │   ├── gym_utils/
│   │   │   ├── task_registry.py           # 任务注册系统
│   │   │   ├── terrain.py                 # 地形生成
│   │   │   ├── math.py                    # 数学工具
│   │   │   └── helpers.py                 # 参数解析等辅助函数
│   │   └── motion_data_configs/
│   │       └── twist_dataset.yaml         # 动捕数据集配置
│   └── logs/                              # 训练日志与 checkpoint 输出目录
│
├── rsl_rl/                        # 🧠 RL 算法库
│   └── rsl_rl/
│       ├── algorithms/
│       │   ├── ppo.py                     # PPO 算法实现
│       │   └── dagger_ppo.py              # DAgger + PPO 混合算法
│       ├── modules/
│       │   ├── actor_critic_mimic.py      # ⭐ 策略网络（MotionEncoder + Actor + Critic）
│       │   └── dagger_actor.py            # DAgger Student Actor
│       ├── runners/
│       │   ├── on_policy_runner_mimic.py  # ⭐ Teacher 训练循环
│       │   └── on_policy_dagger_runner.py # Student DAgger 训练循环
│       ├── storage/
│       │   └── rollout_storage.py         # 经验回放缓冲
│       └── utils/
│           └── normalizer.py              # ⭐ 观测归一化器（JIT 导出必须烘焙）
│
├── pose/                          # 🦴 运动库与运动学
│   └── pose/
│       └── utils/
│           ├── cmg_motion_lib.py          # ⭐ CMG 运动生成器接口（核心）
│           ├── motion_lib_pkl.py          # MoCap PKL 动作加载器
│           ├── forward_kinematics.py      # 前向运动学
│           └── torch_utils.py             # 四元数/旋转工具
│
├── cmg_workspace/                 # 🤖 CMG 神经运动生成器
│   ├── module/
│   │   ├── cmg.py                         # CMG 网络结构
│   │   ├── moe_layer.py                   # Mixture of Experts 层
│   │   └── gating_network.py              # 门控网络
│   ├── dataloader/
│   │   └── cmg_training_data.pt           # CMG 训练数据（322MB）
│   ├── runs/
│   │   └── cmg_20260123_194851/
│   │       └── cmg_final.pt               # 预训练 CMG 模型 checkpoint
│   ├── train.py                           # CMG 训练脚本
│   └── eval_cmg.py                        # CMG 评估脚本
│
├── deploy_real/                   # 🚀 部署（Sim2Sim & Sim2Real）
│   ├── sim2sim_cmg_stu_v2.py             # ⭐ MuJoCo 仿真验证（CMG Student V2）
│   ├── deploy_real_cmg_stu_v2.py         # ⭐ 实机部署（CMG Student V2）
│   ├── server_low_level_g1_sim.py        # 低层仿真服务（Redis，旧管线）
│   ├── server_low_level_g1_real.py       # 低层实机服务（Redis，旧管线）
│   ├── server_high_level_motion_lib.py   # 高层动作播放服务（Redis，旧管线）
│   └── robot_control/                    # 机器人控制封装
│       ├── g1_wrapper.py                  # G1 SDK 封装
│       └── config.py                      # 控制参数
│
├── assets/                        # 🤖 机器人模型文件
│   └── g1/
│       ├── g1_custom_collision_with_fixed_hand.urdf  # ⭐ 当前使用的 23-DOF URDF
│       ├── g1_sim2sim_with_wrist_roll.xml            # MuJoCo sim2sim 场景
│       └── meshes/                                   # 机器人 mesh 文件
│
├── tools/                         # 🔧 可视化与测试工具
│   ├── vis_cmg_ramp.py                    # MuJoCo 可视化 ramp 曲线
│   ├── vis_random_ramp.py                 # 随机 ramp 对比可视化
│   ├── vis_cmg_commands.py                # 实时 command+joint 曲线绘制
│   └── test_cmg_velocity.py              # CMG 速度标定诊断
│
├── tests/                         # 🧪 集成测试
│   └── smoke_test_cmg_integration.py     # CMG 冒烟测试
│
├── docs/                          # 📚 文档
│   ├── cmg_integration.md                 # CMG 集成指南
│   └── CHANGELOG_CMG_Integration.md       # CMG 变更日志
│
├── train_teacher.sh               # Teacher 训练启动脚本（Mocap）
├── train_student.sh               # Student 训练启动脚本（Mocap）
├── train_teacher_cmg.sh           # CMG Teacher 训练启动脚本
├── train_student_cmg_v2.sh        # CMG Student V2 训练启动脚本
├── to_jit.sh                      # Mocap Student → JIT 导出脚本
├── play_teacher.sh                # Teacher 回放脚本
├── play_student.sh                # Student 回放脚本
├── setup.sh                       # 一键环境安装脚本
└── CLAUDE.md                      # Claude Code 项目说明
```

---

## 3. 环境安装

### 硬件要求

- **GPU**：NVIDIA RTX 4090（24GB 显存）或更好
- **训练时间**：单卡约 1~2 天
- **操作系统**：Ubuntu 20.04 / 22.04

### 步骤 1：创建 Conda 环境

```bash
conda create -n twist python=3.8
conda activate twist
```

### 步骤 2：安装 Isaac Gym

从 [NVIDIA 官方](https://developer.nvidia.com/isaac-gym) 下载 Isaac Gym Preview 4，然后：

```bash
cd isaacgym/python && pip install -e .
```

### 步骤 3：安装项目包

```bash
# 安装三个核心包（必须按此顺序）
cd rsl_rl && pip install -e . && cd ..
cd legged_gym && pip install -e . && cd ..
cd pose && pip install -e . && cd ..

# 安装其他依赖
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill \
    gdown hydra-core "imageio[ffmpeg]" mujoco mujoco-python-viewer isaacgym-stubs \
    pytorch-kinematics rich termcolor
pip install "redis[hiredis]"
pip install pyttsx3  # 语音控制（可选）
```

### 步骤 4：启动 Redis（部署用）

```bash
redis-server --daemonize yes
```

### 步骤 5：下载 MoCap 数据集（仅 Mocap 管线需要）

从 [Google Drive](https://drive.google.com/file/d/1bRAGwRAJ3qZV94IBIyuu4cySqZM95XBi/view?usp=sharing) 下载后解压，并修改 `legged_gym/motion_data_configs/twist_dataset.yaml` 中的 `root_path` 指向解压目录。

### 步骤 6：安装实机 SDK（仅 Sim2Real 需要）

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
```

### 验证安装

```python
python -c "import isaacgym; import torch; print('CUDA:', torch.cuda.is_available())"
```

### 一键安装（云实例）

如果你在新的云实例上，可以使用自动化脚本：

```bash
bash setup.sh
# 或自定义 Isaac Gym 路径：
ISAAC_TAR=/path/to/IsaacGym_Preview_4_Package.tar.gz bash setup.sh
```

---

## 4. 核心概念

### 4.1 Teacher-Student 两阶段学习

```
┌────────────────────────────────────────────────────┐
│  阶段 1：Teacher 训练 (PPO)                          │
│                                                      │
│  观测 = 运动参考帧 + 本体感知 + 特权信息              │
│         (1160 dims)   (74 dims)  (84 dims)           │
│                                                      │
│  特权信息包括：                                       │
│    - 真实基座线速度 (3)     ← 部署时拿不到             │
│    - 真实根部高度 (1)       ← 部署时拿不到             │
│    - 关键身体点位置 (27)    ← 部署时拿不到             │
│    - 接触掩码 (2)          ← 部署时拿不到             │
│    - 域随机化参数 (51)     ← 部署时拿不到             │
│                                                      │
│  策略输出 → 23 个关节目标角度（PD 控制器执行）         │
└────────────────────────────────────────────────────┘
                        │
                        ▼ DAgger 蒸馏
┌────────────────────────────────────────────────────┐
│  阶段 2：Student 训练 (DAgger = RL + BC)             │
│                                                      │
│  观测 = 运动参考帧 + 本体感知（无特权信息）            │
│         (CMG V2: 1160 + 77 = 1237 dims)             │
│                                                      │
│  训练目标：                                          │
│    - RL 奖励（和 Teacher 一样）                       │
│    - 模仿 Teacher 输出（BC loss）                     │
│                                                      │
│  DAgger 系数 0.1 → 0.01 退火（逐步减少对 Teacher 依赖）│
└────────────────────────────────────────────────────┘
                        │
                        ▼ TorchScript 导出
┌────────────────────────────────────────────────────┐
│  部署：JIT 推理 (50 Hz)                              │
│                                                      │
│  输入：实时本体感知传感器数据 + CMG 生成的运动参考      │
│  输出：23 个关节目标角度                              │
│  控制频率：50 Hz（每步 0.02s）                        │
└────────────────────────────────────────────────────┘
```

### 4.2 观测空间分解

**Teacher 观测 (1321 dims, CMG 模式)：**

| 组件 | 维度 | 内容 |
|------|------|------|
| `priv_mimic_obs` | 1160 | 20 个未来参考帧 × 58 dims/帧 |
| `proprio` (本体感知) | 77 | 角速度(3) + IMU(2) + 关节角(23) + 关节速度(23) + 上步动作(23) + 速度指令(3) |
| `priv_info` (特权) | 84 | 基座速度(3) + 根高度(1) + 关键体位置(27) + 接触(2) + DR参数(51) |
| **合计** | **1321** | |

**CMG Student V2 观测 (1237 dims)：**

| 组件 | 维度 | 内容 |
|------|------|------|
| `priv_mimic_obs` | 1160 | 与 Teacher 相同的 20 步运动参考 |
| `proprio` | 77 | 与 Teacher 相同的本体感知 |
| **合计** | **1237** | **去掉了 84 维特权信息** |

**单个参考帧 (58 dims)：**
- 根部四元数 + 高度 (8 dims)
- 23 DOF 关节角度 (23 dims)
- 9 个关键身体点 × 3D 位置 (27 dims)

### 4.3 动作空间

- **维度**：23（对应 G1 机器人 23 个 DOF）
- **类型**：关节目标角度偏移量
- **缩放**：`action_scale = 0.5`
- **控制方式**：PD 控制器
  - 力矩 = Kp × (目标角 - 当前角) + Kd × (0 - 当前角速度)
  - 目标角 = 默认关节角 + action × action_scale

---

## 5. 训练流程详解

### 5.1 Mocap 管线

#### 训练 Teacher

```bash
conda activate twist
bash train_teacher.sh <实验名> <GPU设备>

# 例如：
bash train_teacher.sh my_teacher_v1 cuda:0
```

**等价命令：**
```bash
cd legged_gym/legged_gym/scripts
python train.py --task g1_priv_mimic \
                --proj_name g1_priv_mimic \
                --exptid my_teacher_v1 \
                --device cuda:0
```

**参数说明：**
- `--task`：注册的任务名（见[任务注册表](#54-任务注册表)）
- `--proj_name`：W&B 项目名，也决定 log 子目录
- `--exptid`：实验 ID，用于区分不同实验
- `--device`：GPU 设备
- `--debug`：调试模式（1 个环境，开可视化，关 W&B）
- `--resume`：从上次 checkpoint 恢复训练
- `--resumeid xxx`：恢复指定实验 ID 的训练
- `--no_wandb`：关闭 W&B 日志
- `--headless`：无头模式（默认开启）

**输出目录：**
```
legged_gym/logs/g1_priv_mimic/<exptid>/
├── model_500.pt          # 第 500 步 checkpoint
├── model_1000.pt         # 第 1000 步 checkpoint
├── ...
├── model_30000.pt        # 最终 checkpoint
└── config.json           # 训练配置记录
```

**默认参数：**
- `num_envs = 4096`（并行环境数）
- `max_iterations = 30002`
- `save_interval = 500`（每 500 步保存）
- `episode_length_s = 10`（每 episode 10 秒）
- `sim dt = 0.002`（500 Hz 物理仿真）
- `decimation = 10`（策略 50 Hz）

#### 训练 Student

```bash
bash train_student.sh <student实验名> <teacher实验名> <GPU设备>

# 例如（teacher_exptid 必须和上面训练的一致）：
bash train_student.sh my_student_v1 my_teacher_v1 cuda:0
```

**等价命令：**
```bash
cd legged_gym/legged_gym/scripts
python train.py --task g1_stu_rl \
                --proj_name g1_stu_rl \
                --exptid my_student_v1 \
                --teacher_exptid my_teacher_v1 \
                --device cuda:0
```

### 5.2 CMG 管线

#### 训练 CMG Teacher

```bash
bash train_teacher_cmg.sh <速度模式> <实验名> <GPU设备>

# 速度模式：slow (~1m/s) | medium (~2m/s) | fast (~3m/s)
# 例如：
bash train_teacher_cmg.sh medium cmg_teacher_v1 cuda:0
```

**速度范围配置：**

| 模式 | vx 范围 | vy 范围 | yaw 范围 |
|------|---------|---------|----------|
| `slow` | [0.5, 1.5] m/s | [-0.3, 0.3] m/s | [-0.5, 0.5] rad/s |
| `medium` | [1.5, 2.5] m/s | [-0.5, 0.5] m/s | [-0.5, 0.5] rad/s |
| `fast` | [2.5, 3.5] m/s | [-0.5, 0.5] m/s | [-0.5, 0.5] rad/s |

#### 训练 CMG Student V2（推荐）

```bash
bash train_student_cmg_v2.sh <student实验名> <teacher实验名> <GPU设备> [teacher_checkpoint]

# 例如：
bash train_student_cmg_v2.sh cmg_stu_v4 global_obs_v4 cuda:0

# 恢复训练：
bash train_student_cmg_v2.sh cmg_stu_v4 global_obs_v4 cuda:0 -1 \
    --resumeid cmg_stu_v4 --max_iterations 20001
```

**Student V2 关键特性：**
- 与 Teacher 共享完整的 20 步未来参考（1160 dims）
- 额外的本体感知包含速度指令 [vx, vy, yaw_rate]（+3 dims）
- 去掉了 84 维特权信息（Teacher 独有）
- 总观测：1237 = 1160 + 77

### 5.3 回放与评估

```bash
# 回放 Teacher 策略
bash play_teacher.sh <exptid>

# 回放 Student 策略
bash play_student.sh <exptid>

# 或直接调用：
cd legged_gym/legged_gym/scripts
python play.py --task g1_priv_mimic --exptid my_teacher_v1
```

### 5.4 任务注册表

所有任务注册在 `legged_gym/envs/__init__.py`：

| 任务名 | 配置类 | 算法 | 用途 |
|--------|--------|------|------|
| `g1_priv_mimic` | `G1MimicPrivCfg` + `G1MimicPrivCfgPPO` | PPO | Mocap Teacher |
| `g1_stu_rl` | `G1MimicStuRLCfg` + `G1MimicStuRLCfgDAgger` | DAgger | Mocap Student |
| `g1_cmg_slow` | `G1MimicCMGSlowCfg` + `G1MimicCMGSlowCfgPPO` | PPO | CMG Teacher 慢速 |
| `g1_cmg_medium` | `G1MimicCMGMediumCfg` + `G1MimicCMGMediumCfgPPO` | PPO | CMG Teacher 中速 |
| `g1_cmg_fast` | `G1MimicCMGFastCfg` + `G1MimicCMGFastCfgPPO` | PPO | CMG Teacher 快速 |
| `g1_cmg_stu_rl` | `G1MimicCMGStuRLCfg` + `G1MimicCMGStuRLCfgDAgger` | DAgger | CMG Student V1 |
| `g1_cmg_stu_v2` | `G1MimicCMGStuV2Cfg` + `G1MimicCMGStuV2CfgDAgger` | DAgger | CMG Student V2（推荐）|

### 5.5 W&B 日志

训练会自动上传到 Weights & Biases（需要先 `wandb login`）。关闭日志可用 `--no_wandb`。

日志记录的指标包括：
- 各奖励分量
- Episode 长度和总奖励
- 关节跟踪误差
- 速度跟踪误差（CMG）
- 学习率和策略标准差

---

## 6. 模型导出与部署

### 6.1 导出 JIT 模型

#### Mocap Student 导出

```bash
bash to_jit.sh <student_exptid>

# 例如：
bash to_jit.sh my_student_v1
```

输出路径：`legged_gym/logs/g1_stu_rl/<exptid>/traced/<exptid>-<step>-jit.pt`

#### CMG Student V2 导出

```bash
cd legged_gym/legged_gym/scripts
python save_jit_cmg_stu_v2.py --exptid cmg_stu_v4 --device cpu

# 可选参数：
#   --checkpoint -1     # -1 表示最新
#   --proj_name g1_cmg_stu_v2
```

输出路径：`legged_gym/logs/g1_cmg_stu_v2/<exptid>/traced/<exptid>-<step>-jit.pt`

**导出要点**：
- 必须将 Normalizer 的统计量（`_mean`, `_std`, `_eps`, `_clip`）烘焙进 JIT 模型
- 注意属性名是 `_mean`/`_std`，不是 `mean`/`var`

### 6.2 Sim2Sim 验证

#### CMG Student V2（推荐，无需 Redis）

```bash
cd deploy_real
python sim2sim_cmg_stu_v2.py \
    --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \
    --cmd_vx 1.5 \
    --duration 30.0 \
    --device cuda
```

**键盘控制（MuJoCo 窗口中）：**

| 键 | 功能 |
|----|------|
| ↑/↓ | 增加/减少前进速度 (vx) |
| ←/→ | 增加/减少偏航角速度 (yaw) |
| Q/E | 增加/减少侧向速度 (vy) |
| R | 重置 |

#### 旧管线 Sim2Sim（需要 Redis）

```bash
cd deploy_real

# 终端 1：启动低层仿真控制器
python server_low_level_g1_sim.py --policy_path PATH/TO/model.pt

# 终端 2：启动高层动作播放器
python server_high_level_motion_lib.py --motion_file PATH/TO/motion.pkl --vis
```

### 6.3 Sim2Real 实机部署

#### 前置步骤

1. 启动 G1 机器人
2. 以太网连接笔记本和机器人
3. 设置网络接口 IP 为 `192.168.123.222`，子网掩码 `255.255.255.0`
4. 验证连通性：`ping 192.168.123.164`
5. 遥控器按 `L2+R2` 进入开发模式

#### CMG Student V2 实机部署

```bash
cd deploy_real
python deploy_real_cmg_stu_v2.py \
    --policy_path ../legged_gym/logs/g1_cmg_stu_v2/cmg_stu_v4/traced/cmg_stu_v4-21000-jit.pt \
    --cmd_vx 1.0 \
    --net eno1  # 你的网络接口名
```

**手柄操作流程：**
1. `[START]` → 零力矩模式
2. 等待自动回到默认站姿
3. `[A]` → 保持（hold）
4. 再次 `[A]` → 开始执行策略
5. `[Select]` → 安全退出

#### 旧管线实机部署

```bash
cd deploy_real

# 终端 1：低层控制器
python server_low_level_g1_real.py --policy_path PATH/TO/model.pt --net eno1

# 终端 2：动作播放器
python server_high_level_motion_lib.py --motion_file PATH/TO/motion.pkl --vis
```

### 6.4 部署关键常量

部署代码中必须与训练配置一致的常量（来自 `G1MimicCMGStuV2Cfg`）：

| 常量 | 值 | 含义 |
|------|----|------|
| `NUM_OBS` | 1237 | 观测维度 (1160 + 77) |
| `ACTION_SCALE` | 0.5 | 动作缩放系数 |
| `DECIMATION` | 20 | 控制频率 = 仿真频率/20 |
| `CMG_DT` | 0.02 | CMG 运行频率 50 Hz |
| `DEFAULT_DOF_POS` | 见下 | 默认站姿关节角 |

**默认关节角度：**
```python
# 左腿                     右腿
[-0.2, 0, 0, 0.4, -0.2, 0,  -0.2, 0, 0, 0.4, -0.2, 0,
# 腰部
 0, 0, 0,
# 左臂              右臂
 0, 0.4, 0, 1.2,    0, -0.4, 0, 1.2]
```

---

## 7. 配置系统详解

所有训练配置集中在一个文件：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`

### 7.1 配置类继承关系

```
HumanoidMimicCfg                    # 通用人形模仿基类
└── G1MimicPrivCfg                  # G1 Teacher (Mocap)
    ├── G1MimicStuRLCfg             # G1 Student (Mocap)
    │   └── G1MimicStuRLCfgDAgger   # + DAgger 算法参数
    └── G1MimicCMGBaseCfg           # G1 CMG Teacher 基类
        ├── G1MimicCMGSlowCfg       # 慢速 (1 m/s)
        ├── G1MimicCMGMediumCfg     # 中速 (2 m/s)
        ├── G1MimicCMGFastCfg       # 快速 (3 m/s)
        ├── G1MimicCMGStuRLCfg      # CMG Student V1
        └── G1MimicCMGStuV2Cfg      # CMG Student V2（推荐）
```

### 7.2 env 配置子类

```python
class env:
    num_envs = 4096              # 并行环境数（CMG 管线为 1024）
    num_actions = 23             # G1 机器人 DOF 数
    episode_length_s = 10        # episode 长度（CMG 为 16.5s）

    # 观测类型
    obs_type = 'priv'            # 'priv'=Teacher, 'student'=Student

    # 观测维度分解
    n_proprio = 74               # 本体感知维度（CMG: 77, +3 速度指令）
    n_priv_mimic_obs = 1160      # 运动参考帧维度 (20步 × 58)
    n_priv_info = 84             # 特权信息维度
    n_obs_single = 1318          # 单步观测总维度（CMG Teacher: 1321）

    # 参考帧步数（采样未来帧的时间步索引）
    tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                     50, 55, 60, 65, 70, 75, 80, 85, 90, 95]  # 20 步

    # 关节跟踪权重 (23 DOF)
    dof_err_w = [1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # 左腿
                 1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # 右腿
                 0.6, 0.6, 0.6,                    # 腰部
                 0.8, 0.8, 0.8, 1.0,              # 左臂
                 0.8, 0.8, 0.8, 1.0]              # 右臂

    # 终止条件
    enable_early_termination = True
    pose_termination = True
    pose_termination_dist = 0.7  # 关键体位偏差阈值

    # 标准化
    normalize_obs = True

    # CMG 专用
    use_cmd_obs = True           # 将速度指令加入观测
    n_cmd = 3                    # 指令维度 [vx, vy, yaw_rate]
```

### 7.3 terrain 配置子类

```python
class terrain:
    mesh_type = 'trimesh'        # 'trimesh'=起伏地形, 'plane'=平地
    # CMG 模式使用 'plane' + 重力随机化模拟坡度
    height = [0, 0.00]           # 地形高度范围
    horizontal_scale = 0.1       # 地形分辨率
```

### 7.4 control 配置子类

```python
class control:
    # PD 控制器增益（按关节组）
    stiffness = {
        'hip_yaw': 100, 'hip_roll': 100, 'hip_pitch': 100,
        'knee': 150, 'ankle': 40,
        'waist': 150,
        'shoulder': 40, 'elbow': 40,
    }
    damping = {
        'hip_yaw': 2, 'hip_roll': 2, 'hip_pitch': 2,
        'knee': 4, 'ankle': 2,
        'waist': 4,
        'shoulder': 5, 'elbow': 5,
    }
    action_scale = 0.5           # 网络输出 × 0.5 = 目标角偏移
    decimation = 10              # 策略执行频率 = 500/10 = 50 Hz
```

### 7.5 sim 配置子类

```python
class sim:
    dt = 0.002                   # 物理仿真步长 (1/500 s)
    # 策略频率 = 1 / (dt × decimation) = 1 / (0.002 × 10) = 50 Hz
```

### 7.6 domain_rand 配置子类

```python
class domain_rand:
    domain_rand_general = True   # 总开关

    randomize_gravity = True     # 重力方向随机
    gravity_range = (-0.1, 0.1)  # CMG 模式: (-0.86, 0.86) ≈ ±5° 坡度
    gravity_rand_interval_s = 4  # 每 4 秒重新采样

    randomize_friction = True
    friction_range = [0.1, 2.0]  # 摩擦系数范围

    randomize_base_mass = True
    added_mass_range = [-3., 3]  # 额外质量 (kg)

    randomize_base_com = True
    added_com_range = [-0.05, 0.05]  # 质心偏移 (m)

    push_robots = True           # 随机推力扰动
    max_push_vel_xy = 1.0        # 最大推力速度
    push_interval_s = 4

    push_end_effector = True     # 末端执行器推力
    max_push_force_end_effector = 20.0

    randomize_motor = True
    motor_strength_range = [0.8, 1.2]  # 电机强度随机化

    action_delay = True          # 动作延迟
    action_buf_len = 8           # 延迟缓冲区长度
```

### 7.7 noise 配置子类

```python
class noise:
    add_noise = True
    noise_increasing_steps = 3000  # 前 3000 步逐步增加噪声
    class noise_scales:
        dof_pos = 0.01           # 关节角度噪声 (rad)
        dof_vel = 0.1            # 关节速度噪声
        lin_vel = 0.1            # 线速度噪声
        ang_vel = 0.1            # 角速度噪声
        gravity = 0.05           # 重力方向噪声
        imu = 0.1                # IMU 噪声
```

### 7.8 motion 配置子类

```python
class motion:
    # Mocap 模式
    motion_file = "twist_dataset.yaml"
    motion_curriculum = True     # 运动难度课程学习
    key_bodies = [               # 9 个关键追踪身体
        "left_rubber_hand", "right_rubber_hand",
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_knee_link", "right_knee_link",
        "left_elbow_link", "right_elbow_link",
        "head_mocap"
    ]

    # CMG 模式专用
    use_cmg = True
    cmg_model_path = "cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"
    cmg_data_path = "cmg_workspace/dataloader/cmg_training_data.pt"
    cmg_dt = 0.02                # CMG 运行频率 50 Hz

    # 速度范围（按速度模式不同）
    cmg_vx_range = [0.5, 1.5]   # 前向速度范围 (m/s)
    cmg_vy_range = [-0.3, 0.3]  # 侧向速度范围 (m/s)
    cmg_yaw_range = [-0.5, 0.5] # 偏航角速度范围 (rad/s)

    # Ramp 速度曲线参数
    cmg_ramp_enabled = True
    cmg_ramp_up_range = [1.0, 3.0]    # 加速阶段时长 (s)
    cmg_ramp_down_range = [1.5, 4.0]  # 减速阶段时长 (s)
    cmg_ramp_stand_duration = 5.0     # 首尾站立时长 (s)
    cmg_ramp_crawl_range = [0.5, 1.5] # 爬行阶段时长 (s)
    cmg_ramp_crawl_ratio = 0.01       # 爬行速度 = 目标 × 0.01
    cmg_ramp_floor_ratio = 0.1        # 站立速度 = 目标 × 0.1
    cmg_ramp_probability = 1.0        # 100% episode 使用 ramp
    cmg_ramp_min_steady = 3.0         # 最小稳态时长 (s)
```

### 7.9 policy 配置子类（PPO/DAgger）

```python
class policy:
    actor_hidden_dims = [512, 512, 256, 128]   # Actor MLP 层
    critic_hidden_dims = [512, 512, 256, 128]  # Critic MLP 层
    activation = 'silu'          # 激活函数
    motion_latent_dim = 128      # 运动编码器输出维度
    layer_norm = True            # 使用 LayerNorm
    obs_context_len = 11         # 观测历史长度（Student）

    # 初始动作噪声标准差
    action_std = [0.7]*12 + [0.4]*3 + [0.5]*8  # 腿/腰/臂
    init_noise_std = 1.0

class runner:
    max_iterations = 30002       # 最大训练步数
    save_interval = 500          # 每 500 步保存
    policy_class_name = 'ActorCriticMimic'
    algorithm_class_name = 'PPO'         # Teacher: PPO
    # algorithm_class_name = 'DaggerPPO'  # Student: DAgger
    runner_class_name = 'OnPolicyRunnerMimic'

class algorithm:
    # DAgger 专用参数
    dagger_coef = 0.1            # 初始 Teacher 模仿权重
    dagger_coef_min = 0.01       # 退火后的最小值
    dagger_coef_anneal_steps = 60000  # 退火步数
```

---

## 8. 奖励函数详解

奖励函数定义在 `humanoid_mimic.py` 中，权重在 `g1_mimic_distill_config.py` 的 `rewards.scales` 中配置。

### 8.1 核心跟踪奖励

| 奖励函数 | Mocap 权重 | CMG 权重 | 公式 | 含义 |
|---------|-----------|---------|------|------|
| `tracking_joint_dof` | 0.6 | 0.6 | exp(-0.15 × Σ(w_i × Δθ²)) | 关节角度跟踪 |
| `tracking_joint_vel` | 0.2 | 0.2 | exp(-0.01 × Σ(w_i × Δω²)) | 关节速度跟踪 |
| `tracking_root_pose` | 0.6 | 0.0 | exp(-5 × (pos_err + 0.1×rot_err)) | 根部位姿跟踪 |
| `tracking_root_vel` | 1.0 | 0.0 | exp(-1 × (vel_err + 0.5×angvel_err)) | 根部速度跟踪 |
| `tracking_keybody_pos` | 2.0 | 2.0 | exp(-10 × Σ pos_err²) | 关键身体点跟踪 |

**CMG 模式特殊行为：**
- `tracking_keybody_pos` 只跟踪下半身（脚踝+膝盖），因为上半身需要自由平衡
- `tracking_root_pose` 和 `tracking_root_vel` 权重为 0，因为根部位置由速度指令主导

### 8.2 CMG 专用奖励

| 奖励函数 | 权重 | 公式 | 含义 |
|---------|------|------|------|
| `tracking_keybody_pos_upper` | 0.3 | exp(-5 × Σ pos_err²) | 弱上半身跟踪（手、肘、头），exp_scale=5 vs 下半身 10 |
| `tracking_cmd_vel` | 1.5 | exp(-σ × ‖v_robot - v_cmd‖²) | 前向/侧向速度跟踪（对比原始指令，非运动学估计） |
| `tracking_cmd_yaw` | 1.0 | exp(-σ × (yaw_robot - yaw_cmd)²) | 偏航速率跟踪 |
| `action_symmetry` | 0.1 | 左右动作对称性惩罚 | 鼓励左右对称的步态 |

### 8.3 正则化奖励（惩罚项，权重为负）

| 奖励函数 | 权重 | 含义 |
|---------|------|------|
| `feet_slip` | -0.1 | 脚接触地面时滑动惩罚 |
| `feet_contact_forces` | -5e-4 | 过大接触力惩罚（>100N） |
| `feet_stumble` | -1.25 | 脚绊倒惩罚 |
| `dof_pos_limits` | -5.0 | 关节超限惩罚 |
| `dof_torque_limits` | -1.0 | 力矩超限惩罚 |
| `dof_vel` | -1e-4 | 关节速度惩罚 |
| `dof_acc` | -5e-8 | 关节加速度惩罚 |
| `action_rate` | -0.01 | 动作变化率惩罚 |
| `feet_air_time` | 5.0 | 足部腾空时间奖励 |
| `ang_vel_xy` | -0.01 | 横滚/俯仰角速度惩罚 |

### 8.4 奖励计算流程

```
每步仿真:
  1. 计算各奖励分量 r_i = reward_func_i()
  2. 乘以权重 scaled_r_i = scale_i × r_i
  3. 总奖励 R = Σ scaled_r_i
  4. PPO 用 R 更新策略
```

---

## 9. 网络架构详解

### 9.1 ActorCriticMimic

文件：`rsl_rl/rsl_rl/modules/actor_critic_mimic.py`

```
输入观测（以 CMG Teacher 为例，1321 dims）
│
├─ priv_mimic_obs (1160 dims)                   ─→ MotionEncoder ─→ 128D latent
│   └─ 20 步 × (root_quat_h(8) + dof(23) + keybody(27))
│
├─ proprio (77 dims)                             ─→ 直接拼接
│   └─ ang_vel(3) + imu(2) + dof_pos(23)
│      + dof_vel(23) + last_action(23) + cmd(3)
│
└─ priv_info (84 dims, 仅 Teacher)              ─→ 直接拼接

            ┌────────────────────┐
            │   Actor 输入拼接    │
            │                    │
            │ motion_latent(128) │
            │ + first_frame(58)  │  ← 第一帧单独再传一次，强调当前帧
            │ + proprio(77)      │
            │ + priv_info(84)    │  ← Student 没有这部分
            │                    │
            │ = 347 dims         │
            └─────────┬──────────┘
                      │
                      ▼
            ┌────────────────────┐
            │   MLP Actor        │
            │ 347 → 512 (SiLU)  │
            │ 512 → 512 (SiLU)  │
            │ 512 → 256 (SiLU)  │
            │ 256 → 128 (SiLU + LN) │
            │ 128 → 23          │
            └─────────┬──────────┘
                      │
                      ▼
            23 维动作（关节角度偏移）
```

### 9.2 MotionEncoder

将 20 步未来参考帧编码为 128 维 latent：

```python
# 输入: (batch, 20, 58)
Linear(58 → 60)  + ELU           # 投影到通道维度
Conv1D(60 → 40, k=6, s=2) + ELU  # 时间下采样
Conv1D(40 → 20, k=4, s=2) + ELU  # 进一步压缩
Flatten()
Linear(60 → 128)                  # 输出 128D latent
```

### 9.3 Critic 网络

与 Actor 结构相同但独立参数，输出标量值函数。Teacher 训练时 Critic 还能看到额外 3 维信息（`extra_critic_obs`）。

### 9.4 Normalizer

文件：`rsl_rl/rsl_rl/utils/normalizer.py`

训练中对观测进行在线归一化：
```python
normalized_obs = clip((obs - running_mean) / running_std, -clip_val, clip_val)
```

关键属性（导出 JIT 时必须使用这些名字）：
- `_mean`：运行均值
- `_std`：运行标准差
- `_eps`：防除零（默认 1e-5）
- `_clip`：裁剪值（默认 5.0）

---

## 10. 运动参考源详解

### 10.1 Mocap MotionLib

文件：`pose/pose/utils/motion_lib_pkl.py`

从预录制的 PKL 文件加载动捕数据：
```
twist_dataset.yaml → 列出所有 PKL 文件路径和采样权重
                   → MotionLib 加载并提供随机采样接口
```

每个 PKL 文件包含：
- 根部位置/旋转轨迹
- 关节角度轨迹
- 关键身体点位置轨迹
- 帧率信息

### 10.2 CMG (Conditional Motion Generator)

文件：`pose/pose/utils/cmg_motion_lib.py`

CMG 是一个预训练的神经网络，根据速度指令实时生成运动参考：

```
速度指令 [vx, vy, yaw] ──→ CMG 模型 ──→ 29-DOF 关节轨迹
                              │
                              ├─ 自回归生成（50 Hz）
                              ├─ 100 帧前瞻缓冲（2 秒）
                              └─ 根部位置通过 Euler 积分
```

**关键特性：**

**a) DOF 映射（29→23）：**
CMG 输出 29 DOF，G1 只有 23 DOF，需要跳过手腕关节：
```python
CMG_TO_G1_INDICES = [0-18, 22-25]  # 跳过 19,20,21 (左手腕) 和 26,27,28 (右手腕)
```

**b) 左右镜像（50% 随机）：**
每次 episode reset 时，50% 的环境会镜像 CMG 输出：
- 交换左右腿/臂关节
- 翻转 roll/yaw 关节符号
- 翻转 vy 和 yaw_rate 指令

**c) Ramp 速度曲线：**
每个 episode 使用 6 段速度曲线，训练策略的起步/停止能力：

```
速度  ^
      |     ┌──────────────────┐
 v_max|     │   ② steady       │
      |    /│                  │\
      |   / │                  │ \
      | ①/  │                  │  \③
      |──/  │                  │   \──④──
 ~0   |  ⓪  │                  │        ⑤
      └──────────────────────────────────→ 时间
        stand  ramp_up  steady  ramp_down  crawl  stand
```

**d) 轨迹缓冲区：**
- 长度：100 帧（2 秒）
- 当策略查询未来参考帧时，从缓冲区中插值获取
- 每步更新 root 位置并滑动窗口

### 10.3 CMG 模型结构

文件：`cmg_workspace/module/cmg.py`

- **输入**：速度指令 [vx, vy, yaw] + 前一步 29-DOF 输出
- **架构**：Mixture of Experts (MoE) + 门控网络
- **输出**：29-DOF 关节目标位置和速度
- **训练数据**：322MB 的运动序列张量

**运行时数据依赖（`cmg_training_data.pt`）：**

CMG 模型权重（`cmg_final.pt`）只存储网络参数。推理时 `CMGMotionLib._load_cmg_model()` 还需要加载 `cmg_training_data.pt`，从中提取：

1. **`data["stats"]`** — 归一化参数：`motion_mean`, `motion_std`, `command_min`, `command_max` 以及 `motion_dim`, `command_dim`。CMG 模型在标准化空间中运行，所有推理都需要 `_normalize_motion()` / `_denormalize_motion()` / `_normalize_command()` 进行输入输出转换。
2. **`data["samples"]`** — 训练数据中的运动样本，用于 env reset 时提供初始运动状态（`_init_samples`）。

因此**所有使用 `CMGMotionLib` 的场景**（训练、sim2sim、sim2real）都必须能访问此文件。默认路径：`cmg_workspace/dataloader/cmg_training_data.pt`。

---

## 11. 机器人模型详解

### 11.1 Unitree G1 DOF 映射

G1 使用 23 个有效自由度（固定手腕）：

```
索引    关节名                    位置        类型
───────────────────────────────────────────────────
 0     left_hip_pitch_joint       左腿        pitch
 1     left_hip_roll_joint        左腿        roll
 2     left_hip_yaw_joint         左腿        yaw
 3     left_knee_joint            左腿        pitch
 4     left_ankle_pitch_joint     左腿        pitch
 5     left_ankle_roll_joint      左腿        roll
 6     right_hip_pitch_joint      右腿        pitch
 7     right_hip_roll_joint       右腿        roll
 8     right_hip_yaw_joint        右腿        yaw
 9     right_knee_joint           右腿        pitch
10     right_ankle_pitch_joint    右腿        pitch
11     right_ankle_roll_joint     右腿        roll
12     waist_yaw_joint            腰部        yaw
13     waist_roll_joint           腰部        roll
14     waist_pitch_joint          腰部        pitch
15     left_shoulder_pitch_joint  左臂        pitch
16     left_shoulder_roll_joint   左臂        roll
17     left_shoulder_yaw_joint    左臂        yaw
18     left_elbow_joint           左臂        pitch
19     right_shoulder_pitch_joint 右臂        pitch
20     right_shoulder_roll_joint  右臂        roll
21     right_shoulder_yaw_joint   右臂        yaw
22     right_elbow_joint          右臂        pitch
```

### 11.2 关键身体点

9 个被跟踪的身体点（索引用于奖励计算和观测构建）：

| 索引 | 名称 | 分组 | 对应链接 |
|------|------|------|----------|
| 0 | left_hand | 上半身 | left_rubber_hand |
| 1 | right_hand | 上半身 | right_rubber_hand |
| 2 | left_ankle | 下半身 | left_ankle_roll_link |
| 3 | right_ankle | 下半身 | right_ankle_roll_link |
| 4 | left_knee | 下半身 | left_knee_link |
| 5 | right_knee | 下半身 | right_knee_link |
| 6 | left_elbow | 上半身 | left_elbow_link |
| 7 | right_elbow | 上半身 | right_elbow_link |
| 8 | head | 上半身 | head_mocap |

### 11.3 URDF 文件

| 文件 | DOF | 用途 |
|------|-----|------|
| `g1_custom_collision_with_fixed_hand.urdf` | 23 | **当前使用** - 固定手腕 |
| `g1_custom_collision.urdf` | 23 | 自定义碰撞体 |
| `g1_29dof_rev_1_0.urdf` | 29 | 完整 29 DOF（含手腕） |

### 11.4 MuJoCo 场景

| 文件 | 用途 |
|------|------|
| `g1_sim2sim_with_wrist_roll.xml` | **Sim2Sim 用** - 含摩擦调优 (地面 1.0, condim=4) |
| `g1_sim2sim.xml` | 旧版 Sim2Sim |
| `g1_mocap.xml` | 训练时 MoCap 可视化 |

### 11.5 PD 控制参数

| 关节组 | Kp (刚度) | Kd (阻尼) | 惯性矩 (armature) |
|--------|----------|----------|-------------------|
| hip_pitch | 100 | 2 | 0.0103 |
| hip_roll | 100 | 2 | 0.0251 |
| hip_yaw | 100 | 2 | 0.0103 |
| knee | 150 | 4 | 0.0251 |
| ankle | 40 | 2 | 0.003597 |
| waist | 150 | 4 | 0.0103 |
| shoulder | 40 | 5 | 0.003597 |
| elbow | 40 | 5 | 0.003597 |

---

## 12. 可视化与调试工具

### 12.1 MuJoCo Ramp 可视化

```bash
# 固定参数
python tools/vis_cmg_ramp.py --cmd_vx 1.5 --cmd_vy 0.3 --cmd_yaw 0.2

# 训练时的随机范围
python tools/vis_cmg_ramp.py --random
```

### 12.2 随机 Ramp 对比

```bash
# 多次 reset 查看不同随机 ramp profile
python tools/vis_random_ramp.py

# 离线渲染快照
python tools/render_random_ramp.py
```

### 12.3 实时指令曲线

```bash
python tools/vis_cmg_commands.py
# 用 matplotlib 绘制 command 和 joint 角度的实时曲线
```

### 12.4 CMG 速度标定

```bash
python tools/test_cmg_velocity.py
python tools/test_cmg_velocity.py --vx_min 0.5 --vx_max 3.0 --vx_step 0.25
```

### 12.5 CMG 镜像测试

```bash
# MuJoCo 可视化
python test_cmg_mirror.py --cmd_vx 1.5 --cmd_vy 0.3 --cmd_yaw 0.2

# 纯数值验证
python test_cmg_mirror.py --no_viewer
```

### 12.6 训练调试模式

```bash
# 启用调试（1个环境，开可视化，关 W&B）
cd legged_gym/legged_gym/scripts
python train.py --task g1_priv_mimic --exptid debug_test --device cuda:0 --debug
```

---

## 13. 开发指南：如何修改和扩展

### 13.1 添加新的奖励函数

**步骤 1**：在 `humanoid_mimic.py` 中添加奖励方法：

```python
def _reward_my_custom_reward(self):
    """你的自定义奖励。
    必须返回 shape=(num_envs,) 的 tensor。
    """
    # 例如：惩罚基座倾斜
    roll, pitch, _ = euler_from_quaternion(self.root_states[:, 3:7])
    return torch.exp(-5.0 * (roll**2 + pitch**2))
```

**步骤 2**：在配置文件中设置权重：

```python
class rewards:
    class scales:
        my_custom_reward = 0.5  # 正数=奖励，负数=惩罚
```

奖励系统会自动发现以 `_reward_` 开头的方法并与 `scales` 中同名的属性匹配。

### 13.2 修改观测空间

**核心文件**：`humanoid_mimic.py` 中的 `compute_observations()` 方法和 `_get_noise_scale_vec()` 方法。

要添加新的观测：
1. 在 `compute_observations()` 中将新观测拼接到 `obs_buf`
2. 更新配置中的维度计算（`n_proprio`, `n_obs_single`, `num_observations` 等）
3. 在 `_get_noise_scale_vec()` 中为新观测指定噪声级别
4. **同步更新部署代码中的 `NUM_OBS` 常量**

### 13.3 添加新的训练任务

**步骤 1**：在 `g1_mimic_distill_config.py` 中定义新配置：

```python
class MyNewTaskCfg(G1MimicCMGBaseCfg):
    """你的新任务配置"""
    class env(G1MimicCMGBaseCfg.env):
        num_envs = 2048  # 按需调整

    class rewards(G1MimicCMGBaseCfg.rewards):
        class scales(G1MimicCMGBaseCfg.rewards.scales):
            tracking_cmd_vel = 3.0  # 加大速度跟踪权重

class MyNewTaskCfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        max_iterations = 20000
```

**步骤 2**：在 `legged_gym/envs/__init__.py` 中注册：

```python
task_registry.register(
    "my_new_task",
    G1MimicDistill,     # 环境类
    MyNewTaskCfg,        # 环境配置
    MyNewTaskCfgPPO,     # 算法配置
)
```

**步骤 3**：训练：

```bash
cd legged_gym/legged_gym/scripts
python train.py --task my_new_task --exptid my_exp_v1 --device cuda:0
```

### 13.4 修改 CMG 速度范围

在 `g1_mimic_distill_config.py` 中修改对应速度模式的 `motion` 子类：

```python
class G1MimicCMGMediumCfg(G1MimicCMGBaseCfg):
    class motion(G1MimicCMGBaseCfg.motion):
        cmg_vx_range = [1.0, 3.0]    # 修改前向速度范围
        cmg_vy_range = [-1.0, 1.0]   # 修改侧向速度范围
        cmg_yaw_range = [-1.0, 1.0]  # 修改偏航速率范围
```

### 13.5 调整 Ramp 曲线参数

在 `G1MimicCMGBaseCfg.motion` 中修改：

```python
class motion:
    cmg_ramp_enabled = True
    cmg_ramp_up_range = [0.5, 2.0]      # 更快的加速
    cmg_ramp_down_range = [1.0, 3.0]    # 更快的减速
    cmg_ramp_stand_duration = 3.0       # 更短的站立
    cmg_ramp_crawl_range = [0.3, 1.0]   # 更短的爬行
    cmg_ramp_min_steady = 5.0           # 更长的稳态
```

### 13.6 修改策略网络架构

在配置的 `policy` 子类中修改：

```python
class policy:
    actor_hidden_dims = [1024, 512, 256, 128]  # 更大的网络
    critic_hidden_dims = [1024, 512, 256, 128]
    motion_latent_dim = 256                     # 更大的运动编码
    activation = 'elu'                          # 可选: elu, silu, relu
    layer_norm = True
```

如果要修改 MotionEncoder 的结构，需要改 `actor_critic_mimic.py` 中的 `MotionEncoder` 类。

### 13.7 添加新的域随机化

在 `humanoid_mimic.py` 或 `humanoid.py` 中的 `_apply_domain_randomization()` 方法添加逻辑，并在配置的 `domain_rand` 子类中添加对应参数。

### 13.8 训练关键超参数调节建议

| 参数 | 作用 | 调节方向 |
|------|------|----------|
| `num_envs` | 并行环境数 | 显存允许下越大越好，4096 是 Mocap 默认 |
| `action_scale` | 动作幅度 | 太大→抖动，太小→动作不到位 |
| `tracking_joint_dof` | 关节跟踪权重 | 太大→过拟合参考，太小→动作不自然 |
| `tracking_cmd_vel` | 速度跟踪权重 | 太大→速度准确但步态差，太小→速度不精确 |
| `dagger_coef` | Teacher 模仿强度 | 0.1→0.01 退火，太大→Student 过度依赖 Teacher |
| `gravity_range` | 重力扰动 | 越大→鲁棒性越好但训练更难 |
| `push_vel_xy` | 外力扰动 | 越大→抗扰能力越好 |

### 13.9 文件修改速查表

| 想要修改... | 去这个文件 |
|------------|-----------|
| 训练超参数/奖励权重 | `g1_mimic_distill_config.py` |
| 奖励函数实现 | `humanoid_mimic.py` (~line 490+) |
| 观测空间构造 | `humanoid_mimic.py` / `humanoid_char.py` |
| 网络架构 | `actor_critic_mimic.py` |
| PPO 算法 | `rsl_rl/algorithms/ppo.py` |
| DAgger 算法 | `rsl_rl/algorithms/dagger_ppo.py` |
| 训练循环 | `on_policy_runner_mimic.py` / `on_policy_dagger_runner.py` |
| CMG 运动生成 | `cmg_motion_lib.py` |
| MoCap 加载 | `motion_lib_pkl.py` |
| 域随机化 | `humanoid.py` 中的 `_push_robots()` 等方法 |
| 地形生成 | `gym_utils/terrain.py` |
| JIT 导出 | `save_jit_cmg_stu_v2.py` / `save_jit_stu_rlbc.py` |
| Sim2Sim 部署 | `sim2sim_cmg_stu_v2.py` |
| 实机部署 | `deploy_real_cmg_stu_v2.py` |
| 任务注册 | `legged_gym/envs/__init__.py` |

---

## 14. 常见问题

### Q1: CUDA out of memory

减小 `num_envs`。CMG 管线默认 1024，Mocap 管线默认 4096。调试时可用 `--debug` 自动设为 1。

### Q2: NumPy 版本不兼容

TWIST 需要 `numpy==1.23.0`。如果从 GMR 导入 PKL 文件遇到问题，参见 [issue#10](https://github.com/YanjieZe/TWIST/issues/10)。

### Q3: Redis 连接失败

```bash
redis-server --daemonize yes
redis-cli ping  # 应返回 PONG
```
注意：CMG Student V2 的 sim2sim/sim2real **不需要 Redis**。

### Q4: Isaac Gym 导入失败

确保 `LD_LIBRARY_PATH` 包含 conda 环境的 lib 目录：
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Q5: MuJoCo 渲染黑屏

确保安装了 OpenGL 相关库：
```bash
apt install libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libegl1 libglvnd0
```

### Q6: 训练不收敛

- 检查奖励曲线（W&B）中各分量是否正常
- 确认 Teacher 训练时 tracking 奖励上升
- 检查域随机化参数是否过大（先关闭 `domain_rand_general = False` 测试）
- CMG 管线确保 ramp 参数合理

### Q7: Student 效果远差于 Teacher

- 检查 DAgger 系数退火是否过快
- 确保 Teacher checkpoint 已充分训练
- 检查 Student 观测维度是否与 Teacher 匹配（去掉特权信息后）

### Q8: Sim2Sim 行为异常

- 确认 JIT 导出时 Normalizer 正确烘焙
- 确认 `ACTION_SCALE`, `NUM_OBS`, `DECIMATION` 与训练配置一致
- 检查 `DEFAULT_DOF_POS` 是否与训练配置的 `default_joint_angles` 一致
- 确认 `cmg_training_data.pt` 存在（`cmg_workspace/dataloader/` 下）—— CMG 推理需要其中的归一化统计量和初始运动样本，缺失会导致 `CMGMotionLib` 加载失败

---

## 引用

```bibtex
@article{ze2025twist,
  title={TWIST: Teleoperated Whole-Body Imitation System},
  author={Yanjie Ze and Zixuan Chen and João Pedro Araújo and Zi-ang Cao and Xue Bin Peng and Jiajun Wu and C. Karen Liu},
  year={2025},
  journal={arXiv preprint arXiv:2505.02833}
}
```

## 联系

如有问题，请联系 `yanjieze@stanford.edu` 或提交 [GitHub Issue](https://github.com/YanjieZe/TWIST/issues)。
