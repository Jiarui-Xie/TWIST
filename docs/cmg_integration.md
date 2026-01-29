# CMG 动作参考集成指南

## 概述

本文档说明如何使用 CMG (Conditional Motion Generator) 模型生成动作参考，用于 TWIST 教师模型训练。

CMG 是一个基于 Mixture-of-Experts 的神经网络，根据速度命令 (vx, vy, yaw) 自回归生成动作序列，替代传统的动作捕捉数据作为参考。

## 安装

### 1. 安装依赖

```bash
cd /path/to/TWIST
pip install -r legged_gym/requirements.txt
```

### 2. 安装项目包

```bash
pip install -e pose/
pip install -e rsl_rl/
pip install -e legged_gym/
```

### 3. 拉取 LFS 数据

CMG 训练数据文件较大，使用 Git LFS 管理：

```bash
git lfs pull
```

这会拉取 `cmg_workspace/dataloader/cmg_training_data.pt` (约 290MB)。

### 4. 确保 Isaac Gym 已安装

```bash
cd isaacgym/python && pip install -e .
```

## 使用方法

### 训练 CMG 教师模型

提供三档速度的训练配置：

```bash
# 慢速 (1 m/s)
bash train_teacher_cmg.sh slow <experiment_id> <device>

# 中速 (2 m/s)
bash train_teacher_cmg.sh medium <experiment_id> <device>

# 快速 (3 m/s)
bash train_teacher_cmg.sh fast <experiment_id> <device>
```

示例：

```bash
# 在 GPU 0 上训练慢速模型
bash train_teacher_cmg.sh slow cmg_slow_v1 cuda:0

# 在 GPU 1 上训练快速模型
bash train_teacher_cmg.sh fast cmg_fast_v1 cuda:1
```

### 直接使用 train.py

```bash
cd legged_gym/legged_gym/scripts

# 慢速
python train.py --task g1_cmg_slow --proj_name g1_cmg_slow --exptid my_exp --device cuda:0

# 中速
python train.py --task g1_cmg_medium --proj_name g1_cmg_medium --exptid my_exp --device cuda:0

# 快速
python train.py --task g1_cmg_fast --proj_name g1_cmg_fast --exptid my_exp --device cuda:0
```

### 可视化/回放

```bash
python legged_gym/legged_gym/scripts/play.py --task g1_cmg_slow --exptid <experiment_id>
```

## 配置说明

### 速度范围配置

| 配置 | 前向速度 (vx) | 侧向速度 (vy) | 转向速度 (yaw) |
|------|--------------|--------------|----------------|
| Slow | 0.5 ~ 1.5 m/s | -0.3 ~ 0.3 m/s | -0.5 ~ 0.5 rad/s |
| Medium | 1.5 ~ 2.5 m/s | -0.5 ~ 0.5 m/s | -0.8 ~ 0.8 rad/s |
| Fast | 2.5 ~ 3.5 m/s | -0.5 ~ 0.5 m/s | -1.0 ~ 1.0 rad/s |

### 自定义配置

在 `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py` 中创建新配置：

```python
class G1MimicCMGCustomCfg(G1MimicCMGBaseCfg):
    class motion(G1MimicCMGBaseCfg.motion):
        use_cmg = True
        cmg_vx_range = [1.0, 2.0]    # 自定义前向速度范围
        cmg_vy_range = [-0.3, 0.3]   # 自定义侧向速度范围
        cmg_yaw_range = [-0.6, 0.6]  # 自定义转向速度范围
```

然后在 `legged_gym/legged_gym/envs/__init__.py` 中注册：

```python
task_registry.register("g1_cmg_custom", G1MimicDistill, G1MimicCMGCustomCfg(), G1MimicCMGCustomCfgPPO())
```

## 技术细节

### DOF 映射

CMG 模型输出 29 DOF，G1 训练使用 23 DOF。映射关系：

| 部位 | CMG 索引 | G1 索引 | 说明 |
|------|----------|---------|------|
| 左腿 | 0-5 | 0-5 | 直接映射 |
| 右腿 | 6-11 | 6-11 | 直接映射 |
| 腰部 | 12-14 | 12-14 | 直接映射 |
| 左臂 | 15-18 | 15-18 | 直接映射 |
| 左腕 | 19-21 | - | **跳过** |
| 右臂 | 22-25 | 19-22 | 重新索引 |
| 右腕 | 26-28 | - | **跳过** |

### 关键体位置

使用正运动学计算 9 个关键体的 3D 位置：

1. `left_rubber_hand` - 左手
2. `right_rubber_hand` - 右手
3. `left_ankle_roll_link` - 左脚踝
4. `right_ankle_roll_link` - 右脚踝
5. `left_knee_link` - 左膝
6. `right_knee_link` - 右膝
7. `left_elbow_link` - 左肘
8. `right_elbow_link` - 右肘
9. `head_mocap` - 头部

### CMG 工作流程

1. **初始化**：加载 CMG 模型，从训练数据获取归一化统计和初始姿态
2. **重置**：采样速度命令，获取初始动作状态，预生成 2 秒轨迹缓冲
3. **步进**：每步推进缓冲帧索引，必要时重新生成轨迹
4. **查询**：支持当前帧和未来帧（用于 mimic observation）的动作查询

## 文件结构

新增/修改的文件：

```
TWIST/
├── pose/pose/utils/
│   ├── cmg_motion_lib.py      # [新增] CMGMotionLib 类
│   └── forward_kinematics.py  # [新增] 正运动学计算
├── legged_gym/legged_gym/envs/
│   ├── __init__.py            # [修改] 注册 CMG 环境
│   ├── base/
│   │   ├── humanoid_mimic.py         # [修改] 支持 CMG
│   │   └── humanoid_char_config.py   # [修改] CMG 默认配置
│   └── g1/
│       └── g1_mimic_distill_config.py # [修改] CMG 速度配置
├── train_teacher_cmg.sh       # [新增] 训练脚本
└── docs/
    └── cmg_integration.md     # [新增] 本文档
```

## 常见问题

### Q: 运行时报错找不到 pytorch_kinematics

```bash
pip install pytorch_kinematics
```

### Q: LFS 文件未拉取

```bash
git lfs pull
```

### Q: CMG 生成的动作不稳定

- 检查速度命令范围是否在 CMG 训练数据范围内 (vx: 0.3~4.0 m/s)
- 确保 episode 长度不超过 10 秒

### Q: 如何切换回原始 MotionLib

在配置中设置 `use_cmg = False`：

```python
class motion:
    use_cmg = False
    motion_file = "path/to/motion_data.yaml"
```
