# Changelog: CMG 动作参考集成

**日期**: 2026-01-29
**目标**: 使用 CMG (Conditional Motion Generator) 模型生成动作参考，用于 TWIST 教师模型训练

---

## 新增文件

### 1. `pose/pose/utils/cmg_motion_lib.py`
CMG 运动库，提供与 MotionLib 相同的接口：
- 加载 CMG 模型和归一化统计
- 自回归生成动作序列
- 维护轨迹缓冲区（100帧/2秒）支持未来帧查询
- 29 DOF → 23 DOF 映射
- 根节点状态积分（位置、朝向）

### 2. `pose/pose/utils/forward_kinematics.py`
正运动学计算器：
- 使用 `pytorch_kinematics` 从关节角度计算关键体 3D 位置
- 支持 9 个关键体位置计算
- G1 训练 DOF 顺序到 URDF 关节顺序映射

### 3. `cmg_workspace/module/__init__.py`
模块初始化文件，使 `module` 目录成为可导入的 Python 包

### 4. `train_teacher_cmg.sh`
CMG 教师模型训练脚本：
```bash
bash train_teacher_cmg.sh <speed_mode> <exptid> <device>
# speed_mode: slow | medium | fast
```

### 5. `docs/cmg_integration.md`
CMG 集成使用文档

### 6. `docs/CHANGELOG_CMG_Integration.md`
本 changelog 文件

---

## 修改文件

### 1. `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`

新增 CMG 配置类：

```python
# 基础 CMG 配置
class G1MimicCMGBaseCfg(G1MimicPrivCfg)

# 三档速度配置
class G1MimicCMGSlowCfg    # vx: 0.5~1.5 m/s
class G1MimicCMGMediumCfg  # vx: 1.5~2.5 m/s
class G1MimicCMGFastCfg    # vx: 2.5~3.5 m/s

# 对应 PPO 配置
class G1MimicCMGSlowCfgPPO
class G1MimicCMGMediumCfgPPO
class G1MimicCMGFastCfgPPO
```

### 2. `legged_gym/legged_gym/envs/base/humanoid_mimic.py`

修改内容：
- `_load_motions()`: 根据 `cfg.motion.use_cmg` 选择使用 `CMGMotionLib` 或 `MotionLib`
- `_post_physics_step_callback()`: 添加 CMG 步进和根节点状态更新
- `_reset_ref_motion()`: 添加 CMG 重置逻辑

### 3. `legged_gym/legged_gym/envs/base/humanoid_char_config.py`

在 `motion` 类中添加 CMG 默认配置：
```python
use_cmg = False
cmg_model_path = ""
cmg_data_path = ""
cmg_dt = 0.02
cmg_vx_range = [0.5, 1.5]
cmg_vy_range = [-0.3, 0.3]
cmg_yaw_range = [-0.5, 0.5]
```

### 4. `legged_gym/legged_gym/envs/__init__.py`

注册 CMG 环境：
```python
task_registry.register("g1_cmg_slow", ...)
task_registry.register("g1_cmg_medium", ...)
task_registry.register("g1_cmg_fast", ...)
```

### 5. `legged_gym/requirements.txt`

添加依赖：
```
pytorch_kinematics
termcolor
```

---

## 技术细节

### DOF 映射 (29 → 23)

```python
CMG_TO_G1_INDICES = [
    0, 1, 2, 3, 4, 5,       # 左腿 (6)
    6, 7, 8, 9, 10, 11,     # 右腿 (6)
    12, 13, 14,             # 腰部 (3)
    15, 16, 17, 18,         # 左臂 (4)
    22, 23, 24, 25,         # 右臂 (4) - 跳过左腕 19-21
]
# 跳过: 19, 20, 21 (左腕), 26, 27, 28 (右腕)
```

### 速度范围配置

| 档位 | vx (m/s) | vy (m/s) | yaw (rad/s) |
|------|----------|----------|-------------|
| Slow | 0.5 ~ 1.5 | -0.3 ~ 0.3 | -0.5 ~ 0.5 |
| Medium | 1.5 ~ 2.5 | -0.5 ~ 0.5 | -0.8 ~ 0.8 |
| Fast | 2.5 ~ 3.5 | -0.5 ~ 0.5 | -1.0 ~ 1.0 |

### 关键体列表 (9个)

1. left_rubber_hand
2. right_rubber_hand
3. left_ankle_roll_link
4. right_ankle_roll_link
5. left_knee_link
6. right_knee_link
7. left_elbow_link
8. right_elbow_link
9. head_mocap

### 奖励函数位置

- **实现**: `legged_gym/legged_gym/envs/base/humanoid_mimic.py` (第435-544行)
- **权重配置**: `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py` → `class rewards.scales`

主要奖励：
- `tracking_joint_dof`: 0.6
- `tracking_joint_vel`: 0.2
- `tracking_root_pose`: 0.6
- `tracking_root_vel`: 1.0
- `tracking_keybody_pos`: 2.0

---

## 使用方法

### 安装

```bash
pip install -r legged_gym/requirements.txt
pip install -e pose/ -e rsl_rl/ -e legged_gym/
git lfs pull  # 拉取 CMG 训练数据
```

### 训练

```bash
# 慢速 (1 m/s)
bash train_teacher_cmg.sh slow cmg_slow_v1 cuda:0

# 中速 (2 m/s)
bash train_teacher_cmg.sh medium cmg_medium_v1 cuda:0

# 快速 (3 m/s)
bash train_teacher_cmg.sh fast cmg_fast_v1 cuda:0
```

---

## 修复记录

### 2026-01-29 对齐问题修复

**问题**: CMGMotionLib 返回的 body_pos 形状 `(batch, 9, 3)` 与 `_ref_body_pos` 期望的 `(batch, num_rigid_bodies, 3)` 不匹配

**修复**:
1. `humanoid_mimic.py` - 在 CMG 模式下，body_pos 直接赋值到 `_key_body_ids` 对应的位置
2. `cmg_motion_lib.py` - 移除 `_body_link_list` 中的 "pelvis"，保持与 config 中 `key_bodies` 顺序一致

---

## 待验证

- [ ] CMGMotionLib 加载和运行
- [ ] 正运动学计算正确性
- [ ] 三档速度训练效果
- [ ] 轨迹缓冲区重生成逻辑
- [ ] body_pos 形状对齐
