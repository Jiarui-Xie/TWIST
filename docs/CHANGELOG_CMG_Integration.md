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

## 2026-01-30 性能优化与Bug修复

### 问题1: ActorCritic 意外参数警告

**问题**: `ActorCritic.__init__` 打印误导性警告 "got unexpected arguments: ['obs_context_len', 'priv_encoder_dims', 'tanh_encoder_output']"，但实际上这些参数被使用了。

**修复文件**:
- `rsl_rl/rsl_rl/modules/actor_critic.py`
- `rsl_rl/rsl_rl/modules/actor_critic_mimic.py`

**修复方法**: 将这些参数从 `**kwargs` 改为显式参数定义：
```python
def __init__(self, ...,
             priv_encoder_dims=None,
             tanh_encoder_output=False,
             obs_context_len=None,
             **kwargs):
```

### 问题2: 缺少训练进度条

**问题**: 训练时没有 tqdm 进度条显示。

**修复文件**:
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
- `rsl_rl/rsl_rl/runners/on_policy_runner_mimic.py`
- `rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py`

**修复方法**: 添加 tqdm 包装训练循环：
```python
from tqdm import tqdm
for it in tqdm(range(...), desc="Training", ...):
```

### 问题3: CMGMotionLib 性能瓶颈

**问题**: 训练卡在 0%，因为 `_generate_trajectory` 和 `calc_motion_frame` 有 O(n²) 的 Python 循环。

**修复文件**: `pose/pose/utils/cmg_motion_lib.py`

**修复方法**: 向量化关键循环：
- `_generate_trajectory`: 消除 4096 次内层 env 循环
- `calc_motion_frame`: 消除 81920 次 tiled query 循环
- `_get_init_motion`: 简化列表构建

### 问题4: pytorch_kinematics URDF 解析错误

**问题**: `unknown attribute "name" in /robot[@name='g1_29dof_rev_1_0']/link[@name='left_ankle_roll_link']`

**修复文件**: `pose/pose/utils/forward_kinematics.py`

**修复方法**: 用简化的几何近似 FK 替代 `pytorch_kinematics`：
- 不再依赖 URDF 解析
- 使用预定义的连杆长度
- 基于关节角度的近似位置计算

### 训练参数位置说明

| 参数 | 位置 | 说明 |
|------|------|------|
| `max_iterations` | `g1_mimic_distill_config.py:602` | 30002 次迭代 |
| `save_interval` | `g1_mimic_distill_config.py:603` | 基础保存间隔 500 |
| `episode_length_s` | `g1_mimic_distill_config.py:32` | 10 秒/episode |
| `num_steps_per_env` | `humanoid_mimic_config.py:56` | 24 步/迭代 |

### 模型保存策略

```python
# on_policy_runner_mimic.py:242-250
if it < 2500:      保存间隔 = 500   (model_0, 500, 1000, ...)
elif it < 5000:    保存间隔 = 1000  (model_2500, 3500, 4500)
else:              保存间隔 = 2500  (model_5000, 7500, 10000, ...)
```

每次保存新文件 `model_{it}.pt`，不覆盖。

---

## 2026-02-01 四元数格式修复与部分重置修复

### 问题1: 机器人上下颠倒

**问题**: CMG 输出四元数格式为 `[w, x, y, z]` (wxyz)，而 Isaac Gym 期望 `[x, y, z, w]` (xyzw)，导致机器人姿态完全错误。

**修复文件**: `pose/pose/utils/cmg_motion_lib.py`

**修复方法**: 在三个返回点添加格式转换：
```python
# wxyz -> xyzw
root_rot_xyzw = torch.cat([root_rot[:, 1:], root_rot[:, :1]], dim=-1)
```

修改位置：
- `calc_motion_frame()` tiled 查询分支 (第 537-540 行)
- `_calc_current_frame()` (第 579-582 行)
- `_calc_partial_frame()` (新增函数)

### 问题2: 部分环境重置时索引错误

**问题**: 当只有部分环境需要重置时（如 env_ids=[0,3,5]），`calc_motion_frame` 错误地进入 tiled 查询分支，导致 `num_steps = batch_size // num_envs = 0`，产生除零错误。

**修复文件**:
- `pose/pose/utils/cmg_motion_lib.py`
- `legged_gym/legged_gym/envs/base/humanoid_mimic.py`
- `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`

**修复方法**:

1. `cmg_motion_lib.py`:
   - `calc_motion_frame()` 添加 `env_ids` 参数
   - 新增 `_calc_partial_frame()` 函数处理部分查询
   - 三分支判断逻辑：
     ```python
     if batch_size == num_envs:           # 全部环境
         return _calc_current_frame()
     elif env_ids is not None and batch_size < num_envs:  # 部分环境
         return _calc_partial_frame(env_ids)
     else:                                 # tiled 查询（未来帧）
         return _calc_tiled_frame(...)
     ```

2. `humanoid_mimic.py` 和 `g1_mimic_distill.py`:
   - `_reset_ref_motion()` 中调用 CMG 时传入 `env_ids` 参数
   - `g1_mimic_distill.py` 添加缺失的 `self._motion_lib.reset(env_ids)` 调用

### 验证测试

创建 `test_cmg_quaternion.py` 和 `test_cmg_mujoco.py` 验证修复：
```
✓ Quaternion conversion: wxyz [1,0,0,0] -> xyzw [0,0,0,1]
✓ _calc_partial_frame method exists
✓ calc_motion_frame has env_ids parameter
✓ humanoid_mimic.py passes env_ids to calc_motion_frame
✓ g1_mimic_distill.py calls motion_lib.reset(env_ids)
```

---

## 待验证

- [x] CMGMotionLib 加载和运行
- [x] 向量化性能优化
- [x] 简化 FK 实现
- [x] 四元数格式修复 (wxyz -> xyzw)
- [x] 部分环境重置修复
- [ ] 三档速度训练效果
- [ ] 轨迹缓冲区重生成逻辑
- [ ] mean reward 非零验证
