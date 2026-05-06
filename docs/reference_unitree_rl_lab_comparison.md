---
name: TWIST vs unitree_rl_lab CMG Architecture Comparison
description: Comprehensive comparison of TWIST and unitree_rl_lab CMG training architectures — reward design, network, obs, action processing, domain rand, curriculum, and key tricks
type: reference
---

# TWIST vs unitree_rl_lab CMG 架构对比

unitree_rl_lab 位于 `/home/lubuntu/unitree_rl_lab/`，CMG 训练效果优于 TWIST。以下是完整对比分析。

---

## 1. 核心架构差异：全量模仿 vs 残差学习

### TWIST
- Actor 直接输出 23 DOF 关节目标角度
- Actor 输入包含 20 步 CMG 未来参考轨迹（1160 dims）
- CMG 参考仅作为观测和奖励目标

### unitree_rl_lab
- Actor 输出**残差修正量**（相对于 CMG 参考的偏移）
- Actor **不看 CMG 参考**，只看本体感知 + 速度指令
- CMG 参考仅给 Critic 用于价值估计
- Runner 层做加法：`joint_target = CMG_qref + actor_residual`

**关键代码**（`deploy/include/FSM/State_RLResidual.h:59-65`）：
```cpp
auto qr = cmg->get_qref();                         // CMG 参考
auto raw_residual = env->action_manager->action();  // actor 输出
for (size_t i = 0; i < combined.size(); ++i)
    combined[i] = qr[i] + raw_residual[i];          // 最终目标
```

**本质**：feedforward（CMG 开环参考）+ feedback（RL 闭环修正），大幅降低 RL 学习难度。

---

## 2. 网络架构

| | TWIST | unitree_rl_lab |
|---|---|---|
| **Actor** | MLP [512,512,256,128] + Conv1D motion encoder | **LSTM 2×256** |
| **Critic** | MLP [512,512,256,128] | **LSTM 2×256** |
| **Activation** | SiLU + LayerNorm | ELU |
| **Motion Encoder** | Conv1D: 20步参考 → 128D latent | 无（actor 不看 CMG） |
| **Init noise std** | 1.0 | 1.0 |
| **Action std** | per-joint: leg=0.7, waist=0.4, arm=0.5 | 统一（learnable） |

LSTM 能隐式记忆步态周期和接触时序，比 MLP 堆叠历史帧更高效。

---

## 3. 观测空间

### TWIST CMG Teacher（1321 dims）
```
n_priv_mimic_obs = 1160  (20步 × 58 dims: root_z + euler + lin_vel + yaw_vel + 23 dof_pos + 27 key_body_pos)
n_proprio = 77           (ang_vel:3 + imu:2 + dof_pos:23 + dof_vel:23 + actions:23 + cmd:3)
n_priv_info = 84         (base_lin_vel:3 + root_height:1 + key_body_pos:27 + feet_contact:2 + mass:4 + friction:1 + motor:2 + ...)
```

### TWIST CMG Student V2（1237 dims）
```
n_priv_mimic_obs = 1160  (同 teacher)
n_proprio = 77           (同 teacher)
无 priv_info
```

### unitree_rl_lab Actor（475 dims = 95 × 5 步历史）
```
每步 95 dims:
  base_ang_vel:3 (×0.2) + projected_gravity:3 + velocity_commands:3
  + joint_pos_rel:29 + joint_vel_rel:29 (×0.05) + last_action:29
历史 5 步拼接
```

### unitree_rl_lab Critic（额外 privileged）
```
policy obs (475) + gt_linear_velocity (3) + CMG motion ref (58: 29 pos + 29 vel)
```

**关键差异**：unitree_rl_lab 的 actor obs 仅 475 dims（纯本体感知），actor 完全不看 CMG 参考。CMG 参考只给 critic。

---

## 4. 动作空间

| | TWIST | unitree_rl_lab |
|---|---|---|
| **DOF** | 23（无手腕） | 29（含手腕） |
| **Action scale** | 0.5 | 0.5 |
| **Action clipping** | 5.0 / action_scale = 10.0 | clip_actions（config） |
| **Action 含义** | 直接关节角度目标 | 残差偏移（加到 CMG 参考上） |
| **use_default_offset** | N/A（IsaacGym） | **False**（不加默认关节角偏移） |

---

## 5. 奖励函数对比

### 速度跟踪

| 奖励 | TWIST | unitree_rl_lab |
|---|---|---|
| track_lin_vel_xy | tracking_cmd_vel = **1.5** (exp(-2.0 × err²)) | **2.0** (exp(-2.0 × err²)) |
| track_ang_vel_z | tracking_cmd_yaw = **1.0** (exp(-1.0 × err²)) | **1.0** (exp(-1.5 × err²)) |

### CMG 模仿

| 奖励 | TWIST | unitree_rl_lab |
|---|---|---|
| Joint pos tracking | tracking_joint_dof = **0.6** (exp(-0.15 × weighted_err²)) | joint_pos_from_cmg = **1.5** (exp(-0.6 × err²), **gated**) |
| Joint vel tracking | tracking_joint_vel = **0.2** (exp(-0.01 × err²)) | joint_vel_from_cmg = **0.3** (exp(-0.5 × err²), **gated**) |
| Lower body pos | tracking_keybody_pos = **2.0** (exp(-10.0 × err²)) | 无单独 |
| Upper body pos | tracking_keybody_pos_upper = **0.3** (exp(-5.0 × err²)) | 无 |

### 速度门控机制（unitree_rl_lab 独有）

```
vx ≤ 0.4 m/s → Walk 模式：步态模式奖励激活，CMG 模仿奖励关闭
vx ≥ 0.5 m/s → Run 模式：CMG 模仿奖励激活，步态奖励关闭
中间线性插值
```

TWIST 没有这个机制，所有速度使用相同奖励。

### 正则化对比

| 奖励 | TWIST | unitree_rl_lab |
|---|---|---|
| **termination_penalty** | 无显式大惩罚 | **-200.0** |
| **alive** | 无 | **+1.0** |
| **feet_slide/slip** | -0.1 | **-1.0**（10 倍） |
| **action_rate** | -0.01 | **-0.04** |
| **action_smoothness (jerk)** | 无 | **-0.06**（(a_t - 2a_{t-1} + a_{t-2})²） |
| **residual_magnitude** | 无 | **-0.02**（保持残差小） |
| **base_height** | 无（FineTune 才有 0.5） | **-10.0**（维持 0.78m） |
| **flat_orientation** | ang_vel_xy = -0.01 | **-5.0** |
| **energy** | 无 | **-1e-5** |
| **joint_acc** | dof_acc = -5e-8 | **-5e-8** |
| **dof_pos_limits** | -5.0 | **-2.0** |
| **feet_air_time** | 5.0 | 无 |
| **action_symmetry** | 0.1 | 无 |

### Walk 模式专属奖励（unitree_rl_lab 独有，低速时激活）

| 奖励 | Weight |
|---|---|
| gait_walk（步态周期） | **0.5** |
| feet_clearance_walk（抬脚高度） | **1.0** |
| base_height_walk（站高 0.78m） | **-10.0** |
| base_linear_velocity_walk（抑制垂直运动） | **-2.0** |
| joint_vel_walk（关节速度惩罚） | **-0.001** |
| undesired_contacts_walk（非脚接触） | **-1.0** |

### 关节偏差惩罚（unitree_rl_lab）

| 关节组 | Weight |
|---|---|
| Arms（shoulder, elbow, wrist） | -0.1 |
| Waist | -0.5 |
| Legs（hip roll/yaw） | -0.2 |

---

## 6. PPO / 训练参数

| 参数 | TWIST | unitree_rl_lab |
|---|---|---|
| **learning_rate** | 2e-4 | 3e-4 |
| **schedule** | adaptive | adaptive |
| **num_learning_epochs** | 5 | 5 |
| **num_mini_batches** | 4 | 4 |
| **num_steps_per_env** | 24 | 24 |
| **gamma** | 0.99 | 0.99 |
| **lam** | 0.95 | 0.95 |
| **clip_param** | 0.2 | 0.2 |
| **entropy_coef** | 0.005 | 0.005 |
| **desired_kl** | 0.008 | 0.01 |
| **max_grad_norm** | 1.0 | 1.0 |
| **max_iterations** | 30002 | **10000** |
| **num_envs** | 1024 | **4096** |
| **episode_length** | 16.5s | **20s** |
| **save_interval** | 500 | 500 |

---

## 7. Domain Randomization

| 参数 | TWIST | unitree_rl_lab |
|---|---|---|
| **friction** | [0.1, 2.0] | [0.3, 1.0]（更保守） |
| **base_mass** | [-3, +3] kg | [-1, +3] kg（不对称偏重） |
| **motor_strength** | [0.8, 1.2] | 无单独 motor strength |
| **stiffness** | 无随机 | **[0.7, 1.3]**（±30%） |
| **damping** | 无随机 | **[0.7, 1.3]**（±30%） |
| **joint_friction** | 无 | **[0.0, 0.5]** |
| **armature** | 无 | **[0.0, 0.05]** |
| **gravity** | ±5°（0.86 m/s²） | 无 |
| **push_robot** | max_vel=1.0, interval=4s | max_vel=0.8, interval=5s |
| **push_end_effector** | 20N, interval=2s | 无 |
| **base_com** | [-0.05, 0.05] m | 无 |

**差异**：unitree_rl_lab 更关注执行器建模（关节摩擦、刚度、阻尼、转动惯量），TWIST 更关注外部扰动（重力、推力、末端力）。

---

## 8. 速度课程学习

### TWIST
- 固定速度范围，分 slow/medium/fast 三个独立训练
  - Slow: vx=[0.5, 1.5], vy=[-0.3, 0.3]
  - Medium: vx=[1.5, 2.5], vy=[-0.5, 0.5]
  - Fast: vx=[2.5, 3.5], vy=[-0.5, 0.5]
- Ramp profile（起步/停止训练）：stand → ramp_up → steady → ramp_down → crawl → stand

### unitree_rl_lab
- **自适应课程**：从小范围逐步扩展
  - 起始：vx=[-0.1, 0.5], vy=[-0.1, 0.1], yaw=[-0.1, 0.1]
  - 上限：vx=[-0.5, 3.0], vy=[-0.3, 0.3], yaw=[-0.5, 0.5]
- 触发条件：reward > 0.8 × weight 时范围 ± 0.1
- 一次训练覆盖全速度范围

---

## 9. 终止条件

| | TWIST | unitree_rl_lab |
|---|---|---|
| **超时** | 16.5s | 20s（soft terminal） |
| **跌倒高度** | pose_termination_dist = 0.7 | height < 0.2m |
| **姿态异常** | 无单独 | angle > 0.8 rad (≈46°) |

---

## 10. PD 控制参数

| 关节 | TWIST Kp | TWIST Kd |
|---|---|---|
| hip | 100 | 2 |
| knee | 150 | 4 |
| ankle | 40 | 2 |
| waist | 150 | 4 |
| shoulder | 40 | 5 |
| elbow | 40 | 5 |

unitree_rl_lab 使用类似的 PD 参数但**增加了 stiffness/damping 随机化 [0.7, 1.3]**。

---

## 11. 地形

| | TWIST | unitree_rl_lab |
|---|---|---|
| **类型** | plane（flat） | **FLAT_TERRAIN_CFG**（8×8m flat） |
| **border** | N/A | 20m |

两者都用平地训练 CMG。

---

## 12. 噪声

### TWIST
```
dof_pos = 0.01, dof_vel = 0.1, lin_vel = 0.1
ang_vel = 0.1, gravity = 0.05, imu = 0.1
noise_increasing_steps = 3000（逐步增加）
```

### unitree_rl_lab
```
base_ang_vel: [-0.2, 0.2]
projected_gravity: [-0.05, 0.05]
joint_pos_rel: [-0.01, 0.01]
joint_vel_rel: [-1.5, 1.5]
empirical_normalization = True（running mean/std）
history corruption enabled
```

---

## 13. 关键文件路径

### unitree_rl_lab
| 文件 | 用途 |
|---|---|
| `source/.../tasks/locomotion/robots/g1/29dof/RuN_env_cfg.py` | CMG 残差训练主配置 |
| `source/.../tasks/locomotion/mdp/rewards.py` | 所有奖励函数（含门控） |
| `source/.../tasks/locomotion/mdp/observations.py` | CMG 观测 + 步态相位 |
| `source/.../tasks/locomotion/agents/rsl_rl_ppo_residual_cfg.py` | LSTM PPO 配置 |
| `deploy/include/FSM/State_RLResidual.h` | 部署：CMG ref + residual 加法 |
| `deploy/robots/g1_29dof/src/State_RLResidual.cpp` | 部署 C++ 实现 |
| `scripts/rsl_rl/train.py` | 训练入口（OnPolicyRunnerResidual） |

### TWIST
| 文件 | 用途 |
|---|---|
| `legged_gym/.../envs/g1/g1_mimic_distill_config.py` | 所有配置 |
| `legged_gym/.../envs/base/humanoid_mimic.py` | 主环境 + 奖励 |
| `rsl_rl/.../modules/actor_critic_mimic.py` | 网络架构 |
| `pose/pose/utils/cmg_motion_lib.py` | CMG 运动库 |

---

## 14. CMG 自回归迭代方式对比

### TWIST：纯开环自回归

```python
# cmg_motion_lib.py:577 — _generate_trajectory()
current_norm[active] = self._cmg_model(current_norm[active], frame_cmd_norm)
```

- `current_norm` 初始化自上一帧 CMG 的输出（`_current_motion_norm`）
- 每帧迭代：CMG 输出 → 作为下一帧输入 → CMG 输出 → ...
- **完全不看机器人实际状态**，是纯开环 AR 链
- 一次性预生成整段 trajectory buffer，之后逐帧消费
- 风险：CMG 输出不稳定时误差在 AR 链中累积

### unitree_rl_lab：两种模式

#### `forward()`——非自回归（训练时用）
```cpp
// algorithms.h:253-257
auto motion_cmg = usd_to_cmg(joint_pos_usd, joint_vel_usd);  // 每步用机器人真实状态
```
完全闭环，每步都用机器人实际 joint_pos/joint_vel。

#### `forward_ar()`——Leaky 自回归（部署时用）
```cpp
// algorithms.h:305-362
if (!ar_initialized) {
    prev_output_cmg = usd_to_cmg(joint_pos_usd, joint_vel_usd);  // 首帧用真实状态初始化
}
// 用 prev_output_cmg 做 CMG 推理 → motion_ref_cmg

// Leaky AR: 混合 CMG 输出和机器人实际状态
constexpr float ar_leak = 0.05f;
auto actual_cmg = usd_to_cmg(joint_pos_usd, joint_vel_usd);
prev_output_cmg[i] = (1.0f - ar_leak) * motion_ref_cmg[i]
                     + ar_leak * actual_cmg[i];
```

**Leaky AR trick**：95% CMG 自身输出（保持轨迹连贯）+ 5% 机器人实际状态（防止 AR 链漂移/卡死在 clamp 边界）。注释原文：*"prevent the AR chain from getting trapped at clamp fixed points during sharp velocity command transitions"*

### 对比总结

| | TWIST | unitree_rl_lab (forward) | unitree_rl_lab (forward_ar) |
|---|---|---|---|
| **CMG 输入** | 上一步 CMG 输出 | 机器人实际状态 | 0.95×CMG输出 + 0.05×机器人实际 |
| **闭环性** | 纯开环 | 完全闭环 | Leaky 半闭环 |
| **漂移风险** | 高（误差累积） | 无 | 低（leak 修正） |
| **用于** | 训练+部署 | 训练 | 部署 |

---

## 15. 总结：unitree_rl_lab 的核心 Tricks

1. **残差学习**：actor 输出修正量而非完整动作，CMG 提供 feedforward 基础
2. **LSTM 网络**：时序记忆替代历史帧堆叠
3. **速度门控**：低速用结构化步态奖励，高速用 CMG 模仿，平滑过渡
4. **Actor 不看 CMG**：只看本体感知 + 速度指令，CMG ref 仅给 critic（privileged）
5. **强终止惩罚**（-200）+ **alive 奖励**（+1）
6. **二阶动作平滑**：jerk 惩罚 + action_rate + residual_magnitude
7. **自适应速度课程**：一次训练覆盖全速度范围
8. **4096 envs**：更大 batch，梯度更稳定，仅需 10000 iterations
9. **执行器随机化**：stiffness/damping/friction/armature 随机化，更好 sim2real
10. **Walk 模式**专属奖励：步态周期、脚 clearance、高度维持、非脚接触惩罚
11. **Leaky AR**：部署时 CMG 迭代混入 5% 机器人实际状态，防止 AR 链漂移/卡死
