---
name: unitree_mujoco DDS sim2sim/sim2real
description: sim2sim_unitree_mujoco.py uses DDS protocol (unitree_sdk2py) — same script for sim and real, only domain_id and interface differ
type: project
---

`deploy_real/sim2sim_unitree_mujoco.py` 是基于宇树官方 DDS 协议的控制器，与 `unitree_mujoco` 仿真器或真机通信。

**Why:** 原有 `sim2sim_cmg_stu_v2.py` 直接驱动 MuJoCo，控制器代码与实机部署代码不同。DDS 方案让 sim 和 real 共享同一控制器，减少 sim-to-real gap。

**How to apply:**
- Sim2Sim: `--domain_id 1 --interface lo`，需另开终端运行 `unitree_mujoco/simulate_python/unitree_mujoco.py`
- Sim2Real: `--domain_id 0 --interface eno1`（或实际网口）
- 关节映射: TWIST 23-DOF → DDS 29-motor，右臂 TWIST[19-22] → DDS[22-25]，腕关节[19-21,26-28]不控制
- 依赖: `unitree_sdk2py`, `mujoco`, `pygame`；需要 cyclonedds C 库
- 该脚本无手柄安全启动流程，实机慎用（推荐熟练操作者）
