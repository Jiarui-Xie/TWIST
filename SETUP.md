# TWIST One-shot Setup (No Docker, works on non-systemd instances)

本仓库提供 `setup_twist.sh` 用于在 **任意新实例**上“一键初始化”训练环境。
该脚本专门适配以下场景：
- 实例中 **systemd 不可用**（`systemctl` 报错）
- 不能运行 Docker daemon（容器型/受限实例）
- 希望每次启动都能**严格重置**环境，避免旧包污染

---

## 0. 前置要求

- GPU 驱动已可用：`nvidia-smi` 能正常显示 GPU（如 RTX 4090）
- 已经准备好 Isaac Gym 安装包（tar.gz）：
  - 默认路径：`/root/Downloads/IsaacGym_Preview_4_Package.tar.gz`

> 注意：脚本不会从网络下载 Isaac Gym，只会从本地 tar 解压并安装。

---
## 1. 操作步骤
1. 获取仓库

建议不要递归 submodule（避免 HuggingFace 大数据集卡死）：

```bash
cd /root
git clone https://github.com/eziothean-git/TWIST-Docker.git
```

2. 运行一键脚本（严格重置）

脚本位置：仓库根目录 setup_twist.sh

cd /root/TWIST-Docker
chmod +x setup_twist.sh
bash setup_twist.sh

可选：自定义 Isaac Gym tar 路径

如果你的 Isaac Gym tar 不在默认目录，用环境变量覆盖：

ISAAC_TAR=/path/to/IsaacGym_Preview_4_Package.tar.gz bash setup_twist.sh

3. 脚本做了什么

严格按文档初始化流程执行，并额外做“清理”：

apt 安装基础依赖（含 redis、GL/EGL、编译工具）

安装 Miniconda（若未安装）

删除旧 conda 环境 twist（严格重置）

创建新环境：conda create -n twist python=3.8

解压 Isaac Gym tar 到：<repo>/third_party/isaacgym

安装：

pip install -e isaacgym/python

pip install -e rsl_rl

pip install -e legged_gym

pip install -e pose

pip 安装依赖（按 README 列表）

启动 redis（不依赖 systemd）：redis-server --daemonize yes

自检：import isaacgym + torch.cuda.is_available() + redis-cli ping

4. 开始训练
cd /root/TWIST-Docker
conda activate twist
bash train_teacher.sh 0927_twist_teacher cuda:0

5. 常见问题
systemd 不可用怎么办？

脚本不会调用 systemctl, redis 通过 redis-server --daemonize yes 启动。

不想每次都删环境怎么办？

当前脚本是“严格重置”版本。若你想改成增量安装，可自行去掉：

conda env remove -n twist

rm -rf third_party/isaacgym

（建议保留严格重置，避免实例反复调试导致环境污染）


---
1. 把脚本放进 repo：`/root/TWIST-Docker/setup_twist.sh`  
2. 运行：
```bash
cd /root/TWIST-Docker
chmod +x setup.sh
bash setup.sh

之后每个新实例：clone repo + 放好 IsaacGym tar + 运行脚本 就能开训。