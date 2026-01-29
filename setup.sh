#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# TWIST one-shot setup (strict reset)
# Place this file in repo root: TWIST-Docker/setup_twist.sh
#
# Defaults:
#   - Repo dir: inferred from script location
#   - Isaac Gym tar: /root/Downloads/IsaacGym_Preview_4_Package.tar.gz
#   - Extract to: <repo>/third_party/isaacgym
#
# Override via env:
#   ISAAC_TAR=/path/to/IsaacGym_Preview_4_Package.tar.gz
#   ENV_NAME=twist
#   PYTHON_VER=3.8
# -----------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"

ISAAC_TAR="${ISAAC_TAR:-/root/Downloads/IsaacGym_Preview_4_Package.tar.gz}"
ENV_NAME="${ENV_NAME:-twist}"
PYTHON_VER="${PYTHON_VER:-3.8}"

ISAAC_DIR="${REPO_DIR}/third_party/isaacgym"

echo "[0] Repo dir: ${REPO_DIR}"
echo "[0] Isaac Gym tar: ${ISAAC_TAR}"

if [[ ! -d "${REPO_DIR}/rsl_rl" || ! -d "${REPO_DIR}/legged_gym" || ! -d "${REPO_DIR}/pose" ]]; then
  echo "ERROR: This script must live in the TWIST repo root (contains rsl_rl/, legged_gym/, pose/)."
  exit 1
fi

if [[ ! -f "${ISAAC_TAR}" ]]; then
  echo "ERROR: Isaac Gym tar not found: ${ISAAC_TAR}"
  echo "You can override with: ISAAC_TAR=/path/to/isaacgym.tar.gz bash setup_twist.sh"
  exit 1
fi

echo "[1] Install system packages (apt)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  git wget curl ca-certificates unzip \
  ffmpeg \
  redis-server redis-tools \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libegl1 libglvnd0 \
  build-essential

echo "[2] Setup Miniconda if missing..."
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Installing Miniconda (py38) to /root/miniconda ..."
  cd /root
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -O /root/miniconda.sh
  bash /root/miniconda.sh -b -p /root/miniconda
  rm -f /root/miniconda.sh
  export PATH="/root/miniconda/bin:$PATH"
fi

# Make conda available in this shell
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[3] STRICT reset: remove existing conda env '${ENV_NAME}'..."
conda deactivate >/dev/null 2>&1 || true
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true

echo "[4] Create conda env '${ENV_NAME}' with python=${PYTHON_VER}..."
conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip wheel

echo "[5] (Optional) purge pip cache to avoid stale wheels..."
pip cache purge >/dev/null 2>&1 || true

# Ensure conda's libpython is discoverable by Isaac Gym binary extensions
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/ldpath.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF

echo "[6] Install Isaac Gym (fresh extract into repo)..."
rm -rf "${ISAAC_DIR}"
mkdir -p "${ISAAC_DIR}"
tar -xf "${ISAAC_TAR}" -C "${ISAAC_DIR}" --strip-components=1

if [[ ! -d "${ISAAC_DIR}/python" ]]; then
  echo "ERROR: '${ISAAC_DIR}/python' not found after extraction."
  echo "Top-level listing:"
  ls -la "${ISAAC_DIR}" || true
  echo "Hint: check tar structure with: tar -tf \"${ISAAC_TAR}\" | head -n 50"
  exit 1
fi

cd "${ISAAC_DIR}/python"
pip install -e .
cd "${REPO_DIR}"

echo "[7] Install editable packages (as README order)..."
cd rsl_rl && pip install -e . && cd ..
cd legged_gym && pip install -e . && cd ..

pip install "numpy==1.23.0" \
  pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core \
  "imageio[ffmpeg]" mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics \
  rich termcolor

pip install "redis[hiredis]"
pip install pyttsx3

cd pose && pip install -e . && cd ..

echo "[8] Start redis (no systemd required)..."
if redis-cli ping >/dev/null 2>&1; then
  echo "redis already running."
else
  redis-server --daemonize yes
fi
echo "redis ping: $(redis-cli ping || true)"

echo "[9] Quick sanity checks..."
python - << 'PY'
import isaacgym
print("isaacgym: OK")
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "✅ Setup complete."
echo "Next:"
echo "  cd \"${REPO_DIR}\""
echo "  conda activate \"${ENV_NAME}\""
echo "  tmux new -s twist   # optional"
echo "  bash train_teacher.sh 0927_twist_teacher cuda:0"
