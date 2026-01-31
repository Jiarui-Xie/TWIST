#!/usr/bin/env bash
set -euo pipefail

# Simplified setup without sudo (for local testing)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"

ISAAC_TAR="${ISAAC_TAR:-/home/lumi/Downloads/IsaacGym_Preview_4_Package.tar.gz}"
ENV_NAME="${ENV_NAME:-twist}"
PYTHON_VER="${PYTHON_VER:-3.8}"

ISAAC_DIR="${REPO_DIR}/third_party/isaacgym"

echo "[0] Repo dir: ${REPO_DIR}"
echo "[0] Isaac Gym tar: ${ISAAC_TAR}"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    cd /home/lumi
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /home/lumi/miniconda3
    rm -f miniconda.sh
    export PATH="/home/lumi/miniconda3/bin:$PATH"
fi

# Source conda
if [[ -f "/home/lumi/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "/home/lumi/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "[1] Creating conda env '${ENV_NAME}'..."
conda deactivate 2>/dev/null || true
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} already exists, activating..."
else
    conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y
fi
conda activate "${ENV_NAME}"

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "[2] Extracting Isaac Gym..."
if [[ ! -d "${ISAAC_DIR}/python" ]]; then
    mkdir -p "${ISAAC_DIR}"
    tar -xf "${ISAAC_TAR}" -C "${ISAAC_DIR}" --strip-components=1
fi

echo "[3] Installing packages..."
cd "${ISAAC_DIR}/python"
pip install -e . --quiet

cd "${REPO_DIR}"
cd rsl_rl && pip install -e . --quiet && cd ..
cd legged_gym && pip install -e . --quiet && cd ..

pip install --quiet "numpy==1.23.0" \
  pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core \
  "imageio[ffmpeg]" mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics \
  rich termcolor

pip install --quiet "redis[hiredis]"

cd pose && pip install -e . --quiet && cd ..

echo "[4] Sanity check..."
python -c "import isaacgym; print('isaacgym: OK')"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate: source /home/lumi/miniconda3/etc/profile.d/conda.sh && conda activate twist"
echo "To train:    cd ${REPO_DIR}/legged_gym/legged_gym/scripts && python train.py --task g1_cmg_medium --exptid test --debug --no_wandb"
