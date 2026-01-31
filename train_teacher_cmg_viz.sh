#!/bin/bash
# CMG Teacher Training with Visualization
# Usage: bash train_teacher_cmg_viz.sh <speed_mode> <exptid> <device>
# Example: bash train_teacher_cmg_viz.sh medium cmg_v1 cuda:0
#
# This script runs training with 1024 environments and visualization enabled

set -e

SPEED_MODE=${1:-medium}
EXPTID=${2:-cmg_viz_test}
DEVICE=${3:-cuda:0}

# Map speed mode to task
case $SPEED_MODE in
    slow)
        TASK="g1_cmg_slow"
        ;;
    medium)
        TASK="g1_cmg_medium"
        ;;
    fast)
        TASK="g1_cmg_fast"
        ;;
    *)
        echo "Invalid speed mode: $SPEED_MODE"
        echo "Usage: bash train_teacher_cmg_viz.sh <slow|medium|fast> <exptid> <device>"
        exit 1
        ;;
esac

echo "=========================================="
echo "CMG Teacher Training with Visualization"
echo "=========================================="
echo "Speed Mode: $SPEED_MODE"
echo "Task: $TASK"
echo "Experiment ID: $EXPTID"
echo "Device: $DEVICE"
echo "Num Envs: 1024"
echo "Visualization: Enabled"
echo "=========================================="

cd "$(dirname "$0")/legged_gym/legged_gym/scripts"

python train.py \
    --task $TASK \
    --exptid $EXPTID \
    --device $DEVICE \
    --num_envs 1024 \
    --rows 32 \
    --cols 32 \
    --no_wandb \
    --viz

echo "Training complete!"
