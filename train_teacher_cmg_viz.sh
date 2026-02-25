#!/bin/bash
# CMG Teacher Training with Visualization
# Usage: bash train_teacher_cmg_viz.sh <speed_mode> <exptid> <device> [resume_id] [checkpoint]
#
# Speed modes:
#   slow / medium / fast          - Standard CMG (motion imitation dominant)
#   slow_vt / medium_vt / fast_vt - Velocity tracking dominant (fine-tune from walking checkpoint)
#
# Examples:
#   # Train from scratch (standard motion imitation):
#   bash train_teacher_cmg_viz.sh medium cmg_v1 cuda:0
#
#   # Resume from checkpoint (same reward):
#   bash train_teacher_cmg_viz.sh medium cmg_v2 cuda:0 cmg_v1
#
#   # Fine-tune with velocity tracking from existing walking checkpoint:
#   bash train_teacher_cmg_viz.sh medium_vt cmg_vt_v1 cuda:0 cmg_v1 4000

set -e

SPEED_MODE=${1:-medium}
EXPTID=${2:-cmg_viz_test}
DEVICE=${3:-cuda:0}
RESUME_ID=${4:-}
CHECKPOINT=${5:-}

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
    slow_vt)
        TASK="g1_cmg_slow_vt"
        ;;
    medium_vt)
        TASK="g1_cmg_medium_vt"
        ;;
    fast_vt)
        TASK="g1_cmg_fast_vt"
        ;;
    *)
        echo "Invalid speed mode: $SPEED_MODE"
        echo "Usage: bash train_teacher_cmg_viz.sh <slow|medium|fast|slow_vt|medium_vt|fast_vt> <exptid> <device> [resume_id] [checkpoint]"
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

# Build resume arguments
RESUME_ARGS=""
if [ -n "$RESUME_ID" ]; then
    RESUME_ARGS="--resume --resumeid $RESUME_ID"
    echo "Resume from: $RESUME_ID"
    if [ -n "$CHECKPOINT" ]; then
        RESUME_ARGS="$RESUME_ARGS --checkpoint $CHECKPOINT"
        echo "Checkpoint: $CHECKPOINT"
    fi
fi

echo "=========================================="

cd "$(dirname "$0")/legged_gym/legged_gym/scripts"

python train.py \
    --task $TASK \
    --exptid $EXPTID \
    --device $DEVICE \
    --num_envs 1024 \
    --max_iterations 2001 \
    --headless \
    $RESUME_ARGS

echo "Training complete!"
