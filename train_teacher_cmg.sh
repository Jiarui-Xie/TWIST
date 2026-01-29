#!/bin/bash
# Training script for CMG-based teacher models
# Usage: bash train_teacher_cmg.sh <speed_mode> <exptid> <device>
#   speed_mode: slow | medium | fast
#   exptid: experiment ID (e.g., cmg_slow_v1)
#   device: cuda device (e.g., cuda:0)

set -e

cd legged_gym/legged_gym/scripts

speed_mode=${1:-slow}
exptid=${2:-cmg_${speed_mode}_v1}
device=${3:-cuda:0}

# Map speed mode to task name
case $speed_mode in
    slow)
        task_name="g1_cmg_slow"
        proj_name="g1_cmg_slow"
        ;;
    medium)
        task_name="g1_cmg_medium"
        proj_name="g1_cmg_medium"
        ;;
    fast)
        task_name="g1_cmg_fast"
        proj_name="g1_cmg_fast"
        ;;
    *)
        echo "Error: Invalid speed_mode '$speed_mode'. Must be one of: slow, medium, fast"
        exit 1
        ;;
esac

echo "============================================"
echo "CMG Teacher Training"
echo "============================================"
echo "Speed Mode: $speed_mode"
echo "Task Name: $task_name"
echo "Project Name: $proj_name"
echo "Experiment ID: $exptid"
echo "Device: $device"
echo "============================================"

python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}"
