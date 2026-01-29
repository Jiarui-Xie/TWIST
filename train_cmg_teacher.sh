#!/bin/bash

# Train teacher policy using CMG-generated motion
# Usage: bash train_cmg_teacher.sh <experiment_id> <device>
# Example: bash train_cmg_teacher.sh cmg_teacher_v1 cuda:0

cd legged_gym/legged_gym/scripts

exptid=$1
device=$2

task_name="g1_cmg_teacher"
proj_name="g1_cmg_teacher"

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                # --resume \
                # --debug
                # --resumeid xxx
