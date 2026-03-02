#!/bin/bash
# CMG Student Distillation (DAgger: teacher → student)
# Usage: bash train_student_cmg.sh <student_exptid> <teacher_exptid> <device> [teacher_checkpoint]
#
# Examples:
#   # Distill from global_obs_v4 (latest checkpoint):
#   bash train_student_cmg.sh cmg_stu_v1 global_obs_v4 cuda:0
#
#   # Distill from specific teacher checkpoint:
#   bash train_student_cmg.sh cmg_stu_v1 global_obs_v4 cuda:0 8000

set -e

STUDENT_EXPTID=${1:-cmg_stu_v1}
TEACHER_EXPTID=${2:-global_obs_v4}
DEVICE=${3:-cuda:0}
TEACHER_CHECKPOINT=${4:--1}

TASK="g1_cmg_stu_rl"
PROJ_NAME="g1_cmg_stu_rl"

echo "=========================================="
echo "CMG Student Distillation (DAgger)"
echo "=========================================="
echo "Task: $TASK"
echo "Student Experiment: $STUDENT_EXPTID"
echo "Teacher Experiment: $TEACHER_EXPTID"
echo "Teacher Checkpoint: $TEACHER_CHECKPOINT"
echo "Device: $DEVICE"
echo "=========================================="

cd "$(dirname "$0")/legged_gym/legged_gym/scripts"

python train.py \
    --task $TASK \
    --proj_name $PROJ_NAME \
    --exptid $STUDENT_EXPTID \
    --teacher_exptid $TEACHER_EXPTID \
    --teacher_checkpoint $TEACHER_CHECKPOINT \
    --device $DEVICE \
    --num_envs 1024 \
    --max_iterations 10001 \
    --headless

echo "Student distillation complete!"
