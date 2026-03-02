#!/bin/bash
# CMG Student V2 DAgger Training (full 20-step future reference)
#
# Unlike V1 student (1-step mimic + 11-frame history = 1188 dims),
# V2 student gets the same 20-step future reference as the teacher (1237 dims)
# but without privileged info (base_lin_vel, key_body_pos, contacts, DR params).
#
# Usage: bash train_student_cmg_v2.sh <student_exptid> <teacher_exptid> <device> [teacher_checkpoint]
#
# Examples:
#   bash train_student_cmg_v2.sh cmg_stu_v2_test global_obs_v4 cuda:0
#   bash train_student_cmg_v2.sh cmg_stu_v2_test global_obs_v4 cuda:0 8000

set -e

STU_EXPTID=${1:?Usage: bash train_student_cmg_v2.sh <student_exptid> <teacher_exptid> <device> [teacher_checkpoint]}
TEACHER_EXPTID=${2:?Missing teacher experiment ID}
DEVICE=${3:-cuda:0}
TEACHER_CKPT=${4:--1}

TASK="g1_cmg_stu_v2"
PROJ_NAME="g1_cmg_stu_v2"

echo "=========================================="
echo "CMG Student V2 DAgger Training"
echo "=========================================="
echo "Task: $TASK"
echo "Student Experiment: $STU_EXPTID"
echo "Teacher Experiment: $TEACHER_EXPTID (proj: h1)"
echo "Teacher Checkpoint: $TEACHER_CKPT"
echo "Device: $DEVICE"
echo ""
echo "Student obs: priv_mimic(1160) + proprio(77) = 1237"
echo "  - Same 20-step future reference as teacher"
echo "  - No privileged info (84 dims removed)"
echo "=========================================="

cd "$(dirname "$0")/legged_gym/legged_gym/scripts"

python train.py \
    --task $TASK \
    --exptid $STU_EXPTID \
    --proj_name $PROJ_NAME \
    --device $DEVICE \
    --num_envs 1024 \
    --max_iterations 10001 \
    --teacher_exptid $TEACHER_EXPTID \
    --teacher_checkpoint $TEACHER_CKPT \
    --headless \
    "${@:5}"

echo "Student V2 training complete!"
