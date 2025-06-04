#!/bin/bash
export TASK_NAME=$1
export MODEL_NAME=$2
export JOB_NAME=${MODEL_NAME}_${TASK_NAME}
export PARTITION=gpu
export GPUS=1
export CPUS=4

mkdir -p ../${TASK_NAME}
sbatch --export=ALL -p ${PARTITION} -t 72:00:00 --gpus ${GPUS} -J ${JOB_NAME} --cpus-per-task=${CPUS} --output=../${TASK_NAME}/%x_%a_%j.log experiments/slurm_base_single.sh