#!/bin/bash
#SBATCH --job-name=default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[INSERT:EMAIL]
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=4 # -c
#SBATCH --mem=48gb
#SBATCH --gpus=1
#SBATCH --output=%x_%a_%j.log # %x is job-name, %j is job id, %a is array id
#SBATCH --array=0-3

# Submit with e.g. --export=TASK_NAME=mortality24,MODEL_NAME=LGBMClassifier
# Basic experiment variables, please exchange [INSERT] for your experiment parameters

TASK=BinaryClassification # BinaryClassification
YAIB_PATH=/home/vandewrp/projects/YAIB #/dhc/home/robin.vandewater/projects/yaib
EXPERIMENT_PATH=../yaib_logs/${TASK_NAME}_experiment
DATASET_ROOT_PATH= /sc-projects/sc-proj-cc08-cassandra/Prospective_Preprocessed/yaib_format #data/YAIB_Datasets/data

echo "This is a SLURM job named" $SLURM_JOB_NAME "with array id" $SLURM_ARRAY_TASK_ID "and job id" $SLURM_JOB_ID
echo "Resources allocated: " $SLURM_CPUS_PER_TASK "CPUs, " $SLURM_MEM_PER_NODE "GB RAM, " $SLURM_GPUS_PER_NODE "GPUs"
echi "Task type:" ${TASK}
echo "Task: " ${TASK_NAME}
echo "Model: "${MODEL_NAME}
echo "Dataset: "${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "Experiment path: "${EXPERIMENT_PATH}

cd ${YAIB_PATH}

eval "$(conda shell.bash hook)"
conda activate yaib



icu-benchmarks train \
  -d ${DATASET_ROOT_PATH} \
  -n ${DATASETS} \
  -t ${TASK} \
  -tn ${TASK_NAME} \
  -m ${MODEL_NAME} \
  -c \
  -s 1111 \
  -l ${EXPERIMENT_PATH} \
  --tune