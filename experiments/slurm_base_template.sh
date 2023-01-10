#!/bin/bash
#SBATCH --job-name=[INSERT:TASK_NAME,MODEL_NAME]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[INSERT:EMAIL]
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=4 # -c
#SBATCH --mem=48gb
#SBATCH --gpus=1
#SBATCH --output=[INSERT:TASK_NAME,MODEL_NAME]_%a_%j.log # %j is job id, %a is array id
#SBATCH --array=0-3

# Basic experiment variables, please exchange [INSERT] for your experiment parameters
TASK_NAME=[INSERT:TASK_NAME] # mortality24
MODEL_NAME=[INSERT:MODEL_NAME] # LGBM
TASK=[INSERT:TASK_TYPE] # BinaryClassification
YAIB_PATH=[INSERT:YAIB_PATH] #/dhc/home/robin.vandewater/projects/yaib
EXPERIMENT_PATH=../${TASK_NAME}_experiment
DATASET_ROOT_PATH=[INSERT:COHORT_ROOT] #data/YAIB_Datasets/data
DATASETS=(hirid miiv eicu aumc)

cd ${YAIB_PATH}

eval "$(conda shell.bash hook)"
conda activate yaib

icu-benchmarks train \
  -d ${DATASET_ROOT_PATH}/${TASK_NAME}/${DATASETS[$SLURM_ARRAY_TASK_ID]} \
  -n ${DATASETS[$SLURM_ARRAY_TASK_ID]} \
  -t ${TASK} \
  -tn ${TASK_NAME} \
  -m ${MODEL_NAME} \
  -c \
  -s 1111 2222 3333 4444 5555 \
  -l ${EXPERIMENT_PATH} \
  --tune