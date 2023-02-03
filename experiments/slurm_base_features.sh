#!/bin/bash
#SBATCH --job-name=default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[INSERT:EMAIL]
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --output=../%x/%x_%a_%j.log # %x is job-name, %j is job id, %a is array id
#SBATCH --array=0-3

# Submit with e.g. --export=TASK_NAME=mortality24,MODEL_NAME=LGBMClassifier
# Basic experiment variables, please exchange [INSERT] for your experiment parameters
YAIB_PATH=/dhc/home/robin.vandewater/projects/yaib #/dhc/home/robin.vandewater/projects/yaib
cd ${YAIB_PATH}

eval "$(conda shell.bash hook)"
conda activate yaib

TASK=BinaryClassificationNoStatic # BinaryClassification
EXPERIMENT_PATH=../${TASK_NAME}
DATASET_ROOT_PATH=data/YAIB_Datasets/data #data/YAIB_Datasets/data
DATASETS=(aumc hirid eicu miiv)

echo "This is a SLURM job named" $SLURM_JOB_NAME "with array id" $SLURM_ARRAY_TASK_ID "and job id" $SLURM_JOB_ID
echo "Resources allocated: " $SLURM_CPUS_PER_TASK "CPUs, " $SLURM_MEM_PER_NODE "GB RAM, " $SLURM_GPUS_PER_NODE "GPU"
echo "Task:  " ${TASK_NAME}" Model: "${MODEL_NAME}" Dataset:" ${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "Experiment path: "  ${EXPERIMENT_PATH}





icu-benchmarks train \
  -d ${DATASET_ROOT_PATH}/${TASK_NAME}/${DATASETS[$SLURM_ARRAY_TASK_ID]} \
  -n ${DATASETS[$SLURM_ARRAY_TASK_ID]} \
  -t ${TASK} \
  -tn ${TASK_NAME} \
  -m ${MODEL_NAME} \
  -c \
  -s 1111 \
  -l ${EXPERIMENT_PATH} \
  --no-verbose \
  --tune \
  --checkpoint test