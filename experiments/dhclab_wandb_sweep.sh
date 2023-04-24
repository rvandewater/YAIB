#!/bin/bash
#SBATCH --job-name=yaib_experiment
#SBATCH --partition="gpua100" # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=100gb
#SBATCH --output=sepsis_%a_%j.log # %j is job id
#SBATCH --gpus=1

eval "$(conda shell.bash hook)"
conda activate yaib_new
wandb agent --count 1 robinvandewater/yaib-experiments/k0hh4m68


