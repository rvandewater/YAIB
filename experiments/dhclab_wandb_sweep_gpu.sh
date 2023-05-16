#!/bin/bash
#SBATCH --job-name=yaib_experiment
#SBATCH --partition="gpupro,gpu" # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=75gb
#SBATCH --output=los_transformer_tcn_%a_%j.log # %j is job id
#SBATCH --gpus=1
#SBATCH --time=72:00:00

eval "$(conda shell.bash hook)"
conda activate yaib_nvidia
wandb agent --count 1 robinvandewater/yaib-experiments/"$1"


