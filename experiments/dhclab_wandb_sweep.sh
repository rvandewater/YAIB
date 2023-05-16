#!/bin/bash
#SBATCH --job-name=yaib_experiment
#SBATCH --partition="hpcpu,gpupro,gpua100" # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=60gb
#SBATCH --output=demo_los_sweep_%a_%j.log # %j is job id
#SBATCH --gpus=0
#SBATCH --time=72:00:00

eval "$(conda shell.bash hook)"
conda activate yaib_nvidia
wandb agent --count 1 robinvandewater/yaib-experiments/"$1"


