#!/bin/bash
#SBATCH --job-name=yaib_experiment
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=200gb
#SBATCH --output=logs/classification_%a_%j.log # %j is job id
#SBATCH --gpus=1
#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"
conda activate yaib_req_pl
wandb agent --count 1 cassandra_hpi/cassandra/"$1"


