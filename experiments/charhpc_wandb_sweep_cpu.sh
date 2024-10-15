#!/bin/bash
#SBATCH --job-name=yaib_experiment
#SBATCH --partition=pgpu # -p
#SBATCH --cpus-per-task=16 # -c
#SBATCH --mem=100gb
#SBATCH --output=logs/classification_%a_%j.log # %j is job id
#SBATCH --time=24:00:00

source /etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate yaib_req_pl
wandb agent --count 1 cassandra_hpi/cassandra/"$1"

# Debug instance: srun -p gpu --pty -t 5:00:00 --gres=gpu:1 --cpus-per-task=16 --mem=100GB bash