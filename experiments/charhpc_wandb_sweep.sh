#!/bin/bash
#SBATCH --job-name=yaib_experiment
#SBATCH --partition=pgpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=200gb
#SBATCH --output=logs/classification_%a_%j.log # %j is job id
#SBATCH --gpus=1
#SBATCH --time=24:00:00

source /etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate yaib_req_pl
wandb agent --count 1 cassandra_hpi/cassandra/"$1"