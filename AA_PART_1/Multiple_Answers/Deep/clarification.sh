#!/bin/bash
#SBATCH --job-name=deep_baseline
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --account=gusr97847
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=deep_baseline_%j.log

# Load conda and activate your environment
source /projects/0/prjs1482/miniconda/etc/profile.d/conda.sh
conda activate /projects/0/prjs1482/miniconda/envs/deepseek

# Run your script
python clarification.py
