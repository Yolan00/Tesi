#!/bin/bash
#SBATCH --job-name=qwen_clarification
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --account=gusr97847
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --output=qwen_clarification_%j.log

# Load conda and activate your environment
source /projects/0/prjs1482/miniconda/etc/profile.d/conda.sh
conda activate /projects/0/prjs1482/miniconda/envs/qwen


# Run your script
python clarification.py
