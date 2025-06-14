#!/bin/bash
#SBATCH --job-name=probe_all_layers
#SBATCH --output=probe_all_layers.out
#SBATCH --error=probe_all_layers.err
#SBATCH --time=10:00:00
#SBATCH --partition=general
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load modules if needed
# module load python/3.9
# module load cuda/11.8

# Activate your environment
# source venv/bin/activate 

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/net/scratch2/cot_interp/.cache

# Run the probe pipeline
echo "Starting probe pipeline at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python probe.py

echo "Probe pipeline completed at $(date)"

# Optional: Send notification when done
# echo "Probe pipeline completed on $SLURM_NODELIST" | mail -s "Job $SLURM_JOB_ID Complete" jojiao@uchicago.edu 