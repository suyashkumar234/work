#!/bin/bash
#SBATCH --job-name=cowpro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cowpro_%j.out
#SBATCH --error=cowpro_%j.err

module load cuda/12.6.3
export GIT_PYTHON_REFRESH=quiet
cd /scratch/$USER/cowpro

# Use direct python path - bypass conda activation issues
/scratch/suyash.kumar.mec22.itbhu/cowpro_env/bin/python training.py with dataset='SABS_Superpix'