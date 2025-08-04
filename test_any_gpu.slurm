#!/bin/bash
#SBATCH --job-name=cowpro_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=test_%j.out
module load cuda/12.6.3
export GIT_PYTHON_REFRESH=quiet
conda activate /scratch/$USER/cowpro_env
cd /scratch/$USER/cowpro
bash test_ssl_abdominal_ct.sh
