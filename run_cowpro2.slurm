#!/bin/bash
#SBATCH --job-name=cowpro
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu007
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00

module load cuda/12.6.3
export GIT_PYTHON_REFRESH=quiet
#export SACRED_NO_GIT=1  # Disable Sacred's git integration
cd /scratch/$USER/cowpro

PYTHON_PATH="/scratch/suyash.kumar.mec22.itbhu/cowpro_env/bin/python"

CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH training.py with dataset='CHAOST2_Superpix' eval_fold=3 label_sets=0 exclude_cls_list="[2,3]" &
CUDA_VISIBLE_DEVICES=1 $PYTHON_PATH training.py with dataset='CHAOST2_Superpix' eval_fold=4 label_sets=0 exclude_cls_list="[2,3]" &
wait