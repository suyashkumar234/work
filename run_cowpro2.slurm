#!/bin/bash
#SBATCH --job-name=cowpro_labelsets0
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu011
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
# --time=4-00:00:00  # Commented out to use default

module load cuda/12.6.3
export GIT_PYTHON_REFRESH=quiet
cd /scratch/$USER/cowpro

PYTHON_PATH="/scratch/suyash.kumar.mec22.itbhu/cowpro_env/bin/python"

echo "Starting all label_sets=0 training jobs in parallel..."

# Run all 5 jobs in parallel - alternating between GPU 0 and GPU 1
CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH training.py with dataset="CHAOST2_Superpix" eval_fold=0 label_sets=0 exclude_cls_list="[2,3]" &
CUDA_VISIBLE_DEVICES=1 $PYTHON_PATH training.py with dataset="CHAOST2_Superpix" eval_fold=1 label_sets=0 exclude_cls_list="[2,3]" &
# CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH training.py with dataset="CHAOST2_Superpix" eval_fold=2 label_sets=0 exclude_cls_list="[2,3]" &
# CUDA_VISIBLE_DEVICES=1 $PYTHON_PATH training.py with dataset="CHAOST2_Superpix" eval_fold=3 label_sets=0 exclude_cls_list="[2,3]" &
# CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH training.py with dataset="CHAOST2_Superpix" eval_fold=4 label_sets=0 exclude_cls_list="[2,3]" &

wait  # Wait for all background jobs to complete

echo "All label_sets=0 training completed!"