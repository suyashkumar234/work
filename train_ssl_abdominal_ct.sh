#!/bin/bash
# train a model to segment abdominal CT 
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="myexp"
DATASET='SABS_Superpix'
NWORKER=0

ALL_EV=( 0) # 5-fold cross validation (0, 1, 2, 3, 4)
ALL_SCALE=( "MIDDLE") # config of pseudolabels, Medium-grained superpixels (balanced size)

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[2,3]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[1,6]' 

###### Training configs ######
NSTEP=1000 # reduced for fast debugging
DECAY=0.95
# num_epochs = total_iterations / (num_samples / batch_size)
MAX_ITER=100 # reduced for fast debugging
# if batch size = 1, then 500 samples for 500 iterations; if batch size = 4, then 2000 samples for 500 iterations
# set to 5 for debugging and learning code original 1000 
SNAPSHOT_INTERVAL=10 # interval for saving snapshot
SEED='1234'

###### Validation configs ######
SUPP_ID='[6]' # using the additionally loaded scan as support

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}" # running 5-fold cross validation
do # marks the beginning of a fold
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
    PREFIX="train_${DATASET}_lbgroup${LABEL_SETS}_scale_${SUPERPIX_SCALE}_vfold${EVAL_FOLD}"
    echo $PREFIX # prints the prefix for the current fold
    LOGDIR="./exps/${CPT}_${SUPERPIX_SCALE}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir $LOGDIR
    fi

    python3 training.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    batch_size=1 \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID
    done
done
