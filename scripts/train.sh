#!/bin/bash
# json_path = '../csd_db_interact-master/Data.json'

NUM_NODES=1
NUM_GPUS_PER_NODE=2

BATCH_SIZE=128
ACCUM_STEP=2

SAVE_PATH=../output/initial_metalloscribe_run

#Removed: --warmup 0.02\     --limit_train_batches 10\ --limit_val_batches 3 \ --do_train \
#Learning rate was originally 1e-4

set -x
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 python ../train.py \
    --save_path $SAVE_PATH \
    --train_file ../dataset_balanced_2.json\
    --images_folder ../images\
    --use_training_for_validation \
    --train_validation_ratio 0.8 \
    --lr 4e-4 \
    --epochs 10 --eval_per_epoch 1 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_train \
    --do_valid \
    --warmup_ratio 0.02 \
    --gpus $NUM_GPUS_PER_NODE \
    --cache_dir /data/scratch/richwang/.local/bin\