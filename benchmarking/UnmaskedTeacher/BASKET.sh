#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='eval'
OUTPUT_DIR="output/UMT_BASKET"
LOG_DIR="./output/"

PREFIX='your_path_to_BASKET'
DATA_PATH='your_path_to_BASKET'

MODEL_PATH='your_path_to_checkpoint'

sbatch --gpus 2 -J $JOB_NAME --wrap="torchrun --nproc_per_node=2 \
        --master_port ${MASTER_PORT} --nnodes=1 \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'basketball' \
        --split ',' \
        --nb_classes 5 \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 2 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 32 \
        --num_workers 8 \
        --warmup_epochs 2 \
        --tubelet_size 1 \
        --epochs 20 \
        --lr 7e-3 \
        --drop_path 0.1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 1 \
        --test_num_crop 1 \
        --dist_eval \
        --enable_deepspeed \
        --test_best \
        --mixup 0 \
        --eval
"