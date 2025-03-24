#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

export DECORD_EOF_RETRY_MAX=20480

JOB_NAME='videomae_eval'
OUTPUT_DIR="output/VideoMamba_BASKET"

DATA_PATH='your_path_to_BASKET'

MODEL_PATH='your_path_to_checkpoint'

sbatch --gpus 8 --ntasks 1 --wrap="torchrun --nproc_per_node=8 \
        --master_port ${MASTER_PORT} --nnodes=1 \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set basketball \
        --nb_classes 5 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 32 \
        --sampling_rate 412 \
        --num_sample 1 \
        --num_workers 4 \
        --opt adamw \
        --lr 7e-4 \
        --drop_path 0.1 \
        --head_drop_rate 0.0 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --warmup_epochs 5 \
        --epochs 20 \
        --test_num_segment 1 \
        --test_num_crop 1 \
        --dist_eval \
        --enable_deepspeed \
        --mixup 0 \
        --eval
"