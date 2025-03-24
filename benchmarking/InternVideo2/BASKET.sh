#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='eval'
OUTPUT_DIR="output/InternVideo2_BASKET"
LOG_DIR="./output/"

PREFIX='your_path_to_BASKET'
DATA_PATH='your_path_to_BASKET'

MODEL_PATH='your_path_to_checkpoint'

sbatch --gpus 8 --ntasks 1 --wrap="torchrun --nproc_per_node=8 \
        --master_port ${MASTER_PORT} --nnodes=1 \
        run_finetuning.py \
        --model internvideo2_1B_patch14_224 \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'basketball' \
        --split ',' \
        --nb_classes 5 \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --steps_per_print 50 \
        --batch_size 8 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 32 \
        --num_workers 12 \
        --warmup_epochs 0 \
        --tubelet_size 1 \
        --epochs 20 \
        --lr 6e-5 \
        --drop_path 0.3 \
        --layer_decay 0.9 \
        --use_checkpoint \
        --checkpoint_num 24 \
        --layer_scale_init_value 1e-5 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 1 \
        --test_num_crop 1 \
        --dist_eval \
        --enable_deepspeed \
        --bf16 \
        --zero_stage 1 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --test_best \
        --eval
"
