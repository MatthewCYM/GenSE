#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=64

BS=64
NUM_GPU=4
REAL_BS=$(($BS * $NUM_GPU))

PORT_ID=$(expr $RANDOM + 1000)


python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID main.py \
        --train_file data/nli-train.csv \
        --validation_file data/nli-dev.csv \
        --keep_label entailment contradiction \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --model_name_or_path t5-base \
        --max_source_length 96 \
        --max_target_length 48 \
        --per_device_train_batch_size ${BS} \
        --per_device_eval_batch_size 128 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --num_train_epochs 10 \
        --output_dir result/nli-synthesizer-t5-base \
        --seed 42 \
        --metric_for_best_model avg \
        --greater_is_better True \
        --load_best_model_at_end \
        --lr_scheduler_type linearconstant \
        --adafactor \
        --overwrite_output_dir \
        --evaluation_strategy steps \
        --logging_strategy steps \
        --save_strategy steps \
        --logging_steps 500 \
        --save_steps 500 \
        --eval_steps 500 \
        --preprocessing_num_workers 64 \
        --dataloader_num_workers 64