#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPU=4
REAL_BS=$(($BS * $NUM_GPU))

PORT_ID=$(expr $RANDOM + 1000)

export OMP_NUM_THREADS=32

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID main.py \
    --model_name_or_path t5-base \
    --output_dir result/test \
    --train_file data/nli_for_simcse.csv \
    --max_source_length 64 \
    --max_target_length 32 \
    --do_train \
    --do_eval \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --temp 0.05 \
    --lr_scheduler_type linearconstant \
    --adafactor \
    --num_train_epochs 3 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 250 \
    --logging_steps 250 \
    --save_steps 250 \
    --pooler_type cls_before_pooler \
    --mlp_only_train \
    --overwrite_output_dir \
    --model_type seq2seq \
    --dataloader_num_workers 64 \
    --preprocessing_num_workers 64

