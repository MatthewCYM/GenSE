#!/bin/bash

#export CUDA_MPS_PIPE_DIRECTORY=/rsch/tzy/tmp/nvidia-mps
#export CUDA_MPS_LOG_DIRECTORY=/rsch/tzy/tmp/nvidia-log
export CUDA_VISIBLE_DEVICES=0


TESTFILE=data/c4_for_generation.txt
OUTPUTDIR=generated_data/c4

python generation.py \
    --model_name_or_path nli-synthesizer-t5-base \
    --train_file data/nli-train.csv \
    --test_file ${TESTFILE} \
    --max_source_length 96 \
    --max_target_length 64 \
    --num_beams 1 \
    --do_sample \
    --top_p 0.9 \
    --do_predict \
    --per_device_eval_batch_size 384 \
    --output_dir ${OUTPUTDIR} \
    --predict_with_generate \
    --cache_dir cache \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 32 \
    --eval_accumulation_steps 500

python merge_data.py \
    --input_file ${TESTFILE} \
    --output_dir ${OUTPUTDIR}
