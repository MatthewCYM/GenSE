#!/bin/bash

#export CUDA_MPS_PIPE_DIRECTORY=/rsch/tzy/tmp/nvidia-mps
#export CUDA_MPS_LOG_DIRECTORY=/rsch/tzy/tmp/nvidia-log
export CUDA_VISIBLE_DEVICES=0



MODEL=nli-synthesizer-t5-base

TESTFILE=generated_data/c4/test.csv
OUTPUTDIR=generated_data/c4/

python discrimination.py \
    --model_name_or_path $MODEL \
    --train_file data/nli-train.csv \
    --test_file ${TESTFILE} \
    --max_source_length 128 \
    --max_target_length 64 \
    --do_predict \
    --per_device_eval_batch_size 512 \
    --output_dir ${OUTPUTDIR} \
    --cache_dir cache \
    --preprocessing_num_workers 128

python filter_data.py --output_dir ${OUTPUTDIR}