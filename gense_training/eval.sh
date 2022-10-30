#!/bin/bash



export CUDA_VISIBLE_DEVICES=0


python evaluation.py \
        --model_name_or_path mattymchen/gense-base \
        --pooler cls_before_pooler \
        --mode test \
        --task_set sts \
        --add_prompt


