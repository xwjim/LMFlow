#!/bin/bash

deepspeed_args="--master_port=29501"      # Default argument

CUDA_VISIBLE_DEVICES=2 \
    deepspeed ${deepspeed_args} examples/evaluate.py \
    --answer_type text \
    --model_name_or_path ~/finetune_qasum_8000/checkpoint-12/ \
    --dataset_path data/BELLE/ \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 3 \
    --metric accuracy \
    --use_ram_optimized_load False \
