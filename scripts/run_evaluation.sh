#!/bin/bash

deepspeed_args="--master_port=29501"      # Default argument

deepspeed ${deepspeed_args} examples/evaluation.py \
    --answer_type text \
    --model_name_or_path ./output_models/lawerv7/checkpoint-800/ \
    --dataset_path data/lawer_citation_test \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy \
    --max_new_tokens 1024 \
    --repetition_penalty 1.05 \
    --temperature 0.1 \
    --prompt_structure "{input}" \
    --use_ram_optimized_load False \
