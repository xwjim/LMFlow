#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=report_v7_2
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/report_v7/ #/home/zhfeng/WuDaoCorpus2.0_base_200G/WuDaoCorpus2.0_textonly #${project_dir}/data/documentqa/

mkdir -p ${output_dir} ${log_dir}

CUDA_VISIBLE_DEVICES=0,1 deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path output_models/pretrain_12000_clean_0.2M_ch_en_v3/checkpoint-2800/ \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 4 \
    --learning_rate 6e-6 \
    --block_size 2048 \
    --per_device_train_batch_size 7 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name $exp_id \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 40 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err

    # --lr_scheduler_type constant \ --streaming \
#     --resume_from_checkpoint output_models/report/checkpoint-30 \