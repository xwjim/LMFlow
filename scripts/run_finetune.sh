#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

# Parses arguments
exp_id=finetune
model_name_or_path=~/github/llm_ckpt/Ziya-LLaMA-13B/
dataset_path=data/lawerv3/
output_dir=output_models/${exp_id}
deepspeed_args="--master_port=11000"

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --block_size 4196 \
    --per_device_train_batch_size 3 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --gradient_checkpointing \
    --truncate_to_model_max_length False \
    --do_rope_scaling True \
    --rope_pi_ratio 1 \
    --rope_ntk_ratio 4 \
    --ddp_timeout 72000 \
    --save_steps 400 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err

  # --use_flash_attention \