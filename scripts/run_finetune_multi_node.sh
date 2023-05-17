#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

export NUM_GPUS_PER_WORKER=1
export NUM_WORKERS=2

export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NET_GDR_LEVEL=2

export MASTER_ADDR=10.10.255.128
export MASTER_PORT=31228

exp_id=finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/belle/

mkdir -p ${output_dir} ${log_dir}

CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_nodes $NUM_WORKERS \
    --num_gpus $NUM_GPUS_PER_WORKER \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --hostfile $project_dir/configs/hostfile \
  examples/finetune.py \
    --model_name_or_path ../llm_ckpt/belle-ext-7b/ \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --block_size 512 \
    --per_device_train_batch_size 60 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --streaming \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 2000 \
    --gradient_checkpointing \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
