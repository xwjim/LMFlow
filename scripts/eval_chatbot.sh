#!/bin/bash


CUDA_VISIBLE_DEVICES=2 python ./examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path ../llm_ckpt/belle-ext-13b/  --prompt_structure "###Human: {input_text}###Assistant:" --end_string "#" --max_new_tokens 200 --repetition_penalty 1.02 --do_sample --top_p 0.9 --use_ram_optimized_load False