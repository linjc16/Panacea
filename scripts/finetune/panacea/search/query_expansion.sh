CUDA_VISIBLE_DEVICES=5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=2 --main_process_port 12234 src/finetune/finetune.py recipes/panacea/finetune/search/query_expansion.yaml