CUDA_VISIBLE_DEVICES=2,3,5,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port 1234 src/instruction_tuning/sft.py recipes/panacea/sft/config_full.yaml