CUDA_VISIBLE_DEVICES=2,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=2 src/finetune/finetune.py recipes/mistral/finetune/summarization/single_trial.yaml