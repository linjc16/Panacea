MODEL_NAME=meta-llama/Llama-2-7b-hf

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 2 --main_process_port 1234 src/pretrain/pretrain.py \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 128 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --output_dir /data/linjc/trialfm/models-new/pretrain-v1 \
    --model_name $MODEL_NAME \
    --warmup_ratio 0.01 \
    --save_steps 200 \
    --weight_decay 0.05 \

# /data/linjc/trialfm/models-new/pretrain-v1/runs
# tensorboard --logdir=./Feb01_01-53-05_sunlab-serv-04.cs.illinois.edu --port 8909