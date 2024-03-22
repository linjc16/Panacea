TASK=$1
MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.1
CACHE_DIR=/data/linjc/hub
SAVE_DIR=data/downstream/design/results/$TASK/
MODEL_NAME=mistral-7b

CUDA_VISIBLE_DEVICES=2 python src/eval/design/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test