TASK=$1
MODEL_PATH=NousResearch/Llama-2-13b-chat-hf
CACHE_DIR=/data/linjc/hub
SAVE_DIR=data/downstream/design/results/$TASK/
MODEL_NAME=llama2-13b

CUDA_VISIBLE_DEVICES=0 python src/eval/design/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test