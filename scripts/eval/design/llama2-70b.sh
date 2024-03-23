TASK=$1
MODEL_PATH=NousResearch/Llama-2-70b-chat-hf
CACHE_DIR=/data/linjc/hub
SAVE_DIR=data/downstream/design/results/$TASK/
MODEL_NAME=llama2-70b

CUDA_VISIBLE_DEVICES=1,7 python src/eval/design/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test