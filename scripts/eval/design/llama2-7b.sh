TASK=$1
EXP_ID=$2 # results
MODEL_PATH=NousResearch/Llama-2-7b-chat-hf
CACHE_DIR=/data/linjc/hub
SAVE_DIR=data/downstream/design/results$EXP_ID/$TASK/
MODEL_NAME=llama2-7b

CUDA_VISIBLE_DEVICES=3 python src/eval/design/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test \
    --sample True