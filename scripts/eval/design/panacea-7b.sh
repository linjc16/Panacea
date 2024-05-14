TASK=$1
EXP_ID=$2 # results
MODEL_PATH=/shared/jl254/data/linjc/trialfm/sft/panacea-chat-v2
CACHE_DIR=/data/linjc/hub
SAVE_DIR=data/downstream/design/results$EXP_ID/$TASK/
MODEL_NAME=panacea-7b

CUDA_VISIBLE_DEVICES=1   python src/eval/design/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test