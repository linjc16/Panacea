TASK=$1
EXP_ID=$2 # results
MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
CACHE_DIR=/data/linjc/hub
SAVE_DIR=data/downstream/design/results$EXP_ID/$TASK/
MODEL_NAME=llama3-8b

CUDA_VISIBLE_DEVICES=3 python src/eval/design/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test \
    --sample True