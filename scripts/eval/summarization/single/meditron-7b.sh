EXP_ID=$1
MODEL_PATH=epfl-llm/meditron-7b
CACHE_DIR=/data/linjc/hub
FILE_DIR=data/downstream/summazization/single-trial
SAVE_DIR=data/downstream/summazization/single-trial/results$EXP_ID
MODEL_NAME=meditron-7b

CUDA_VISIBLE_DEVICES=3 python src/eval/summarization/single/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --file_dir $FILE_DIR \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test