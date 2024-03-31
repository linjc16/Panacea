MODEL_PATH=epfl-llm/meditron-7b
CACHE_DIR=/data/linjc/hub
FILE_DIR=data/downstream/search/query_expansion
SAVE_DIR=data/downstream/search/query_expansion/results
MODEL_NAME=meditron-7

CUDA_VISIBLE_DEVICES=7 python src/eval/search/query_expansion/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --file_dir $FILE_DIR \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test