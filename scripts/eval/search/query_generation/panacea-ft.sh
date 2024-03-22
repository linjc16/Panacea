MODEL_PATH=/data/linjc/trialfm/finetuning/query_generation/panacea
CACHE_DIR=/data/linjc/hub
FILE_DIR=data/downstream/search/query_generation
SAVE_DIR=data/downstream/search/query_generation/results
MODEL_NAME=panacea-ft

CUDA_VISIBLE_DEVICES=3 python src/eval/search/query_generation/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --file_dir $FILE_DIR \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test