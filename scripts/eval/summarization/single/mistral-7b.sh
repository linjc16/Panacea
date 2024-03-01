MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.1
CACHE_DIR=/data/linjc/hub
FILE_DIR=data/downstream/summazization
SAVE_DIR=data/downstream/summazization/results
MODEL_NAME=mistral-7b

CUDA_VISIBLE_DEVICES=4 python src/eval/summarization/single/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --file_dir $FILE_DIR \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test