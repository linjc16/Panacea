MODEL_PATH=NousResearch/Llama-2-7b-chat-hf
CACHE_DIR=/data/linjc/hub
FILE_DIR=data/downstream/search/query_generation
SAVE_DIR=data/downstream/search/query_generation/results
MODEL_NAME=llama2-7b

CUDA_VISIBLE_DEVICES=5 python src/eval/search/query_generation/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --file_dir $FILE_DIR \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test