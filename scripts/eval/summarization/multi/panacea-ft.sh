MODEL_PATH=/data/linjc/trialfm/finetuning/multi-trial-sum/panacea
CACHE_DIR=/data/linjc/hub
FILE_DIR=data/downstream/summazization/multi-trial
SAVE_DIR=data/downstream/summazization/multi-trial/results
MODEL_NAME=panacea-ft

CUDA_VISIBLE_DEVICES=0 python src/eval/summarization/multi/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --file_dir $FILE_DIR \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test