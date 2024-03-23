TASK=$1
MODEL_PATH=/shared/jl254/data/linjc/trialfm/finetuning/trial_design/panacea
CACHE_DIR=/data/linjc/hub
SAVE_DIR=data/downstream/design/results/$TASK/
MODEL_NAME=panacea-ft

CUDA_VISIBLE_DEVICES=4 python src/eval/design/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test