MODEL_PATH=/data/linjc/trialfm/models-mistral/pretrain-v3
CACHE_DIR=/data/linjc/hub
DATASET=$1
SAVE_DIR=data/downstream/matching/patient2trial/$DATASET/results
MODEL_NAME=panacea-base

CUDA_VISIBLE_DEVICES=1 python src/eval/matching/patient2trial/eval_base.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test