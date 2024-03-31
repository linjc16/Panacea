MODEL_PATH=/shared/jl254/data/linjc/trialfm/finetuning/patient2trial/panacea
CACHE_DIR=/data/linjc/hub
DATASET=$1
SAVE_DIR=data/downstream/matching/patient2trial/$DATASET/results
MODEL_NAME=panacea-ft

CUDA_VISIBLE_DEVICES=7 python src/eval/matching/patient2trial/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test