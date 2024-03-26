MODEL_PATH=
CACHE_DIR=/data/linjc/hub
DATASET=$1
SAVE_DIR=data/downstream/matching/patient2trial/$DATASET/results
MODEL_NAME=panacea-ft

CUDA_VISIBLE_DEVICES=4 python src/eval/matching/patient2trial/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test