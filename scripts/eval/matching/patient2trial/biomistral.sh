MODEL_PATH=BioMistral/BioMistral-7B
CACHE_DIR=/data/linjc/hub
DATASET=$1
SAVE_DIR=data/downstream/matching/patient2trial/$DATASET/results
MODEL_NAME=biomistral

CUDA_VISIBLE_DEVICES=0 python src/eval/matching/patient2trial/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test