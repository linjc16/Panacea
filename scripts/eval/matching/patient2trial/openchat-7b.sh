MODEL_PATH=openchat/openchat-3.5-0106
CACHE_DIR=/data/linjc/hub
DATASET=cohort
SAVE_DIR=data/downstream/matching/patient2trial/cohort/results
MODEL_NAME=openchat-7b

CUDA_VISIBLE_DEVICES=4 python src/eval/matching/patient2trial/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test