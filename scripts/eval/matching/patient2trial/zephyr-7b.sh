MODEL_PATH=HuggingFaceH4/zephyr-7b-beta
CACHE_DIR=/data/linjc/hub
DATASET=cohort
SAVE_DIR=data/downstream/matching/patient2trial/$DATASET/results
MODEL_NAME=zephyr-7b

CUDA_VISIBLE_DEVICES=6 python src/eval/matching/patient2trial/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test