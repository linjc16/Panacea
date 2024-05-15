DATASET=$1
EXP_ID=$2 # results
MODEL_PATH=NousResearch/Llama-2-7b-chat-hf
CACHE_DIR=/data/linjc/hub
SAVE_DIR=data/downstream/matching/patient2trial/$DATASET/results$EXP_ID
MODEL_NAME=llama2-7b

CUDA_VISIBLE_DEVICES=2 python src/eval/matching/patient2trial/eval.py \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test