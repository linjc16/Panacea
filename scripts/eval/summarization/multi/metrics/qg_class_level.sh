MODEL_NAME=$1

python src/eval/summarization/multi/metrics/jaccard_qg_class_level.py \
    --model_name $MODEL_NAME \