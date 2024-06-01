MODEL_NAME=$1

python src/eval/summarization/single/metrics/jaccard_qg_class_level.py \
    --model_name $MODEL_NAME \