MODEL_NAME=$1

python src/eval/summarization/multi/metrics/jaccard_qe.py \
    --model_name $MODEL_NAME \