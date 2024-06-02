MODEL_NAME=$1

python src/eval/summarization/single/metrics/jaccard_qe.py \
    --model_name $MODEL_NAME \