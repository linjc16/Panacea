MODEL_NAME=$1

python src/eval/summarization/multi/metrics/rouge.py \
    --model_name $MODEL_NAME