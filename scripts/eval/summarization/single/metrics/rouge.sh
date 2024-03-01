MODEL_NAME=$1

python src/eval/summarization/single/metrics/rouge.py \
    --model_name $MODEL_NAME