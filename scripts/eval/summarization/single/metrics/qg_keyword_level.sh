MODEL_NAME=$1

python src/eval/summarization/single/metrics/jaccard_query_gen.py \
    --model_name $MODEL_NAME \