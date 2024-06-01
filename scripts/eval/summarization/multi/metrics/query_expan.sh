MODEL_NAME=$1
EXP_ID=$2
FILE_DIR=data/downstream/summazization/multi-trial/results/query_gen

CUDA_VISIBLE_DEVICES=1 python src/eval/summarization/multi/metrics/query_expan.py \
    --model_name $MODEL_NAME \
    --file_dir $FILE_DIR \