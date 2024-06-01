MODEL_NAME=$1
EXP_ID=$2
FILE_DIR=data/downstream/summazization/single-trial/results/query_gen

CUDA_VISIBLE_DEVICES=5 python src/eval/summarization/single/metrics/query_expan.py \
    --model_name $MODEL_NAME \
    --file_dir $FILE_DIR \