MODEL_NAME=$1
EXP_ID=$2
RES_DIR=data/downstream/summazization/single-trial/results$EXP_ID

CUDA_VISIBLE_DEVICES=3 python src/eval/summarization/single/metrics/query_gen.py \
    --model_name $MODEL_NAME \
    --res_dir $RES_DIR