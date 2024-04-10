MODEL_NAME=$1
TASK=$2
RES_DIR=data/downstream/design/results/$TASK

CUDA_VISIBLE_DEVICES=1 python src/eval/design/metrics/bertscore.py \
    --model_name $MODEL_NAME \
    --res_dir $RES_DIR