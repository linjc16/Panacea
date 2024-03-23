MODEL_NAME=$1
TASK=$2
RES_DIR=data/downstream/design/results/$TASK

python src/eval/design/metrics/rouge.py \
    --model_name $MODEL_NAME \
    --res_dir $RES_DIR