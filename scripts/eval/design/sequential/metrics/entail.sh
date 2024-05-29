MODEL_NAME=$1
TASK=$2
EXP_ID=$3
RES_DIR=data/downstream/design/results$EXP_ID/sequential/$TASK/eval_entail

python src/eval/design/metrics/entail.py \
    --model_name $MODEL_NAME \
    --res_dir $RES_DIR