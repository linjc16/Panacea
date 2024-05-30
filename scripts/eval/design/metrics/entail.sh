MODEL_NAME=$1
TASK=$2
EXP_ID=$3
RES_DIR=data/downstream/design/results$EXP_ID/$TASK/eval_entail
# RES_DIR=$3

python src/eval/design/metrics/entail.py \
    --model_name $MODEL_NAME \
    --res_dir $RES_DIR \
    --task $TASK