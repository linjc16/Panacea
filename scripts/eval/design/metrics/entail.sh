MODEL_NAME=$1
TASK=$2
# RES_DIR=data/downstream/design/results/$TASK/eval_entail
RES_DIR=$3

python src/eval/design/metrics/entail.py \
    --model_name $MODEL_NAME \
    --res_dir $RES_DIR