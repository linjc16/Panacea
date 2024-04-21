TASK=$1
MODEL_NAME=$2
RES_DIR=data/downstream/design/results/$TASK/eval_entail
# RES_DIR=$3

python src/eval/design/metrics/entail.py \
    --model_name $MODEL_NAME \
    --res_dir $RES_DIR