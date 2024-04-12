TASK=$1
RES_DIR=data/downstream/design/results/$TASK/eval_entail

python src/eval/design/metrics/plot/entail.py \
    --res_dir $RES_DIR \
    --task $TASK