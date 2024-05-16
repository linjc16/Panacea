MODEL_NAME=$1

for EXP_ID in 1 2
do
    RES_DIR=data/downstream/design/results$EXP_ID/outcome_measures
    python src/eval/design/metrics/entail/outcome_measures.py \
        --model_name $MODEL_NAME \
        --res_dir $RES_DIR
done