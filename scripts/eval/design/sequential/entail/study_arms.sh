MODEL_NAME=$1

# RES_DIR=data/downstream/design/results/sequential/study_arms
# python src/eval/design/metrics/entail/study_arms.py \
#     --model_name $MODEL_NAME \
#     --res_dir $RES_DIR

for EXP_ID in 1 2
do
    RES_DIR=data/downstream/design/results$EXP_ID/sequential/study_arms
    python src/eval/design/metrics/entail/study_arms.py \
        --model_name $MODEL_NAME \
        --res_dir $RES_DIR
done