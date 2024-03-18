DATASET=cohort
RES_DIR=data/downstream/matching/patient2trial/$DATASET/results
MODEL_NAME=$1

python src/eval/matching/patient2trial/metrics/cls_metrics.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --res_dir $RES_DIR