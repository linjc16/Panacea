DATASET=$1
MODEL_NAME=$2
EXP_ID=$3
RES_DIR=data/downstream/matching/patient2trial/$DATASET/results$EXP_ID



python src/eval/matching/patient2trial/metrics/cls_metrics.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --res_dir $RES_DIR