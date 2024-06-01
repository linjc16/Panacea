DATASET=$1
MODEL_NAME=$2
EXP_ID=$3
RES_DIR=data/downstream/matching/patient2trial/$DATASET/results$EXP_ID



python src/vis/analysis/downstream/matching/hallucination/eval_hall.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --res_dir $RES_DIR