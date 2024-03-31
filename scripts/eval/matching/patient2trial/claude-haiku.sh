DATASET=$1
SAVE_DIR=data/downstream/matching/patient2trial/$DATASET/results
MODEL_NAME=claude-haiku

python src/eval/matching/patient2trial/eval_aws.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --split test