TASK=$1
SAVE_DIR=data/downstream/design/results/$TASK/
MODEL_NAME=claude-sonnet

python src/eval/design/eval_aws.py \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test