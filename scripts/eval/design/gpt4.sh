TASK=$1
SAVE_DIR=data/downstream/design/results/$TASK/
MODEL_NAME=gpt-4

python src/eval/design/eval_gpt.py \
    --task $TASK \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --split test