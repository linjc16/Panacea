MODEL_NAME=$1
EXP_ID=$2
RES_DIR=data/downstream/summazization/single-trial/results$EXP_ID

python src/eval/summarization/single/metrics/gpt_eval_conclusion.py \
    --model_name $MODEL_NAME \
    --res_dir $RES_DIR