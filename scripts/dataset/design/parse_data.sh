TASK=$1
SPLIT=$2

python src/dataset/design/parse_chat_data.py \
    --task $TASK \
    --split $SPLIT