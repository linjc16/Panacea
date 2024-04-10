import os
import argparse
import openai
import json
import pdb
from tqdm import tqdm
import sys

sys.path.append('./')
from src.utils.claude_aws import chat_haiku


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results/study_arms')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()


    with open(os.path.join(args.res_dir, f'{args.model_name}.json'), 'r') as f:
        results = json.load(f)

    preds = []
    groundtruth = []
    for key, value in results.items():
        preds.extend(value['model_response'])
        groundtruth.extend(value['groundtruth'])

    assert len(preds) == len(groundtruth)

    