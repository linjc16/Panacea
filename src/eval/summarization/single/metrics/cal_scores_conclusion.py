import json
import re
import argparse
import os
import pdb

def extract_last_digit_eval(data):

    last_digits = []

    for key, value in data.items():
        eval_text = value.get('eval', '')
        match = re.search(r'(\d)$', eval_text)
        if match:
            last_digits.append(match.group(1))
        else:
            last_digits.append('-1')
    
    return last_digits



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    parser.add_argument('--file_path', type=str, default='data/downstream/summazization/single-trial/results/gpt_eval_conclusion')
    args = parser.parse_args()
    

    args.file_path = os.path.join(args.file_path, f'{args.model_name}.json')
    # load groundtruth, data/downstream/summazization/single-trial/results/gpt_eval_conclusion/groundtruth.json
    with open('data/downstream/summazization/single-trial/results/gpt_eval_conclusion/groundtruth.json', 'r') as f:
        groundtruth = json.load(f)
    
    eval_res_groundtruth = extract_last_digit_eval(groundtruth)

    with open(args.file_path, 'r') as f:
        preds = json.load(f)
    
    eval_res_pred = extract_last_digit_eval(preds)

    # # only compare top 100
    # eval_res_groundtruth = eval_res_groundtruth[:500]
    # eval_res_pred = eval_res_pred[:500]

    # calculate matching rate
    matching_rate = sum([1 for i, j in zip(eval_res_pred, eval_res_groundtruth) if i == j]) / len(eval_res_pred)

    
    print(f'Matching rate: {matching_rate:.4f}')