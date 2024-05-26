import os
import argparse
import openai
import json
import pdb
from tqdm import tqdm
import sys
import pandas as pd

sys.path.append('./')
from src.utils.gpt_azure import gpt_chat_35_msg
from src.utils.gpt import gpt_chat_35


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/summazization/single-trial/results')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()
    
    preds = pd.read_csv(os.path.join(args.res_dir, f'{args.model_name}.csv'))

    groundtruth = pd.read_csv("data/downstream/summazization/single-trial/test.csv")
    
    preds['summary'] = preds['summary'].apply(lambda x: str(x))
    preds['summary'] = preds['summary'].apply(lambda x: x.strip())
    
    groundtruth.rename(columns={'summary_text': 'summary'}, inplace=True)
    
    assert len(preds) == len(groundtruth)
    
    
    # load prompt src/eval/summarization/single/metrics/prompt.txt
    with open('src/eval/summarization/single/metrics/prompt.txt', 'r') as f:
        prompt = f.read()
    

    eval_results = {}

    save_dir = 'data/downstream/summazization/single-trial/results'
    save_dir = os.path.join(save_dir, 'gpt_eval')
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(len(preds))):
        target_summary = groundtruth.iloc[i]['summary']
        input_text = preds.iloc[i]['summary']
        
        prompt_text = prompt.replace('{input}', input_text).replace('{groundtruth}', target_summary)
        eval_output = gpt_chat_35(prompt_text)

        eval_results[i] = {
            'summary': target_summary,
            'model_output': input_text,
            'summary': eval_output
        }
        
        if i % 100 == 0:
            with open(os.path.join(save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(eval_results, f, indent=4)

    with open(os.path.join(save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)
    