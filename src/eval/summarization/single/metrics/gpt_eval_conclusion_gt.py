import os
import argparse
import openai
import json
import pdb
from tqdm import tqdm
import sys
import pandas as pd

sys.path.append('./')
from src.utils.claude_aws import chat_haiku, chat_sonnet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/summazization/single-trial/results')
    args = parser.parse_args()

    args.model_name = 'groundtruth'
    
    
    groundtruth = pd.read_csv("data/downstream/summazization/single-trial/test.csv")

    
    groundtruth.rename(columns={'summary_text': 'summary'}, inplace=True)
    
    
    # load prompt src/eval/summarization/single/metrics/prompt.txt
    prompt = (
        "Summary: {input}\n"
        "Based on this summary, is this trial study effective or not. ”"
        "If effective, output 1, otherwise output 0."
        'Directly output the number.'
        'Output:'
    )    

    
    eval_results = {}

    save_dir = args.res_dir
    save_dir = os.path.join(save_dir, 'gpt_eval_conclusion')
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(len(groundtruth))):
        target_summary = groundtruth.iloc[i]['summary']
        
        prompt_text = prompt.replace('{input}', target_summary)

        attempt = 0
        while attempt < 20:
            try:
                eval_output = chat_sonnet(prompt_text)
                break
            except:
                attempt += 1
                continue
                
        eval_results[i] = {
            'summary': target_summary,
            'eval': eval_output
        }
        
        if i % 100 == 0:
            with open(os.path.join(save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(eval_results, f, indent=4)

    with open(os.path.join(save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)
    