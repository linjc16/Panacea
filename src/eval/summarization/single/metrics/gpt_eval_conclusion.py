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
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()
    
    preds = pd.read_csv(os.path.join(args.res_dir, f'{args.model_name}.csv'))

    groundtruth = pd.read_csv("data/downstream/summazization/single-trial/test.csv")
    
    preds['summary'] = preds['summary'].apply(lambda x: str(x))
    preds['summary'] = preds['summary'].apply(lambda x: x.strip())
    
    groundtruth.rename(columns={'summary_text': 'summary'}, inplace=True)
    
    assert len(preds) == len(groundtruth)
    
    
    # load prompt src/eval/summarization/single/metrics/prompt.txt
    prompt = (
        "Summary: {input}\n"
        "Based on this summary, is this trial study successuful or not. ”"
        "If successful, output 1, otherwise output 0."
        'Directly output the number.'
        'Output:'
    )
    
    # if args.model_name == 'zephyr-7b':
    #     # for each summary, only extract text after "Summary:"
    #     preds['summary'] = preds['summary'].apply(lambda x: x.split('Summary:')[1].strip())
    
    
    eval_results = {}

    save_dir = args.res_dir
    save_dir = os.path.join(save_dir, 'gpt_eval_conclusion')
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(len(preds))):
        target_summary = groundtruth.iloc[i]['summary']
        input_text = preds.iloc[i]['summary']
        
        prompt_text = prompt.replace('{input}', input_text).replace('{groundtruth}', target_summary)

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
            'model_output': input_text,
            'eval': eval_output
        }
        
        if i % 100 == 0:
            with open(os.path.join(save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(eval_results, f, indent=4)

    with open(os.path.join(save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)
    