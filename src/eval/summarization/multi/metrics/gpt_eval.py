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
    parser.add_argument('--res_dir', type=str, default='data/downstream/summazization/multi-trial/results')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()
    
    preds = pd.read_csv(os.path.join(args.res_dir, f'{args.model_name}.csv'))

    with open('data/downstream/summazization/multi-trial/test.json', 'r') as f:
        data = json.load(f)
    
    # for each key, extract value['target'], merge into a dataframe
    groundtruth = {'id': [], 'summary': []}
    for key, value in tqdm(data.items()):
        groundtruth['id'].append(key)
        groundtruth['summary'].append(value['target'])
    groundtruth = pd.DataFrame(groundtruth)
    
    preds['summary'] = preds['summary'].apply(lambda x: str(x))
    preds['summary'] = preds['summary'].apply(lambda x: x.strip())
    
    assert len(preds) == len(groundtruth)
    
    
    with open('src/eval/summarization/multi/metrics/prompt_eval.txt', 'r') as f:
        prompt = f.read()
    
    # if args.model_name == 'zephyr-7b':
    #     # for each summary, only extract text after "Summary:"
    #     preds['summary'] = preds['summary'].apply(lambda x: x.split('Summary:')[1].strip())
    
    
    eval_results = {}

    save_dir = args.res_dir
    save_dir = os.path.join(save_dir, 'gpt_eval')
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
    