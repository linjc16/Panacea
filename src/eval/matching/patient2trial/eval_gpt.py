import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
import pdb
import json

tqdm.pandas()

sys.path.append('./')

from src.utils.gpt_azure import gpt_chat_35, gpt_chat_4



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='claude-haiku')
    parser.add_argument('--save_dir', type=str, default='/data/linjc/trialfm/downstream/summarization/results')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='cohort')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    

    with open(f'data/downstream/matching/patient2trial/{args.dataset}/test.json', 'r') as f:
        inputs = json.load(f)

    i = 0
    output_dict = {}
    for key, value in tqdm(inputs.items()):
        prompt = value['input']
        if args.model_name == 'gpt-3.5':
            try:
                decoded = gpt_chat_35(prompt, {})
            except:
                decoded = ""
        elif args.model_name == 'gpt-4':
            try:
                decoded = gpt_chat_4(prompt, {  })
            except:
                decoded = ""
        
        output_dict[key] = {
            'output': decoded,
            'label': value['label']
        }

        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(output_dict, f, indent=4)
    
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)