import os
import argparse
import openai
import json
import pdb
from tqdm import tqdm
import sys

sys.path.append('./')
from src.utils.claude_aws import chat_haiku, chat_sonnet



def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    
    return data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='claude-haiku')
    parser.add_argument('--task', type=str, default='study_arms')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    filepath = f'data/downstream/design/parsed/{args.task}/{args.split}.json'

    data_dict = load_dataset(filepath)



    output_reults = {}
    count = 0
    for key, value in tqdm(data_dict.items()):
        groudtruth = []
        out_response = []

        if len(value) % 2 != 0:
            value = value[:-1]

        for i in range(3, len(value) // 2):
            try:
                # [0], [0, 1, 2], [0, 1, 2, 3, 4]
                input = value[:i * 2 + 1]
                
                if args.model_name == 'claude-haiku':
                    response = chat_haiku(input)
                elif args.model_name == 'claude-sonnet':
                    response = chat_sonnet(input)
                else:
                    raise ValueError(f"Model name {args.model_name} is not supported.")
                out_response.append(response)
                # pdb.set_trace()

            except:
                out_response.append('')
            
            groudtruth.append(value[i * 2 + 1]['content'])

        
        output_reults[key] = {
            'model_response': out_response,
            'groundtruth': groudtruth,
        }

        if count % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(output_reults, f, indent=4)
        
        count += 1

    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(output_reults, f, indent=4)