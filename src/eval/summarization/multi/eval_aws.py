import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict

sys.path.append('./')

from src.utils.claude_aws import chat_haiku, chat_sonnet


def load_dataset(file_dir, split='test'):
    # load data/downstream/summazization/multi-trial/{split}.json
    with open(os.path.join(file_dir, f'{split}.json'), 'r') as f:
        data = json.load(f)
    
    output_data = defaultdict(list)
    for key, value in tqdm(data.items()):
        output_data['id'].append(key)
        # merge title and abstract list within the same paper (index), then add prefix "Study #x:"
        study_text = ""
        for i in range(len(value['title'])):
            study_text += f"Study #{i+1}: {value['title'][i]}. {value['abstract'][i]}.\n\n"
        # remove the last \n\n
        study_text = study_text[:-2]
        output_data['study_text'].append(study_text)
        output_data['target'].append(value["target"])
    
    df = pd.DataFrame(output_data)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-3.5')
    parser.add_argument('--file_dir', type=str, default='data/downstream/summazization/multi-trial')
    parser.add_argument('--save_dir', type=str, default='data/downstream/summazization/multi-trial/results')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    instruction_prompt = "Your task is to synthesize the key findings from a collection of study abstracts related to a specific clinical trial related research question."
    instruction_prompt += "\nCombine the insights from the provided abstracts into a cohesive summary. Your summary should integrate the findings rather than listing them separately. It's crucial to maintain the scientific integrity of the original studies while ensuring the summary is accessible and informative."
    instruction_prompt += "\nThe output should only be the summary. Do not explain how you summarize it."
    instruction_prompt += "\n\nStudy Abstracts: {Text}"
    instruction_prompt += "\n\nSummary:"
    
    if not os.path.exists(os.path.join(args.save_dir, f'{args.model_name}.csv')):
        with open(os.path.join(args.save_dir, f'{args.model_name}.csv'), 'w') as f:
            f.write('id,summary\n')
    
    df = load_dataset(args.file_dir, args.split)
    
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        id = row['id']
        input_text = row['study_text']
        
        if args.model_name == 'claude-haiku':
            try:
                prediction = chat_haiku(instruction_prompt.format(Text=input_text))
            except:
                prediction = ""
        elif args.model_name == 'claude-sonnet':
            try:
                prediction = chat_sonnet(instruction_prompt.format(Text=input_text))
            except:
                prediction = ""
        else:
            raise ValueError(f"Model name {args.model_name} is not supported.")

        results = pd.DataFrame(columns=['id', 'summary'])
        results.loc[0] = [id, prediction]
        results.to_csv(os.path.join(args.save_dir, f'{args.model_name}.csv'), mode='a', header=False, index=False)