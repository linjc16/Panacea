import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.append('./')

from src.utils.gpt_azure import gpt_chat_35, gpt_chat_4


def load_dataset(file_dir, split='test'):
    df = pd.read_csv(os.path.join(file_dir, split + '.csv'))
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-3.5')
    parser.add_argument('--file_dir', type=str, default='data/downstream/summazization/single-trial')
    parser.add_argument('--save_dir', type=str, default='data/downstream/summazization/single-trial/results')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    instruction_prompt = "Your task is to create a clear, concise, and accurate summary of the provided clinical trial document. The summary should capture the key aspects of the trial."
    instruction_prompt += "\nThe output should only be the summarization of the given trial. Do not explain how you summarize it."
    instruction_prompt += "\nInput Text: {Text}"
    instruction_prompt += "\nSummary: "
    
    if not os.path.exists(os.path.join(args.save_dir, f'{args.model_name}.csv')):
        with open(os.path.join(args.save_dir, f'{args.model_name}.csv'), 'w') as f:
            f.write('id,summary\n')
    
    df = load_dataset(args.file_dir, args.split)

    for i in tqdm(range(len(df))):
        input_text = df.iloc[i]['input_text']
        if args.model_name == 'gpt-3.5':
            summary = gpt_chat_35(instruction_prompt, {"Text": input_text})
        elif args.model_name == 'gpt-4':
            summary = gpt_chat_4(instruction_prompt, {"Text": input_text})
        else:
            raise ValueError(f"Model name {args.model_name} is not supported.")
        
        results = pd.DataFrame(columns=['id', 'summary'])
        results.loc[0] = [i, summary]
        results.to_csv(os.path.join(args.save_dir, f'{args.model_name}.csv'), mode='a', header=False, index=False)