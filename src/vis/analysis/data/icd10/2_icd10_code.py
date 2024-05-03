import sys
sys.path.append('./')

from src.utils.gpt_azure import gpt_chat_35_msg as gpt_chat
import pandas as pd
import os
import pdb
import json
import argparse
from tqdm import tqdm
import glob

import multiprocessing as mp

def load_ctgov_data_subset(filename):
    filepath = f'/data/linjc/trialfm/ctgov_20231231/{filename}.txt'
    df = pd.read_csv(filepath, sep='|', dtype=str)
    
    return df


def worker(input):
    args, df, process_id, prompt, save_dir = input

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        input_disease = row['name']

        if input_disease in data_generated:
            continue
        
            
        attempts = 0
        while attempts < 20:
            output = gpt_chat(prompt.replace('{input_disease}', input_disease))
            try:
                icd_dict = json.loads(output)
                break
            except:
                attempts += 1
                continue

        
        if attempts == 20:
            print(f"Failed to generate ICD-10-CM hierarchy for {input_disease}")
            continue
        
        data_dict = {
            'condition': input_disease,
            'icd10_hierarchy': icd_dict
        }

        with open(os.path.join(save_dir, f'{args.dataset}_{process_id}.json'), 'a') as f:
            f.write(json.dumps(data_dict) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ctgov')
    args = parser.parse_args()

    with open(f'data/analysis/icd10/conditions/{args.dataset}_conditions.json', 'r') as f:
        conditions_dict = json.load(f)
    
    save_dir = f'data/analysis/icd10/raw/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)

    output_filepaths = glob.glob(os.path.join(save_dir, '*.json'))
    data_generated = dict()
    for filepath in output_filepaths:
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                data_generated[data['condition']] = data
    
    prompt = (
        'Given a medical condition, '
        'provide the ICD-10-CM hierarchy from the broadest category down to the specific category that this condition belongs to. '
        'List the hierarchy levels in a json format, starting with the broadest group, '
        'followed by subcategories down to the specific code that the condition falls under.'
        'For example, given the condition "Her-2 Negative Breast Cancer With Leptomeningeal Metastasis", '
        'the hierarchy level json format output is {"code": "C00-D49", "description": "Neoplasms", "subcategories": [{"code": "C50-C50", "description": "Malignant neoplasms of breast", "subcategories": {"code": "C50", "description": "Malignant neoplasm of breast"}}]}. '
        '\n\n'
        'Medical Condition: {input_disease}\n'
        'Now, output the ICD-10-CM hierarchy in json format. Directly output the json format without any additional text.'
    )
    
    # remove conditions that have been generated
    for condition in data_generated:
        if condition in conditions_dict:
            del conditions_dict[condition]

    # transform conditions_dict to df
    df = pd.DataFrame(conditions_dict.items(), columns=['name', 'count'])

    # remove rows with count less than 5
    df = df[df['count'] >= 5]

    num_processes = 10
    chunk_size = len(df) // num_processes

    inputs = [(args, df.iloc[i*chunk_size:(i+1)*chunk_size], i, prompt, save_dir) for i in range(num_processes)]

    # worker(inputs[0])
    with mp.Pool(num_processes) as pool:
        pool.map(worker, inputs)