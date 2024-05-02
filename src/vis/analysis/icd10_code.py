import sys
sys.path.append('./')

from src.utils.gpt import gpt_chat
import pandas as pd
import os
import pdb
import json
import argparse
from tqdm import tqdm

import multiprocessing as mp

def load_ctgov_data_subset(filename):
    filepath = f'/data/linjc/trialfm/ctgov_20231231/{filename}.txt'
    df = pd.read_csv(filepath, sep='|', dtype=str)
    
    return df


def worker(input):
    args, df, process_id, prompt, save_dir = input

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        input_disease = row['name']

        if input_disease == 'healthy':
            continue

        attempts = 0
        while attempts < 20:
            output = gpt_chat('gpt-3.5-turbo-0125', prompt.replace('{input_disease}', input_disease))
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
            'id': row['id'],
            'nct_id': row['nct_id'],
            'condition': input_disease,
            'icd10_hierarchy': icd_dict
        }

        with open(os.path.join(save_dir, f'{args.dataset}_{process_id}.json'), 'a') as f:
            f.write(json.dumps(data_dict) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ctgov')
    args = parser.parse_args()

    if args.dataset == 'ctgov':
        df = load_ctgov_data_subset('conditions')
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    save_dir = 'data/analysis/icd10/raw'
    os.makedirs(save_dir, exist_ok=True)

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


    num_processes = 20
    chunk_size = len(df) // num_processes

    inputs = [(args, df.iloc[i*chunk_size:(i+1)*chunk_size], i, prompt, save_dir) for i in range(num_processes)]

    with mp.Pool(num_processes) as pool:
        pool.map(worker, inputs)