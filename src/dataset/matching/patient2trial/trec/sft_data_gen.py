import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
import pdb
import json
from multiprocessing import Pool

tqdm.pandas()

sys.path.append('./')

from src.utils.claude_aws import chat_haiku, chat_sonnet

def worker(args):
    inputs, process_id, args = args
    i = 0
    output_dict = {}

    for key, value in tqdm(inputs.items()):
        if key in output_dict:
            i += 1
            continue

        prompt = value['input']
        try:
            decoded = chat_haiku(prompt)
        except:
            decoded = ""
        
        output_dict[key] = {
            'input': value['input'],
            'output': decoded,
            'label': value['label'],
            'nct_id': value['nct_id'],
            'patient_id': value['patient_id']
        }
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}_sft_data_{process_id}.json'), 'w') as f:
                json.dump(output_dict, f, indent=4)
    
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}_sft_data_{process_id}.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='claude-haiku')
    parser.add_argument('--save_dir', type=str, default='data/downstream/matching/patient2trial/TREC2021/raw/sft_data')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='TREC2021')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    

    with open(f'data/downstream/matching/patient2trial/{args.dataset}/train_cot.json', 'r') as f:
        inputs = json.load(f)
    
    num_processes = 10

    # Split the data into chunks
    data = list(inputs.items())
    chunk_size = len(data) // num_processes

    chunks = []
    for i in range(num_processes):
        if i == num_processes - 1:
            chunks.append((dict(data[i * chunk_size:]), i, args))
        else:
            chunks.append((dict(data[i * chunk_size:(i + 1) * chunk_size]), i, args))
    
    
    # pdb.set_trace()
    # worker(chunks[0])
    with Pool(num_processes) as p:
        p.map(worker, chunks)
