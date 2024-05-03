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


def worker(input):
    df, process_id, prompt, save_dir = input

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title = row['Title']
        abstract = row['Abstract']
        attempts = 0
        while attempts < 20:
            try:
                output = gpt_chat(prompt.replace('{title}', title).replace('{abstract}', abstract))
                output_list = json.loads(output)
                break
            except:
                attempts += 1
                continue
        
        output_dict = {
            'title': title,
            'mesh_terms': output_list
        }

        with open(os.path.join(save_dir, f'mesh_terms_{process_id}.json'), 'a') as f:
            f.write(json.dumps(output_dict) + '\n')

if __name__ == '__main__':

    file_dir = '/data/linjc/trialfm/code/ctr_crawl/0_final_data/papers/embase/*.csv'
    filepaths = glob.glob(file_dir)

    save_dir = 'data/analysis/icd10/mesh/raw/embase'
    os.makedirs(save_dir, exist_ok=True)

    df_list = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df_list.append(df)

    df = pd.concat(df_list)

    prompt = (
        "Identify and list relevant MeSH terms from the given scientific paper's title and abstract. "
        "Focus on extracting key medical and procedural terms that would be useful for indexing this paper in medical databases. "
        "Directly output the MeSH term in a json dict format."
        "Output Example: {'mesh_terms': ['Aged, 80 and over', 'Child', ...]}}"
        "\n\n"
        "Title: {title}"
        'Abstract: {abstract}'
        '\n\n'
        'Now, output the MeSH terms in a dict format. '
        'Directly output the json format without any additional text.'
    )

    num_processes = 10

    chunk_size = len(df) // num_processes
    
    inputs = [(df.iloc[i*chunk_size:(i+1)*chunk_size], i, prompt, save_dir) for i in range(num_processes)]
    
    with mp.Pool(num_processes) as pool:
        pool.map(worker, inputs)
    # worker(inputs[0])