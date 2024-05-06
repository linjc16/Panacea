import pandas as pd
import glob
import os
import pdb
# from Bio import Entrez
import re
import multiprocessing as mp
import pubmed_parser as pp
import json
import random
import time
import multiprocessing as mp
from tqdm import tqdm

from xml.etree import ElementTree as ET
import requests

save_dir = 'data/analysis/icd10/mesh/raw'
os.makedirs(save_dir, exist_ok=True)

def get_pmc_id_from_pubmed(pubmed_id, process_id):
    mesh_details = []

    attempts = 0
    while attempts < 100:
        if pubmed_id == '22382203':
            pdb.set_trace()
        try:
            response = requests.get(f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pubmed_id}&retmode=xml&rettype=medline')
            tree = ET.fromstring(response.content)
            for mesh in tree.findall('.//MeshHeading'):
                descriptor = mesh.find('DescriptorName')
                if descriptor is not None:
                    ui = descriptor.attrib['UI']
                    term = descriptor.text
                    mesh_details.append((ui, term))

            with open(os.path.join(save_dir, f'mesh_terms_{process_id}.json'), 'a') as f:
                f.write(json.dumps({'pubmed_id': pubmed_id, 'mesh_terms': mesh_details}) + '\n')
            
            time.sleep(random.randint(1, 3)/10)
            return 

        except:
            attempts += 1
        
    print(f'Error with {pubmed_id}')


def process_row(input):
    df, process_id = input

    for index, row in tqdm(df.iterrows(), total=len(df)):
        pubmed_id = row['pubmed_id']
        get_pmc_id_from_pubmed(pubmed_id, process_id)
    


if __name__ == '__main__':
    filepaths = glob.glob('/data/linjc/trialfm/code/ctr_crawl/0_final_data/papers/pubmed/*.csv')
    
    df_list = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df_list.append(df)
    
    df = pd.concat(df_list)
    
    num_processes = 2
    
    chunk_size = int(len(df)/num_processes)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    inputs = [(chunk, i) for i, chunk in enumerate(chunks)]

    with mp.Pool(num_processes) as pool:
        pool.map(process_row, inputs)

    # process_row(inputs[0])
