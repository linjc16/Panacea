import pandas as pd
import glob
import os
import pdb
# from Bio import Entrez
import re
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
import pubmed_parser as pp
import json
import random
import time

from xml.etree import ElementTree as ET
import requests

save_dir = 'data/analysis/icd10/mesh/'

def get_pmc_id_from_pubmed(pubmed_id):
    mesh_details = []

    try:
        response = requests.get(f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pubmed_id}&retmode=xml&rettype=medline')
        tree = ET.fromstring(response.content)
        for mesh in tree.findall('.//MeshHeading'):
            descriptor = mesh.find('DescriptorName')
            if descriptor is not None:
                ui = descriptor.attrib['UI']
                term = descriptor.text
                mesh_details.append((ui, term))

        with open(os.path.join(save_dir, 'mesh_terms.json'), 'a') as f:
            f.write(json.dumps({'pubmed_id': pubmed_id, 'mesh_terms': mesh_details}) + '\n')
        
        time.sleep(random.randint(1, 3)/10)

    except:
        return


def process_row(pubmed_id):
    return get_pmc_id_from_pubmed(pubmed_id)


if __name__ == '__main__':
    filepaths = glob.glob('/data/linjc/trialfm/code/ctr_crawl/0_final_data/papers/pubmed/*.csv')
    
    df_list = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df_list.append(df)
    
    df = pd.concat(df_list)

    # df = df.iloc[:10]

    # for each row, get the pmc id
    # df['pmc_id'] = df['pubmed_id'].apply(lambda x: get_pmc_id_from_pubmed(x))
    
    pmc_ids = process_map(process_row, df['pubmed_id'], max_workers=16)
    # process_row(df['pubmed_id'])

    # pdb.set_trace()

    # # count the number of papers with pmc id
    # print('There are {} papers with pmc id'.format(len(df[df['pmc_id'].notnull()])))
    
    # save_dir = '3_crawl_paper/pubmed/data/raw_pmc/pubmed_raw.csv'
    # os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # # save the df to 3_crawl_paper/pubmed/data/raw_pmc/
    # df.to_csv(save_dir, index=False)
