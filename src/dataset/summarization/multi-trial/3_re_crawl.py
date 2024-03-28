import pandas as pd
import glob
import os
import pubmed_parser as pp
import pdb
from collections import defaultdict
import re
from tqdm import tqdm
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', help='train or test')
    args = parser.parse_args()

    with open(f'data/downstream/summazization/multi-trial/raw/pm_{args.split}.csv', 'r') as f:
        df = pd.read_csv(f)
    
    pmids = df['pmid'].tolist()

    cleaned_list = []
    for idx, pmid in enumerate(tqdm(pmids)):
        try:
            dict_out = pp.parse_xml_web(pmid, save_xml=False)
            title = dict_out['title']
            abstract = dict_out['abstract']
        except:
            title = df[df['pmid'] == pmid]['title'].values[0]
            abstract = df[df['pmid'] == pmid]['abstract'].values[0]
            print(f'Error: {pmid}')
        
        cleaned_list.append([pmid, title, abstract])

        if idx % 1000 == 0:
            cleaned_df = pd.DataFrame(cleaned_list, columns=['pmid', 'title', 'abstract'])
            cleaned_df.to_csv(f'data/downstream/summazization/multi-trial/raw/cleaned/pm_{args.split}_cleaned.csv', index=False)

    
    cleaned_df = pd.DataFrame(cleaned_list, columns=['pmid', 'title', 'abstract'])

    cleaned_df.to_csv(f'data/downstream/summazization/multi-trial/raw/cleaned/pm_{args.split}_cleaned.csv', index=False)