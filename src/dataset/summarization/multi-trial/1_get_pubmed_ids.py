import glob
import pandas as pd
import re
from tqdm import tqdm
import json
import pdb


# pdb.set_trace()
def get_pubmed_id(df):
    # Accession Number, extract the id starting with 'PUBMED', like PUBMED 35341362
    # The full part will be like: PUBMED 35341362; EMBASE 2015419831, remove other parts except PUBMED
    # use regex to extract the pubmed id
    df['pubmed_id'] = df['Accession Number'].apply(lambda x: re.findall(r'PUBMED \d+', x)[0].split(' ')[1])

    return df

if __name__ == '__main__':
    filepaths = glob.glob('/data/linjc/trialfm/code/ctr_crawl/0_final_data/papers/pubmed/*.csv')
    
    df_list = []
    for filepath in tqdm(filepaths):
        df = pd.read_csv(filepath)
        df_list.append(df)
    
    df = pd.concat(df_list)

    df = get_pubmed_id(df)

    # save the pubmed_id column into a json file
    pubmed_id_list = df['pubmed_id'].tolist()
    pubmed_id_list = list(set(pubmed_id_list))

    with open('data/downstream/summazization/multi-trial/pubmed_ids.json', 'w') as f:
        json.dump(pubmed_id_list, f)