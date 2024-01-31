import pandas as pd
import os
import glob
import pdb
import re
from tqdm import tqdm

def load_embase(file_dir):
    """
    Load the embase data.
    """
    file_list = glob.glob(os.path.join(file_dir, '*.csv'))
    print('File list: ', file_list)
    df_list = []
    for file in tqdm(file_list):
        df = pd.read_csv(file)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df[['Title', 'Abstract']]

    paper_list = []
    for i in tqdm(range(len(df))):
        title = df.iloc[i]['Title']
        abstract = df.iloc[i]['Abstract']
        paper_list.append(title + '\n\nAbstract:\n' + abstract)

    return paper_list

def load_pubmed(file_dir):
    """
    Load the pubmed data.
    """
    file_list = glob.glob(os.path.join(file_dir, '*.csv'))
    print('File list: ', file_list)
    df_list = []
    for file in tqdm(file_list):
        df = pd.read_csv(file)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df[['Title', 'Abstract', 'full_text']]
    
    paper_list = []
    for i in tqdm(range(len(df))):
        title = df.iloc[i]['Title']
        abstract = df.iloc[i]['Abstract']
        full_text = '\n\n'+ df.iloc[i]['full_text'] if pd.isna(df.iloc[i]['full_text']) == False else ''
        paper_list.append(str(title) + '\n\nAbstract:\n' + str(abstract) + str(full_text))

    return paper_list

if __name__ == '__main__':
    # file_dir = '/data/linjc/ctr_crawl/0_final_data/papers/embase'
    # output = load_embase(file_dir)

    file_dir = '/data/linjc/trialfm/final_data/papers/pubmed'
    output = load_pubmed(file_dir)

    pdb.set_trace()