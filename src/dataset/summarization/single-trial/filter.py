import os
import pandas as pd
from tqdm import tqdm
import argparse
import pdb

def load_data(split):
    data_dir = '/data/linjc/trialfm/downstream/summarization'
    df = pd.read_csv(os.path.join(data_dir, f'{split}.csv'))
    return df


def curate_train_data():
    df = load_data('train')

    # apply on df to get the word length of the summary text, add a new column
    df['word_length'] = df['summary_text'].apply(lambda x: len(str(x).split()))
    # sort the df by the word length
    df = df.sort_values(by='word_length')

    with open('data/downstream/summazization/pre-select.txt', 'r') as f:
        pre_select = f.readlines()
    
    pre_select = set([x.strip() for x in pre_select])
    # keep the trials that are in the pre-select list
    df = df[df['nct_id'].isin(pre_select)]

    # first remove those less than 50 words
    df = df[df['word_length'] > 50]
    
    # calculate the median of the word length
    median = df['word_length'].median()
    # pre-select the trials that are near the median
    pre_select_df = df[df['word_length'] > median + 100]

    # sample 5000 trials
    pre_select_df = pre_select_df.sample(5000, random_state=42)
    
    # for summary text, replace '~' with '\n'
    pre_select_df['summary_text'] = pre_select_df['summary_text'].apply(lambda x: x.replace('~', '\n'))

    # save to csv to 'data/downstream/summazization/train.csv'
    pre_select_df.to_csv('data/downstream/summazization/train.csv', index=False)

def curate_test_data():
    df = load_data('test')
    # apply on df to get the word length of the summary text, add a new column
    df['word_length'] = df['summary_text'].apply(lambda x: len(str(x).split()))
    # sort the df by the word length
    df = df.sort_values(by='word_length')
    # first remove those less than 50 words
    df = df[(df['word_length'] > 100) & (df['word_length'] < 400)]
    # calculate the median of the word length
    median = df['word_length'].median()

    # randomly select 1000 trials to curate the test data
    df = df.sample(1000, random_state=42)

    # for summary text, replace '~' with '\n'
    df['summary_text'] = df['summary_text'].apply(lambda x: x.replace('~', '\n'))

    # save to csv to 'data/downstream/summazization/test.csv'
    df.to_csv('data/downstream/summazization/test.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()

    if args.split == 'train':
        curate_train_data()
    else:
        curate_test_data()

    pdb.set_trace()