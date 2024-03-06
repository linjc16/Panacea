import pandas as pd
import glob
from tqdm import tqdm
import random
import json
import pdb


def load_mesh_terms():
    filepath = '/data/linjc/trialfm/ctgov_20231231/browse_conditions.txt'

    df = pd.read_csv(filepath, sep='|', dtype=str)

    return df

def load_calculated_values():
    filepath = '/data/linjc/trialfm/ctgov_20231231/calculated_values.txt'
    df = pd.read_csv(filepath, sep='|', dtype=str)

    return df

def split_data(df_cal_vals, split='train'):
    if split == 'train':
        nct_ids = df_cal_vals[df_cal_vals['registered_in_calendar_year'] <= '2022']['nct_id'].tolist()
    else:
        nct_ids = df_cal_vals[df_cal_vals['registered_in_calendar_year'] == '2023']['nct_id'].tolist()
    
    return nct_ids

if __name__ == '__main__':
    # split data by date, find mesh_term, deduplicate, then find greated than 5

    df_mesh_terms = load_mesh_terms()
    df_cal_vals = load_calculated_values()

    nct_ids_train = split_data(df_cal_vals, 'train')
    nct_ids_test = split_data(df_cal_vals, 'test')

    # find mesh terms for train and test
    # for df_mesh_terms, groupby nct_id, then join the mesh terms into a list

    df_mesh_terms_train = df_mesh_terms[df_mesh_terms['nct_id'].isin(nct_ids_train)]
    df_mesh_terms_test = df_mesh_terms[df_mesh_terms['nct_id'].isin(nct_ids_test)]

    df_mesh_terms_train = df_mesh_terms_train.groupby('nct_id')['mesh_term'].apply(list).reset_index()
    df_mesh_terms_test = df_mesh_terms_test.groupby('nct_id')['mesh_term'].apply(list).reset_index()

    # deduplicate
    df_mesh_terms_train['mesh_term'] = df_mesh_terms_train['mesh_term'].apply(lambda x: list(set(x)))
    df_mesh_terms_test['mesh_term'] = df_mesh_terms_test['mesh_term'].apply(lambda x: list(set(x)))

    # find greater than 5
    df_mesh_terms_train = df_mesh_terms_train[df_mesh_terms_train['mesh_term'].apply(lambda x: len(x) > 5)]
    df_mesh_terms_test = df_mesh_terms_test[df_mesh_terms_test['mesh_term'].apply(lambda x: len(x) > 5)]
    
    # for each row, select some mesh terms as the input, leave the rest as the output, save to json

    train_mesh_terms_dict = {}
    test_mesh_terms_dict = {}
    for i in tqdm(range(len(df_mesh_terms_train))):
        nct_id = df_mesh_terms_train['nct_id'].iloc[i]
        mesh_terms = df_mesh_terms_train['mesh_term'].iloc[i]
        # select some mesh terms as the input, leave the rest as the output, save to json
        input_mesh_terms = mesh_terms[:5]
        output_mesh_terms = mesh_terms[5:]
        train_mesh_terms_dict[nct_id] = {
            'nct_id': nct_id,
            'input': input_mesh_terms, 
            'output': output_mesh_terms
        }
    
    for i in tqdm(range(len(df_mesh_terms_test))):
        nct_id = df_mesh_terms_test['nct_id'].iloc[i]
        mesh_terms = df_mesh_terms_test['mesh_term'].iloc[i]
        # select some mesh terms as the input, leave the rest as the output, save to json
        input_mesh_terms = mesh_terms[:5]
        output_mesh_terms = mesh_terms[5:]
        test_mesh_terms_dict[nct_id] = {
            'nct_id': nct_id,
            'input': input_mesh_terms, 
            'output': output_mesh_terms
        }

    # randomly select 100000 from train 
    
    random.seed(0)
    train_mesh_terms_dict = dict(random.sample(train_mesh_terms_dict.items(), 100000))
    
    # save to json
    with open('data/downstream/search/query_expansion/train.json', 'w') as f:
        json.dump(train_mesh_terms_dict, f, indent=4)
    
    with open('data/downstream/search/query_expansion/test.json', 'w') as f:
        json.dump(test_mesh_terms_dict, f, indent=4)
    
    

