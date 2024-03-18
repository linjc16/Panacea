import os
import glob
import pdb
import argparse
from tqdm import tqdm
import pandas as pd
import json

def get_damaged_nctid(file_dir, split):
    damaged_nctid_list = []
    damaged_dict_list = []

    filepaths = glob.glob(os.path.join(file_dir, split, '*.json'))
    # for each file, load the json
    # each row is a dict

    output_list = []
    for filepath in tqdm(filepaths):
        with open(filepath, 'r') as f:
            # each row is a dict, read line by line
            for line in f:
                ctgov_dict = json.loads(line)
                if ctgov_dict['arms_and_interventions'] == "":
                    damaged_nctid_list.append(ctgov_dict['nct_id'])
                    damaged_dict_list.append(ctgov_dict)

    return damaged_nctid_list, damaged_dict_list

def load_refined_files(file_dir_refined, split):
    filepaths = glob.glob(os.path.join(file_dir_refined, split, '*.json'))
    refined_dict_list = []
    for filepath in tqdm(filepaths):
        with open(filepath, 'r') as f:
            # each row is a dict, read line by line
            for line in f:
                ctgov_dict = json.loads(line)
                refined_dict_list.append(ctgov_dict)

    return refined_dict_list

def load_original_files(file_dir, split):
    filepaths = glob.glob(os.path.join(file_dir, split, '*.json'))
    original_dict_list = []
    for filepath in tqdm(filepaths):
        with open(filepath, 'r') as f:
            # each row is a dict, read line by line
            for line in f:
                ctgov_dict = json.loads(line)
                original_dict_list.append(ctgov_dict)

    return original_dict_list

def merge(file_dir, file_dir_refined, split, num_process):
    refined_dict_list = load_refined_files(file_dir_refined, split)
    original_dict_list = load_original_files(file_dir, split)

    # for original_dict_list, transfer to a dict, key is nct_id, value is the dict
    original_dict = {}
    for original_dict_curr in tqdm(original_dict_list):
        original_dict[original_dict_curr['nct_id']] = original_dict_curr
    
    # traverse the refined_dict_list, if nct_id in original_dict, then update the original_dict
    for refined_dict_curr in tqdm(refined_dict_list):
        if refined_dict_curr['nct_id'] in original_dict:
            original_dict[refined_dict_curr['nct_id']] = refined_dict_curr

    # write the original_dict to file, each line is the value of the dict, which is a dict
    # first, convert the dict to a list
    # then split the list into 128 parts
    # then write the list to file

    original_dict_list = list(original_dict.values())
    num_process = num_process
    original_dict_list_list = [original_dict_list[i::num_process] for i in range(num_process)]

    for i in tqdm(range(num_process)):
        with open(os.path.join(save_dir, f'merged_{i}.json'), 'a') as f:
            for original_dict_curr in original_dict_list_list[i]:
                json.dump(original_dict_curr, f)
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, default='data/downstream/design/raw/selected_step1')
    parser.add_argument('--file_dir_refined', type=str, default='data/downstream/design/raw/selected_step1/refined/')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_process', type=int, default=128)
    args = parser.parse_args()

    save_dir = os.path.join(args.file_dir, 'merged', args.split)
    os.makedirs(save_dir, exist_ok=True)
    
    merge(args.file_dir, args.file_dir_refined, args.split, args.num_process)
    # damaged_nctid_list, damaged_dict_list = get_damaged_nctid(args.file_dir, args.split)