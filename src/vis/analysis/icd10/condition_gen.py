import pandas as pd
import argparse
import os
import json
import pdb
from collections import defaultdict
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aus_zealand')
    args = parser.parse_args()

    save_dir = 'data/analysis/icd10/conditions'

    if args.dataset == 'ctgov':
        filepath = f'/data/linjc/trialfm/ctgov_20231231/conditions.txt'
        df = pd.read_csv(filepath, sep='|', dtype=str)
        conditions = df['name'].tolist()

        # for each unique condition, count the number of times it appears in the dataset
        # save the count in a dictionary
        condition_count = defaultdict(int)
        for condition in conditions:
            condition_count[condition] += 1

    elif args.dataset == 'german':
        data_dir = '/data/linjc/trialfm/code/ctr_crawl/0_final_data/visualization'
        df = pd.read_csv(os.path.join(data_dir, f'{args.dataset}.csv'))
        conditions = df['Conditions']

        
        condition_count = defaultdict(int)
        for condition in conditions:
            if isinstance(condition, float):
                continue
            condition = re.sub(r'<.*?>', '', condition)

            condition = condition.replace(';', ',').replace('\n', ',')
            condition_curr = condition.split(',')

            for cond in condition_curr:
                if 'ICD10::' in cond:
                    cond = cond.split('::')[1].strip().lower()
                    if cond and cond != "":
                        condition_count[cond] += 1

    
    else:
        data_dir = '/data/linjc/trialfm/code/ctr_crawl/0_final_data/visualization'
        df = pd.read_csv(os.path.join(data_dir, f'{args.dataset}.csv'))
        conditions = df['Conditions']
        
        # for each unique condition, count the number of times it appears in the dataset
        # save the count in a dictionary
        condition_count = defaultdict(int)
        for condition in conditions:
            if isinstance(condition, float):
                continue
            # repalce <...> with '', including text between <>
            condition = re.sub(r'<.*?>', '', condition)

            condition = condition.replace(';', ',').replace('\n', ',')
            condition_curr = condition.split(',')
            for cond in condition_curr:
                cond = cond.strip().lower()
                if cond == "": 
                    continue
                if cond and cond != "" and len(cond.split()) < 7:
                    condition_count[cond] += 1
    
    # rank the conditions by the number of times they appear in the dataset
    condition_count = dict(sorted(condition_count.items(), key=lambda x: x[1], reverse=True))

    with open(os.path.join(save_dir, f'{args.dataset}_conditions.json'), 'w') as f:
        json.dump(condition_count, f, indent=4)
    

    print('Number of data points:', len(df))
    print(f"Unique conditions: {len(condition_count)}")
    # pdb.set_trace()