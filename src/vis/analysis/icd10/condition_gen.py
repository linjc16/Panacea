import pandas as pd
import argparse
import os
import json
import pdb
from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aus_zealand')
    args = parser.parse_args()

    save_dir = 'data/analysis/icd10/conditions'

    if args.dataset == 'ctgov':
        filepath = f'/data/linjc/trialfm/ctgov_20231231/conditions.txt'
        df = pd.read_csv(filepath, sep='|', dtype=str)
        conditions = df['name'].tolist()

        # unique_conditions = set(conditions)

        # for each unique condition, count the number of times it appears in the dataset
        # save the count in a dictionary
        condition_count = defaultdict(int)
        for condition in conditions:
            condition_count[condition] += 1


    else:
        data_dir = '/data/linjc/trialfm/code/ctr_crawl/0_final_data/visualization'
        df = pd.read_csv(os.path.join(data_dir, f'{args.dataset}.csv'))
        conditions = df['Conditions']

        # # for each row, split the conditions by ', ', and save the unique conditions
        # unique_conditions = set()
        # for condition in conditions:
        #     # if float, then skip
        #     if isinstance(condition, float):
        #         continue
        #     condition = condition.replace(';', ',').replace('\n', ',')
        #     condition_curr = condition.split(',')
        #     for cond in condition_curr:
        #         if cond:
        #             unique_conditions.add(cond.strip().lower())
        
        # for each unique condition, count the number of times it appears in the dataset
        # save the count in a dictionary
        condition_count = defaultdict(int)
        for condition in conditions:
            if isinstance(condition, float):
                continue
            condition = condition.replace(';', ',').replace('\n', ',')
            condition_curr = condition.split(',')
            for cond in condition_curr:
                if cond:
                    condition_count[cond.strip().lower()] += 1
        
    # rank the conditions by the number of times they appear in the dataset
    condition_count = dict(sorted(condition_count.items(), key=lambda x: x[1], reverse=True))

    with open(os.path.join(save_dir, f'{args.dataset}_conditions.json'), 'w') as f:
        json.dump(condition_count, f, indent=4)
    

    print('Number of data points:', len(df))
    print(f"Unique conditions: {len(condition_count)}")
    # pdb.set_trace()