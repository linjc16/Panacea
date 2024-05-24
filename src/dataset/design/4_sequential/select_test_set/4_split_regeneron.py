import pandas as pd
import glob
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import os
import json


if __name__ == '__main__':
    filepath = 'data/downstream/design/ctgov/calculated_values.txt'
    df_cal_vals = pd.read_csv(filepath, sep='|', dtype=str)

    with open('data/downstream/design/nctid_test.txt', 'r') as f:
        nct_ids = f.read().splitlines()
    
    # split nct_ids by 'registered_in_calendar_year' before 2023 and after 2023
    nct_ids_before_2023 = []
    nct_ids_after_2023 = []
    
    for nct_id in nct_ids:
        if nct_id in df_cal_vals['nct_id'].values:
            if int(df_cal_vals[df_cal_vals['nct_id'] == nct_id]['registered_in_calendar_year']) < 2023:
                nct_ids_before_2023.append(nct_id)
            else:
                nct_ids_after_2023.append(nct_id)
        else:
            print(f'{nct_id} not in calculated_values.txt')
    
    print(f'Number of nct_ids before 2023: {len(nct_ids_before_2023)}')
    print(f'Number of nct_ids after 2023: {len(nct_ids_after_2023)}')