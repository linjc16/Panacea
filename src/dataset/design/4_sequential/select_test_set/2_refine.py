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
    for filepath in tqdm(filepaths[:]):
        with open(filepath, 'r') as f:
            # each row is a dict, read line by line
            for line in f:
                ctgov_dict = json.loads(line)
                if ctgov_dict['arms_and_interventions'] == "| Participant Group/Arm | Intervention/Treatment |\n| --- | --- |\n":
                    damaged_nctid_list.append(ctgov_dict['nct_id'])
                    damaged_dict_list.append(ctgov_dict)

    return damaged_nctid_list, damaged_dict_list

def refine(input):
    damaged_dict_list, process_id = input
    interventions_filepath = '/data/linjc/trialfm/ctgov_20231231/interventions.txt'
    interventions_df = pd.read_csv(interventions_filepath, sep='|', dtype=str)
    

    # for each dict, refine the arms_and_interventions
    for ctgov_dict in tqdm(damaged_dict_list):
        interventions_df_curr = interventions_df[interventions_df['nct_id'] == ctgov_dict['nct_id']]
        
        # | Intervention/Treatment |
        # | --- |
        # | intervention_type: name\n*description |
        interventions_text = ""
        interventions_text += "| Intervention/Treatment |\n| --- |\n"
        for i in range(len(interventions_df_curr)):
            intervention_type = interventions_df_curr.iloc[i]['intervention_type']
            name = interventions_df_curr.iloc[i]['name']
            description = interventions_df_curr.iloc[i]['description']
            interventions_text += "|{}: {}|{}|\n".format(intervention_type, name, description)
        
        ctgov_dict['arms_and_interventions'] =  interventions_text if interventions_text != "| Intervention/Treatment |\n| --- |\n" else ""



    with open(os.path.join(output_dir, f'refined_{process_id}.json'), 'a') as f:
        for ctgov_dict in damaged_dict_list:
            json.dump(ctgov_dict, f)
            f.write('\n')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, default='data/downstream/design/raw/selected_step1')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_process', type=int, default=64)
    args = parser.parse_args()
    
    output_dir = os.path.join(args.file_dir, 'refined', args.split)
    os.makedirs(output_dir, exist_ok=True)
    
    _, damaged_dict_list = get_damaged_nctid(args.file_dir, args.split)


    num_process = args.num_process

    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]
    
    nct_ids_chunks = chunkify(damaged_dict_list, num_process)

    import multiprocessing as mp
    # add process id
    nct_ids_chunks = [(chunk, i) for i, chunk in enumerate(nct_ids_chunks)]

    # refine(nct_ids_chunks[0])

    with mp.Pool(num_process) as p:
        p.map(refine, nct_ids_chunks)