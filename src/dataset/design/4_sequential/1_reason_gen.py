import os
import sys
import json
import argparse

sys.path.append('./')
from src.utils.gpt import gpt_chat
from tqdm import tqdm
import multiprocessing as mp
import glob

import pdb

def gen_reasons(input):
    args, ctgov_dict_list, i = input
    output_dict = {}

    save_path = args.save_path.split('.json')[0] + f'_{i}.json'

    i = 0
    for ctgov_dict in tqdm(ctgov_dict_list):
        prompt_curr = prompt.format(**ctgov_dict)
        try:
            response = gpt_chat('gpt-3.5-turbo-0125', prompt_curr, seed=44)
        except:
            response = ""

        output_dict[ctgov_dict['nct_id']] = response

        if i % 50 == 0:
            with open(save_path, 'w') as f:
                json.dump(output_dict, f, indent=4)
        i += 1

    with open(save_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sequential')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    prompt = (
        'Given the information below about a clinical trial, please analyze and provide reasons '
        'for (1) the design of each criterion listed under the "Criteria" section. '
        'For each criterion (both inclusion and exclusion criteria), '
        'explain why it is reasonable and necessary for the goals and structure of this trial; '
        '(2) the design of each study arms listed under the "Study Arms" section. '
        'For each study arm, focus on Participant Group/Arm, Intervention/Treatment, and so on. '
        'Explain why they are reasonable and necessary for the goals and structure of this trial; '
        '(3) the design of each outcome measure listed under the "Primary Outcome Measure" and "Second Outcome Measure" sections. '
        'For each outcome measure, focus on Outcome Measure, Measure Description, Time Frame, and so on. '
        'Explain why they are reasonable and necessary for the goals and structure of this trial.'
        '\n\n'
        'Title: {brief_title}\n'
        'Official Title: {official_title}\n'
        'Conditions: {conditions}\n'
        'Intervention / Treatment: {interventions}\n'
        'Study Type: {study_type}\n'
        'Phase: {phase}\n\n'
        'Brief Summary: {brief_summary}\n\n'
        'Criteria: {eligibility_criteria}\n\n'
        'Study Arms: {arms_and_interventions}\n\n'
        "Design Details: {design_details}\n\n"
        "Primary Outcome Measure: {primary_outcome_measures}\n\n"
        "Second Outcome Measure: {secondary_outcome_measures}"
    )
    
    save_path = f'data/downstream/design/raw/reasons/{args.task}/{args.split}'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'reasons_{args.split}.json')
    args.save_path = save_path
    
    ctgov_dict_list = []
    merge_files = glob.glob(f'data/downstream/design/raw/selected_step1/merged/{args.split}/*.json')

    for merge_file in tqdm(merge_files):
        with open(merge_file, 'r') as f:
            for line in f:
                ctgov_dict = json.loads(line)
                ctgov_dict_list.append(ctgov_dict)
    
    if args.split == 'train':
        # select the first 1/3 of the data
        ctgov_dict_list = ctgov_dict_list[2.5 * len(ctgov_dict_list) // 3:]
    
    num_processes = 10
    
    # split into num_processes chunks
    ctgov_dict_list_chunks = [ctgov_dict_list[i::num_processes] for i in range(num_processes)]

    # for each chunk, add the args and process id (i)
    
    for i, ctgov_dict_list_chunk in enumerate(ctgov_dict_list_chunks):
        ctgov_dict_list_chunks[i] = (args, ctgov_dict_list_chunk, i)
    
    pool = mp.Pool(processes=num_processes)

    pool.map(gen_reasons, ctgov_dict_list_chunks)
    pool.close()