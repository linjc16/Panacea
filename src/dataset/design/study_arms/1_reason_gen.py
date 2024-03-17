import os
import sys
import json
import argparse

sys.path.append('./')
from src.utils.gpt import gpt_chat
from tqdm import tqdm

import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='study_arms')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    prompt = (
        'Given the information below about a clinical trial, please analyze and provide reasons '
        'for the design of each study arms listed under the "Study Arms" section. '
        'For each study arm, focus on Participant Group/Arm, Intervention/Treatment, and so on. '
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
        'Study Arms: {arms_and_interventions}'
    )
    
    save_path = f'data/downstream/design/raw/reasons/{args.task}/reasons_{args.split}.json'

    
    output_dict = {}
    ctgov_dict_list = []
    with open(f'data/downstream/design/raw/selected_step1/merged/{args.split}/merged.json', 'r') as f:
        for line in f:
            ctgov_dict = json.loads(line)
            ctgov_dict_list.append(ctgov_dict)

    
    i = 0
    for ctgov_dict in tqdm(ctgov_dict_list):
        prompt_curr = prompt.format(**ctgov_dict)
        response = gpt_chat('gpt-3.5-turbo-0125', prompt_curr, seed=44)

        output_dict[ctgov_dict['nct_id']] = response

        if i % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(output_dict, f)
        i += 1

    with open(save_path, 'w') as f:
        json.dump(output_dict, f)
            
    
