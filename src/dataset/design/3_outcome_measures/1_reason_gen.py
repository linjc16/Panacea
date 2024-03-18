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
    parser.add_argument('--task', type=str, default='outcome_measures')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    prompt = (
        'Given the information below about a clinical trial, please analyze and provide reasons '
        'for the design of each outcome measure listed under the "Primary Outcome Measure" and "Second Outcome Measure" sections. '
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
    
    save_path = f'data/downstream/design/raw/reasons/{args.task}'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'reasons_{args.split}.json')

    
    output_dict = {}
    ctgov_dict_list = []
    with open(f'data/downstream/design/raw/selected_step1/merged/{args.split}/merged.json', 'r') as f:
        for line in f:
            ctgov_dict = json.loads(line)
            ctgov_dict_list.append(ctgov_dict)
    
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            output_dict = json.load(f)

    i = 0
    for ctgov_dict in tqdm(ctgov_dict_list):
        if ctgov_dict['nct_id'] in output_dict:
            continue
        prompt_curr = prompt.format(**ctgov_dict)
        try:
            response = gpt_chat('gpt-3.5-turbo-0125', prompt_curr, seed=44)
        except:
            response = ""

        output_dict[ctgov_dict['nct_id']] = response

        if i % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(output_dict, f)
        i += 1
    
    with open(save_path, 'w') as f:
        json.dump(output_dict, f)
            
    
