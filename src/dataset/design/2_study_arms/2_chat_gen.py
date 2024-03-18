import os
import sys
import json
import argparse
import random
import re

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
        'Given the information below about a clinical trial, please generate multi-turn conversation data used for training models. '
        'The generated conversation should revolve around study arm design, including the participant group/arm and intervention/treatment. '
        'Moreover, the generated conversation should contain interactions between users and chatbots: '
        '{interaction_prompt}'
        'In such way, they complete the design of all of the study arms one by one and step by step. '
        '\n\nBelow is the information about the clinical trial:\n\n'
        'Title: {brief_title}\n'
        'Official Title: {official_title}\n'
        'Conditions: {conditions}\n'
        'Intervention / Treatment: {interventions}\n'
        'Study Type: {study_type}\n'
        'Phase: {phase}\n\n'
        'Brief Summary: {brief_summary}\n\n'
        'Criteria: {eligibility_criteria}\n\n'
        'Design Details: {design_details}\n\n'
        'Study Arms: {arms_and_interventions}\n\n'
        'Reasons for the design of each study arm: {reasons} \n\n'
        'Now generate the conversation data for the design of the study arms. '
        'The information the user should implicitly provide includes the following: '
        'Title, Conditions, Intervention / Treatment, Study Type, Phase, Criteria, Design Details and so on.'
        'In the final part of the conversation, '
        'the conversation should output the full study arms provided above. Note that all the information in output full study arms can be exactly found from the conversation. '
        'Note that you should fully leverage the reasons provided for the design of each study arm '
        'in some smart way to generate the conversation data. '
        'The role in the generated conversation should be "User" and "Chatbot". '
    )
    
    save_path = f'data/downstream/design/raw/reasons/{args.task}/chat_{args.split}.json'

    output_dict = {}

    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            output_dict = json.load(f)

    ctgov_dict_list = []
    with open(f'data/downstream/design/raw/selected_step1/merged/{args.split}/merged.json', 'r') as f:
        for line in f:
            ctgov_dict = json.loads(line)
            ctgov_dict_list.append(ctgov_dict)
    
    with open(f'data/downstream/design/raw/reasons/{args.task}/reasons_{args.split}.json', 'r') as f:
        reasons_dict = json.load(f)
    
    i = 0
    for ctgov_dict in tqdm(ctgov_dict_list):
        if ctgov_dict['nct_id'] in output_dict:
            i += 1
            continue
            
        if random.random() > 0.3:
            interaction_prompt = (
                'most of the time, chatbot gives advice on study arm design; '
                'when there is something needed to be clarified, users can provide some ideas to the chatbot. '
                'Also, somethimes when chatbot asks the user for ideas, the user may have no idea and then chatbot should give some suggestions. '
            )
        else:
            interaction_prompt = 'almost all the advice and design ideas are provided by the chatbot, and the user just follows the chatbot. '

        prompt_curr = prompt.format(**ctgov_dict, reasons=reasons_dict[ctgov_dict['nct_id']], interaction_prompt=interaction_prompt)
        
        for j in range(5):
            try:
                response = gpt_chat('gpt-3.5-turbo-0125', prompt_curr)
            except:
                response = ''
            
            # detect whether the response contains "[xxx study arms xxx]"
            if not re.search(r'\[.*study arms.*\]', response):
                break
                
        
        output_dict[ctgov_dict['nct_id']] = response
        
        if i % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(output_dict, f, indent=4)
        i += 1
    
    with open(save_path, 'w') as f:
        json.dump(output_dict, f, indent=4)