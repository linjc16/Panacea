import os
import sys
import json
import argparse
import random
import re
import copy

sys.path.append('./')
from src.utils.gpt_azure import gpt_chat_35
from tqdm import tqdm

import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='criteria')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    prompt = (
        'Given the information below about a clinical trial, please generate multi-turn conversation data used for training models. '
        'The generated conversation should revolve around criteria design, including the inclusion and exclusion criteria. '
        'Moreover, the generated conversation should contain interactions between users and chatbots: '
        '{interaction_prompt}'
        'In such way, they complete the design of all of the criteria one by one and step by step. '
        '\n\nBelow is the information about the clinical trial:\n\n'
        'Title: {brief_title}\n'
        'Official Title: {official_title}\n'
        'Conditions: {conditions}\n'
        'Intervention / Treatment: {interventions}\n'
        'Study Type: {study_type}\n'
        'Phase: {phase}\n\n'
        'Brief Summary: {brief_summary}\n\n'
        'Criteria: {eligibility_criteria}\n\n'
        'Reasons for the design of each criterion: {reasons} \n\n'
        'Now generate the conversation data for the design of the criteria. '
        'The information the user should implicitly provide includes the following: '
        'Title, Conditions, Intervention / Treatment, Study Type, Phase. '
        'In the final part of the conversation, '
        'the conversation should output the full criteria provided above. Note that all the information in output full criteria can be exactly found from the conversation. '
        'Note that you should fully leverage the reasons provided for the design of each criterion '
        'in some smart way to generate the conversation data. '
        'The role in the generated conversation should be "User" and "Chatbot". '
    )
    
    gen_path = f'data/downstream/design/raw/reasons/{args.task}/{args.split}/chat/chat_{args.split}_refined_v1.json'
    save_path = f'data/downstream/design/raw/reasons/{args.task}/{args.split}/chat/chat_{args.split}.json'
    
    output_dict = {}

    if os.path.exists(gen_path):
        with open(gen_path, 'r') as f:
            output_dict = json.load(f)

    # pdb.set_trace()
    
    ctgov_dict_list = []
    with open(f'data/downstream/design/raw/selected_step1/merged/{args.split}/merged.json', 'r') as f:
        for line in f:
            ctgov_dict = json.loads(line)
            ctgov_dict_list.append(ctgov_dict)
    
    with open(f'data/downstream/design/raw/reasons/{args.task}/{args.split}/reasons_{args.split}.json', 'r') as f:
        reasons_dict = json.load(f)
    
    output_dict_new = copy.deepcopy(output_dict)

    nct_ids_error = [
        'NCT03912259', 'NCT03564340', 'NCT01363440', 'NCT06121180', 'NCT05133531', 'NCT01963598',
        'NCT02540369', 'NCT05505448', 'NCT02120950', 'NCT06128629', 'NCT05092581', 'NCT06191315',
        'NCT02017639', 'NCT05828511', 'NCT02776735', 'NCT05338879'
    ]

    i = 0
    for ctgov_dict in tqdm(ctgov_dict_list):
        if ctgov_dict['nct_id'] not in nct_ids_error:
            i += 1
            continue
        
        if random.random() > 0.3:
            interaction_prompt = (
                'most of the time, chatbot gives advice on criteria design; '
                'when there is something needed to be clarified, users can provide some ideas to the chatbot. '
                'Also, somethimes when chatbot asks the user for ideas, the user may have no idea and then chatbot should give some suggestions. '
            )
        else:
            interaction_prompt = 'almost all the advice and design ideas are provided by the chatbot, and the user just follows the chatbot. '

        query_dict = {**ctgov_dict, 'reasons': reasons_dict[ctgov_dict['nct_id']], 'interaction_prompt': interaction_prompt}

        for j in range(10):
            try:
                response = gpt_chat_35(prompt, query_dict)
            except:
                response = ''
            
            # detect whether the response contains "[xxx criteria xxx]"
            if not re.search(r'\[.*criteria.*\]', response) and not re.search(r'\(.*criterion.*\)', response) and \
                not re.search(r'\[.*conversation].*\]', response) and not re.search(r'\(.*conversation].*\)', response):
                break
                
        
        output_dict_new[ctgov_dict['nct_id']] = response
        
        if i % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(output_dict_new, f, indent=4)
        i += 1
    
    with open(save_path, 'w') as f:
        json.dump(output_dict_new, f, indent=4)