import os
import sys
import json
import argparse
import random
import re
import glob

sys.path.append('./')
from src.utils.gpt_azure import gpt_chat_35
from tqdm import tqdm
import multiprocessing as mp

import pdb

def gen_chat_data(input):
    args, ctgov_dict_list, i = input
    output_dict = {}

    save_path = args.save_path.split('.json')[0] + f'_{i}.json'
    
    i = 0
    for ctgov_dict in tqdm(ctgov_dict_list):
        if ctgov_dict['nct_id'] in output_dict:
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

        # prompt_curr = prompt.format(**ctgov_dict, reasons=reasons_dict[ctgov_dict['nct_id']], interaction_prompt=interaction_prompt)
        
        # merge the dict
        query_dict = {**ctgov_dict, 'reasons': reasons_dict[ctgov_dict['nct_id']], 'interaction_prompt': interaction_prompt}

        for j in range(5):
            try:
                response = gpt_chat_35(prompt, query_dict)
            except:
                response = ''
            
            # detect whether the response contains "[xxx criteria xxx]"
            if not re.search(r'\[.*criteria.*\]', response):
                break
                
        
        output_dict[ctgov_dict['nct_id']] = response
        
        if i % 50 == 0:
            with open(save_path, 'w') as f:
                json.dump(output_dict, f, indent=4)
        i += 1
    
    with open(save_path, 'w') as f:
        json.dump(output_dict, f, indent=4)


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
        'The information the user should provide at the beginning of the conversation includes the following: '
        'Title, Conditions, Intervention / Treatment, Study Type, Phase. '
        'In the final part of the conversation, '
        'the conversation should output the full criteria provided above. Note that all the information in output full criteria can be exactly found from the conversation. '
        'Note that you should fully leverage the reasons provided for the design of each criterion '
        'in some smart way to generate the conversation data. '
        'The role in the generated conversation should be "User" and "Chatbot". '
    )
    
    save_path = f'data/downstream/design/raw/reasons/{args.task}/{args.split}/chat/chat_{args.split}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    args.save_path = save_path

    output_dict = {}

    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            output_dict = json.load(f)

    ctgov_dict_list = []
    merge_files = glob.glob(f'data/downstream/design/raw/selected_step1/merged/{args.split}/*.json')

    for merge_file in tqdm(merge_files):
        with open(merge_file, 'r') as f:
            for line in f:
                ctgov_dict = json.loads(line)
                ctgov_dict_list.append(ctgov_dict)
    
    reason_filedir = f"data/downstream/design/raw/reasons/{args.task}/{args.split}/*.json"
    reason_files = glob.glob(reason_filedir)

    reasons_dict = {}
    for reason_file in tqdm(reason_files):
        with open(reason_file, 'r') as f:
            reasons_dict.update(json.load(f))
    
    # remove the key with empty value
    reasons_dict = {k: v for k, v in reasons_dict.items() if v}

    # remove the item in ctgov_dict_list if the key is not in reasons_dict
    ctgov_dict_list = [ctgov_dict for ctgov_dict in ctgov_dict_list if ctgov_dict['nct_id'] in reasons_dict]

    num_processes = 10

    # split into num_processes chunks
    ctgov_dict_list_chunks = [ctgov_dict_list[i::num_processes] for i in range(num_processes)]

    # for each chunk, add the args and process id (i)
    
    for i, ctgov_dict_list_chunk in enumerate(ctgov_dict_list_chunks):
        ctgov_dict_list_chunks[i] = (args, ctgov_dict_list_chunk, i)

    pool = mp.Pool(processes=num_processes)

    pool.map(gen_chat_data, ctgov_dict_list_chunks)
    pool.close()