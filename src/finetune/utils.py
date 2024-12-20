import pandas as pd
import json
import os
from tqdm import tqdm
from collections import defaultdict
import pdb

import sys
sys.path.append('./')
from src.eval.search.query_generation.eval import json_schema


def load_single_trial_summarization_data(data_path):
    df_data = pd.read_csv(data_path)
    input_text = df_data['input_text'].tolist()
    summary_text = df_data['summary_text'].tolist()
    instruction_prompt = "Your task is to create a clear, concise, and accurate summary of the provided clinical trial document. The summary should capture the key aspects of the trial."
    instruction_prompt += "\nThe output should only be the summarization of the given trial. Do not explain how you summarize it."
    instruction_prompt += "\nInput Text: {Text}"
    data_list = []
    for i in range(len(input_text)):
        source = {"content": instruction_prompt.format(Text=input_text[i]), 'role': 'user'}
        target = {"content": f"{summary_text[i]}", 'role': 'assistant'}
        data_list.append([source, target])
    
    return data_list


def load_multi_trial_summarization_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    output_data = defaultdict(list)
    for key, value in tqdm(data.items()):
        if len(value['title']) < 2:
            continue
        output_data['id'].append(key)
        # merge title and abstract list within the same paper (index), then add prefix "Study #x:"
        study_text = ""
        for i in range(len(value['title'])):
            study_text += f"Study #{i+1}: {value['title'][i]}. {value['abstract'][i]}.\n\n"
        # remove the last \n\n
        study_text = study_text[:-2]
        output_data['study_text'].append(study_text)
        output_data['target'].append(value["target"])
    
    df_data = pd.DataFrame(output_data)
    input_text = df_data['study_text'].tolist()
    target_text = df_data['target'].tolist()

    instruction_prompt = "Your task is to synthesize the key findings from a collection of study abstracts related to a specific clinical trial related research question. In some cases, you will also be provided with a review background detailing the research question of the given studies."
    instruction_prompt += "\nCombine the insights from the provided abstracts into a cohesive summary. Your summary should integrate the findings rather than listing them separately. It's crucial to maintain the scientific integrity of the original studies while ensuring the summary is accessible and informative."
    instruction_prompt += "\nThe output should only be the summary. Do not explain how you summarize it."
    instruction_prompt += "\n\nStudy Abstracts: {Text}"
    data_list = []

    for i in range(len(input_text)):
        source = {"content": instruction_prompt.format(Text=input_text[i]), 'role': 'user'}
        target = {"content": f"{target_text[i]}", 'role': 'assistant'}
        data_list.append([source, target])
    

    return data_list

def load_query_generation_data(data_path):

    with open(data_path, 'r') as f:
        data = json.load(f)

    instruction_prompt = "Given a query used for searching clinical trials in a database, conduct exact extracttion of related entities from the query and then generate a JSON object that can be used to query the database. If a field is not provided, leave it empty fiiled with 'N/A'."
    instruction_prompt += '\n\nQuery: {query}'
    instruction_prompt += '\nResult:'

    # SEARCH_TEMPLATE = """{prompt}\nOutput result in the following JSON schema format:\n{schema}"""
    # instruction_prompt = """Result: {response}"""

    data_list = []
    for key, value in tqdm(data.items()):
        try:
            parsed_dict = json.loads(value['parsed_dict'])
        except:
            print(key)
            continue
        # transform the parsed_dict to a string, key: value, if value is a list, join them with ','
        output = ''
        for k, v in parsed_dict.items():
            if isinstance(v, list):
                v = ', '.join(v)
            elif isinstance(v, dict):
                # out_dict = []
                # for k1, v1 in v.items():
                #     if not v1:
                #         v1 = 'N/A'
                #     out_dict.append(f"{k1}: {v1}")
                # v = ', '.join(out_dict)
                continue
            
            if not v:
                v = 'N/A'
            output += f"{k}: {v}\n"
        user_input = instruction_prompt.format(query=value['query'])
        # user_input = SEARCH_TEMPLATE.format(prompt=user_input, schema=json_schema)
        # assistant_response = RESPONSE_TEMPLATE.format(response=value['parsed_dict'])
        assistant_response = output
        source = {"content": user_input, 'role': 'user'}
        target = {"content": assistant_response, 'role': 'assistant'}
        data_list.append([source, target])
    

    return data_list

def load_query_expansion_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    output_data = {}

    for key, value in data.items():
        query = ', '.join(value['input'])
        expanded = ', '.join(value['output'])
        output_data[key] = {'query': query, 'expanded': expanded}

    instruction_prompt = "Given MeSH Terms used for searching clinical trials in a database, expand the input MeSH terms and then generate a JSON object that contains the expanded MeSH terms. Don't include the original MeSH terms in the expanded MeSH terms."
    instruction_prompt += '\n\nInput MeSH Terms: {query}. Now expand the input MeSH terms and generate the expanded MeSH terms.'
    
    data_list = []
    for key, value in tqdm(output_data.items()):
        source = {"content": instruction_prompt.format(query=value['query']), 'role': 'user'}
        target = {"content": value['expanded'], 'role': 'assistant'}
        data_list.append([source, target])
    
    return data_list

def load_trial_design_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    data_list = []
    for key, value in tqdm(data.items()):
        data_list.append(value)

    return data_list

def load_patient2trial_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    label_dict = {
        0: 'Excluded',
        1: 'Not relevant',
        2: 'Eligible'
    }
    
    data_list = []
    for key, value in tqdm(data.items()):
        source = {"content": value['input'], 'role': 'user'}
        
        # target = {"content": f"Trial-level eligibility: {value['label']}) {label_dict[value['label']]}.", 'role': 'assistant'}
        target = {'content': value['output'], 'role': 'assistant'}
        data_list.append([source, target])
    
    return data_list

if __name__ == '__main__':
    # load_multi_trial_summarization_data('data/downstream/summazization/multi-trial/train.json')
    # load_query_generation_data('data/downstream/search/query_generation/train.json')
    # load_query_expansion_data('data/downstream/search/query_expansion/test.json')
    # data_list = load_trial_design_data('data/downstream/design/parsed/study_arms/test.json')
    load_patient2trial_data('data/downstream/matching/patient2trial/TREC2021/train.json')