import pandas as pd
import json
import os
from tqdm import tqdm
from collections import defaultdict
import pdb

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
        output_data['id'].append(key)
        # merge title and abstract list within the same paper (index), then add prefix "Study #x:"
        study_text = ""
        for i in range(min(len(value['title']), 3)):
            study_text += f"Study #{i+1}: {value['title'][i]}. {value['abstract'][i]}.\n\n"
        # remove the last \n\n
        study_text = study_text[:-2]
        output_data['study_text'].append(study_text)
        output_data['background'].append(value["background"])
        output_data['target'].append(value["target"])
    
    df_data = pd.DataFrame(output_data)
    input_text = df_data['study_text'].tolist()
    bg_text = df_data['background'].tolist()
    target_text = df_data['target'].tolist()

    instruction_prompt = "Your task is to synthesize the key findings from a collection of study abstracts related to a specific clinical trial related research question. In some cases, you will also be provided with a review background detailing the research question of the given studies."
    instruction_prompt += "\nCombine the insights from the provided abstracts into a cohesive summary. Your summary should integrate the findings rather than listing them separately. It's crucial to maintain the scientific integrity of the original studies while ensuring the summary is accessible and informative."
    instruction_prompt += "\nThe output should only be the summary. Do not explain how you summarize it."
    instruction_prompt += "\n\nReview Background: {Background}"
    instruction_prompt += "\n\nStudy Abstracts: {Text}"
    data_list = []

    for i in range(len(input_text)):
        source = {"content": instruction_prompt.format(Text=input_text[i], Background=bg_text[i]), 'role': 'user'}
        target = {"content": f"{target_text[i]}", 'role': 'assistant'}
        data_list.append([source, target])
    
    return data_list

if __name__ == '__main__':
    load_multi_trial_summarization_data('data/downstream/summazization/multi-trial/test.json')