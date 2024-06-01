import json
import re
import os
import argparse
import pdb
from tqdm import tqdm

import sys
sys.path.append('./')
from src.utils.claude_aws import chat_haiku, chat_sonnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--res_dir', type=str, default='/data/linjc/trialfm/downstream/summarization/results')
    parser.add_argument('--dataset', type=str, default='TREC2021')
    args = parser.parse_args()

    args.res_dir = os.path.join(args.res_dir, f'{args.model_name}.json')

    with open(args.res_dir, 'r') as f:
        results = json.load(f)
    
    # extract pred from value['output']
    preds = []
    labels = []
    for key, value in results.items():
        # extract pred from value['output']
        if 'eligibility: 0)' in value['output']:
            preds.append(0)
        elif 'eligibility: 1)' in value['output']:
            preds.append(1)
        elif 'eligibility: 2)' in value['output']:
            preds.append(2)
        else:
            preds.append(-1)
        
        labels.append(value['label'])
    
    
    assert len(preds) == len(labels)
    
    if args.dataset == 'TREC2021':
        label_mapping = {
            0: "0) Excluded (patient meets inclusion criteria, but is excluded on the grounds of the trial's exclusion criteria)",
            1: "1) Not relevant (patient does not have sufficient information to qualify for the trial)",
            2: "2) Eligible (patient meets inclusion criteria and exclusion criteria do not apply)"
        }
    
    prompt = (
        'Given a patient note, clinical tiral eligibility criteria, and a label, help me understand why the model made a wrong prediction. '
        'You can choose one or more reasons from the following list: '
        '1. The model hallucinated additional criteria that do not exist in the input clinical trial eligibility criteria. '
        '2. The model hallucinated additional patient information that do not exist in the input patient note. '
        '3. Other reasons.\n\n'
        'Patient note: {patient_note}\n'
        'Clinical trial eligibility criteria: {criteria}\n'
        'Label: {label}\n'
        'Model prediction: {pred}\n\n'
        'Output the error reason without explanation in a list format in the last line, e.g., Reason type: [1] or [1, 2].\n\n'
    )

    with open('src/vis/analysis/downstream/matching/hallucination/inputs.json', 'r') as f:
        inputs = json.load(f)

    reasons_dict = {}
    i = 0
    for i in tqdm(range(len(preds))):
        if preds[i] == -1:
            reasons_dict[i] = 'Non-responsive Error'
            if i % 100 == 0:
                with open(f'src/vis/analysis/downstream/matching/hallucination/results/{args.model_name}.json', 'w') as f:
                    json.dump(reasons_dict, f, indent=4)
                
            i += 1
            continue
            
        if preds[i] != labels[i]:
            patient_notes = inputs[str(i)]['patient_notes']
            criteria = inputs[str(i)]['clinical_trial']
            model_output = results[str(i)]['output']
            label = label_mapping[labels[i]]

            prompt = prompt.format(patient_note=patient_notes, criteria=criteria, label=label, pred=model_output)
            attempt = 0
            while attempt < 10:
                try:
                    reason = chat_sonnet(prompt)
                    break
                except:
                    attempt += 1
                    continue
            
            reasons_dict[i] = reason
        
        if i % 100 == 0:
            with open(f'src/vis/analysis/downstream/matching/hallucination/results/{args.model_name}.json', 'w') as f:
                json.dump(reasons_dict, f, indent=4)
            
        i += 1

    with open(f'src/vis/analysis/downstream/matching/hallucination/results/{args.model_name}.json', 'w') as f:
        json.dump(reasons_dict, f, indent=4)
    