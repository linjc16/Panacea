import json
import re
import os
import argparse
import pdb

import sys
sys.path.append('./')
from src.utils.claude_aws import chat_haiku, chat_sonnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--res_dir', type=str, default='/data/linjc/trialfm/downstream/summarization/results')
    parser.add_argument('--dataset', type=str, default='cohort')
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
        'Output the error reason in a list format, e.g., [1] or [1, 2].\n\n'
    )

    reasons_dict = {}
    for i in range(len(preds)):
        if preds[i] == -1:
            reasons_dict[i] = 'Non-responsive Error'
            continue

        if preds[i] != labels[i]:
            model_output = results[str(i)]['output']

    # for each sample, if the pred is wrong, use chat_haiku to analize the error reason
    