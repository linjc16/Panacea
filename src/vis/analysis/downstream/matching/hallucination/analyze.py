import argparse
import json
import re
import os
from collections import Counter

import pdb


def process_errors(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    results = {}
    non_responsive_pattern = "Non-responsive Error"
    reason_type_pattern = r"Reason type: (\[[^\]]+\])"

    for key, text in data.items():
        if non_responsive_pattern in text:
            results[key] = [-1]
        else:
            match = re.search(reason_type_pattern, text)
            if match:
                # Convert the matched string to a list using eval
                results[key] = eval(match.group(1))
            else:
                results[key] = None  # In case no reason type is found

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TREC2021')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    parser.add_argument('--res_dir', type=str, default='data/downstream/matching/patient2trial/TREC2021/results1')
    parser.add_argument('--hallucination_file_dir', type=str, default='src/vis/analysis/downstream/matching/hallucination/results')
    args = parser.parse_args()

    filepath = os.path.join(args.hallucination_file_dir, f'{args.model_name}.json')

    err_results = process_errors(filepath)
    

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


    eligible_err_key_list = []
    non_eligible_err_key_list = []
    # find label == 2, but output others
    for i in range(len(preds)):
        if labels[i] == 2 and preds[i] != 2:
            eligible_err_key_list.append(str(i))
    
    print('Type 1 error number:', len(eligible_err_key_list))
    
    # find label != 2, but output 2
    for i in range(len(preds)):
        if labels[i] != 2 and preds[i] == 2:
            non_eligible_err_key_list.append(str(i))
        
    print('Type 2 error number:', len(non_eligible_err_key_list))
    
    eligible_err_results = [err_results[key] for key in eligible_err_key_list]
    non_eligible_err_results = [err_results[key] for key in non_eligible_err_key_list]

    # merge lists in eligible_err_results
    merged_eligible_err_results = []
    for err_list in eligible_err_results:
        merged_eligible_err_results.extend(err_list)
    
    # merge lists in non_eligible_err_results
    merged_non_eligible_err_results = []
    for err_list in non_eligible_err_results:
        merged_non_eligible_err_results.extend(err_list)

    # count the frequency of each reason
    eligible_err_counter = Counter(merged_eligible_err_results)
    non_eligible_err_counter = Counter(merged_non_eligible_err_results)
    

    print(f"Eligible errors: {eligible_err_counter}")
    print(f"Non-eligible errors: {non_eligible_err_counter}")

