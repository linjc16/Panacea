import json
import re
import argparse
import os
import pdb
from tqdm import tqdm
from collections import defaultdict

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

import matplotlib.pyplot as plt

def extract_number(text):
    try:
        number = re.search(r"Number of Criteria: (\d+)", text).group(1)
    except AttributeError:
        try:
            number = re.search(r"\d+", text).group(0)
        except:
            number = 0
    return int(number)

def eval_subset(results_subset):
    # extract pred from value['output']
    preds = []
    labels = []
    for key, value in results_subset.items():
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

    accuracy = accuracy_score(labels, preds)

    balanced_accuracy = balanced_accuracy_score(labels, preds)
    
    # calculate kappa score
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(labels, preds)

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    F1_score = report['weighted avg']['f1-score']

    return balanced_accuracy, kappa, F1_score, accuracy


def calculate_inclu_exclu_dict(model_name):
    res_path = os.path.join(args.res_dir, f'{model_name}.json')
    with open(res_path, 'r') as f:
        results = json.load(f)
    
    
    data_by_inclusion_count = defaultdict(list)
    data_by_exclusion_count = defaultdict(list)
    # for each data in inputs, check nct_id, use extracted_data to get the count and append the input key to data_by_inclusion_count
    for key, value in inputs.items():
        nct_id = value['nct_id']
        if nct_id in extracted_data:
            data_by_inclusion_count[extracted_data[nct_id]['inclusion']].append(key)
            data_by_exclusion_count[extracted_data[nct_id]['exclusion']].append(key)
        
    # sorted the dict by key
    data_by_inclusion_count = dict(sorted(data_by_inclusion_count.items()))
    data_by_exclusion_count = dict(sorted(data_by_exclusion_count.items()))
    
    

    inclusion_results_by_count = {}
    
    for count, keys in data_by_inclusion_count.items():
        # get the subset of results by keys
        results_subset = {key: results[key] for key in keys}
        inclusion_results_by_count[count] = eval_subset(results_subset)
    

    exclusion_results_by_count = {}
    for count, keys in data_by_exclusion_count.items():
        # get the subset of results by keys
        results_subset = {key: results[key] for key in keys}
        exclusion_results_by_count[count] = eval_subset(results_subset)

    return inclusion_results_by_count, exclusion_results_by_count

def plot_results(results_by_count_dict, metric_index, metric_name):
    plt.figure(figsize=(12, 8))
    for model_name, results in results_by_count_dict.items():
        # x_inclusion = list(results['inclusion'].keys())
        # y_inclusion = [results['inclusion'][k][metric_index] for k in x_inclusion]
        # plt.plot(x_inclusion, y_inclusion, label=f'{model_name} Inclusion')

        x_exclusion = list(results['exclusion'].keys())
        y_exclusion = [results['exclusion'][k][metric_index] for k in x_exclusion]
        plt.plot(x_exclusion, y_exclusion, label=f'{model_name} Exclusion', linestyle='--')
        

    plt.title(f'{metric_name} by Number of Criteria')
    plt.xlabel('Number of Criteria')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{args.res_dir}/{metric_name}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/matching/patient2trial/TREC2021/results')
    parser.add_argument('--dataset', type=str, default='TREC2021')
    args = parser.parse_args()

    with open('src/vis/analysis/downstream/matching/results/criteria_counts.json') as file:
        data = json.load(file)

    extracted_data = {}
    for trial_id, criteria in data.items():
        extracted_data[trial_id] = {
            'inclusion': extract_number(criteria['inclusion']),
            'exclusion': extract_number(criteria['exclusion'])
        }
    
    with open(f'data/downstream/matching/patient2trial/{args.dataset}/test.json', 'r') as f:
        inputs = json.load(f)
    

    model_name_list = ['llama2-7b', 'mistral-7b', 'zephyr-7b', 'biomistral-7b', 'medalpaca-7b', 
                       'meditron-7b', 'panacea-base', 'panacea-7b']


    results_by_count_dict = {}
    for model_name in model_name_list:
        inclusion_results_by_count, exclusion_results_by_count = calculate_inclu_exclu_dict(model_name)
        results_by_count_dict[model_name] = {'inclusion': inclusion_results_by_count, 'exclusion': exclusion_results_by_count}
    
    # plot line chart, x-axis is the number of criteria, y-axis is the balanced accuracy, kappa or F1-score, totall 3 figures
    
    # Plotting results
    metrics = {
        'balanced_accuracy': 'Balanced Accuracy',
        'kappa': 'Cohen Kappa Score',
        'F1_score': 'F1 Score'
    }
    
    
    # Plotting results
    metrics = [(0, 'Balanced Accuracy'), (1, 'Cohen Kappa Score'), (2, 'F1 Score'), (3, 'Accuracy')]
    
    for index, name in metrics:
        plot_results(results_by_count_dict, index, name)