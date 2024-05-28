import json
import re
import argparse
import os
import pdb
from tqdm import tqdm
from collections import defaultdict
import numpy as np

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

    try:
        F1_score_eligible = report['2']['f1-score']
    except:
        F1_score_eligible = 0

    return balanced_accuracy, kappa, F1_score, F1_score_eligible


def calculate_inclu_exclu_dict(model_name, num_bins=10):
    res_path = os.path.join(args.res_dir, f'{model_name}.json')
    with open(res_path, 'r') as f:
        results = json.load(f)
    
    
    # data_by_inclusion_count = defaultdict(list)
    # data_by_exclusion_count = defaultdict(list)
    # # for each data in inputs, check nct_id, use extracted_data to get the count and append the input key to data_by_inclusion_count
    # for key, value in inputs.items():
    #     nct_id = value['nct_id']
    #     if nct_id in extracted_data:
    #         data_by_inclusion_count[extracted_data[nct_id]['inclusion']].append(key)
    #         data_by_exclusion_count[extracted_data[nct_id]['exclusion']].append(key)
        

    data_by_inclusion_count = defaultdict(list)
    data_by_exclusion_count = defaultdict(list)
    
    inclusion_counts = [extracted_data[value['nct_id']]['inclusion'] for key, value in inputs.items() if value['nct_id'] in extracted_data]
    exclusion_counts = [extracted_data[value['nct_id']]['exclusion'] for key, value in inputs.items() if value['nct_id'] in extracted_data]
    
    # Calculate bin edges for both inclusion and exclusion counts
    # inclusion_bins = np.histogram_bin_edges(inclusion_counts, bins=num_bins)
    # exclusion_bins = np.histogram_bin_edges(exclusion_counts, bins=num_bins)

    # # remove 0 from inclusion_counts and exclusion_counts
    # inclusion_counts = [count for count in inclusion_counts if count != 0]
    # exclusion_counts = [count for count in exclusion_counts if count != 0]
    
    inclusion_bins = np.percentile(inclusion_counts, np.linspace(0, 100, num_bins + 1))
    exclusion_bins = np.percentile(exclusion_counts, np.linspace(0, 100, num_bins + 1))
    

    # Assigning inputs to bins based on their count
    for key, value in inputs.items():
        nct_id = value['nct_id']
        if nct_id in extracted_data:
            inclusion_count = extracted_data[nct_id]['inclusion']
            exclusion_count = extracted_data[nct_id]['exclusion']
            # Find the appropriate bin for the inclusion and exclusion counts
            inclusion_bin_index = np.digitize(inclusion_count, inclusion_bins) - 1  # -1 because bins are 1-indexed in output
            exclusion_bin_index = np.digitize(exclusion_count, exclusion_bins) - 1
            data_by_inclusion_count[inclusion_bin_index].append(key)
            data_by_exclusion_count[exclusion_bin_index].append(key)
    
    # Convert defaultdict to dict for final output
    data_by_inclusion_count = dict(data_by_inclusion_count)
    data_by_exclusion_count = dict(data_by_exclusion_count)

    # sorted the dict by key
    data_by_inclusion_count = dict(sorted(data_by_inclusion_count.items()))
    data_by_exclusion_count = dict(sorted(data_by_exclusion_count.items()))

    # Map old indices to the range strings
    new_data_by_inclusion_count = {}
    new_data_by_exclusion_count = {}
    
    for bin_index in sorted(data_by_inclusion_count.keys()):
        if bin_index < len(inclusion_bins) - 1:
            key = f"{inclusion_bins[bin_index]}-{inclusion_bins[bin_index + 1]}"
            new_data_by_inclusion_count[key] = data_by_inclusion_count[bin_index]
    
    for bin_index in sorted(data_by_exclusion_count.keys()):
        if bin_index < len(exclusion_bins) - 1:
            key = f"{exclusion_bins[bin_index]}-{exclusion_bins[bin_index + 1]}"
            new_data_by_exclusion_count[key] = data_by_exclusion_count[bin_index]
    
    data_by_inclusion_count = new_data_by_inclusion_count
    data_by_exclusion_count = new_data_by_exclusion_count

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
        x_inclusion = list(results['inclusion'].keys())
        y_inclusion = [results['inclusion'][k][metric_index] for k in x_inclusion]
        plt.plot(x_inclusion, y_inclusion, label=f'{model_name} Inclusion')
        
        # x_exclusion = list(results['exclusion'].keys())
        # y_exclusion = [results['exclusion'][k][metric_index] for k in x_exclusion]
        # plt.plot(x_exclusion, y_exclusion, label=f'{model_name} Exclusion', linestyle='--')
        

    plt.title(f'{metric_name} by Number of Criteria')
    plt.xlabel('Number of Criteria')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visulization/line_chart/{metric_name}.png')


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
    metrics = [(0, 'Balanced Accuracy'), (1, 'Cohen Kappa Score'), (2, 'F1 Score'), (3, 'F1 Score Eligible')]
    
    for index, name in metrics:
        plot_results(results_by_count_dict, index, name)