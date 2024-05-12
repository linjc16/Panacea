import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

import plot_settings_bar
import plot_utils

import argparse


data_labels = ["Ours", "Bio\nOpen", "General\nOpen", "General\nClosed"]

model_colors = {
    'LLaMA-2-7B': '#009fff',
    'LLaMA-3-8B': '#63bff0',
    'Mistral-7B': '#ec9a7e',
    'OpenChat-7B': '#cb6156',
    'Zephyr-7B': '#fad3bc',
    'BioMistral-7B': '#1b5a98',
    'MedAlpaca-7B': '#3b80ae',
    'Meditron-7B': '#b3d3e6',
    'GPT-3.5': '#d1d2d4',
    'GPT-4': '#a4a2a8',
    'Claude 3 Haiku': '#7f8081',
    'Claude 3 Sonnet': '#4a4b4c',
    'Panacea': '#a62b35',
    'Panacea-Base': '#d75050'
}


def bar_plot(nested_data, data_labels, name, y_lim=None):

    
    ax = plot_settings_bar.get_wider_axis(2, 4)


    plot_utils.grouped_barplot(ax, nested_data, data_labels, None, metric, model_colors,
                    xscale='linear', yscale='linear', min_val=0, invert_axes=False, tickloc_top=False, rotangle=0, anchorpoint='center', y_lim=y_lim)

    
    plot_utils.format_ax(ax)
    

    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', 
                            ncols=1)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)

    # set y-axis label size, don't set name again
    
    
    plt.savefig(f'visulization/bar_{args.task}_{name}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'visulization/bar_{name}_ablation.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='single_trial_sum')
    args = parser.parse_args()
    
    filepaths_dict = {
        'query_expansion': 'src/vis/results/query_expansion.csv',
        'single_trial_sum': 'src/vis/results/single_trial_sum.csv',
        'multi_trial_sum': 'src/vis/results/multi_trial_sum.csv',
        'query_generation': 'src/vis/results/query_generation.csv',
        'patient_trial_matching_trec2021': 'src/vis/results/patient_trial_matching_trec2021.csv',
        'patient_trial_matching_sigir': 'src/vis/results/patient_trial_matching_sigir.csv',
        'criteria_design': 'src/vis/results/criteria_design.csv',
        'study_arm_design': 'src/vis/results/study_arm_design.csv',
        'outcome_measure_design': 'src/vis/results/outcome_measure_design.csv',
    }

    y_lim_dict = {
        'query_expansion': {'Precision': (0, 0.7), 'Recall': (0, 0.7), 'F1': (0, 0.7)},
        'single_trial_sum': {'ROUGE-1': (0, 0.5), 'ROUGE-2': (0, 0.2), 'ROUGE-L': (0, 0.25)},
        'multi_trial_sum': {'ROUGE-1': (0, 0.3), 'ROUGE-2': (0, 0.06), 'ROUGE-L': (0, 0.18)},
        'query_generation': {'Precision': (0, 0.9), 'Recall': (0, 0.9), 'F1': (0, 0.9)},
        'patient_trial_matching_trec2021': {'BACC': (0, 0.6), 'Recall (Eligible)': (0, 0.8), 'F1': (0, 0.7)},
        'patient_trial_matching_sigir': {'BACC': (0, 0.6), 'Recall (Eligible)': (0, 0.6), 'F1': (0, 0.7)},
        'criteria_design': {'BLEU-4': (0, 0.3), 'ROUGE-L': (0, 0.5), 'Clinical Accuracy': (0, 0.9)},
        'study_arm_design': {'BLEU-4': (0, 0.4), 'ROUGE-L': (0, 0.6), 'Clinical Accuracy': (0, 0.7)},
        'outcome_measure_design': {'BLEU-4': (0, 0.4), 'ROUGE-L': (0, 0.6), 'Clinical Accuracy': (0, 0.6)},
    }

    data = pd.read_csv(filepaths_dict[args.task])
    metrics = data.columns[1:]

    full_dict = {}

    for metric in metrics:
        # Create the nested data structure for each metric
        nested_data = [
            [('Panacea', data[data['Model'] == 'Panacea (Ours)'][metric].values[0]),
            ('Panacea-Base', data[data['Model'] == 'Panacea-Base'][metric].values[0])],
            [('BioMistral-7B', data[data['Model'] == 'BioMistral-7B'][metric].values[0]),
            ('MedAlpaca-7B', data[data['Model'] == 'MedAlpaca-7B'][metric].values[0]),
            ('Meditron-7B', data[data['Model'] == 'Meditron-7B'][metric].values[0])],
            [('OpenChat-7B', data[data['Model'] == 'OpenChat-7B'][metric].values[0]),
            ('Mistral-7B', data[data['Model'] == 'Mistral-7B'][metric].values[0]),
            ('Zephyr-7B', data[data['Model'] == 'Zephyr-7B'][metric].values[0]),
            ('LLaMA-3-8B', data[data['Model'] == 'LLaMA-3-8B'][metric].values[0]),
            ('LLaMA-2-7B', data[data['Model'] == 'LLaMA-2-7B'][metric].values[0])],
            [('Claude 3 Sonnet', data[data['Model'] == 'Claude 3 Sonnet'][metric].values[0]),
            ('Claude 3 Haiku', data[data['Model'] == 'Claude 3 Haiku'][metric].values[0]),
            ('GPT-4', data[data['Model'] == 'GPT-4'][metric].values[0]),
            ('GPT-3.5', data[data['Model'] == 'GPT-3.5'][metric].values[0])]
        ]
        full_dict[metric] = nested_data
    
    for metric in metrics:
        bar_plot(full_dict[metric], data_labels, metric, y_lim=y_lim_dict[args.task].get(metric.strip(), ""))

