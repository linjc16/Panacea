import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append('./src/vis/src')

import plot_settings_bar
import plot_utils


# Define model colors
# model_colors = {
#     'LLaMA-2-7B': '#90e0ef',
#     'LLaMA-3-8B': '#48cae4',
#     'Mistral-7B': '#0096c7',
#     'OpenChat-7B': '#023e8a',
#     'Zephyr-7B': '#00b4d8',
#     'BioMistral-7B': '#0077b6',
#     'MedAlpaca-7B': '#ade8f4',
#     'Meditron-7B': '#caf0f8',
#     'Panacea': '#03045e',
#     'Panacea-Base': '#d75050'
# }

model_colors = {
    'LLaMA-2-7B': '#e7ad61',
    'LLaMA-3-8B': '#e0c17d',
    'Mistral-7B': '#daf0ed',
    'OpenChat-7B': '#30988e',
    'Zephyr-7B': '#f8e6c3',
    'BioMistral-7B': '#81ccc1',
    'MedAlpaca-7B': '#8c5109',
    'Meditron-7B': '#53300e',
    'Panacea': '#00675e',
    'Panacea-Base': '#bf802d'
}

# model_colors = {
#     'LLaMA-2-7B': '#bf802d',
#     'LLaMA-3-8B': '#8c5109',
#     'Mistral-7B': '#00675e',
#     'OpenChat-7B': '#81ccc1',
#     'Zephyr-7B': '#53300e',
#     'BioMistral-7B': '#30988e',
#     'MedAlpaca-7B': '#e0c17d',
#     'Meditron-7B': '#f8e6c3',
#     'Panacea': '#daf0ed',
#     'Panacea-Base': '#e7ad61'
# }

def bar_plot(nested_data, data_labels, name, y_lim=None):

    ax = plot_settings_bar.get_wider_axis(4, 4)

    # Data labels
    data_labels = ['Criteria', 'Study Arms', 'Outcome Measures']

    # Plotting the  data
    plot_utils.grouped_barplot(ax, nested_data, data_labels, None, 
                            name, model_colors, xscale='linear', yscale='linear', 
                            min_val=0, invert_axes=False, tickloc_top=False,  rotangle=0, anchorpoint='center', y_lim=y_lim)
    plot_utils.format_ax(ax)

    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', 
                            ncols=1)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)

    # Show the plot
    plt.savefig(f'visulization/bar_design_{name}.png', dpi=300, bbox_inches='tight')


def get_nested_data_single(data):
    nested_data = [
        ('Panacea', data[data['Model'] == 'Panacea'][metric].values[0]),
        ('OpenChat-7B', data[data['Model'] == 'OpenChat-7B'][metric].values[0]),
        ('BioMistral-7B', data[data['Model'] == 'BioMistral-7B'][metric].values[0]),
        ('Mistral-7B', data[data['Model'] == 'Mistral-7B'][metric].values[0]),
        ('Zephyr-7B', data[data['Model'] == 'Zephyr-7B'][metric].values[0]),
        ('LLaMA-3-8B', data[data['Model'] == 'LLaMA-3-8B'][metric].values[0]),
        ('LLaMA-2-7B', data[data['Model'] == 'LLaMA-2-7B'][metric].values[0]),
        ('Panacea-Base', data[data['Model'] == 'Panacea-Base'][metric].values[0]),
        ('MedAlpaca-7B', data[data['Model'] == 'MedAlpaca-7B'][metric].values[0]),
        ('Meditron-7B', data[data['Model'] == 'Meditron-7B'][metric].values[0]),
    ]

    return nested_data

if __name__ == '__main__':
    # Load the CSV files
    criteria_data = pd.read_csv('src/vis/results/criteria_design.csv')
    study_arm_data = pd.read_csv('src/vis/results/study_arm_design.csv')
    outcome_measure_data = pd.read_csv('src/vis/results/outcome_measure_design.csv')

    # remove ['GPT-3.5‘, 'GPT-4', 'Claude 3 Haiku', 'Claude 3 Sonnet'] from the data
    criteria_data = criteria_data[~criteria_data['Model'].isin(['GPT-3.5', 'GPT-4', 'Claude 3 Haiku', 'Claude 3 Sonnet'])]
    study_arm_data = study_arm_data[~study_arm_data['Model'].isin(['GPT-3.5', 'GPT-4', 'Claude 3 Haiku', 'Claude 3 Sonnet'])]
    outcome_measure_data = outcome_measure_data[~outcome_measure_data['Model'].isin(['GPT-3.5', 'GPT-4', 'Claude 3 Haiku', 'Claude 3 Sonnet'])]

    y_lim_dict = {'BLEU-4': (0, 0.4), 'ROUGE-L': (0, 0.6), 'Clinical Accuracy': (0, 0.6)}
    metrics = criteria_data.columns[1:]

    for metric in metrics:
        nested_data = [
            get_nested_data_single(criteria_data),
            get_nested_data_single(study_arm_data),
            get_nested_data_single(outcome_measure_data),
        ]
        bar_plot(nested_data, None, metric)#, y_lim=y_lim_dict[metric])