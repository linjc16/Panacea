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

def bar_plot(nested_data, data_labels, name, nested_errs, y_lim=None):

    ax = plot_settings_bar.get_wider_axis(2, 4)

    # Data labels
    data_labels = ['SIGIR', 'TREC 2021']

    # Plotting the  data
    plot_utils.grouped_barplot(ax, nested_data, data_labels, None, 
                            name, model_colors, xscale='linear', yscale='linear', 
                            min_val=0, invert_axes=False, tickloc_top=False,  rotangle=0, anchorpoint='center', y_lim=y_lim, nested_errs=nested_errs)
    plot_utils.format_ax(ax)

    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', 
                            ncols=1)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # set y-label font size
    plt.ylabel(name, fontsize=16)

    # Show the plot
    plt.savefig(f'visulization/bar_matching_{name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'visulization/bar_matching_{name}.pdf', dpi=300, bbox_inches='tight')


def get_nested_data_single(data):
    nested_data = [
        ('Panacea', data[data['Model'] == 'Panacea'][metric].values[0]),
        # ('OpenChat-7B', data[data['Model'] == 'OpenChat-7B'][metric].values[0]),
        ('BioMistral-7B', data[data['Model'] == 'BioMistral-7B'][metric].values[0]),
        ('Mistral-7B', data[data['Model'] == 'Mistral-7B'][metric].values[0]),
        ('Zephyr-7B', data[data['Model'] == 'Zephyr-7B'][metric].values[0]),
        # ('LLaMA-3-8B', data[data['Model'] == 'LLaMA-3-8B'][metric].values[0]),
        ('LLaMA-2-7B', data[data['Model'] == 'LLaMA-2-7B'][metric].values[0]),
        ('Panacea-Base', data[data['Model'] == 'Panacea-Base'][metric].values[0]),
        ('MedAlpaca-7B', data[data['Model'] == 'MedAlpaca-7B'][metric].values[0]),
        ('Meditron-7B', data[data['Model'] == 'Meditron-7B'][metric].values[0]),
    ]

    return nested_data

def get_nested_data_err(data):
    # nested_errs = [[5, 8], [7, 6]]
    nested_errs = [
        data[data['Model'] == 'Panacea'][metric].values[0],
        # data[data['Model'] == 'OpenChat-7B'][metric].values[0],
        data[data['Model'] == 'BioMistral-7B'][metric].values[0],
        data[data['Model'] == 'Mistral-7B'][metric].values[0],
        data[data['Model'] == 'Zephyr-7B'][metric].values[0],
        # data[data['Model'] == 'LLaMA-3-8B'][metric].values[0],
        data[data['Model'] == 'LLaMA-2-7B'][metric].values[0],
        data[data['Model'] == 'Panacea-Base'][metric].values[0],
        data[data['Model'] == 'MedAlpaca-7B'][metric].values[0],
        data[data['Model'] == 'Meditron-7B'][metric].values[0],
    ]

    return nested_errs

def load_mean_err(dataset):
    frames = []
    for i in [0, 1, 2]:
        df = pd.read_csv(f'src/vis/results/{i}/patient_trial_matching_{dataset}.csv')
        frames.append(df)
    combined = pd.concat(frames)
    mean = combined.groupby('Model').mean()
    error = combined.groupby('Model').std() / np.sqrt(len(mean))

    # 'Model' is the first column name
    mean = mean.reset_index()
    error = error.reset_index()

    # replace column names (F1 (Class 0) -> F1 (Excluded)), (F1 (Class 2) -> F1 (ELigible))
    mean.columns = ['Model'] + [metric.replace('Class 0', 'Excluded').replace('Class 2', 'Eligible') for metric in mean.columns[1:]]
    error.columns = ['Model'] + [metric.replace('Class 0', 'Excluded').replace('Class 2', 'Eligible') for metric in error.columns[1:]]

    # replace (Class 1) -> (Others)
    mean.columns = ['Model'] + [metric.replace('Class 1', 'Others') for metric in mean.columns[1:]]
    error.columns = ['Model'] + [metric.replace('Class 1', 'Others') for metric in error.columns[1:]]
    
    return mean, error

def remove_unwanted_models(data):
    return data[~data['Model'].isin(['LLaMA-3-8B', 'OpenChat-7B', 'GPT-3.5', 'GPT-4', 'Claude 3 Haiku', 'Claude 3 Sonnet'])]

if __name__ == '__main__':
    # Load the CSV files

    mean_sigir, error_sigir = load_mean_err('sigir')
    mean_trec2021, error_trec2021 = load_mean_err('trec2021')
    
    sigir_data = remove_unwanted_models(mean_sigir)
    trec2021_data = remove_unwanted_models(mean_trec2021)

    error_sigir = remove_unwanted_models(error_sigir)
    error_trec2021 = remove_unwanted_models(error_trec2021)
    
    metrics = sigir_data.columns[1:]
    # metrics = ['BACC', 'F1', 'KAPPA']

    for metric in metrics:
        nested_data = [
            get_nested_data_single(sigir_data),
            get_nested_data_single(trec2021_data),
        ]
    
        error_data = [
            get_nested_data_err(error_sigir),
            get_nested_data_err(error_trec2021),
        ]

        bar_plot(nested_data, None, metric, nested_errs=error_data)