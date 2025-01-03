import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append('./src/vis/src')

import plot_settings_bar
import plot_utils

plt.rc('font', family='Helvetica')

import matplotlib as mpl


# set basic parameters
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({"ytick.color" : "black",
                     "xtick.color" : "black",
                     "axes.labelcolor" : "black",
                     "axes.edgecolor" : "black"})

mpl.rcParams.update({
    "pdf.use14corefonts": True
})

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

name_mapping_dict = {
    'Jaccard (Diseases)': 'Disease',
    'Jaccard (Interventions)': 'Intervention',
    'Jaccard (Sponsor)': 'Sponsor',
    'Jaccard (Status)': 'Status',
    'Jaccard (Phase)': 'Phase',
    'Jaccard (Study Type)': 'Study Type',
    'Jaccard': '',
}

def bar_plot(nested_data, data_labels, name, nested_errs, y_lim=None):

    ax = plot_settings_bar.get_wider_axis(2.5, 4)

    # Data labels

    y_label = 'Jaccard index'
    # name only extract strings between brackets   
    label = name_mapping_dict.get(name, name)
    data_labels = [label]

    # Plotting the  data
    plot_utils.grouped_barplot(ax, nested_data, data_labels, None, 
                            name, model_colors, xscale='linear', yscale='linear', 
                            min_val=0, invert_axes=False, tickloc_top=False,  rotangle=0, anchorpoint='center', y_lim=y_lim, nested_errs=nested_errs)
    plot_utils.format_ax(ax)

    # plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', 
    #                         ncols=1)
    # plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))

    plt.xlabel('')
    plt.xticks(fontsize=25)
    if y_lim:
        plt.yticks(np.arange(0, y_lim[1], 0.2), fontsize=25)
    else:
        plt.yticks(fontsize=25)
    
    plt.ylabel(y_label, fontsize=25)

    # set the ysticks 0, 0.2, 0.4, 0.6, 0.8, ..., accoridng to the y_lim

    plt.tick_params(axis='x', length=0)
    
    # Show the plot
    plt.savefig(f'visulization/bar_query_gen_{name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'visulization/bar_query_gen_{name}.pdf', dpi=300, bbox_inches='tight')


def get_nested_data_single(data):
    nested_data = [
        ('Panacea', data[data['Model'] == 'Panacea'][metric].values[0]),
        # ('OpenChat-7B', data[data['Model'] == 'OpenChat-7B'][metric].values[0]),
        ('BioMistral-7B', data[data['Model'] == 'BioMistral-7B'][metric].values[0]),
        ('Mistral-7B', data[data['Model'] == 'Mistral-7B'][metric].values[0]),
        ('Zephyr-7B', data[data['Model'] == 'Zephyr-7B'][metric].values[0]),
        # ('LLaMA-3-8B', data[data['Model'] == 'LLaMA-3-8B'][metric].values[0]),
        ('LLaMA-2-7B', data[data['Model'] == 'LLaMA-2-7B'][metric].values[0]),
        # ('Panacea-Base', data[data['Model'] == 'Panacea-Base'][metric].values[0]),
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
        # data[data['Model'] == 'Panacea-Base'][metric].values[0],
        data[data['Model'] == 'MedAlpaca-7B'][metric].values[0],
        data[data['Model'] == 'Meditron-7B'][metric].values[0],
    ]

    return nested_errs

def load_mean_err(mode):
    frames = []
    for i in [0]:
        df = pd.read_csv(f'src/vis/results/{i}/query_generation.csv')
        frames.append(df)
    combined = pd.concat(frames)
    mean = combined.groupby('Model').mean()
    error = combined.groupby('Model').std() / np.sqrt(len(mean))

    # 'Model' is the first column name
    mean = mean.reset_index()
    error = error.reset_index()
    
    return mean, error

def remove_unwanted_models(data):
    return data[~data['Model'].isin(['LLaMA-3-8B', 'OpenChat-7B', 'GPT-3.5', 'GPT-4', 'Claude 3 Haiku', 'Claude 3 Sonnet', 'Panacea-Base'])]

if __name__ == '__main__':
    # Load the CSV files

    mean_single, error_single = load_mean_err('single')
    # mean_multi, error_multi = load_mean_err('multi')

    single_data = remove_unwanted_models(mean_single)
    # multi_data = remove_unwanted_models(mean_multi)

    error_single = remove_unwanted_models(error_single)
    # error_multi = remove_unwanted_models(error_multi)
    
    metrics = single_data.columns[1:]
    # metrics = ['BACC', 'F1', 'KAPPA']
    

    for data in [single_data]:
        for metric in metrics:
            nested_data = [
                get_nested_data_single(data),
            ]

            error_data = [
                get_nested_data_err(error_single),
            ]

            y_lim_dict = {
                'Jaccard (Interventions)': (0, 1.01),
                'Jaccard (Phase)': (0, 1.01),
                'Jaccard (Status)': (0, 1.01),
                'Jaccard (Study Type)': (0, 1.01),
                'Jaccard (Phase)': (0, 1.01),
            }

            bar_plot(nested_data, None, metric, nested_errs=error_data, y_lim=y_lim_dict.get(metric, None))