import pandas as pd
import pdb
import json
import numpy as np
import argparse
import glob

import matplotlib.pyplot as plt

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns


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

MEDIUM_SIZE = 8
SMALLER_SIZE = 6
BIGGER_SIZE = 25
plt.rc('font', family='Helvetica', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 3
FIG_WIDTH = 5

model_colors = {
    'LLaMA-2': '#e7ad61',
    'Mistral': '#daf0ed',
    'Zephyr': '#f8e6c3',
    'BioMistral': '#81ccc1',
    'MedAlpaca': '#8c5109',
    'Meditron': '#53300e',
    'Panacea': '#30988e',
    'Panacea-Base': '#bf802d'
}

filename_to_model_name = {
    'biomistral': 'BioMistral',
    # 'claude-haiku': 'Claude 3 Haiku',
    # 'claude-sonnet': 'Claude 3 Sonnet',
    # 'gpt-4': 'GPT 4',
    # 'gpt-35': 'GPT 3.5',
    'llama2-7b': 'LLaMA-2',
    # 'llama3-8b': 'LLaMA-3-8B',
    'medalpaca': 'MedAlpaca',
    'meditron-7b': 'Meditron',
    'mistral-7b': 'Mistral',
    # 'openchat-7b': 'OpenChat-7B',
    'panacea-7b': 'Panacea',
    'panacea-base': 'Panacea-Base',
    'zephyr-7b': 'Zephyr',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='Jaccard')
    args = parser.parse_args()
    
    files = glob.glob(f'src/vis/analysis/downstream/search/query_generation/results/{args.metric}/*.json')

    results_dict = {}
    for file in files:
        model_name = file.split('/')[-1].split('.')[0]
        
        with open(file, 'r') as f:
            results = json.load(f)
        
        results_dict[model_name] = results
    
    groups = []
    scores = []
    order = ['Panacea',  'BioMistral', 'Mistral', 'Zephyr', 'LLaMA-2', 'Panacea-Base', 'MedAlpaca', 'Meditron']

    for key, value in results_dict.items():
        if key not in filename_to_model_name:
            continue
        groups.extend([filename_to_model_name[key]] * len(value))
        scores.extend(value)

    plt.clf()
    sns.set_palette([model_colors[model] for model in order if model in model_colors])
    fig, ax = plt.subplots(figsize=(1*FIG_WIDTH, FIG_HEIGHT))
    sns.boxplot(x=groups, y=scores, hue=groups, order=order, width=0.4, showfliers=False, palette=model_colors, legend=False)
    
    ax.set_xlabel('')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    

    metric_name_dict = {
        'precision': 'Precision',
        'recall': 'Recall',
        'F1': '$F_1$',
        'Jaccard': 'Jaccard index',
    }
    # set y-label font size
    plt.ylabel(metric_name_dict.get(args.metric, args.metric), fontsize=15)
    
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 800
    plt.savefig(f'visulization/box_plot_query_gen_{args.metric}.png')
    plt.savefig(f'visulization/box_plot_query_gen_{args.metric}.pdf')