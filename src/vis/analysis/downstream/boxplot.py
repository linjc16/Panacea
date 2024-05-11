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
FIG_HEIGHT = 5
FIG_WIDTH = 5


filename_to_model_name = {
    'biomistral-7b': 'BioMistral-7B',
    # 'claude-haiku': 'Claude 3 Haiku',
    # 'claude-sonnet': 'Claude 3 Sonnet',
    # 'gpt-4': 'GPT 4',
    # 'gpt-35': 'GPT 3.5',
    'llama2-7b': 'LLaMA-2-7B',
    'llama3-8b': 'LLaMA-3-8B',
    'medalpaca-7b': 'MedAlpaca-7B',
    'meditron-7b': 'Meditron-7B',
    'mistral-7b': 'Mistral-7B',
    'openchat-7b': 'OpenChat-7B',
    'panacea-7b': 'Panacea',
    'panacea-base': 'Panacea-Base',
    'zephyr-7b': 'Zephyr-7B',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='criteria')
    parser.add_argument('--metric', type=str, default='BLEU')
    args = parser.parse_args()
    
    files = glob.glob(f'src/vis/analysis/downstream/design/results/{args.task}/{args.metric}/*.json')

    results_dict = {}
    for file in files:
        model_name = file.split('/')[-1].split('.')[0]
        
        if args.metric == 'BLEU':
            with open(file, 'r') as f:
                results = json.load(f)
        elif args.metric == 'ROUGE':
            with open(file, 'r') as f:
                data = json.load(f)
            results = [item['rougeL'] for item in data]
        
        results_dict[model_name] = results
    
    groups = []
    scores = []
    order = ['Panacea', 'Panacea-Base', 'LLaMA-2-7B', 'LLaMA-3-8B', 'BioMistral-7B', 'Mistral-7B', 'MedAlpaca-7B', 'Meditron-7B', 'Zephyr-7B', 'OpenChat-7B']

    for key, value in results_dict.items():
        if key not in filename_to_model_name:
            continue
        groups.extend([filename_to_model_name[key]] * len(value))
        scores.extend(value)

    df_criteria = pd.DataFrame({
        'Model': groups,
        'Score': scores
    })

    df_study_arms = df_criteria.copy()
    df_outcome_measures = df_criteria.copy()

    df_criteria['Group'] = 'Criteria'
    df_study_arms['Group'] = 'Study Arms'
    df_outcome_measures['Group'] = 'Outcome Measures'

    df_all = pd.concat([df_criteria, df_study_arms, df_outcome_measures])

    plt.figure(figsize=(18, 6))
    for i, group in enumerate(['Criteria', 'Study Arms', 'Outcome Measures']):
        group_data = df_all[df_all['Group'] == group]
        sns.boxplot(x='Model', y='Score', data=group_data, order=order, ax=plt.subplot(1, 3, i + 1))
        plt.title(group)
        plt.xticks(rotation=45)
        if i > 0:
            plt.ylabel('')  # 只在第一个subplot显示y轴标签

    plt.tight_layout()
    plt.savefig(f'{args.metric}.png')  # 保存图片


    # plt.clf()
    # palette1 = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0"] * 3
    # # sns.set_palette(palette=palette1)
    # fig, ax = plt.subplots(figsize=(1*FIG_WIDTH, FIG_HEIGHT))
    # sns.boxplot(x=groups, y=scores, order=order, width=0.3, showfliers=False, palette=palette1)
    
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # ax.set_ylabel(f'{args.metric}')
    
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    
    # plt.tight_layout()
    # plt.rcParams['savefig.dpi'] = 800
    # # plt.savefig(f'{args.metric}.pdf')
    # plt.savefig(f'{args.metric}.png')