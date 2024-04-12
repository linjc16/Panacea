import json
import argparse
import re
import os
import pdb
import glob
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='criteria')
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results/criteria/eval_entail_v1')
    args = parser.parse_args()

    filepaths = glob.glob(os.path.join(args.res_dir, f'*.json'))

    final_results = defaultdict(dict)
    for filepath in tqdm(filepaths):
        model_name = os.path.basename(filepath).split('_')[0]
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        back_steps = range(-4, 0)
        for back_step in back_steps:
            evals = []
            for key, value in results.items():
                eval_res_list = value['eval_results']
                # extract the number after "Match prediction: " from the list
                match_pred = []
                for eval_res in eval_res_list[back_step:]:
                    try:
                        pred = re.findall(r'Match prediction: (\d)', eval_res)
                        match_pred.append(int(pred[0]))
                    except:
                        match_pred.append(0)

                evals.extend(match_pred)
        
            # calculate the accuracy
            accuracy = sum(evals) / len(evals)

            final_results[back_step][model_name] = accuracy
    
    # plot the results, x axis is the back steps, y axis is the accuracy for each model

    back_steps = list(final_results.keys())
    models = list(final_results[-1].keys())

    plt.figure(figsize=(10, 6))

    for model in models:
        accuracies = [final_results[step][model] for step in back_steps]
        plt.plot(back_steps, accuracies, label=model, marker='o')

    plt.xlabel('Back Steps')
    plt.ylabel('Clinical Accuracy')
    plt.title(f'Clinical Accuracy Over Back Steps ({args.task})')
    plt.legend()
    plt.grid(True)
    plt.xticks(back_steps)
    plt.savefig(f'{args.task}_entail.png', dpi=300)