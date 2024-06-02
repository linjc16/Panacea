import json
import glob
import argparse
import os
import pdb
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append('./')
from src.eval.summarization.multi.metrics.rouge import calculate_rouge_scores
from src.eval.design.metrics.bleu import calculate_bleu_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='ROUGE-L')
    args = parser.parse_args()
    
    args.save_dir = 'src/vis/analysis/downstream/summarization/multi'
    args.save_dir = os.path.join(args.save_dir, 'results', args.metric)
    os.makedirs(args.save_dir, exist_ok=True)
    

    file_dir = f'data/downstream/summazization/multi-trial/results'
    files = glob.glob(f'{file_dir}/*.csv')

    
    
    for file in files:
        model_name = file.split('/')[-1].split('.')[0]

        preds = pd.read_csv(file)

        with open('data/downstream/summazization/multi-trial/test.json', 'r') as f:
            data = json.load(f)
        
        # for each key, extract value['target'], merge into a dataframe
        groundtruth = {'id': [], 'summary': []}
        for key, value in tqdm(data.items()):
            groundtruth['id'].append(key)
            groundtruth['summary'].append(value['target'])
        groundtruth = pd.DataFrame(groundtruth)
        
        
        preds['summary'] = preds['summary'].apply(lambda x: str(x))
        preds['summary'] = preds['summary'].apply(lambda x: x.strip())


        assert len(preds) == len(groundtruth)
        
        if args.metric == 'ROUGE-L':
            _, scores = calculate_rouge_scores(preds['summary'].tolist(), groundtruth['summary'].tolist())
            scores_list = [score['rougeL'] for score in scores]
        elif args.metric == 'BLEU':
            _, scores = calculate_bleu_scores(preds['summary'].tolist(), groundtruth['summary'].tolist())
            scores_list = scores

        
        
        # save results
        with open(os.path.join(args.save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(scores_list, f)
