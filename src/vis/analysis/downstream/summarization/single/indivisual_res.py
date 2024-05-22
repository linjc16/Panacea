import json
import glob
import argparse
import os
import pdb
import pandas as pd

import sys
sys.path.append('./')
from src.eval.summarization.single.metrics.rouge import calculate_rouge_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='ROUGE-L')
    args = parser.parse_args()
    
    args.save_dir = 'src/vis/analysis/downstream/summarization/single'
    args.save_dir = os.path.join(args.save_dir, 'results', args.metric)
    os.makedirs(args.save_dir, exist_ok=True)
    

    file_dir = f'data/downstream/summazization/single-trial/results'
    files = glob.glob(f'{file_dir}/*.csv')

    
    for file in files:
        model_name = file.split('/')[-1].split('.')[0]


        
        preds = pd.read_csv(file)

        groundtruth = pd.read_csv("data/downstream/summazization/single-trial/test.csv")
        
        preds['summary'] = preds['summary'].apply(lambda x: str(x))
        preds['summary'] = preds['summary'].apply(lambda x: x.strip())
        
        groundtruth.rename(columns={'summary_text': 'summary'}, inplace=True)
        
        assert len(preds) == len(groundtruth)
        
        _, scores = calculate_rouge_scores(preds['summary'].tolist(), groundtruth['summary'].tolist())

        scores_rougeL = [score['rougeL'] for score in scores]
        
        # save results
        with open(os.path.join(args.save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(scores_rougeL, f)
