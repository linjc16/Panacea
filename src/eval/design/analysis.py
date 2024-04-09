import argparse
import json
import os
import sys
sys.path.append('./')
from src.eval.design.metrics.bleu import calculate_bleu_scores
from src.eval.design.metrics.rouge import calculate_rouge_scores

import glob
from tqdm import tqdm
import pdb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='study_arms')
    
    args = parser.parse_args()

    res_dir = f'data/downstream/design/results/{args.task}'

    filepaths = glob.glob(f'{res_dir}/*.json')

    

    for turn_id in [-2, -3, -4, -5, -6, -7, -8, -9, -10]:
        
        res_dict = {}
        for filepath in filepaths:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            model_name = os.path.basename(filepath).split('.')[0]
            
            preds = []
            groundtruth = []
            for key, value in results.items():
                preds.extend(value['model_response'][turn_id:-1])
                groundtruth.extend(value['groundtruth'][turn_id:-1])

            assert len(preds) == len(groundtruth)

            # Calculate the BLEU score
            bleu_score = calculate_bleu_scores(preds, groundtruth)
            # Calculate the ROUGE score
            rouge_score, _ = calculate_rouge_scores(preds, groundtruth)

            res_dict[model_name] = {
                'bleu': bleu_score,
                'rouge': rouge_score
            }
        print(f"Turn {turn_id}")

        for key, value in res_dict.items():
            print(f'Model: {key}, BLEU: {value["bleu"]}, ROUGE: {value["rouge"]}')