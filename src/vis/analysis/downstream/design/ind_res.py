import json
import glob
import argparse
import os
import pdb

import sys
sys.path.append('./')
from src.eval.design.metrics.rouge import calculate_rouge_scores
from src.eval.design.metrics.bleu import calculate_bleu_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='criteria')
    parser.add_argument('--metric', type=str, default='BLEU')
    args = parser.parse_args()
    
    args.save_dir = 'src/vis/analysis/downstream/design'
    args.save_dir = os.path.join(args.save_dir, 'results', args.task, args.metric)
    os.makedirs(args.save_dir, exist_ok=True)
    

    file_dir = f'data/downstream/design/results/{args.task}'
    files = glob.glob(f'{file_dir}/*.json')

    with open(f'data/downstream/design/parsed/{args.task}/del_end_sent.json', 'r') as f:
        del_end_sent = json.load(f)
    
    # merge del_end_sent to a dict
    del_end_sent = {key: value for item in del_end_sent for key, value in item.items()}

    eval_results = {}

    for file in files:
        model_name = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            results = json.load(f)

        preds = []
        groundtruth = []
        for key, value in results.items():
            model_output = value['model_response']
            gt = value['groundtruth']
            if del_end_sent[key]:
                model_output = model_output[:-1]
                gt = gt[:-1]
            
            preds.extend(model_output)
            groundtruth.extend(gt)

        assert len(preds) == len(groundtruth)

        if args.metric == 'BLEU':
            _, results = calculate_bleu_scores(preds, groundtruth)
        elif args.metric == 'ROUGE':
            _, results = calculate_rouge_scores(preds, groundtruth)

        # save results
        with open(os.path.join(args.save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(results, f, indent=4)
