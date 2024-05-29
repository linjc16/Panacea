from tqdm.contrib.concurrent import process_map
import multiprocessing
from tqdm import tqdm
import pandas as pd
import argparse
import pdb
import os
import json
import re


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/summazization/multi-trial/results/gpt_eval')
    parser.add_argument('--model_name', type=str, default='llama2')
    args = parser.parse_args()

    # read json file
    with open (os.path.join(args.res_dir, f'{args.model_name}.json'), 'r') as f:
        preds = json.load(f)
    
    # select top 100 from preds dict
    # preds = {k: v for k, v in preds.items() if int(k) < 50}
    
    goal_scores = []
    recruiting_scores = []
    study_arm_scores = []
    conclusion_scores = []

    for key, value in preds.items():
        eval_res = value['eval']

        if 'Topic Alignment: 0' in eval_res:
            goal_scores.append(0)
        else:
            goal_scores.append(1)
        
        if 'Patient Recruiting Method: 0' in eval_res:
            recruiting_scores.append(0)
        else:
            recruiting_scores.append(1)
        
        if 'Study Arm Consistency: 0' in eval_res:
            study_arm_scores.append(0)
        else:
            study_arm_scores.append(1)
        
        if 'Conclusion Similarity: 0' in eval_res:
            conclusion_scores.append(0)
        else:
            conclusion_scores.append(1)
    
    goal_score = sum(goal_scores) / len(goal_scores)
    # recruiting_score = sum(recruiting_scores) / len(recruiting_scores)
    # study_arm_score = sum(study_arm_scores) / len(study_arm_scores)
    # conclusion_score = sum(conclusion_scores) / len(conclusion_scores)

    print(f'Model: {args.model_name}, Goal Alignment: {goal_score:.4f}')

    # print(f'Model: {args.model_name}, Goal Alignment: {goal_score:.4f}, Patient Recruiting Method: {recruiting_score:.4f}, Study Arm Consistency: {study_arm_score:.4f}, Conclusion Similarity: {conclusion_score:.4f}')

    # # calculate all scores
    # avg_scores = (goal_score + recruiting_score + study_arm_score) / 3

    # print(f'Model: {args.model_name}, Average Score: {avg_scores:.4f}')