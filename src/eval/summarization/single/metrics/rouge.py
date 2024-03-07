from rouge_score import rouge_scorer
from tqdm.contrib.concurrent import process_map
import multiprocessing
from tqdm import tqdm
import pandas as pd
import argparse
import pdb
import os

def calculate_individual_rouge_scores(args):
    predicted_caption, ground_truth_caption = args
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    predicted_caption = str(predicted_caption)
    ground_truth_caption = str(ground_truth_caption)

    scores = scorer.score(ground_truth_caption, predicted_caption)
    
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }

def calculate_rouge_scores(predicted_caption_list, ground_truth_caption_list):
    results = process_map(calculate_individual_rouge_scores, zip(predicted_caption_list, ground_truth_caption_list), max_workers=multiprocessing.cpu_count())

    total_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    
    for score in results:
        for key in total_scores:
            total_scores[key] += score[key]
    
    avg_scores = {key: value / len(results) for key, value in total_scores.items()}

    return avg_scores, results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/summazization/single-trial/results')
    parser.add_argument('--model_name', type=str, default='llama2')
    args = parser.parse_args()
    
    preds = pd.read_csv(os.path.join(args.res_dir, f'{args.model_name}.csv'))

    groundtruth = pd.read_csv("data/downstream/summazization/single-trial/test.csv")
    
    preds['summary'] = preds['summary'].apply(lambda x: str(x))
    preds['summary'] = preds['summary'].apply(lambda x: x.strip())

    groundtruth.rename(columns={'summary_text': 'summary'}, inplace=True)

    assert len(preds) == len(groundtruth)
    
    scores, _ = calculate_rouge_scores(preds['summary'].tolist(), groundtruth['summary'].tolist())

    print(f'Model: {args.model_name}, ROUGE-? F1 Score: {scores}')
    