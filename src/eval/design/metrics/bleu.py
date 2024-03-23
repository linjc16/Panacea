import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm.contrib.concurrent import process_map  # Import process_map for multiprocessing
import multiprocessing
import argparse
import json
import os
import pdb


def calculate_individual_bleu_score(args):
    predicted_caption, ground_truth_captions = args
    pred_tokens = predicted_caption.split()
    truth_tokens = [ground_truth_captions.split()]
    smooth = SmoothingFunction().method4  # Method 4 smoothing
    # Note: ground_truth_captions should be a list of lists (each inner list is one reference caption)
    bleu_score = sentence_bleu(truth_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return bleu_score


# Function to calculate the average BLEU score across all predicted and ground truth caption pairs
def calculate_bleu_scores(predicted_caption_list, ground_truth_caption_list):
    # Use process_map for parallel processing
    scores = process_map(calculate_individual_bleu_score, zip(predicted_caption_list, ground_truth_caption_list), max_workers=multiprocessing.cpu_count())
    
    average_bleu_score = sum(scores) / len(scores)
    return average_bleu_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results/study_arms')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()
    
    with open(os.path.join(args.res_dir, f'{args.model_name}.json'), 'r') as f:
        results = json.load(f)

    preds = []
    groundtruth = []
    for key, value in results.items():
        preds.extend(value['model_response'])
        groundtruth.extend(value['groundtruth'])

    assert len(preds) == len(groundtruth)
    
    # Calculate the BLEU score
    bleu_score = calculate_bleu_scores(preds, groundtruth)
    print("Average BLEU score:", bleu_score)
