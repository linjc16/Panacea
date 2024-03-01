from rouge_score import rouge_scorer
from tqdm.contrib.concurrent import process_map
import multiprocessing
from tqdm import tqdm
import pandas as pd
import argparse

def calculate_individual_rouge_score(args):
    predicted_caption, ground_truth_caption = args
    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)

    predicted_caption = str(predicted_caption)
    ground_truth_caption = str(ground_truth_caption)

    rouge_score = scorer.score(ground_truth_caption, predicted_caption)
    return rouge_score["rouge2"].fmeasure

def calculate_rouge_score(predicted_caption_list, ground_truth_caption_list):
    # Use process_map for multiprocessing with progress bar
    scores = process_map(calculate_individual_rouge_score, zip(predicted_caption_list, ground_truth_caption_list), max_workers=multiprocessing.cpu_count())

    rouge_score_avg = sum(scores) / len(scores)

    return rouge_score_avg, scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/summazization/results')
    parser.add_argument('--model_name', type=str, default='llama2')
    args = parser.parse_args()
    
    llama2_res = pd.read_csv("llama2.csv")
    panacea_res = pd.read_csv("panacea.csv")

    groundtruth = pd.read_csv("/data/linjc/trialfm/downstream/summarization/test.csv")
    
    # add id column
    groundtruth['id'] = range(len(groundtruth))

    # load index
    indexes = pd.read_csv("rough_new_index.csv")
    
    # for llama2_res, panacea_res, groundtruth, select the index
    llama2_res = llama2_res[llama2_res['id'].isin(indexes['id'].tolist())]
    panacea_res = panacea_res[panacea_res['id'].isin(indexes['id'].tolist())]
    groundtruth = groundtruth[groundtruth['id'].isin(indexes['id'].tolist())]

    # post processing llama2
    # each data use strip()

    llama2_res['summary'] = llama2_res['summary'].apply(lambda x: x.strip())
    # transform to str
    llama2_res['summary'] = llama2_res['summary'].apply(lambda x: str(x))

    # for panacea, remove [/INST] and then strip()
    panacea_res['summary'] = panacea_res['summary'].apply(lambda x: x.replace("[/INST]", "").strip())
    # transform to str
    panacea_res['summary'] = panacea_res['summary'].apply(lambda x: str(x))

    groundtruth.rename(columns={'summary_text': 'summary'}, inplace=True)

    min_len = min(len(llama2_res), len(panacea_res), len(groundtruth))

    llama2_res = llama2_res[:min_len]
    panacea_res = panacea_res[:min_len]
    groundtruth = groundtruth[:min_len]

    llama2_res_scores, llama2_scores_list = calculate_rouge_score(llama2_res['summary'].tolist(), groundtruth['summary'].tolist())
    panacea_res_scores, panacea_scores_list = calculate_rouge_score(panacea_res['summary'].tolist(), groundtruth['summary'].tolist())

    print(llama2_res_scores)
    print(panacea_res_scores)

    # # save the score list to csv, column name is id, llama2, panacea
    # scores = pd.DataFrame({'id': range(len(llama2_scores_list)), 'llama2': llama2_scores_list, 'panacea': panacea_scores_list})
    # scores.to_csv("meteor_score_list.csv", index=False)
    