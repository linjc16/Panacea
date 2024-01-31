from transformers import AutoTokenizer
import argparse
import os
import sys
sys.path.append('./')

from src.utils.load_trial_data import load_aus_zealand, load_brazil, load_chictr, load_dutch, \
    load_euctr, load_german, load_iran, load_isrctn, load_japan, load_korea, \
    load_pan_african, load_sri_lanka, load_thai

from src.utils.load_ctgov import load_ctgov
from src.utils.load_paper_data import load_embase, load_pubmed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def count_tokens_with_tokenizer(text_list, model_name, cache_dir):
    """
    Count the number of tokens in each text in the list using a specific tokenizer.

    :param text_list: List of strings.
    :param model_name: Name of the model whose tokenizer will be used.
    :return: List of integers representing the number of tokens in each text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    token_counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in tqdm(text_list)]

    return token_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aus_zealand', help='Dataset name.')
    parser.add_argument('--split', type=str, default='train', help='Split name.')
    args = parser.parse_args()
    
    if args.dataset == 'aus_zealand':
        texts = load_aus_zealand('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'brazil':
        texts = load_brazil('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'chictr':
        texts = load_chictr('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'dutch':
        texts = load_dutch('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'euctr':
        texts = load_euctr('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'german':
        texts = load_german('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'iran':
        texts = load_iran('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'isrctn':
        texts = load_isrctn('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'japan':
        texts = load_japan('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'korea':
        texts = load_korea('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'pan_african':
        texts = load_pan_african('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'sri_lanka':
        texts = load_sri_lanka('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'thai':
        texts = load_thai('/data/linjc/ctr_crawl/0_final_data/trials')
    elif args.dataset == 'embase':
        texts = load_embase('/data/linjc/ctr_crawl/0_final_data/papers/embase')
    elif args.dataset == 'pubmed':
        texts = load_pubmed('/data/linjc/trialfm/final_data/papers/pubmed')
    elif args.dataset == 'ctgov':
        texts = load_ctgov('/data/linjc/trialfm/ctgov/merged', args.split)
    
    model_name = 'meta-llama/Llama-2-7b-hf'
    token_counts = count_tokens_with_tokenizer(texts, model_name, cache_dir='/data/linjc/huggingface/hub')
    
    print('Token counts: ', sum(token_counts))

    # plot histogram
    plt.hist(token_counts, bins=np.arange(0, 30000, 1000))
    plt.title('Token counts')
    plt.xlabel('Number of tokens')
    plt.ylabel('Number of texts')
    os.makedirs('temp', exist_ok=True)
    plt.savefig(f'temp/token_counts_{args.dataset}.png')
