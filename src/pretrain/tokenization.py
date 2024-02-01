import argparse
import os
import transformers
import logging
from typing import Dict, Optional, Sequence
import multiprocessing
from dataclasses import dataclass, field
import pickle

import torch
import random
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import copy
import json
tqdm.pandas()

import sys
sys.path.append('./')

from src.utils.load_ctgov import load_ctgov
from src.utils.load_paper_data import load_embase, load_pubmed
from src.utils.load_trial_data import load_aus_zealand, load_brazil, load_chictr, load_dutch, \
    load_euctr, load_german, load_iran, load_isrctn, load_japan, load_korea, \
    load_pan_african, load_sri_lanka, load_thai
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM

import pdb

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

random.seed(42)

trial_filename_sets = {
    'ctgov': '/data/linjc/trialfm/ctgov_fixed_md/merged',
    'aus_zealand': '/data/linjc/ctr_crawl/0_final_data/trials',
    'brazil': '/data/linjc/ctr_crawl/0_final_data/trials',
    'chictr': '/data/linjc/ctr_crawl/0_final_data/trials',
    'dutch': '/data/linjc/ctr_crawl/0_final_data/trials',
    'euctr': '/data/linjc/ctr_crawl/0_final_data/trials',
    'german': '/data/linjc/ctr_crawl/0_final_data/trials',
    'iran': '/data/linjc/ctr_crawl/0_final_data/trials',
    'isrctn': '/data/linjc/ctr_crawl/0_final_data/trials',
    'japan': '/data/linjc/ctr_crawl/0_final_data/trials',
    'korea': '/data/linjc/ctr_crawl/0_final_data/trials',
    'pan_african': '/data/linjc/ctr_crawl/0_final_data/trials',
    'sri_lanka': '/data/linjc/ctr_crawl/0_final_data/trials',
    'thai': '/data/linjc/ctr_crawl/0_final_data/trials'
}

paper_filename_sets = {
    'embase': '/data/linjc/ctr_crawl/0_final_data/papers/embase',
    'pubmed': '/data/linjc/trialfm/final_data/papers/pubmed'
}

def load_dataset():
    data_list = []
    # Load the trial data
    print('Loading the trial data...')
    data_list += load_ctgov(trial_filename_sets['ctgov'], 'train')
    data_list += load_aus_zealand(trial_filename_sets['aus_zealand'])
    data_list += load_brazil(trial_filename_sets['brazil'])
    data_list += load_chictr(trial_filename_sets['chictr'])
    data_list += load_dutch(trial_filename_sets['dutch'])
    data_list += load_euctr(trial_filename_sets['euctr'])
    data_list += load_german(trial_filename_sets['german'])
    data_list += load_iran(trial_filename_sets['iran'])
    data_list += load_isrctn(trial_filename_sets['isrctn'])
    data_list += load_japan(trial_filename_sets['japan'])
    data_list += load_korea(trial_filename_sets['korea'])
    data_list += load_pan_african(trial_filename_sets['pan_african'])
    data_list += load_sri_lanka(trial_filename_sets['sri_lanka'])
    data_list += load_thai(trial_filename_sets['thai'])
    
    # Load the paper data
    print('Loading the paper data...')
    data_list += load_embase(paper_filename_sets['embase'])
    data_list += load_pubmed(paper_filename_sets['pubmed'])

    # shuffle the data
    random.shuffle(data_list)

    return data_list

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    labels = copy.deepcopy(input_ids)
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    labels_lens = copy.deepcopy(input_ids_lens)

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_chunk(data_chunk, tokenizer, args, chunk_index):
    """
    Tokenize a chunk of data and save it to a separate file.
    """
    # Tokenize the data chunk
    tokenized_data = [_tokenize_fn([text], tokenizer) for text in tqdm(data_chunk)]
    # Flatten the list of dicts into a single dict
    tokenized_data_flattened = {
        key: sum((example[key] for example in tokenized_data), [])
        for key in tokenized_data[0].keys()
    }
    # pdb.set_trace()
    # Save the tokenized data chunk to a file
    output_file = os.path.join(args.output_dir, f"tokenized_data_chunk_{chunk_index}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(tokenized_data_flattened, f)

def worker_process(data_chunk, tokenizer, args, chunk_index):
    """
    Worker process to handle tokenization and saving of a data chunk.
    """
    preprocess_chunk(data_chunk, tokenizer, args, chunk_index)


def add_special_tokens(tokenizer: transformers.PreTrainedTokenizer):
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    tokenizer.add_special_tokens(special_tokens_dict)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument("--cache_dir", type=str, default='/data/linjc/hub/')
    parser.add_argument("--output_dir", type=str, default='/data/linjc/trialfm/tokenized_data/pretrain/')
    parser.add_argument("--max_length", type=int, default=4096)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Load the data
    data_list = load_dataset()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir,
        use_auth_token=True,
        add_eos_token=True,
        model_max_length=args.max_length,
        use_fast=False
    )
    add_special_tokens(tokenizer)

    # Setup multiprocessing
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Determine the size of each chunk based on the total data size and number of processes
    chunk_size = len(data_list) // num_processes + (len(data_list) % num_processes > 0)

    # Create tasks for multiprocessing
    tasks = [(data_list[i:i + chunk_size], tokenizer, args, idx) for idx, i in enumerate(range(0, len(data_list), chunk_size))]

    # worker_process(tasks[0][0], tasks[0][1], tasks[0][2], tasks[0][3])
    # Process data in parallel and save in chunks
    for task in tasks:
        pool.apply_async(worker_process, task)

    pool.close()
    pool.join()

    # pdb.set_trace()


if __name__ == '__main__':
    run()