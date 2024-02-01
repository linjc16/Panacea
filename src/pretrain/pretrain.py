import argparse
import os
import transformers
import logging
import glob
import pickle
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field

import torch
import random
from accelerate import Accelerator
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import copy
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

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


class PretrainingDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(PretrainingDataset, self).__init__()

        print("Loading tokenized inputs... ")
        filepaths = glob.glob('/data/linjc/trialfm/tokenized_data/pretrain/*.pkl')
        
        # Load the tokenized inputs
        data_dict = {
            'input_ids': [],
            'labels': [],
            'input_ids_lens': [],
            'labels_lens': []
        }

        for filepath in tqdm(filepaths, desc="Loading PKL files"):
            with open(filepath, 'rb') as f:
                data_chunk = pickle.load(f)
                for key in data_dict.keys():
                    data_dict[key].extend(data_chunk[key])
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForDataset(object):

    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument("--cache_dir", type=str, default='/data/linjc/hub/')
    parser.add_argument("--output_dir", type=str, default='/data/linjc/goldminer/output/')
    parser.add_argument('--dataset_path', type=str, default='data/temp/sft_dataset.csv')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--report_to", type=str, default='tensorboard')
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--bf16_full_eval", type=bool, default=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    # Step 1: Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir,
        use_auth_token=True,
        add_eos_token=True,
        model_max_length=args.max_length,
        use_fast=False
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir,
        device_map=None,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # Step 2: Load the dataset
    train_dataset = PretrainingDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForDataset(tokenizer=tokenizer)    

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        report_to=args.report_to,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        bf16_full_eval=args.bf16_full_eval,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay
    )

    # Step 4: Define the trainer
    # data_collator = DataCollatorForDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == '__main__':
    main()