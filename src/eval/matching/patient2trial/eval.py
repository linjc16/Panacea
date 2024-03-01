from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
sys.path.append('./')
from src.eval.matching.patient2trial.utils import format_input
import pdb
import json

tqdm.pandas()


def load_model(model_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        )
    
    model.eval()

    return tokenizer, model
    

def read_trec_qrels(trec_qrel_file):
    '''
    Read a TREC style qrel file and return a dict:
        QueryId -> docName -> relevance
    '''
    qrels = []
    with open(trec_qrel_file) as fh:
        for line in fh:
            try:
                query, zero, doc, relevance = line.strip().split()
                qrels.append((query, doc, int(relevance)))
            except Exception as e:
                print ("Error: unable to split line in 4 parts", line)
                raise e
    return qrels


def load_dataset(args):
    df_criteria = pd.read_csv('data/downstream/matching/patient2trial/cohort/criteria.csv')

    if args.dataset == 'cohort':
        df_notes = pd.read_csv('data/downstream/matching/patient2trial/cohort/patient_notes.csv')
        qrels = read_trec_qrels('data/downstream/matching/patient2trial/cohort/qrels-clinical_trials.txt')
    
    return df_notes, df_criteria, qrels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default='/data/linjc/hub')
    parser.add_argument('--lora_dir', type=str, default='/data/linjc/trialfm')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--save_dir', type=str, default='/data/linjc/trialfm/downstream/summarization/results')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='cohort')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path
    cache_dir = args.cache_dir

    tokenizer, model = load_model(model_path, cache_dir)

    df_notes, df_criteria, qrels = load_dataset(args)

    if os.path.exists(f'data/downstream/matching/patient2trial/{args.dataset}/input.json'):
        with open(f'data/downstream/matching/patient2trial/{args.dataset}/input.json', 'r') as f:
            inputs = json.load(f)
    else:
        inputs = format_input(df_notes, df_criteria, qrels)

    i = 0
    output_dict = {}
    for key, value in tqdm(inputs.items()):
        prompt = value['input']
        messages = [
                {"role": "user", "content": prompt},
            ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(model.device)

        
        generated_ids = model.generate(model_inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(generated_ids)[0]
        
        # split by [/INST]
        if args.model_name.startswith('openchat'):
            decoded = decoded.split('<|end_of_turn|>')[1].strip()
        else:
            decoded = decoded.split('[/INST]')[-1].strip()
        
        output_dict[key] = {
            'output': decoded,
            'label': value['label']
        }

        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(output_dict, f, indent=4)
    
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)