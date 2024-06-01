from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm
from jsonformer import Jsonformer
import json
import pdb

tqdm.pandas()

json_schema={
    "type": "object",
    "properties": {
        "Expanded Terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Generated terms that are expanded from the input terms.",
        },
    },
}

def load_model(model_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1000000000000000019884624838656

    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        )
    
    model.config.pad_token_id = tokenizer.eos_token_id
    # model.resize_token_embeddings(len(tokenizer))
    
    model.eval()

    return tokenizer, model


def load_dataset(filepath):

    with open(os.path.join(filepath), 'r') as f:
        data = json.load(f)
    
    output_data = {}
    i = 0
    for key, value in data.items():
        diseases = list(set(value['diseases']))
        query = ', '.join(diseases)
        output_data[key] = {'query': query}
        i += 1
    
    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/shared/jl254/data/linjc/trialfm/sft/panacea-chat-v2')
    parser.add_argument('--cache_dir', type=str, default='/data/linjc/hub')
    parser.add_argument('--model_name', type=str, default='llama2')
    
    parser.add_argument('--file_dir', type=str, default='data/downstream/summazization/multi-trial/results/query_gen')
    parser.add_argument('--save_dir', type=str, default='data/downstream/summazization/multi-trial/results/query_expan')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path
    cache_dir = args.cache_dir

    data = load_dataset(os.path.join(args.file_dir, f'{args.model_name}.json'))

    tokenizer, model = load_model(model_path, cache_dir)
    
    
    instruction_prompt = "Given Terms used for searching clinical trials in a database, expand the input terms and then generate a JSON object that contains the expanded terms. Don't include the original terms in the expanded terms."

    instruction_prompt += '\n`For example, the input Terms are: "Neurocognitive Disorders, Tauopathies, Movement Disorders, Dementia, Synucleinopathies", then Expanded Terms are: Central Nervous System Diseases, Basal Ganglia Diseases, Brain Diseases, Alzheimer Disease, Lewy Body Disease, Nervous System Diseases.`'
    
    instruction_prompt += '\n\nInput Terms: {query}. Now expand the input terms and generate the expanded terms.'
    instruction_prompt += ' Split each expanded term by ", ".'
    instruction_prompt += '\n\nExpanded Terms:'

    
    outputs = {}
    
    i = 0
    for key, value in tqdm(data.items()):
        merged_input_text = instruction_prompt.format(query=value['query'])

        messages = [
            {"role": "user", "content": merged_input_text},
        ]
        
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        generated_ids = model.generate(encodeds, max_new_tokens=512, do_sample=False)
        generated_data_ori = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            generated_data = generated_data_ori[generated_data_ori.find('<|assistant|>') + len('<|assistant|>'):].strip()
            generated_data = generated_data.split(', ')
        except:
            generated_data = ''
        
        outputs[key] = generated_data
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(outputs, f, indent=4)
        
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(outputs, f, indent=4)