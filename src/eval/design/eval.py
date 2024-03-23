import argparse
import os
import json
import pdb

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def load_model(model_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1000000000000000019884624838656
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        )
    
    model.eval()

    return tokenizer, model

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    
    return data_dict

role_dict = {
    'openchat-7b': ['GPT4 Correct User', 'GPT4 Correct Assistant'],
    'mistral-7b': ['[INST]', '[/INST]'],
    'llama2-7b': ['[INST]', '[/INST]'],
    'llama2-13b': ['[INST]', '[/INST]'],
    'llama2-70b': ['[INST]', '[/INST]'],
}

def format_dialogue(content, model_name):
    """
    Format the conversation content into dialogue pairs of User and Chatbot without relying on explicit line breaks.
    """

    user_role, assis_role = role_dict[model_name]
    if model_name.startswith('mistral') or model_name.startswith('llama2'):
        content = content.replace(user_role, f'{user_role}:').replace(assis_role, f'{assis_role}:')

    dialogue_pairs = []
    current_pair = {}
    prev_role = None

    # Split the content by role identifiers, keeping the delimiter
    parts = content.split(f'{user_role}:')
    for part in parts[1:]:
        sub_parts = part.split(f'{assis_role}:')
        user_text = sub_parts[0].strip()
        if user_text:
            if prev_role == user_role:
                dialogue_pairs.append(current_pair)
                current_pair = {}
            current_pair[user_role] = user_text
            prev_role = user_role

        for sub_part in sub_parts[1:]:
            assis_text, next_user_text = sub_part.rsplit(f'{user_role}:', 1) if f'{user_role}:' in sub_part else (sub_part, "")
            assis_text = assis_text.strip()
            if assis_text:
                current_pair[assis_role] = assis_text
                dialogue_pairs.append(current_pair)
                current_pair = {}
                prev_role = assis_role

            if next_user_text:
                current_pair[user_role] = next_user_text.strip()
                prev_role = user_role

    # Append the last pair if it exists
    if current_pair.get(user_role) or current_pair.get(assis_role):
        dialogue_pairs.append(current_pair)

    return dialogue_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default='/data/linjc/hub')
    parser.add_argument('--lora_dir', type=str, default='/data/linjc/trialfm')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--task', type=str, default='study_arms')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    filepath = f'data/downstream/design/parsed/{args.task}/{args.split}.json'
    
    model_path = args.model_path
    cache_dir = args.cache_dir

    data_dict = load_dataset(filepath)

    tokenizer, model = load_model(model_path, cache_dir)
    

    output_reults = {}
    count = 0
    for key, value in tqdm(data_dict.items()):
        groudtruth = []
        out_response = []

        if len(value) % 2 != 0:
            value = value[:-1]

        for i in range(3, len(value) // 2):
            try:
                # [0], [0, 1, 2], [0, 1, 2, 3, 4]
                input = value[:i * 2 + 1]
                
                if args.model_name == 'openchat-7b':
                    encodeds = tokenizer.apply_chat_template(input, return_tensors="pt", add_generation_prompt=True).to(model.device)
                else:
                    encodeds = tokenizer.apply_chat_template(input, return_tensors="pt").to(model.device)
                
                generated_ids = model.generate(encodeds, max_new_tokens=512, do_sample=False)
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # pdb.set_trace()
                
                dialogue_pairs = format_dialogue(response, args.model_name)
                try:
                    out_response.append(dialogue_pairs[i][role_dict[args.model_name][1]])
                except:
                    out_response.append(dialogue_pairs[-1][role_dict[args.model_name][1]])
            except:
                out_response.append('')
            
            groudtruth.append(value[i * 2 + 1]['content'])

        
        output_reults[key] = {
            'model_response': out_response,
            'groundtruth': groudtruth,
        }

        if count % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(output_reults, f, indent=4)
        
        count += 1

    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(output_reults, f, indent=4)
