from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pandas as pd
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
import pdb

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
    

def load_dataset(file_dir, split='test'):
    # load data/downstream/summazization/multi-trial/{split}.json
    with open(os.path.join(file_dir, f'{split}.json'), 'r') as f:
        data = json.load(f)
    
    output_data = defaultdict(list)
    for key, value in tqdm(data.items()):
        output_data['id'].append(key)
        # merge title and abstract list within the same paper (index), then add prefix "Study #x:"
        study_text = ""
        for i in range(len(value['title'])):
            study_text += f"Study #{i+1}: {value['title'][i]}. {value['abstract'][i]}.\n\n"
        # remove the last \n\n
        study_text = study_text[:-2]
        output_data['study_text'].append(study_text)
        output_data['background'].append(value["background"])
        output_data['target'].append(value["target"])
    
    df = pd.DataFrame(output_data)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default='/data/linjc/hub')
    parser.add_argument('--lora_dir', type=str, default='/data/linjc/trialfm')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--file_dir', type=str, default='data/downstream/summazization/multi-trial')
    parser.add_argument('--save_dir', type=str, default='data/downstream/summazization/multi-trial/results')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path
    cache_dir = args.cache_dir

    df = load_dataset(args.file_dir, args.split)

    tokenizer, model = load_model(model_path, cache_dir)

    instruction_prompt = "Your task is to synthesize the key findings from a collection of study abstracts related to a specific clinical trial related research question. In some cases, you will also be provided with a review background detailing the research question of the given studies."
    instruction_prompt += "\nCombine the insights from the provided abstracts into a cohesive summary. Your summary should integrate the findings rather than listing them separately. It's crucial to maintain the scientific integrity of the original studies while ensuring the summary is accessible and informative."
    instruction_prompt += "\nThe output should only be the summary. Do not explain how you summarize it."
    instruction_prompt += "\n\nReview Background: {Background}"
    instruction_prompt += "\n\nStudy Abstracts: {Text}"
    instruction_prompt += "\n\nSummary:"
    
    
    if not os.path.exists(os.path.join(args.save_dir, f'{args.model_name}.csv')):
        with open(os.path.join(args.save_dir, f'{args.model_name}.csv'), 'w') as f:
            f.write('id,summary\n')

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        id = row['id']
        input_text = row['study_text']
        bg_text = row['background']

        merged_input_text = instruction_prompt.format(Text=input_text, Background=bg_text)

        messages = [
            {"role": "user", "content": merged_input_text},
        ]
        
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        generated_ids = model.generate(encodeds, max_new_tokens=512, do_sample=False)
        summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            prediction = summary[summary.find('Summary:') + len('Summary:'):]
        except:
            prediction = ""
        prediction = prediction.strip()
        
        results = pd.DataFrame(columns=['id', 'summary'])
        
        # add id and prediction to a row
        results.loc[0] = [id, prediction]
        results.to_csv(os.path.join(args.save_dir, f'{args.model_name}.csv'), mode='a', header=False, index=False)
    
    
    # batch_size = 1
    # transform the df into batches, each batch has ids and study_text
    # df = df.groupby(df.index // batch_size).agg({'id': lambda x: list(x), 'study_text': lambda x: list(x), 'background': lambda x: list(x)}) 
    
    
    # for i in tqdm(range(len(df))):
    #     row = df.iloc[i]
    #     ids = row['id']
    #     study_text = row['study_text']
    #     background = row['background']
        
    #     merged_input_text = []
    #     for input_text, bg_text in zip(study_text, background):
    #         merged_input_text.append(instruction_prompt.format(Text=input_text, Background=bg_text))

    #     # tokenize the input_text
    #     inputs = tokenizer(merged_input_text, padding=True, return_tensors="pt")
    #     inputs = inputs.to(model.device)

    #     # pdb.set_trace()

    #     with torch.no_grad():
    #         generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    #     summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    #     predictions = []
    #     for summary in summaries:
    #         # for the generated text, remove the input_text, find its index, and get the text after it
    #         try:
    #             summary = summary[summary.find('Summary:') + len('Summary:'):]
    #         except:
    #             pdb.set_trace()
    #             summary = ""
            
    #         predictions.append(summary.strip())

    #     # save id and prediction to dataframe
    #     for j in range(len(ids)):
    #         results = pd.DataFrame(columns=['id', 'summary'])
    #         id = ids[j]
    #         prediction = predictions[j]
            
    #         # add id and prediction to a row
    #         results.loc[j] = [id, prediction]
    #         results.to_csv(os.path.join(args.save_dir, f'{args.model_name}.csv'), mode='a', header=False, index=False)
    
