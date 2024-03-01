from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm
import pdb

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
    

def load_dataset(file_dir, split='test'):
    df = pd.read_csv(os.path.join(file_dir, split + '.csv'))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default='/data/linjc/hub')
    parser.add_argument('--lora_dir', type=str, default='/data/linjc/trialfm')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--file_dir', type=str, default='data/downstream/summazization')
    parser.add_argument('--save_dir', type=str, default='/data/linjc/trialfm/downstream/summarization/results')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path
    cache_dir = args.cache_dir

    tokenizer, model = load_model(model_path, cache_dir)
    df = load_dataset(args.file_dir, args.split)

    instruction_prompt = "Your task is to create a clear, concise, and accurate summary of the provided clinical trial document. The summary should capture the key aspects of the trial."
    instruction_prompt += "\nThe output should only be the summarization of the given trial. Do not explain how you summarize it."
    instruction_prompt += "\nInput Text: {Text}"
    instruction_prompt += "\nSummary: "

    
    if not os.path.exists(os.path.join(args.save_dir, f'{args.model_name}.csv')):
        with open(os.path.join(args.save_dir, f'{args.model_name}.csv'), 'w') as f:
            f.write('id,summary\n')


    # for each data, add a column of id
    df['id'] = df.index

    batch_size = 1
    # transform the df into batches, each batch has ids and input_text
    df = df.groupby(df.index // batch_size).agg({'id': lambda x: list(x), 'input_text': lambda x: list(x)})

    
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        ids = row['id']
        input_texts = row['input_text']

        merged_input_text = []
        for input_text in input_texts:
            merged_input_text.append(instruction_prompt.format(Text=input_text))

        # tokenize the input_text
        inputs = tokenizer(merged_input_text, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        # pdb.set_trace()

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        predictions = []
        for summary in summaries:
            # for the generated text, remove the input_text, find its index, and get the text after it
            try:
                summary = summary[summary.find('Summary:') + len('Summary:'):]
            except:
                summary = ""
            
            predictions.append(summary.strip())

        # save id and prediction to dataframe
        for j in range(len(ids)):
            results = pd.DataFrame(columns=['id', 'summary'])
            id = ids[j]
            prediction = predictions[j]
            
            # add id and prediction to a row
            results.loc[j] = [id, prediction]
            results.to_csv(os.path.join(args.save_dir, f'{args.model_name}.csv'), mode='a', header=False, index=False)
    
    # for i in tqdm(range(len(df))):
    #     row = df.iloc[i]
    #     id = row['id']
    #     input_text = row['input_text']
        
    #     pdb.set_trace()

    #     merged_input_text = instruction_prompt.format(Text=input_text)

    #     messages = [
    #         {"role": "user", "content": merged_input_text},
    #     ]
        
    #     encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    #     generated_ids = model.generate(encodeds, max_new_tokens=512, do_sample=False)
    #     summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    #     try:
    #         prediction = summary[summary.find('Summary:') + len('Summary:'):]
    #     except:
    #         prediction = ""
    #     prediction = prediction.strip()
        
    #     results = pd.DataFrame(columns=['id', 'summary'])
        
    #     # add id and prediction to a row
    #     results.loc[0] = [id, prediction]
    #     results.to_csv(os.path.join(args.save_dir, f'{args.model_name}.csv'), mode='a', header=False, index=False)
    