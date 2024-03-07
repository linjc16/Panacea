from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
from tqdm import tqdm
import pdb

PRMOPT = (
    "Verify whether the 'brief summary' section of the clinical trial protocol can be obtained from the other sctions. "
    "The 'brief summary' should not contain any information that can not be found in the other sections.\n"
    "First analyze the clinical trial protocol and then compare it with the 'brief summary' section.\n"
    "In the last line of your response, please directly answer with the format 'The answer is: xx', replace 'xx' with 'Yes' or 'No'.\n"
    "Here is the clinical trial protocol: \n"
    "{protocol}"
    "\n\n"
    "Here is the 'brief summary' section of the clinical trial protocol: \n"
    "{summary}"
    "\n\n"
    "Give me your response:\n"
)

def load_model():
    model_name = "openchat/openchat-3.5-0106"
    cache_dir = '/data/linjc/hub'
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation="flash_attention_2"
    )
    return tokenizer, model

def load_data():
    data_dir = '/data/linjc/trialfm/downstream/summarization'
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    return df

def pre_select():
    tokenizer, model = load_model()
    df = load_data()

    save_file = 'data/downstream/summazization/pre-select.txt'
    
    for i in tqdm(range(len(df))):
        protocol = df['input_text'][i]
        summary = df['summary_text'][i]
        prompt = PRMOPT.format(protocol=protocol, summary=summary)
        messages = [
            {"role": "user", "content": prompt},
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(model.device)
        generated_ids = model.generate(model_inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        try:
            decoded_processed = decoded.split('The answer is')[-1].strip()
        except:
            continue
        if 'yes' in decoded_processed.lower():
            with open(save_file, 'a') as f:
                # write the nct id
                f.write(df['nct_id'][i] + '\n')

if __name__ == '__main__':
    pre_select()