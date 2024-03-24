import os
import argparse
import openai
import json
import pdb
from tqdm import tqdm
with open('./openai_api_azure.key', 'r') as f:
    api_key = f.read().strip()
    

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://zifeng-gpt-2.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import JsonOutputParser

def gpt_chat(messages, model='gpt-35'):
    """
        messages: [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Knock knock."},
            {"role": "assistant", "content": "Who's there?"},
            {"role": "user", "content": "Orange."},
        ]
    """
    client = openai.AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=api_key,
        api_version="2023-09-01-preview"
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=256,
    )
    output = response.choices[0].message.content
    
    return output

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    
    return data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-35')
    parser.add_argument('--task', type=str, default='study_arms')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    filepath = f'data/downstream/design/parsed/{args.task}/{args.split}.json'

    data_dict = load_dataset(filepath)



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
                
                response = gpt_chat(input, model=args.model_name)
                out_response.append(response)
                # pdb.set_trace()

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