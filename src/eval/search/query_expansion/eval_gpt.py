import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import json
import pdb

sys.path.append('./')
from pydantic.v1 import BaseModel, Field, validator
from typing import List, Literal, Optional
from enum import Enum

with open('./openai_api_azure.key', 'r') as f:
    api_key = f.read().strip()
    
    
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://trialmind.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import JsonOutputParser


class ClinicalTrialQuery(BaseModel):
    expanded_MeSH_terms: Optional[List[str]] = Field(description="Generated MeSH terms that are expanded from the input MeSH terms.")



def gpt_chat_json_parser(prompt, query_dict, model_name):
    parser = JsonOutputParser(pydantic_object=ClinicalTrialQuery)
    
    prompt = PromptTemplate(
        template=prompt,
        input_variables=["query"],
        partial_variables={"schema": parser.get_format_instructions()},
    )

    if model_name == 'gpt-3.5':
        model = AzureChatOpenAI(
            deployment_name="gpt-35", # "gpt-35"
            model_name='gpt-35-turbo'
        )
    elif model_name == 'gpt-4':
        model = AzureChatOpenAI(
            deployment_name="gpt-4", # "gpt-35"
            model_name='gpt-4'
        )
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    
    chain = prompt | model | parser
    
    return chain.invoke(query_dict)


def load_dataset(file_dir, split='test'):

    with open(os.path.join(file_dir, split + '.json'), 'r') as f:
        data = json.load(f)
    
    output_data = {}
    i = 0
    for key, value in data.items():
        if i >= 5000:
            break
        query = ', '.join(value['input'])
        output_data[key] = {'query': query}
        i += 1
    
    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-3.5')
    parser.add_argument('--file_dir', type=str, default='data/downstream/search/query_expansion')
    parser.add_argument('--save_dir', type=str, default='data/downstream/search/query_expansion/results')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    PROMPT_TEMPLATE = ('Given MeSH Terms used for searching clinical trials in a database, expand the input MeSH terms and then generate a JSON object that contains the expanded MeSH terms. '
    'Don\'t include the original MeSH terms in the expanded MeSH terms.'
    '\n`For example, the input MeSH Terms are: "Neurocognitive Disorders, Tauopathies, Movement Disorders, Dementia, Synucleinopathies", then Expanded MeSH Terms are: Central Nervous System Diseases, Basal Ganglia Diseases, Brain Diseases, Alzheimer Disease, Lewy Body Disease, Nervous System Diseases.`'
    '\n\nnInput MeSH Terms: {query}. Now expand the input MeSH terms and generate the expanded MeSH terms.'
    '\nOutput result in the following JSON schema format:\n{schema}\nResult:"'
    )
    
    
    data = load_dataset(args.file_dir, args.split)

    outputs = {}
    
    i = 0
    for key, value in tqdm(data.items()):
        query=value['query']
        try:
            generated_data = gpt_chat_json_parser(PROMPT_TEMPLATE, {"query": query}, args.model_name)
        except:
            generated_data = {}
        
        outputs[key] = generated_data
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(outputs, f, indent=4)
        
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(outputs, f, indent=4)
    