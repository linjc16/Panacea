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
        "diseases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The disease, disorder, syndrome, illness, or injury that is being studied. On ClinicalTrials.gov, conditions may also include other health-related issues, such as lifespan, quality of life, and health risks.",
        },
        "interventions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A process or action that is the focus of a clinical study. Interventions include drugs, medical devices, procedures, vaccines, and other products that are either investigational or already available. Interventions can also include noninvasive approaches, such as education or modifying diet and exercise.",
        },
        "sponsor": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Sponsor name"
        },
        "status": {
            "enum": ["", "RECRUITING", "TERMINATED", "APPROVED_FOR_MARKETING", "COMPLETED", "ENROLLING_BY_INVITATION"],
            "type": "string",
            "description": "Overall status of the study"
        },
        "phase": {
            "enum": ["", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"],
            "type": "string",
            "description": "Phase of the study"
        },
        "study_type": {
            "enum": ["", "INTERVENTIONAL", "OBSERVATIONAL"],
            "type": "string",
            "description": "Type of the study"
        },
        "person_name": {
            "type": "string",
            "description": "Name of the investigator"
        },
        "nctid": {
            "type": "array",
            "items": {"type": "string"},
            "description": "NCT ID of the clinical trial study"
        },
        "locations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Name of the country or city"
        },
        "start_year": {
            "type": "object",
            "properties": {
                "YEAR": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Year"
                },
                "OPERATOR": {
                    "type": "string",
                    "enum": ["before", "after", "on"],
                    "description": "Enable detect before and after the year"
                }
            },
            "required": ["YEAR", "OPERATOR"],
            "description": "Start year of the trial"
        },
        "end_year": {
            "type": "object",
            "properties": {
                "YEAR": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Year"
                },
                "OPERATOR": {
                    "type": "string",
                    "enum": ["before", "after", "on"],
                    "description": "Enable detect before and after the year"
                }
            },
            "required": ["YEAR", "OPERATOR"],
            "description": "End year of the trial"
        }
    },
}

def load_model(model_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        )
    
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    model.eval()

    return tokenizer, model


def load_dataset(file_dir, split='test'):

    with open(os.path.join(file_dir, split + '.json'), 'r') as f:
        data = json.load(f)
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default='/data/linjc/hub')
    parser.add_argument('--lora_dir', type=str, default='/data/linjc/trialfm')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--file_dir', type=str, default='data/downstream/search/query_generation')
    parser.add_argument('--save_dir', type=str, default='data/downstream/search/query_generation/results')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path
    cache_dir = args.cache_dir

    tokenizer, model = load_model(model_path, cache_dir)
    data = load_dataset(args.file_dir, args.split)

    instruction_prompt = "Given a query used for searching clinical trials in a database, conduct exact extracttion of related entities from the query and then generate a JSON object that can be used to query the database. If a field is not provided, leave it empty fiiled with 'N/A'."

    instruction_prompt += '\n\nQuery: {query}'

    outputs = {}
    
    i = 0
    for key, value in tqdm(data.items()):
        jsonformer = Jsonformer(model, tokenizer, json_schema, instruction_prompt.format(query=value['query']))
        generated_data = jsonformer()
        outputs[key] = generated_data
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(outputs, f, indent=4)
        
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(outputs, f, indent=4)