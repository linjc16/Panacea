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

with open('./claude_api_aws.key', 'r') as f:
    # two lines, one for AWS_ACCESS_KEY_ID, the next for AWS_SECRET_ACCESS_KEY
    keys = f.readlines()
    AWS_ACCESS_KEY_ID = keys[0].strip()
    AWS_SECRET_ACCESS_KEY = keys[1].strip()


AWS_DEFAULT_REGION = "us-west-2"
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import BedrockChat
import boto3

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_DEFAULT_REGION,
)

class YearOperator(BaseModel):
    YEAR: int = Field(ge=0, description="Year")
    OPERATOR: Literal["before", "after", "on"] = Field(description="Enable detect before and after the year")


class ClinicalTrialQuery(BaseModel):

    class PhaseEnum(str, Enum): # ZW: can use Literal instead in the future
        DEFAULT = ""
        EARLY_PHASE1 = "EARLY_PHASE1"
        PHASE1 = "PHASE1"
        PHASE2 = "PHASE2"
        PHASE3 = "PHASE3"
        PHASE4 = "PHASE4"

    class StudyTypeEnum(str, Enum):
        DEFAULT = ""
        INTERVENTIONAL = "INTERVENTIONAL"
        OBSERVATIONAL = "OBSERVATIONAL"

    class OverallStatusEnum(str, Enum):
        DEFAULT = ""
        RECRUITING = "RECRUITING"
        TERMINATED = "TERMINATED"
        APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"
        COMPLETED = "COMPLETED"
        ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    
    diseases: Optional[List[str]] = Field(description="The disease, disorder, syndrome, illness, or injury that is being studied. On ClinicalTrials.gov, conditions may also include other health-related issues, such as lifespan, quality of life, and health risks.")
    interventions: Optional[List[str]] = Field(description="A process or action that is the focus of a clinical study. Interventions include drugs, medical devices, procedures, vaccines, and other products that are either investigational or already available. Interventions can also include noninvasive approaches, such as education or modifying diet and exercise.")
    sponsor: Optional[List[str]] = Field(description="Sponsor name")
    
    status: Optional[List[OverallStatusEnum]] = Field(description="Overall status of the study")
    phase: Optional[List[PhaseEnum]] = Field(description="Phase of the study")
    study_type: Optional[StudyTypeEnum] = Field(description="Type of the study")
    
    person_name: Optional[str] = Field(description="Name of the investigator")
    nctid: Optional[List[str]] = Field(description="NCT ID of the clinical trial study")
    locations: Optional[List[str]] = Field(description="Name of the country or city")
    start_year: Optional[YearOperator] = Field(description="Start year of the trial")
    end_year: Optional[YearOperator] = Field(description="End year of the trial")

    

def gpt_chat_json_parser(prompt, query_dict, model_name):
    parser = JsonOutputParser(pydantic_object=ClinicalTrialQuery)
    
    prompt = PromptTemplate(
        template=prompt,
        input_variables=["query"],
        partial_variables={"schema": parser.get_format_instructions()},
    )

    if model_name == 'claude-haiku':
        model = BedrockChat(
            client=bedrock_runtime,
            model_id='anthropic.claude-3-haiku-20240307-v1:0',
        )
    elif model_name == 'claude-sonnet':
        model = BedrockChat(
            client=bedrock_runtime,
            model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        )
    
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    
    chain = prompt | model | parser
    
    return chain.invoke(query_dict)


def load_dataset(file_dir, split='test'):

    with open(os.path.join(file_dir, split + '.json'), 'r') as f:
        data = json.load(f)
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-3.5')
    parser.add_argument('--file_dir', type=str, default='data/downstream/search/query_generation')
    parser.add_argument('--save_dir', type=str, default='data/downstream/search/query_generation/results')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    PROMPT_TEMPLATE = 'Given a query used for searching clinical trials in a database, conduct exact extracttion of related entities from the query and then generate a JSON object that can be used to query the database. If a field is not provided, leave it empty fiiled with "N/A".\n\nQuery: "{query}\nOutput result in the following JSON schema format:\n{schema}\nResult:"'
    
    
    data = load_dataset(args.file_dir, args.split)

    outputs = {}

    i = 0
    for key, value in tqdm(data.items()):
        query=value['query']
        
        for _ in range(5):
            try:
                generated_data = gpt_chat_json_parser(PROMPT_TEMPLATE, {"query": query}, args.model_name)
                break
            except:
                generated_data = {}
        
        outputs[key] = generated_data
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(outputs, f, indent=4)
        
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(outputs, f, indent=4)
    