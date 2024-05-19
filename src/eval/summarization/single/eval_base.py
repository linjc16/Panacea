from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm
import pdb

tqdm.pandas()

def load_model(model_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, padding_side='left', use_fast=False)
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



def load_sft_mistral_model_tokenizer(args):
    base_model_name = args.base_model_name
    cache_dir = args.cache_dir
    lora_dir = args.lora_dir

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16, 
        device_map='auto', 
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1000000000000000019884624838656
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapter and merge
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload()

    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return model, tokenizer

def load_dataset(file_dir, split='test'):
    df = pd.read_csv(os.path.join(file_dir, split + '.csv'))
    return df


one_shot_example = """
The Interaction Between Metformin and Microbiota - the MEMO Study.

Study Overview
=================

Official Title
-----------------
The Interaction Between Metformin and Microbiota - the Reason for Gastrointestinal Side Effects?

Conditions
-----------------
Type 2 Diabetes Mellitus

Intervention / Treatment
-----------------
* Drug: Metformin


Participation Criteria
=================
Eligibility Criteria
-----------------
Inclusion criteria: Type 2 diabetes (diagnosis set within the last 12 months) planned metformin treatment Age: 40 - 80 years Have provided written informed consent. Exclusion criteria: already started treatment with Metformin intestinal disease incl. irritable bowel syndrome treatment with antibiotics in the last 3 months Inflammatory disorder t.ex. rheumatoid arthritis anemia, haemoglobinopathy alcohol or drug abuse cancer disease under treatment

Ages Eligible for Study
-----------------
Minimum Age: 40 Years
Maximum Age: 80 Years

Sexes Eligible for Study
-----------------
All

Accepts Healthy Volunteers
-----------------
No

Study Plan
=================
How is the study designed?
-----------------

Arms and Interventions

| Participant Group/Arm | Intervention/Treatment |
| --- | --- |
| prospective cohort<br>All participants receive Metformin in accordance to the ordinary therapy practice. They will be part of one subject group/cohort. After onset of metformin treatment the subjects who develop gastrointestinal side effects will be compared with cases without gastrointestinal side effects. | Drug: Metformin<br>* Receiving Metformin is no study intervention. The investigators have no influence on the Metformin treatment. All participants receive Metformin provided by the health care system. The individual case determine dose and frequency of the treatment as part of the usual therapy guidelines.<br>|

What is the study measuring?
-----------------
Primary Outcome Measures

| Outcome Measure | Measure Description | Time Frame |
| --- | --- | --- |
| change from baseline in the fecal microbiota composition by detecting bacteria families with help of relative abundance (%) and diversity metrics | Faeces samples will be collected at baseline, after 2 months and after 4 months of Metformin treatment and analysed using 16S rRNA sequencing, whole genome shotgun sequencing and metagenomic analyses | 4 months | 

Secondary Outcome Measures

| Outcome Measure | Measure Description | Time Frame |
| --- | --- | --- |
| genetic correlation between Microbiota and gastrointestinal side effects | The microbiota genome and the relation to the appearance of gastrointestinal side effects will be analysed. | 4 months | 
| number of patients with gastrointestinal side effects | A questionnaire will be answered at baseline, after 2 and 4 months and refers to a period of 2 weeks before study visit. Six different side effects will be monitored: loss of appetite, nausea, vomiting, diarrhea, meteorism, stomach ache. | 4 months | 
| rate of gastrointestinal side effects | Changes in the rate of side effects using questionnaire at baseline, after 2 and 4 months.The answer possibilities are: never, one or a few times, daily, multiple times a day. | 4 months | 
| time to first appearance of gastrointestinal side effects | How many months till the onset of side effects during metformin treatment: depending on the time of assessment, 2 or 4 months | 4 months | 
| time to first appearance of side effects which requires dose changes in the Metformin treatment | Analysing changes in metformin treatment and relation to appearance of side effects | 4 months | 
| time to first appearance of side effects which requires termination of the Metformin treatment | Analysing length of metformin treatment and relation to appearance of side effects | 4 months | 
| correlation between microbiota and beneficial glucose lowering response | Plasma glucose will be analysed at baseline, after 2 and 4 months of Metformin treatment and correlated to the changes in the composition of microbiota. | 4 months | 

 Terms related to the study
=================
Keywords Provided by Centre Hospitalier Valida
-----------------
Metformin, Gastrointestinal side effects, Microbiota

"""

summary_example = (
    "Metformin has been used in Sweden since 1957, and it is recommended as first line therapy for type 2-diabetes (T2D) in national and international guidelines. However, adverse effects involving diarrhea, constipation, bloating, and abdominal pain are common which leads to discontinuation of medication or not being able to reach therapeutic doses. Here the investigators will perform a prospective study to investigate whether i) participants with T2D who experience adverse events following metformin treatment have an altered microbiota at baseline compared to participants without adverse events and ii) if the microbiota is altered in participants during onset of adverse events. The investigators hypothesis is that adverse effects associated with metformin are caused by an altered gut microbiota, either at base line or following metformin treatment. The study design is a nested case-cohort study. The investigators will recruit 600 patients and expect 200 individuals to have side effects and 400 without during a 24-month study period. Fecal samples will be collected at baseline, 2 months, and 4 months or when gastrointestinal symptoms occur. All fecal samples will be sequenced by 16s rRNA (ribosomal ribonucleic acid) sequencing to obtain a baseline microbiota profile; a subpopulation consisting of homogenous groups of participants will be in depth-analyzed using shotgun sequencing. If the hypothesis is confirmed this project may lead to bacterial therapies that will allow more patients tolerate metformin."
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default='/data/linjc/hub')
    parser.add_argument('--lora_dir', type=str, default='/data/linjc/trialfm')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--file_dir', type=str, default='data/downstream/summazization/single-trial')
    parser.add_argument('--save_dir', type=str, default='data/downstream/summazization/single-trial/results')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--sample', type=bool, default=True)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    model_path = args.model_path
    cache_dir = args.cache_dir
    
    tokenizer, model = load_model(model_path, cache_dir)
    
    # if args.model_name == 'llama3-8b':


    df = load_dataset(args.file_dir, args.split)

    instruction_prompt = "Your task is to create a clear, concise, and accurate summary of the provided clinical trial document. The summary should capture the key aspects of the trial."
    instruction_prompt += "\nThe output should only be the summarization of the given trial. Do not explain how you summarize it."
    instruction_prompt += '\n\nExample: \n\nInput Text: {one_shot_example}\nSummary: {summary_example}\n'
    instruction_prompt += "\nInput Text: {Text}"
    instruction_prompt += "\nSummary: "
    
    
    if not os.path.exists(os.path.join(args.save_dir, f'{args.model_name}.csv')):
        with open(os.path.join(args.save_dir, f'{args.model_name}.csv'), 'w') as f:
            f.write('id,summary\n')
    
    # for each data, add a column of id
    df['id'] = df.index

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        id = row['id']
        input_text = row['input_text']


        merged_input_text = instruction_prompt.format(Text=input_text, one_shot_example=one_shot_example, summary_example=summary_example)

        encodeds = tokenizer(merged_input_text, return_tensors="pt").to(model.device)
        

        generated_ids = model.generate(**encodeds, max_new_tokens=1024, do_sample=args.sample)
        summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        

        try:
            prediction = summary[len(merged_input_text):]
        except:
            prediction = ""
        prediction = prediction.strip()
        
        results = pd.DataFrame(columns=['id', 'summary'])
        
        # add id and prediction to a row
        results.loc[0] = [id, prediction]
        results.to_csv(os.path.join(args.save_dir, f'{args.model_name}.csv'), mode='a', header=False, index=False)
    
