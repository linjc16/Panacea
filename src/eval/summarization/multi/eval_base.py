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
        output_data['target'].append(value["target"])
    
    df = pd.DataFrame(output_data)
    return df

one_shot_example = """Study #1 Prophylactic indomethacin therapy in the first twenty-four hours of life for the prevention of patent ductus arteriosus in preterm infants treated prophylactically with surfactant in the delivery room. To determine whether a course of low-dose indomethacin therapy, when initiated within 24 hours of birth, would decrease ductal shunting in premature infants who received prophylactic surfactant in the delivery room. Ninety infants, with birth weights of 600 to 1250 gm, were entered into a prospective, randomized, controlled trial to receive either indomethacin, 0.1 mg/kg per dose, or placebo less than 24 hours and again every 24 hours for six doses. Echocardiography was performed on day 1 before treatment and on day 7, 24 hours after treatment. A hemodynamically significant patent ductus arteriosus (PDA) was confirmed with an out-of-study echocardiogram, and the nonresponders were treated with standard indomethacin or ligation. Forty-three infants received indomethacin (birth weight, 915 +/- 209 gm; gestational age, 26.4 +/- 1.6 weeks; 25 boys), and 47 received placebo (birth weight, 879 +/- 202 gm; gestational age, 26.4 +/- 1.8 weeks; 22 boys) (P = not significant). Of 90 infants, 77 (86%) had a PDA by echocardiogram on the first day of life before study treatment; 84% of these PDAs were moderate or large in size in the indomethacin-treated group compared with 93% in the placebo group. Nine of forty indomethacin-treated infants (21%) were study-dose nonresponders compared with 22 (47%) of 47 placebo-treated infants (p < 0.018). There were no significant differences between both groups in any of the long-term outcome variables, including intraventricular hemorrhage, duration of oxygen therapy, endotracheal intubation, duration of stay in neonatal intensive care unit, time to regain birth weight or reach full caloric intake, incidence of bronchopulmonary dysplasia, and survival. No significant differences were noted in the incidence of oliguria, elevated plasma creatinine concentration, thrombocytopenia, pulmonary hemorrhage, or necrotizing enterocolitis. The prophylactic use of low doses of indomethacin, when initiated in the first 24 hours of life in low birth weight infants who receive prophylactic surfactant in the delivery room, decreases the incidence of left-to-right shunting at the level of the ductus arteriosus.
Study #2 Indomethacin reduces the risks of severe intraventricular hemorrhage. A prospective, random selection, double-blind clinical trial was carried out to determine the efficacy of indomethacin in preventing periventricular-intraventricular hemorrhage (PV-IVH). Babies who were born in our institution, had birth weights less than or equal to 1500 gm, and had no PV-IVH or grade 1 PV-IVH were given either placebo (n = 70) or indomethacin (n = 71), 0.2 mg/kg intravenously at 6 hours of age and 0.1 mg/kg at 18 and 30 hours. Two major outcomes were determined: the development of grades 2 to 4 PV-IVH and the development of severe PV-IVH (i.e., hemorrhages with blood filling greater than 50% of the ventricles and in some cases with associated parenchymal echodensities). Grades 2 to 4 PV-IVH occurred in 16 (23%) of the indomethacin group and 27 (39%) of the placebo group (p less than 0.03). The incidence of severe PV-IVH was 3% in the indomethacin-treated babies and 14% in the control group (p less than 0.02). The influence of other perinatal factors on the incidence of grades 2 to 4 or severe PV-IVH was determined by stepwise logistic regression. Placebo use, early grade 1 PV-IVH, lower birth weight, and higher fraction of inspired oxygen at 6 hours of life were associated with higher estimated odds of the development of grades 2 to 4 PV-IVH. Placebo use, male gender, lower 5-minute Apgar score, and a large base deficit were predictive of severe PV-IVH. Estimated odds ratios of severe PV-IVH with placebo use and male gender were 11.25:1 and 9:1, respectively. Thus indomethacin prophylaxis reduced the relative risk of grades 2 to 4 PV-IVH and severe PV-IVH, but other perinatal variables contributed significantly to the overall risk of PV-IVH.
Study #3 Administration of indomethacin for the prevention of periventricular-intraventricular hemorrhage in high-risk neonates. One hundred twenty-two preterm infants were enrolled in a placebo-controlled, double-blind trial using intravenous indomethacin for the prevention of periventricular-intraventricular hemorrhage (PVH-IVH). Before random assignment, data on the infants were stratified according to low-weight (500 to 999 g) or high-weight (1000 to 1500 g) subgroups. Cranial sonography was used to document the absence of PVH-IVH before enrollment and the occurrence of PVH-IVH during the 7-day protocol. Indomethacin, 0.1 mg/kg, or placebo was administered before 12 hours of age and at 24, 48, and 72 hours of age. Five patients receiving indomethacin and six receiving placebo were withdrawn before completion of the study. In the remaining 111 patients, the indomethacin and placebo groups were comparable with respect to gestational ages, maternal complications, Apgar scores, ventilatory requirements, complications of prematurity, and mortality rate. PVH-IVH developed in six of 56 infants who received indomethacin and 11 of 55 infants who received placebo (P = 0.174). Analysis of the individual strata showed that the indomethacin-treated infants in the low-weight subgroup sustained a higher mortality rate (11/17 vs 3/16; P = 0.008) without a reduction in the incidence of PVH-IVH. Infants in the indomethacin-treated high-weight subgroup demonstrated a significantly lower incidence of PVH-IVH (2/39 vs 8/39; P = 0.04), but the frequency of high-grade hemorrhages was comparable for both indomethacin- and placebo-treated groups. In summary, the prophylactic administration of intravenous indomethacin for the prevention of PVH-IVH cannot be recommended for infants less than 1000 g. In preterm infants between 1000 and 1500 g birth weight, indomethacin significantly reduced the incidence of PVH-IVH.
"""

summary_example = 'Prophylactic indomethacin has short-term benefits for preterm infants including a reduction in the incidence of symptomatic PDA, PDA surgical ligation, and severe intraventricular haemorrhage. However, there is no evidence of effect on mortality or neurodevelopment.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default='/data/linjc/hub')
    parser.add_argument('--lora_dir', type=str, default='/data/linjc/trialfm')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--file_dir', type=str, default='data/downstream/summazization/multi-trial')
    parser.add_argument('--save_dir', type=str, default='data/downstream/summazization/multi-trial/results')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--sample', type=bool, default=True)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path
    cache_dir = args.cache_dir
    
    df = load_dataset(args.file_dir, args.split)
    
    tokenizer, model = load_model(model_path, cache_dir)
    
    instruction_prompt = "Your task is to synthesize the key findings from a collection of study abstracts related to a specific clinical trial related research question."
    instruction_prompt += "\nCombine the insights from the provided abstracts into a cohesive summary. Your summary should integrate the findings rather than listing them separately. It's crucial to maintain the scientific integrity of the original studies while ensuring the summary is accessible and informative."
    instruction_prompt += "\nThe output should only be the summary. Do not explain how you summarize it."
    # instruction_prompt += "\nExample:"
    # instruction_prompt += f"\nStudy Abstracts: {one_shot_example}"
    # instruction_prompt += f"\nSummary: {summary_example}"
    instruction_prompt += "\nStudy Abstracts: {Text}"
    instruction_prompt += "\nSummary: "
    
    if not os.path.exists(os.path.join(args.save_dir, f'{args.model_name}.csv')):
        with open(os.path.join(args.save_dir, f'{args.model_name}.csv'), 'w') as f:
            f.write('id,summary\n')
    
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        id = row['id']
        input_text = row['study_text']

        merged_input_text = instruction_prompt.format(Text=input_text)

        encodeds = tokenizer(merged_input_text, return_tensors='pt').to(model.device)
        
        generated_ids = model.generate(**encodeds, max_new_tokens=512, do_sample=args.sample)
        summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            prediction = summary[len(merged_input_text):].strip()
        except:
            prediction = ""
        prediction = prediction.strip()
        
        results = pd.DataFrame(columns=['id', 'summary'])
        
        # add id and prediction to a row
        results.loc[0] = [id, prediction]
        results.to_csv(os.path.join(args.save_dir, f'{args.model_name}.csv'), mode='a', header=False, index=False)
    