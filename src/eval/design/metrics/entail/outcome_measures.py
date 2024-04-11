import os
import argparse
import openai
import json
import pdb
from tqdm import tqdm
import sys

sys.path.append('./')
from src.utils.claude_aws import chat_haiku


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results/outcome_measures')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()
    
    args.save_dir = os.path.join(args.res_dir, 'eval_entail')
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.res_dir, f'{args.model_name}.json'), 'r') as f:
        results = json.load(f)

    
    instruction_prompts = (
        "Act as an impartial judge and evaluate whether the outcome measures mentioned in a model's output are present in the full table of groundtruth outcome measures. "
        "Output '1' or '0', where '1' means the outcome measures mentioned in the model's output are fully included in the groundtruth outcome measures table, and '0' means the outcome measures from the model's output are not included in the groundtruth. "
        "You should provide an explanation for the evaluation."
        "\n\n"
        "Example:\n"
        "[Model Output]\nAbsolutely. To measure the recruitment rate, we can track the number of participants who enroll in the study within a specified time frame. For this trial, we can monitor the recruitment rate up to 8 weeks after recruitment first opens. The goal is to achieve a recruitment rate of at least 70% to ensure the feasibility of conducting the full-scale trial.\n[End of Model Output]\n"
        "[Groundtruth Primary Outcome Measures]\n| Outcome Measure | Measure Description | Time Frame |\n| --- | --- | --- |\n| Feasibility and safety | No adverse impacts of the study procedures on participants | Up to 3 weeks post-surgery | \n| Recruitment | Recruitment rate of at least 70% | Up to 8 weeks after recruitment first opens | \n| Randomization | Ability to randomize patients to one of two groups | Baseline | \n| Data collection of stapler reload model | Ability to collect the type of stapler reloads used | Up to 3 weeks post-surgery | \n| Data collection of stapler quantities | Ability to collect the number of stapler reloads used | Up to 3 weeks post-surgery | \n| Data collection of energy sealing data | Ability to collect the sealing time in seconds | Up to 3 weeks post-surgery | \n| Data collection of energy device data | Ability to collect the generator setting of the energy device | Up to 3 weeks post-surgery | \n[End of Groundtruth Primary Outcome Measures]\n"
        "[Groundtruth Secondary Outcome Measures]\n| Outcome Measure | Measure Description | Time Frame |\n| --- | --- | --- |\n| Adverse events (AEs) and complications | Short-term clinical outcomes, as measured by postoperative AEs and complications, will be recorded during patient follow-ups. | 3 weeks post-surgery | \n| Intraoperative costs of stapler or energy device use | Surgical device (stapler or energy) costs per surgery will be collected and evaluated in Canadian dollars. | Up to 3 weeks following hospital discharge | \n| Hospitalization costs based on length of hospital stay | Inpatient hospitalization costs per day following surgery will be collected in Canadian dollars. | From admission to discharge, up to 14 days |\n[End of Groundtruth Secondary Outcome Measures]\n"
        "Match prediction: 1"
        "\n\n"
        "Now, evaluate the following model output and groundtruth outcome measures table. "
        "You should first output the 'match prediction' at the beginning of the response by `Match prediction: `, e.g., `Match prediction: 1`. Then, provide an explanation for your evaluation. "
        "\n\n"
        "[Model Output]\n{model_output}\n[End of Model Output]\n\n"
        "[Groundtruth Primary Outcome Measures]\n{prim_out_meas}\n[End of Groundtruth Primary Outcome Measures]\n\n"
        "[Groundtruth Secondary Outcome Measures]\n{sec_out_meas}\n[End of Groundtruth Secondary Outcome Measures]"
    )
    

    # data/downstream/design/raw/selected_step1/merged/test/merged.json
    with open('data/downstream/design/raw/selected_step1/merged/test/merged.json', 'r') as f:
        # read row by row, each row is a dict
        test_data_list = [json.loads(line) for line in f]
    
    # transfer the list to a dict, key is the nct_id
    test_data_dict = {}
    for data in test_data_list:
        test_data_dict[data['nct_id']] = data
    
    # read data/downstream/design/parsed/outcome_measures/del_end_sent.json
    with open('data/downstream/design/parsed/outcome_measures/del_end_sent.json', 'r') as f:
        del_end_sent = json.load(f)
    
    # merge del_end_sent to a dict
    del_end_sent = {key: value for item in del_end_sent for key, value in item.items()}

    eval_results = {}

    i = 0
    for key, value in tqdm(results.items()):
        model_output = value['model_response']
        if del_end_sent[key]:
            model_output = model_output[:-1]
        prim_out_meas = test_data_dict[key]['primary_outcome_measures']
        sec_out_meas = test_data_dict[key]['secondary_outcome_measures']
        
        model_preds = []

        for resp in model_output:
            prompt = instruction_prompts.format(model_output=resp, prim_out_meas=prim_out_meas, sec_out_meas=sec_out_meas)
            response = chat_haiku(prompt)
            model_preds.append(response)

        eval_results[key] = {
            'model_response': model_output,
            'eval_results': model_preds
        }

        if i % 20 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}_eval.json'), 'w') as f:
                json.dump(eval_results, f, indent=4)
        
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}_eval.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)