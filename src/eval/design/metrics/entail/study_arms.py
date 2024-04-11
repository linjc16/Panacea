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
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results/study_arms')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()
    
    args.save_dir = os.path.join(args.res_dir, 'eval_entail')
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.res_dir, f'{args.model_name}.json'), 'r') as f:
        results = json.load(f)

    
    instruction_prompts = (
        "Act as an impartial judge and evaluate whether the study arms mentioned in a model's output are present in the full table of groundtruth study arms. "
        "Output '1' or '0', where '1' means the study arms mentioned in the model's output are fully included in the groundtruth study arm table, and '0' means the study arms from the model's output are not included in the groundtruth. "
        "You should provide an explanation for the evaluation."
        "\n\n"
        "Example:\n"
        "[Model Output]\nThe placebo comparator arm, which we'll call \"Control: Placebo,\" will also include obese subjects with Type 2 Diabetes at risk of Nonalcoholic Steatohepatitis. Participants in this arm will receive a placebo, which will be designed to mimic the appearance of the active treatment but will not contain any active drug. The primary purpose of this arm is to compare the safety and efficacy of HU6 to the placebo, to determine if any observed effects are due to the active treatment or could be attributed to other factors.\n[End of Model Output]\n"
        "[Groundtruth Study Arm]\n| Participant Group/Arm | Intervention/Treatment |\n| --- | --- |\n| Experimental: Active Treatment: HU6 Planned doses of HU6<br> | Drug: HU6<br>* HU6 is being evaluated for its efficacy in improving liver fat content in obese subjects with Type 2 Diabetes at risk of Nonalcoholic Steatohepatitis (NASH)<br>|\n| Placebo Comparator: Placebo Comparator Non-active study drug<br> | Other: Placebo<br>* Placebo<br>|\n[End of Groundtruth Study Arm]\n"
        "Match prediction: 1"
        "\n\n"
        "Now, evaluate the following model output and groundtruth study arm table. "
        "You should first output the 'match prediction' at the beginning of the response by `Match prediction: `, e.g., `Match prediction: 1`. Then, provide an explanation for your evaluation. "
        "\n\n"
        "[Model Output]\n{model_output}\n[End of Model Output]\n\n"
        "[Groundtruth Study Arm]\n{groundtruth}\n[End of Groundtruth Study Arm]"
    )
    

    # data/downstream/design/raw/selected_step1/merged/test/merged.json
    with open('data/downstream/design/raw/selected_step1/merged/test/merged.json', 'r') as f:
        # read row by row, each row is a dict
        test_data_list = [json.loads(line) for line in f]
    
    # transfer the list to a dict, key is the nct_id
    test_data_dict = {}
    for data in test_data_list:
        test_data_dict[data['nct_id']] = data
    
    # read data/downstream/design/parsed/study_arms/del_end_sent.json
    with open('data/downstream/design/parsed/study_arms/del_end_sent.json', 'r') as f:
        del_end_sent = json.load(f)
    
    # merge del_end_sent to a dict
    del_end_sent = {key: value for item in del_end_sent for key, value in item.items()}

    eval_results = {}

    i = 0
    for key, value in tqdm(results.items()):
        model_output = value['model_response']
        if del_end_sent[key]:
            model_output = model_output[:-1]
        groundtruth = test_data_dict[key]['arms_and_interventions']
        
        model_preds = []

        for resp in model_output:
            prompt = instruction_prompts.format(model_output=resp, groundtruth=groundtruth)
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