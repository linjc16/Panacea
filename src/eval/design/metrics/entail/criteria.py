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
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results/criteria')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()
    
    args.save_dir = os.path.join(args.res_dir, 'eval_entail')
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.res_dir, f'{args.model_name}.json'), 'r') as f:
        results = json.load(f)


    instruction_prompts = (
        "Act as an impartial judge and evaluate whether the criteria mentioned in a model's output are present in the full list of the groundtruth criteria. "
        "Output '1' or '0', where '1' means the criteria mentioned in the model's output are fully included in the groundtruth criteria list, and '0' means the criteria from the model's output are not included in the groundtruth. "
        "You should provide an explanation for the evaluation. "
        "\n\n"
        "Example:\n"
        "[Model Output]\nExcellent! Moving on to the third criterion, I propose \"Ability to provide written informed consent.\" Informed consent is a fundamental ethical requirement in clinical research. Participants must fully understand the trial and voluntarily agree to participate.\n[End of Model Output]\n"
        "[Groundtruth Criteria list]\nInclusion Criteria:~Age between 18 and 120 years at time of consent~Ability to speak and understand English~Clinical stage I, II or IIIa NSCLC~Candidate for RTS segmentectomy, as determined by the operating surgeon~Exclusion Criteria:~Anticoagulation with inability to cease anticoagulant therapy prior to surgery~Incurable coagulopathy~Systemic vascular disease or vasculitis~Not a candidate for RTS segmentectomy\n[End of Groundtruth Criteria]\n"
        "Match prediction: 0"
        "\n\n"
        "Now, evaluate the following model output and groundtruth criteria list. Provide match prediction and an explanation for your evaluation. "
        "Finally, you should always output the 'match prediction' in the last line by `Match prediction: `, e.g., `Match prediction: 1`."
        "\n\n"
        "[Model Output]\n{model_output}\n[End of Model Output]\n\n"
        "[Groundtruth Criteria list]\n{groundtruth}\n[End of Groundtruth Criteria]"
    )

    

    # data/downstream/design/raw/selected_step1/merged/test/merged.json
    with open('data/downstream/design/raw/selected_step1/merged/test/merged.json', 'r') as f:
        # read row by row, each row is a dict
        test_data_list = [json.loads(line) for line in f]
    
    # transfer the list to a dict, key is the nct_id
    test_data_dict = {}
    for data in test_data_list:
        test_data_dict[data['nct_id']] = data
    
    # read data/downstream/design/parsed/criteria/del_end_sent.json
    with open('data/downstream/design/parsed/criteria/del_end_sent.json', 'r') as f:
        del_end_sent = json.load(f)
    
    # merge del_end_sent to a dict
    del_end_sent = {key: value for item in del_end_sent for key, value in item.items()}

    eval_results = {}

    i = 0
    for key, value in tqdm(results.items()):
        model_output = value['model_response']
        if del_end_sent[key]:
            model_output = model_output[:-1]
        groundtruth = test_data_dict[key]['eligibility_criteria']
        
        model_preds = []

        for resp in model_output:
            prompt = instruction_prompts.format(model_output=resp, groundtruth=groundtruth)
            response = chat_haiku(prompt)
            model_preds.append(response)

        eval_results[key] = {
            'model_response': model_output,
            'eval_results': model_preds
        }

        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}_eval.json'), 'w') as f:
                json.dump(eval_results, f, indent=4)
        
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}_eval.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)