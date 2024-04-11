import json
import argparse
import re
import os
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results/criteria/eval_entail')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()

    with open(os.path.join(args.res_dir, f'{args.model_name}_eval.json'), 'r') as f:
        results = json.load(f)

    evals = []
    for key, value in results.items():
        eval_res_list = value['eval_results']
        # extract the number after "Match prediction: " from the list
        match_pred = []
        for eval_res in eval_res_list[-1:]:
            try:
                pred = re.findall(r'Match prediction: (\d)', eval_res)
                match_pred.append(int(pred[0]))
            except:
                match_pred.append(0)

        evals.extend(match_pred)
    
    # calculate the accuracy
    accuracy = sum(evals) / len(evals)

    print(f"Accuracy: {accuracy}")