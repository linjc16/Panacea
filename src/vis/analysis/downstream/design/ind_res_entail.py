import json
import argparse
import re
import os
import pdb

def calculate_entail_acc(results):
    evals = []
    for key, value in results.items():
        eval_res_list = value['eval_results']
        # extract the number after "Match prediction: " from the list
        match_pred = []
        for eval_res in eval_res_list[-2:]:
            try:
                pred = re.findall(r'Match prediction: (\d)', eval_res)
                match_pred.append(int(pred[0]))
            except:
                match_pred.append(0)

        evals.extend(match_pred)
    
    # print accuracy
    accuracy = sum(evals) / len(evals)
    print(f"Accuracy: {accuracy}")
    
    return evals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='criteria')
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results1/sequential/criteria/eval_entail')
    args = parser.parse_args()

    # model name list
    model_names = ['mistral-7b', 'zephyr-7b', 'panacea-7b']
    
    eval_dict = {}

    for model_name in model_names:
        print(f"Model: {model_name}")
        with open(os.path.join(args.res_dir, f'{model_name}_eval.json'), 'r') as f:
            results = json.load(f)
        
        evals = calculate_entail_acc(results)

        eval_dict[model_name] = evals
    
    # find all the indexes where panacea is better than the other models
    
    index_set = []
    index_set.append(191)
    for i in range(len(eval_dict['panacea-7b'])):
        if eval_dict['panacea-7b'][i] > eval_dict['mistral-7b'][i] and eval_dict['panacea-7b'][i] > eval_dict['zephyr-7b'][i]:
            index_set.append(i)
    
    print(f"Panacea is better than the other models at indexes: {index_set}")
    print(f"Total number of indexes: {len(index_set)}")

    
    # output the response for each model in the index set

    with open(f'data/downstream/design/parsed/{args.task}/del_end_sent.json', 'r') as f:
        del_end_sent = json.load(f)
    
    # merge del_end_sent to a dict
    del_end_sent = {key: value for item in del_end_sent for key, value in item.items()}

    response_dir = f'data/downstream/design/results1/sequential/{args.task}'
    results_dict = {}
    for model_name in model_names:
        with open(os.path.join(response_dir, f'{model_name}.json'), 'r') as f:
            results = json.load(f)
        
        # according to del_end_sent, remove the last sentence
        results_new = {}
        for key, value in results.items():
            model_output = value['model_response']
            gt = value['groundtruth']
            if del_end_sent[key]:
                model_output = model_output[:-1]
                gt = gt[:-1]
            
            results_new[key] = {'model_response': model_output, 'groundtruth': gt}

        results_dict[model_name] = results_new


    # obtain key list
    key_list = list(results_dict[model_name].keys())

    # output the response for each model in the index set
    for index in index_set:
        try:
            print(f"Index: {index}")
            for model_name in model_names:
                print(f"{model_name}:")
                print(f"Model Response: {results_dict[model_name][key_list[index]]['model_response'][-2]}")
            print(f"Groundtruth: {results_dict[model_name][key_list[index]]['groundtruth'][-1]}")
            print("\n")
        except:
            continue

    # pdb.set_trace()