import argparse
import json
import os
import glob
import pdb


def cal_scores(preds, groundtruth):
    precision_list = []
    recall_list = []
    f_score_list = []
    
    i = 0
    for key, item in groundtruth.items():
        if i >= 2500:
            break
        try:
            input = item['input']
            gt_keywords = set(item['output'])
            
            if key in preds:
                pred_dict = preds[key]

                if not pred_dict:
                    pred_keywords = []
                else:
                    try:
                        pred_keywords = pred_dict['Expanded MeSH Terms']
                    except:
                        try:
                            pred_keywords = pred_dict['expanded_MeSH_terms']
                        except:
                            if isinstance(pred_dict, list):
                                pred_keywords = pred_dict
                
                # remove input list from pred_keywords list
                pred_keywords = set([x for x in pred_keywords if x not in input])
                
                true_positives = gt_keywords & pred_keywords
                precision = len(true_positives) / len(pred_keywords) if pred_keywords else 0
                recall = len(true_positives) / len(gt_keywords) if gt_keywords else 0
                f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
                
                precision_list.append(precision)
                recall_list.append(recall)
                f_score_list.append(f_score)
                i += 1
        except json.JSONDecodeError:
            print('Error in parsing the groundtruth JSON. Skipping the current instance.')
            continue

    
    return precision_list, recall_list, f_score_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/search/query_expansion/results')
    args = parser.parse_args()


    file_dir = f'data/downstream/search/query_expansion/results'
    files = glob.glob(f'{file_dir}/*.json')

    for file in files:
        model_name = file.split('/')[-1].split('.')[0]

        pred_filename = os.path.join(args.res_dir, f'{model_name}.json')
        with open(pred_filename, 'r') as f:
            preds = json.load(f)
        
        gt_filename = 'data/downstream/search/query_expansion/test.json'
        with open(gt_filename, 'r') as f:
            groundtruth = json.load(f)
        
        precision_list, recall_list, f_score_list = cal_scores(preds, groundtruth)
        

        # save f_score_list
        save_dir = 'src/vis/analysis/downstream/search/query_expansion/results/F1'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(f_score_list, f)

        # save precision_list
        save_dir = 'src/vis/analysis/downstream/search/query_expansion/results/precision'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(precision_list, f)

        # save recall_list
        save_dir = 'src/vis/analysis/downstream/search/query_expansion/results/recall'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(recall_list, f)
            