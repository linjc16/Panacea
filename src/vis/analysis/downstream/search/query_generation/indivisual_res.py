import argparse
import json
import os
import glob
import pdb


def cal_scores(preds, groundtruth):
    precision_list = []
    recall_list = []
    f_score_list = []
    jaccard_list = []
    
    for key, item in groundtruth.items():
        try:
            parsed_dict = json.loads(item['parsed_dict'])
            
            gt_keywords = set()
            for k, v in parsed_dict.items():
                if k not in ['start_year', 'end_year']:
                    if isinstance(v, list):
                        gt_keywords.update([v.lower() for v in v if v.lower() != "n/a"])
                    else:
                        if v.lower() != "n/a":
                            gt_keywords.add(v.lower())
                else:
                    if v['YEAR'] != 0:
                        gt_keywords.add(str(int(v['YEAR'])))
                        gt_keywords.add(str(v['OPERATOR']))

            if key in preds:
                pred_dict = preds[key]
                pred_keywords = set()

                    
                if pred_dict:
                    for k, v in pred_dict.items():
                        if k not in ['start_year', 'end_year']:
                            if isinstance(v, list):
                                try:
                                    pred_keywords.update([v_.lower() for v_ in v if v_.lower() != "n/a"])
                                except:
                                    continue
                            elif isinstance(v, dict):
                                if v:
                                    # add the values of the first key in the dict
                                    for k1, v1 in v.items():
                                        if isinstance(v1, list):
                                            pred_keywords.update([v.lower() for v in v1 if v.lower() != "n/a"])
                                        else:
                                            continue
                            else:
                                try:
                                    if v.lower() != "n/a":
                                        pred_keywords.add(v.lower())
                                except:
                                    continue
                        else:
                            try:
                                if v['YEAR'] != 0 and str(v['YEAR']).lower() != 'n/a':
                                    
                                    pred_keywords.add(str(int(v['YEAR'])))
                                    pred_keywords.add(str(v['OPERATOR']))
                            except:
                                continue
                
                true_positives = gt_keywords & pred_keywords
                precision = len(true_positives) / len(pred_keywords) if pred_keywords else 0
                recall = len(true_positives) / len(gt_keywords) if gt_keywords else 0
                f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
                jaccard = len(true_positives) / len(gt_keywords | pred_keywords) if (gt_keywords | pred_keywords) else 0
                
                precision_list.append(precision)
                recall_list.append(recall)
                f_score_list.append(f_score)
                jaccard_list.append(jaccard)
        except json.JSONDecodeError:
            print('Error in parsing the groundtruth JSON. Skipping the current instance.')
            continue
    
    return precision_list, recall_list, f_score_list, jaccard_list

    # Averaging the Precision, Recall, and F-scores
    # save precision_lsit, recall_list, f_score_list    

    # save_dir = 'src/vis/analysis/downstream/search/query_generation/results'
    # os.makedirs(save_dir, exist_ok=True)
    # with open(os.path.join(save_dir, 'precision_list.json'), 'w') as f:
    #     json.dump(precision_list, f)
    # with open(os.path.join(save_dir, 'recall_list.json'), 'w') as f:
    #     json.dump(recall_list, f)
    # with open(os.path.join(save_dir, 'f_score_list.json'), 'w') as f:
    #     json.dump(f_score_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/search/query_generation/results')
    args = parser.parse_args()

    file_dir = f'data/downstream/search/query_generation/results'
    files = glob.glob(f'{file_dir}/*.json')

    for file in files:
        model_name = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            results = json.load(f)

        pred_filename = os.path.join(args.res_dir, f'{model_name}.json')
        with open(pred_filename, 'r') as f:
            preds = json.load(f)
        
        gt_filename = 'data/downstream/search/query_generation/test.json'
        with open(gt_filename, 'r') as f:
            groundtruth = json.load(f)
        
        precision_list, recall_list, f_score_list, jaccard_list = cal_scores(preds, groundtruth)
        
        # save f_score_list
        save_dir = 'src/vis/analysis/downstream/search/query_generation/results/F1'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(f_score_list, f)

        # save precision_list
        save_dir = 'src/vis/analysis/downstream/search/query_generation/results/precision'
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(precision_list, f)
        
        # save recall_list
        save_dir = 'src/vis/analysis/downstream/search/query_generation/results/recall'
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(recall_list, f)
        
        # save jaccard_list
        save_dir = 'src/vis/analysis/downstream/search/query_generation/results/jaccard'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{model_name}.json'), 'w') as f:
            json.dump(jaccard_list, f)
