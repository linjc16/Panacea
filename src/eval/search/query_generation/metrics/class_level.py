import argparse
import json
import os
import pdb


def cal_scores(preds, groundtruth, class_name):
    precision_list = []
    recall_list = []
    f_score_list = []
    
    for key, item in groundtruth.items():
        try:
            parsed_dict = json.loads(item['parsed_dict'])

            # only keep the key that is equal to the class_name
            parsed_dict = {k: v for k, v in parsed_dict.items() if k == class_name}

            # if the value is None, skip the current instance
            if not parsed_dict:
                continue
            
            
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
                # only keep the key that is equal to the class_name
                pred_dict = {k: v for k, v in pred_dict.items() if k == class_name}
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
                
                precision_list.append(precision)
                recall_list.append(recall)
                f_score_list.append(f_score)
        except json.JSONDecodeError:
            print('Error in parsing the groundtruth JSON. Skipping the current instance.')
            continue

    # Averaging the Precision, Recall, and F-scores
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    avg_f_score = sum(f_score_list) / len(f_score_list) if f_score_list else 0

    return avg_precision, avg_recall, avg_f_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    parser.add_argument('--res_dir', type=str, default='data/downstream/search/query_generation/results')
    args = parser.parse_args()

    pred_filename = os.path.join(args.res_dir, f'{args.model_name}.json')
    with open(pred_filename, 'r') as f:
        preds = json.load(f)
    
    gt_filename = 'data/downstream/search/query_generation/test.json'
    with open(gt_filename, 'r') as f:
        groundtruth = json.load(f)
    

    for class_name in ['diseases', 'interventions', 'sponsor', 'status', 'phase', 'study_type']:
        avg_precision, avg_recall, avg_f_score = cal_scores(preds, groundtruth, class_name)

        print(f'Class: {class_name}')
        print(f'Model: {args.model_name}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f_score:.4f}')
        print('\n')