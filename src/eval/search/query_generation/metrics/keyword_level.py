import argparse
import json
import os
import pdb


def cal_scores(preds, groundtruth):
    precision_list = []
    recall_list = []
    f_score_list = []
    
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

                for k, v in pred_dict.items():
                    if k not in ['start_year', 'end_year']:
                        if isinstance(v, list):
                            pred_keywords.update([v.lower() for v in v if v.lower() != "n/a"])
                        else:
                            if v.lower() != "n/a":
                                pred_keywords.add(v.lower())
                    else:
                        if v['YEAR'] != 0:
                            pred_keywords.add(str(int(v['YEAR'])))
                            pred_keywords.add(str(v['OPERATOR']))
                
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
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--res_dir', type=str, default='data/downstream/search/query_generation/results')
    args = parser.parse_args()

    pred_filename = os.path.join(args.res_dir, f'{args.model_name}.json')
    with open(pred_filename, 'r') as f:
        preds = json.load(f)
    
    gt_filename = 'data/downstream/search/query_generation/test.json'
    with open(gt_filename, 'r') as f:
        groundtruth = json.load(f)
    
    avg_precision, avg_recall, avg_f_score = cal_scores(preds, groundtruth)

    print(f'Model: {args.model_name}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f_score:.4f}')