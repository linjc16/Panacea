import argparse
import json
import os
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

    # Averaging the Precision, Recall, and F-scores
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    avg_f_score = sum(f_score_list) / len(f_score_list) if f_score_list else 0
    
    return avg_precision, avg_recall, avg_f_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--res_dir', type=str, default='data/downstream/search/query_expansion/results')
    args = parser.parse_args()
    
    pred_filename = os.path.join(args.res_dir, f'{args.model_name}.json')
    with open(pred_filename, 'r') as f:
        preds = json.load(f)
    
    gt_filename = 'data/downstream/search/query_expansion/test.json'
    with open(gt_filename, 'r') as f:
        groundtruth = json.load(f)
    
    avg_precision, avg_recall, avg_f_score = cal_scores(preds, groundtruth)

    print(f'Model: {args.model_name}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f_score:.4f}')