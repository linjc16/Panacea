import json
import re
import os
import argparse

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--res_dir', type=str, default='/data/linjc/trialfm/downstream/summarization/results')
    parser.add_argument('--dataset', type=str, default='cohort')
    args = parser.parse_args()

    args.res_dir = os.path.join(args.res_dir, f'{args.model_name}.json')

    with open(args.res_dir, 'r') as f:
        results = json.load(f)

    # extract pred from value['output']
    preds = []
    labels = []
    for key, value in results.items():
        # extract pred from value['output']
        if 'eligibility: 0)' in value['output']:
            preds.append(0)
        elif 'eligibility: 1)' in value['output']:
            preds.append(0)
        elif 'eligibility: 2)' in value['output']:
            preds.append(1)
        else:
            preds.append(-1)
        

        if value['label'] == 1 or value['label'] == 2:
            labels.append(1)
        else:
            labels.append(0)
    
    
    assert len(preds) == len(labels)

    # calculate accuracy using sklearn

    accuracy = accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy:.4f}")

    # calculate precision, recall, f1-score using sklearn and the results using two decimal places

    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    # output F1-score, not weighted average
    print(f"F1-score: {report['1']['f1-score']:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall: {report['1']['recall']:.4f}")
    
    data_temp = []
    for cls in [0, 1]:
        if str(cls) in report:
            print(f"Recall for class {cls}: {report[str(cls)]['recall']:.4f}")
            print(f'Precision for class {cls}: {report[str(cls)]["precision"]:.4f}')
            print(f'F1-score for class {cls}: {report[str(cls)]["f1-score"]:.4f}')

            data_temp.append(f'{report[str(cls)]["f1-score"]:.4f}')
    
    # print(','.join(data_temp))
    # print weighted accuracy

    balanced_accuracy = balanced_accuracy_score(labels, preds)
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    
    # calculate kappa score
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(labels, preds)
    print(f"Kappa: {kappa:.4f}")

    # calculate Krippendorff's alpha-reliability