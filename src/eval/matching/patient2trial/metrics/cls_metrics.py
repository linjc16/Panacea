import json
import re
import os
import argparse

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score


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
            preds.append(1)
        elif 'eligibility: 2)' in value['output']:
            preds.append(2)
        else:
            preds.append(-1)
        
        labels.append(value['label'])
    
    
    assert len(preds) == len(labels)

    # calculate accuracy using sklearn

    accuracy = accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy:.4f}")

    # calculate precision, recall, f1-score using sklearn and the results using two decimal places

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    
    for cls in [0, 1, 2]:
        if str(cls) in report:
            print(f"Recall for class {cls}: {report[str(cls)]['recall']:.4f}")
            print(f'Precision for class {cls}: {report[str(cls)]["precision"]:.4f}')
            print(f'F1-score for class {cls}: {report[str(cls)]["f1-score"]:.4f}')
    # print weighted accuracy

    balanced_accuracy = balanced_accuracy_score(labels, preds)
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")