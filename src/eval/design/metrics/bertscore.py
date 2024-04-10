from evaluate import load
import argparse
import json
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='data/downstream/design/results/study_arms')
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    args = parser.parse_args()

    bertscore = load("bertscore")

    with open(os.path.join(args.res_dir, f'{args.model_name}.json'), 'r') as f:
        results = json.load(f)

    preds = []
    groundtruth = []
    for key, value in results.items():
        preds.extend(value['model_response'])
        groundtruth.extend(value['groundtruth'])

    assert len(preds) == len(groundtruth)
    
    bert_scores = bertscore.compute(predictions=preds, references=groundtruth, model_type="distilbert-base-uncased", nthreads=256, batch_size=256)
    bert_f1 = bert_scores['f1']
    # average of list
    bert_f1 = sum(bert_f1) / len(bert_f1)

    bert_precision = bert_scores['precision']
    bert_recall = bert_scores['recall']

    bert_precision = sum(bert_precision) / len(bert_precision)
    bert_recall = sum(bert_recall) / len(bert_recall)
    
    print(f'Model: {args.model_name}, BERTScore F1 Score: {bert_f1:.4f}')
    print(f'Model: {args.model_name}, BERTScore Precision: {bert_precision:.4f}')
    print(f'Model: {args.model_name}, BERTScore Recall: {bert_recall:.4f}')