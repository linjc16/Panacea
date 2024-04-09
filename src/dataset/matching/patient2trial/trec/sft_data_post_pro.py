import json
import glob
from collections import defaultdict



if __name__ == '__main__':
    file_paths = glob.glob('data/downstream/matching/patient2trial/TREC2021/raw/sft_data/*')
    output_dict = {}

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            output_dict.update(data)
    
    label_count = defaultdict(int)
    train_data = {}
    for key, value in output_dict.items():
        # extract pred from value['output']
        if 'eligibility: 0)' in value['output']:
            preds = 0
        elif 'eligibility: 1)' in value['output']:
            preds= 1
        elif 'eligibility: 2)' in value['output']:
            preds = 2
        else:
            preds = -1
        
        if preds == value['label']:
            if value['label'] == 0 and label_count[0] > 3000: 
                continue
            label_count[value['label']] += 1
            train_data[key] = value
    
    # print the number of correct predictions
    print(f"Number of correct predictions: {len(train_data)}")

    # count the number of 0,1,2 in the labels
    labels = [value['label'] for value in train_data.values()]



    with open('data/downstream/matching/patient2trial/TREC2021/train.json', 'w') as f:
        json.dump(train_data, f, indent=4)