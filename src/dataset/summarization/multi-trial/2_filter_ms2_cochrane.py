from datasets import load_dataset
import json
from tqdm import tqdm

import pdb


def filter_data(ms2_data, pubmed_ids):
    filtered_data = []
    word_count = []
    count = 0
    for i in tqdm(range(len(ms2_data))):
        pmids_curr = ms2_data[i]['pmid']
        # check if the pmids are in the pubmed_ids
        pmids_curr_new = list(set(pmids_curr) & pubmed_ids)
        if len(pmids_curr_new) > 2 or len(pmids_curr_new) == len(pmids_curr) and len(pmids_curr) > 1:
            # count the number of word in abstract + target + background
            text_list = []
            text_list.extend(ms2_data[i]['abstract'])
            text_list.append(ms2_data[i]['target'])
            text = '\n'.join(text_list)
            

            if len(text.split()) > 8000 or len(text.split()) < 1000:
                continue
            word_count.append(len(text.split()))
            filtered_data.append(ms2_data[i])
            count += 1
    
    print('data count:', count)
    print('average word count:', sum(word_count) / len(word_count))
    print('max word count:', max(word_count))
    print('min word count:', min(word_count))

    return filtered_data

if __name__ == '__main__':
    ms2_data = load_dataset('allenai/mslr2022', 'cochrane', cache_dir='/data/linjc/cache')

    pubmed_ids = json.load(open('data/downstream/summazization/multi-trial/pubmed_ids.json', 'r'))
    pubmed_ids = set(pubmed_ids)

    
    # construct training data
    ms2_train_data = ms2_data['train']

    print('Constructing training data')
    filtered_train_data = filter_data(ms2_train_data, pubmed_ids)

    # save to json dict
    filtered_train_data_dict = {}
    for i in range(len(filtered_train_data)):
        filtered_train_data_dict[i] = filtered_train_data[i]
    
    with open('data/downstream/summazization/multi-trial/train.json', 'w') as f:
        json.dump(filtered_train_data_dict, f, indent=4)

    # construct test data, use the validation set as my test set
    ms2_val_data = ms2_data['validation']
    filtered_val_data = filter_data(ms2_val_data, pubmed_ids)
    
    filtered_val_data_dict = {}
    for i in range(len(filtered_val_data)):
        filtered_val_data_dict[i] = filtered_val_data[i]
    
    with open('data/downstream/summazization/multi-trial/test.json', 'w') as f:
        json.dump(filtered_val_data_dict, f, indent=4)