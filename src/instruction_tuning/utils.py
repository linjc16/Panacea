from datasets import Dataset
import sys
import os
import pdb
from datasets import load_dataset, DatasetDict, concatenate_datasets

sys.path.append('./')
from src.finetune.utils import (
    load_single_trial_summarization_data,
    load_multi_trial_summarization_data,
    load_query_generation_data,
    load_query_expansion_data,
    load_trial_design_data,
    load_patient2trial_data,
)

data_path = {
    'single-trial summarization': 'data/downstream/summazization/single-trial/train.csv',
    'multi-trial summarization': 'data/downstream/summazization/multi-trial/train.json',
    'query generation': 'data/downstream/search/query_generation/train.json',
    'query expansion': 'data/downstream/search/query_expansion/train.json',
    'trial design': 'data/downstream/design/parsed',
    'patient2trial': 'data/downstream/matching/patient2trial/TREC2021/train.json'
}

fracs_dict = {
    'single-trial summarization': 1.0,
    'multi-trial summarization': 1.0,
    'query generation': 0.6,
    'query expansion': 1.0,
    'trial design': 1.0,
    'patient2trial': 1.0,
    'ultra_chat': 1.0
}

def load_all_data(shuffle=True):
    # single-trial summarization
    data_list_sing_ts = load_single_trial_summarization_data(data_path['single-trial summarization'])
    raw_datasets_sing_ts = convert_to_dataset(data_list_sing_ts)

    # multi-trial summarization
    data_list_multi_ts = load_multi_trial_summarization_data(data_path['multi-trial summarization'])
    raw_datasets_multi_ts = convert_to_dataset(data_list_multi_ts)

    # query generation
    data_list_query_gen = load_query_generation_data(data_path['query generation'])
    raw_datasets_query_gen = convert_to_dataset(data_list_query_gen)

    # query expansion
    data_list_query_exp = load_query_expansion_data(data_path['query expansion'])
    raw_datasets_query_exp = convert_to_dataset(data_list_query_exp)

    # trial design
    data_list_trial_design = []
    task_names = ['criteria', 'study_arms', 'outcome_measures']
    filepaths = [os.path.join(data_path['trial design'], task_name, 'train.json') for task_name in task_names]
    data_list_trial_design = []
    for filepath in filepaths:
        data_list_trial_design.extend(load_trial_design_data(filepath))
    raw_datasets_trial_design = convert_to_dataset(data_list_trial_design)
    
    # patient2trial
    data_list_patient2trial = load_patient2trial_data(data_path['patient2trial'])
    raw_datasets_patient2trial = convert_to_dataset(data_list_patient2trial)

    raw_datasets_ultra_chat_ori = load_dataset('HuggingFaceH4/ultrachat_200k') 
    # change the split names, train_sft, test_sft -> train, test
    new_splits = {
        'train': raw_datasets_ultra_chat_ori['train_sft'],
        'test': raw_datasets_ultra_chat_ori['test_sft']
    }
    raw_datasets_ultra_chat = DatasetDict(new_splits)
    # remove 'prompt', 'prompt_id' columns
    raw_datasets_ultra_chat = raw_datasets_ultra_chat.remove_columns(['prompt', 'prompt_id'])

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    
    raw_train_datasets.append(raw_datasets_sing_ts['train'])
    raw_train_datasets.append(raw_datasets_multi_ts['train'])
    raw_train_datasets.append(raw_datasets_query_gen['train'])
    raw_train_datasets.append(raw_datasets_query_exp['train'])
    raw_train_datasets.append(raw_datasets_trial_design['train'])
    raw_train_datasets.append(raw_datasets_patient2trial['train'])
    raw_train_datasets.append(raw_datasets_ultra_chat['train'])

    raw_val_datasets.append(raw_datasets_sing_ts['test'])
    raw_val_datasets.append(raw_datasets_multi_ts['test'])
    raw_val_datasets.append(raw_datasets_query_gen['test'])
    raw_val_datasets.append(raw_datasets_query_exp['test'])
    raw_val_datasets.append(raw_datasets_trial_design['test'])

    fracs = [fracs_dict['single-trial summarization'], fracs_dict['multi-trial summarization'], fracs_dict['query generation'], fracs_dict['query expansion'], fracs_dict['trial design'], fracs_dict['patient2trial'], fracs_dict['ultra_chat']]

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    return raw_datasets
    

def convert_to_dataset(data_list):
    data_dict = {'messages': data_list}
    raw_datasets = Dataset.from_dict(data_dict, split='train')
    raw_datasets = raw_datasets.train_test_split(test_size=0.15, seed=42)
    return raw_datasets

if __name__ == '__main__':
    datasets = load_all_data()
    print(datasets)
    pdb.set_trace()
