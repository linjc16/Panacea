import json
import pandas as pd
import pdb
from tqdm import tqdm

import sys
sys.path.append('./')
from src.dataset.matching.patient2trial.utils import read_trec_qrels

from src.utils.claude_aws import chat_sonnet

import multiprocessing


def worker(inputs):
    process_id, nct_ids_union = inputs


    try:
        # read existing criteria_count json files from src/vis/analysis/downstream/matching/results/raw/ process_id
        with open(f'src/vis/analysis/downstream/matching/results/raw/criteria_count_{process_id}.json', 'r') as f:
            criteria_count_dict = json.load(f)
    except:
        criteria_count_dict = {}
        
    i = 0
    for nct_id in tqdm(nct_ids_union):
        if nct_id in criteria_count_dict:
            continue
        if nct_id not in criteria_dict:
            continue
        inclusion = criteria_dict[nct_id]['inclusion']
        exclusion = criteria_dict[nct_id]['exclusion']

        # count the number of criteria, inclusion and exclusion separately

        # inclusion
        input_str = inclusion

        attempts = 0
        while attempts < 10:
            try:
                response = chat_sonnet(prompt.format(input=input_str))
                criteria_count_dict[nct_id] = {'inclusion': response}
                break
            except:
                attempts += 1
                continue
        

        # exclusion
        input_str = exclusion

        attempts = 0
        while attempts < 10:
            try:
                response = chat_sonnet(prompt.format(input=input_str))
                criteria_count_dict[nct_id]['exclusion'] = response
                break
            except:
                attempts += 1
                continue
                
        
        if i % 100 == 0:
            with open(f'src/vis/analysis/downstream/matching/results/raw/criteria_count_{process_id}.json', 'w') as f:
                json.dump(criteria_count_dict, f, indent=4)

        i += 1


    with open(f'src/vis/analysis/downstream/matching/results/raw/criteria_count_{process_id}.json', 'w') as f:
        json.dump(criteria_count_dict, f, indent=4)



if __name__ == '__main__':
    df_criteria = pd.read_csv('data/downstream/matching/patient2trial/cohort/criteria.csv')

    # transform to a dict, key is nct_id, value is a dict with 'inclusion' and 'exclusion' keys
    criteria_dict = {}
    for i in range(len(df_criteria)):
        nct_id = df_criteria.iloc[i]['nct_id']
        inclusion = df_criteria.iloc[i]['inclusion_criteria']
        exclusion = df_criteria.iloc[i]['exclusion_criteria']

        criteria_dict[nct_id] = {'inclusion': inclusion, 'exclusion': exclusion}

    
    qrels_trec2021 = read_trec_qrels('data/downstream/matching/patient2trial/TREC2021/qrels-clinical_trials.txt')
    qrel_sigir = read_trec_qrels('data/downstream/matching/patient2trial/cohort/qrels-clinical_trials.txt')
    

    # obtain the set of nct_ids in both, each element in the list is like ('20154', 'NCT02109419', 1)
    nct_ids_trec2021 = set([x[1] for x in qrels_trec2021])
    nct_ids_sigir = set([x[1] for x in qrel_sigir])

    # union
    nct_ids_union = nct_ids_trec2021.union(nct_ids_sigir)



    

    # for each nct_id, use sonnet to obtain the number of criteria
    prompt = (
        "Given a clinical trial eligibility criteria, please count the number of criteria.\n"
        "For example, ``(i) adult patients (aged 18 years or order); (ii) patients with suspected or newly diagnosed or previously treated malignant tumors...``"
        "The number of criteria is 2.\n\n"
        "Eligibility Criteria:\n"
        "{input}\n\n"
        "Directly output the number of criteria.\n"
        "Number of Criteria: "   
    )

    
    num_process = 10
    nct_ids_union = list(nct_ids_union)

    nct_ids_union_split = []
    chunk_size = len(nct_ids_union) // num_process

    for i in range(num_process):
        if i == num_process - 1:
            nct_ids_union_split.append((i, nct_ids_union[i * chunk_size:]))
        else:
            nct_ids_union_split.append((i, nct_ids_union[i * chunk_size: (i + 1) * chunk_size]))
    
    with multiprocessing.Pool(num_process) as p:
        p.map(worker, nct_ids_union_split)
    # worker(nct_ids_union_split[0])
