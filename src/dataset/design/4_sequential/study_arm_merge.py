import os
import sys
import json
import argparse
import random
import re
import glob

sys.path.append('./')
from tqdm import tqdm
import multiprocessing as mp

import pdb


if __name__ == '__main__':
    criteria_path = 'data/downstream/design/parsed/criteria/test.json'
    study_arm_path = 'data/downstream/design/parsed/study_arms/test.json'
    outcome_measure_path = 'data/downstream/design/parsed/outcome_measures/test.json'

    with open(criteria_path, 'r') as f:
        criteria_data = json.load(f)
    
    with open(study_arm_path, 'r') as f:
        study_arm_data = json.load(f)

    with open(outcome_measure_path, 'r') as f:
        outcome_measure_data = json.load(f)
    
    # get the intersection of the keys
    keys = set(criteria_data.keys()) & set(study_arm_data.keys()) & set(outcome_measure_data.keys())


    study_arm_data_new = {}
    for key in tqdm(keys):
        study_arm_conversation = study_arm_data[key]
        Exist = False
        # find 'criteria' in the conversation by user role, replace the whole with [Criteria Output from Last]
        for i in range(0, len(study_arm_conversation), 2):
            if 'criteria' in study_arm_conversation[i]['content']:
                Exist = True
                study_arm_conversation[i]['content'] = '[Criteria Output from Last]'
        if not Exist:
            # insert to the second user message
            study_arm_conversation[4]['content'] = study_arm_conversation[4]['content'] + ' [Criteria Output from Last]'
        
        study_arm_data_new[key] = study_arm_conversation

    save_path = 'data/downstream/design/parsed/sequential/study_arms/test.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(study_arm_data_new, f, indent=4)
