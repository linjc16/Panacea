import json
import sys
sys.path.append('./')

from src.dataset.matching.patient2trial.trec.data_gen import one_shot_exmaple
from tqdm import tqdm

if __name__ == '__main__':
    filename = 'data/downstream/matching/patient2trial/TREC2021/test_one_shot.json'
    with open(filename, 'r') as f:
        data = json.load(f)
    
    
    new_data = {}
    for key, value in data.items():
        # delete one_shot_exapmle from value['input], delete the string imported from data_gen
        one_shot_exmaple_str = one_shot_exmaple
        new_input = value['input'].replace(one_shot_exmaple_str, '').replace('``````\n\n', '')
        new_data[key] = {'input': new_input, 'label': value['label'], 'nct_id': value['nct_id'], 'patient_id': value['patient_id']}
    
    with open('data/downstream/matching/patient2trial/TREC2021/test.json', 'w') as f:
        json.dump(new_data, f, indent=4)