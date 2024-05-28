import glob
import json
import pdb


if __name__ == '__main__':
    # src/vis/analysis/downstream/matching/results/raw
    # read all criteria_count json files
    criteria_count_dict = {}
    for file in glob.glob('src/vis/analysis/downstream/matching/results/raw/criteria_count_*.json'):
        with open(file, 'r') as f:
            criteria_count_dict.update(json.load(f))
    

    # save to src/vis/analysis/downstream/matching/results  
    with open('src/vis/analysis/downstream/matching/results/criteria_count.json', 'w') as f:
        json.dump(criteria_count_dict, f, indent=4)