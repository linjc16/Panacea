import json
import re
import pdb

def extract_number(text):
    try:
        number = re.search(r"Number of Criteria: (\d+)", text).group(1)
    except AttributeError:
        try:
            number = re.search(r"\d+", text).group(0)
        except:
            number = 0
    return number



if __name__ == '__main__':

    with open('src/vis/analysis/downstream/matching/results/criteria_counts.json') as file:
        data = json.load(file)

    extracted_data = {}
    for trial_id, criteria in data.items():
        extracted_data[trial_id] = {
            'inclusion': extract_number(criteria['inclusion']),
            'exclusion': extract_number(criteria['exclusion'])
        }
    
    
    pdb.set_trace()