import json
import glob
import os
import pdb

def count_codes(data):
    code_counts = {}

    def process_category(key, category):
        # Process the main code
        if 'code' not in category or 'description' not in category:
            return
        try:
            code = category['code']
        except:
            pdb.set_trace()
        description = category['description']
        if code in code_counts:
            code_counts[code]['count'] += icd10_conditions[key]
        else:
            code_counts[code] = {'description': description, 'count': icd10_conditions[key]}

        # Recursively process subcategories
        subcategories = category.get('subcategories', [])
        if isinstance(subcategories, list):
            for sub in subcategories:
                process_category(key, sub)
        elif isinstance(subcategories, dict):
            process_category(key, subcategories)

    # Start the recursive processing
    for key, item in data.items():
        process_category(key, item)
    
    return code_counts

if __name__ == '__main__':
    # data/analysis/icd10/icd10_conditions.json
    with open('data/analysis/icd10/icd10_conditions.json', 'r') as f:
        icd10_conditions = json.load(f)
    
    file_dir = 'data/analysis/icd10/merged'
    save_dir = 'data/analysis/icd10'

    filepaths = glob.glob(f'{file_dir}/*.json')

    icd_merge_dict = {}
    for file in filepaths:
        with open(file, 'r') as f:
            data = json.load(f)
            # all key to lower case
            data = {key.lower(): value for key, value in data.items()}
            icd_merge_dict.update(data)



    icd_10_counts = count_codes(icd_merge_dict)

    # sort by count
    icd_10_counts = dict(sorted(icd_10_counts.items(), key=lambda x: x[1]['count'], reverse=True))

    with open('data/analysis/icd10/icd10_counts.json', 'w') as f:
        json.dump(icd_10_counts, f, indent=4)