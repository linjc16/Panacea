import json
import glob
import os
import pdb

def count_codes(data):
    code_counts = {}

    def process_category(category):
        # Process the main code
        if 'code' not in category or 'description' not in category:
            return
        try:
            code = category['code']
        except:
            pdb.set_trace()
        description = category['description']
        if code in code_counts:
            code_counts[code]['count'] += 1
        else:
            code_counts[code] = {'description': description, 'count': 1}

        # Recursively process subcategories
        subcategories = category.get('subcategories', [])
        if isinstance(subcategories, list):
            for sub in subcategories:
                process_category(sub)
        elif isinstance(subcategories, dict):
            process_category(subcategories)

    # Start the recursive processing
    for item in data.values():
        process_category(item)

    return code_counts

if __name__ == '__main__':

    file_dir = 'data/analysis/icd10/merged'
    save_dir = 'data/analysis/icd10'

    filepahts = glob.glob(f'{file_dir}/*.json')
    
    icd_10_counts = {} # key is the code, value is a dictionary with description and count

    for file in filepahts:
        with open(file, 'r') as f:
            data = json.load(f)
            icd_10_counts[file.split('/')[-1].split('.')[0]] = count_codes(data)

    icd_10_merged = {}

    # merge the counts for all datasets
    for dataset, counts in icd_10_counts.items():
        for code, data in counts.items():
            if code in icd_10_merged:
                icd_10_merged[code]['count'] += data['count']
            else:
                icd_10_merged[code] = data
    
    # rank the codes by key
    icd_10_merged = dict(sorted(icd_10_merged.items(), key=lambda x: x[0]))
    # remove those not start with a letter
    icd_10_merged = {k: v for k, v in icd_10_merged.items() if k[0].isalpha()}

    # rank the codes by count
    icd_10_merged = dict(sorted(icd_10_merged.items(), key=lambda x: x[1]['count'], reverse=True))

    # sum all the counts to get the total count
    total_count = sum([v['count'] for v in icd_10_merged.values()])
    print(f'Total count: {total_count}')
    
    with open(f'{save_dir}/icd10_counts.json', 'w') as f:
        json.dump(icd_10_merged, f, indent=4)