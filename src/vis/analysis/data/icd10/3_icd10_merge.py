import json
import glob
import os


dataset_list = [
    "aus_zealand", "brazil", "dutch", "german", "iran", "isrctn", "japan", "korea", "pan_african", "sri_lanka", "thai",
    'ctgov', 'chictr'
]


if __name__ == '__main__':
    file_dir = 'data/analysis/icd10/raw'

    save_dir = 'data/analysis/icd10/merged'
    
    for dataset in dataset_list:
        file_list = glob.glob(f'{file_dir}/{dataset}/*.json')
        print(f'Processing {dataset} with {len(file_list)} files')

        merged_data = {}
        for file in file_list:
            with open(file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    merged_data[data['condition']] = data['icd10_hierarchy']

        with open(f'{save_dir}/{dataset}.json', 'w') as f:
            json.dump(merged_data, f, indent=4)