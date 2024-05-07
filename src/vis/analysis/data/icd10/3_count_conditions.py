import glob
import json



if __name__ == '__main__':
    # data/analysis/icd10/conditions
    filepahts = glob.glob('data/analysis/icd10/conditions/*.json')

    condition_dict = {}

    # merge the counts for all datasets, notice the lower case and upper case, use some string manipulation to make them consistent
    for file in filepahts:
        with open(file, 'r') as f:
            data = json.load(f)
            for condition, count in data.items():
                condition = condition.lower()
                if condition in condition_dict:
                    condition_dict[condition] += count
                else:
                    condition_dict[condition] = count
    
    # sort by count
    condition_dict = dict(sorted(condition_dict.items(), key=lambda x: x[1], reverse=True))

    # save to data/analysis/icd10
    with open('data/analysis/icd10/icd10_conditions.json', 'w') as f:
        json.dump(condition_dict, f, indent=4)