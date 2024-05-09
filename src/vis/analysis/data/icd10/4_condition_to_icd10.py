import json
import glob
import os
import pdb
import pandas as pd

def count_codes(data):
    code_counts = {}

    def process_category(key, category):
        # Process the main code
        if 'code' not in category or 'description' not in category:
            return
        
        if 'subcategories' not in category:
            try:
                code = category['code']
            except:
                pdb.set_trace()
            description = category['description']
            if description in code_counts:
                code_counts[description]['count'] += icd10_conditions[key]
            else:
                code_counts[description] = {'count': icd10_conditions[key]}

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


def load_icd10_hierarchy():
    # data/analysis/icd10/icd10_diagnosis_hierarchy_2024.csv
    icd10_hierarchy = pd.read_csv('data/analysis/icd10/icd10_diagnosis_hierarchy_2024.csv')
    chapter_section_hierarchy = icd10_hierarchy.groupby(['chapter', 'chapter_desc', 'section', 'section_desc']).size().reset_index(name='counts')

    hierarchy_dict = {}
    for idx, row in chapter_section_hierarchy.iterrows():
        chapter = row['chapter']
        section = row['section']
        if chapter not in hierarchy_dict:
            hierarchy_dict[chapter] = {
                'description': row['chapter_desc'].replace(f'({chapter})', '').strip(),
                'sections': {}
            }
        hierarchy_dict[chapter]['sections'][section] = row['section_desc'].replace(f'({section})', '').strip()

    return hierarchy_dict

def save_icd10_dict():
    icd10_hierarchy = pd.read_csv('data/analysis/icd10/icd10_diagnosis_hierarchy_2024.csv')

    hierarchy_csv_dict = {}
    # key is description, value is all the other columns
    for idx, row in icd10_hierarchy.iterrows():
        hierarchy_csv_dict[row['description']] = {
            'chapter': row['chapter'],
            'chapter_desc': row['chapter_desc'].replace(f'({row["chapter"]})', '').strip(),
            'section': row['section'],
            'section_desc': row['section_desc'].replace(f'({row["section"]})', '').strip(),
        }
    
    # save
    with open('data/analysis/icd10/icd10_hierarchy_csv.json', 'w') as f:
        json.dump(hierarchy_csv_dict, f, indent=4)
    
    return hierarchy_csv_dict

if __name__ == '__main__':
    # data/analysis/icd10/icd10_conditions.json
    with open('data/analysis/icd10/icd10_conditions.json', 'r') as f:
        icd10_conditions = json.load(f)


    if os.path.exists('data/analysis/icd10/icd10_hierarchy_csv.json'):
        with open('data/analysis/icd10/icd10_hierarchy_csv.json', 'r') as f:
            hierarchy_csv_dict = json.load(f)
    else:
        hierarchy_csv_dict = save_icd10_dict()


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
    

    # for each key in icd_10_counts, find the corresponding chapter and section by looking up the hierarchy_csv_dict
    for key, value in icd_10_counts.items():
        description = key
        descp_lower = description.lower()
        if descp_lower in hierarchy_csv_dict:
            chapter = hierarchy_csv_dict[descp_lower]['chapter']
            section = hierarchy_csv_dict[descp_lower]['section']
            icd_10_counts[key]['chapter'] = chapter
            icd_10_counts[key]['section'] = section
            icd_10_counts[key]['lower_case'] = descp_lower
        else:
            # if major depressive disorder, single episode, check major depressive disorder, single episode, unspecified
            try:
                if descp_lower == 'major depressive disorder, single episode':
                    descp_lower = 'major depressive disorder, single episode, unspecified'
                elif descp_lower == 'Human immunodeficiency virus [HIV] disease resulting in malignant neoplasms':
                    descp_lower = 'human immunodeficiency virus [hiv] disease'
                elif descp_lower == 'chronic kidney disease':
                    descp_lower = 'chronic kidney disease (ckd)'
                elif descp_lower == 'seropositive rheumatoid arthritis':
                    descp_lower = 'rheumatoid arthritis with rheumatoid factor'
                elif descp_lower == 'acute lymphoblastic leukemia':
                    descp_lower = 'acute lymphoblastic leukemia [all]'
                chapter = hierarchy_csv_dict[descp_lower]['chapter']
                section = hierarchy_csv_dict[descp_lower]['section']
                icd_10_counts[key]['chapter'] = chapter
                icd_10_counts[key]['section'] = section
                icd_10_counts[key]['lower_case'] = descp_lower
            except:
                icd_10_counts[key]['chapter'] = None
                icd_10_counts[key]['section'] = None

    # delete key with None chapter and section
    icd_10_counts = {key: value for key, value in icd_10_counts.items() if value['chapter'] is not None and value['section'] is not None}

    # for each key that are the same in the lower_case, merge the counts
    icd_10_counts_new = {}
    for key, value in icd_10_counts.items():
        lower_case = value['lower_case']
        if lower_case in icd_10_counts_new:
            icd_10_counts_new[lower_case]['count'] += value['count']
        else:
            icd_10_counts_new[lower_case] = value

    with open('data/analysis/icd10/icd10_counts.json', 'w') as f:
        json.dump(icd_10_counts_new, f, indent=4)
    
    # # select the top 100 conditions, the first 100 conditions
    # top_100_conditions = {k: v for k, v in list(icd_10_counts_new.items())[:100]}

    # chapter_dict = {}
    # for key, value in top_100_conditions.items():
    #     chapter = value['chapter']
    #     section = value['section']

    #     # chapter[section][key] = value['count']
    #     if chapter not in chapter_dict:
    #         chapter_dict[chapter] = {
    #             'sections': {}
    #         }
    #     if section not in chapter_dict[chapter]['sections']:
    #         chapter_dict[chapter]['sections'][section] = {}
    #     chapter_dict[chapter]['sections'][section][key] = value['count']

        

        
    # # save chapter_dict and section_dict
    # with open('data/analysis/icd10/chapter_dict.json', 'w') as f:
    #     json.dump(chapter_dict, f, indent=4)
    

    # # obtain chapter dict, key is chapter, value is chapter description
    # chapter_name_dict = {value['chapter']: value['chapter_desc'] for value in hierarchy_csv_dict.values()}
    
    # with open('data/analysis/icd10/chapter_name_dict.json', 'w') as f:
    #     json.dump(chapter_name_dict, f, indent=4)