import pandas as pd
import json
from collections import defaultdict
import pdb

# Load the CSV file
data = pd.read_csv('data/analysis/icd10/icd10_diagnosis_hierarchy_2024.csv')

# Initialize the dictionary for the ICD-10 hierarchy
icd10_tree_refined = {}


# key icd10_code, value decsription in df (data)
icd10_code_to_desp_dict = defaultdict(str)
icd10_code_to_desp_dict.update(dict(zip(data['icd10_code'], data['description'])))

# Loop through each row of the dataframe to populate the hierarchy
for _, row in data.iterrows():
    chapter = row['chapter']
    if chapter not in icd10_tree_refined:
        icd10_tree_refined[chapter] = {
            'description': row['chapter_desc'].replace(f'({chapter})', '').strip(),
            'codes': {}
        }

    section = row['section']
    if section not in icd10_tree_refined[chapter]['codes']:
        try:
            icd10_tree_refined[chapter]['codes'][section] = {
                'description': icd10_code_to_desp_dict[section].replace(f'({section})', '').strip(),
                'codes': {}
            }
        except:
            icd10_tree_refined[chapter]['codes'][section] = {
                'description': row['section_desc'].replace(f'({section})', '').strip(),
                'codes': {}
            }
    
    category = row['category']
    if category not in icd10_tree_refined[chapter]['codes'][section]['codes']:
        icd10_tree_refined[chapter]['codes'][section]['codes'][category] = {
            'description': icd10_code_to_desp_dict[category].replace(f'({category})', '').strip(),
            'codes': {}
        }

    # Set the path for inserting codes
    current_codes = icd10_tree_refined[chapter]['codes'][section]['codes'][category]['codes']

    # Handle each subcategory, ensuring no code includes itself
    path = [row['subcategory_1'], row['subcategory_2'], row['subcategory_3']]
    # descriptions = [row['subcategory_1_desc'], row['subcategory_2_desc'], row['subcategory_3_desc']]
    descriptions = [icd10_code_to_desp_dict[row['subcategory_1']], icd10_code_to_desp_dict[row['subcategory_2']], icd10_code_to_desp_dict[row['subcategory_3']]]
    target_dict = current_codes

    for i, code in enumerate(path):
        if pd.notna(code) and code != row['icd10_code'] and code not in target_dict:
            target_dict[code] = {
                'description': descriptions[i],
                'codes': {}
            }
            target_dict = target_dict[code]['codes']

    # Add the ICD-10 code to the correct location
    if row['icd10_code'] not in target_dict:
        target_dict[row['icd10_code']] = {
            'description': row['description'],
            'codes': {}
        }

def remove_redundant_entries(hierarchy):
    keys_to_remove = []
    # Check each key in the current level of the dictionary
    for key, value in hierarchy.items():
        if 'codes' in value:
            # Identify if the key itself exists in its 'codes' dictionary
            if key in value['codes']:
                keys_to_remove.append(key)
            # Recursively process the sub-dictionary
            remove_redundant_entries(value['codes'])

    # Remove the identified redundant keys
    for key in keys_to_remove:
        del hierarchy[key]['codes'][key]

remove_redundant_entries(icd10_tree_refined)



# Convert the dictionary to a JSON string
icd10_refined_json = json.dumps(icd10_tree_refined, indent=4)

# Save the JSON to a file
with open('data/analysis/icd10/icd10_tree.json', 'w') as file:
    file.write(icd10_refined_json)

