import pandas as pd
import glob
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import os
import json
import pdb

def load_data(ctgov_dir='data/downstream/design/ctgov'):
    filepaths = glob.glob(os.path.join(ctgov_dir, '*.txt'))
    file_dict = {}
    
    for filepath in tqdm(filepaths):
        # load data all str
        df = pd.read_csv(filepath, sep='|', dtype=str)
        filename = filepath.split('/')[-1].split('.')[0]
        file_dict[filename] = df
    
    return file_dict

def split_data(file_dict, split='train'):
    df_cal_vals = file_dict['calculated_values']
    if split == 'train':
        nct_ids = df_cal_vals[df_cal_vals['registered_in_calendar_year'] <= '2022']['nct_id'].tolist()
    else:
        nct_ids_test = df_cal_vals[df_cal_vals['registered_in_calendar_year'] > '2022']['nct_id'].tolist()
        with open('data/downstream/design/nctid_test.txt', 'r') as f:
            nct_ids_regeneron = f.read().splitlines()
            # only keep the nct_ids that are in the test set
            nct_ids_regeneron_after_2023 = []
            for nct_id in nct_ids_regeneron:
                if nct_id in nct_ids_test:
                    nct_ids_regeneron_after_2023.append(nct_id)
        
        # randomly select 600 from nct_ids_regeneron and merge with nct_ids_regeneron_after_2023
        nct_ids = list(set(nct_ids_regeneron_after_2023 + nct_ids_test[:600]))
    
    print(f'Number of nct_ids in {split}: {len(nct_ids)}')
    
    return nct_ids

def extract_nct_data(nct_id, file_dict):
    data_dict = {}
    
    data_dict['nct_id'] = nct_id
    try:
        data_dict['phase'] = file_dict['studies'][file_dict['studies']['nct_id'] == nct_id]['phase'].values[0]
    except:
        data_dict['phase'] = ''
    
    try:
        data_dict['study_type'] = file_dict['studies'][file_dict['studies']['nct_id'] == nct_id]['study_type'].values[0]
    except:
        data_dict['study_type'] = ''

    try:
        data_dict['brief_title'] = file_dict['studies'][file_dict['studies']['nct_id'] == nct_id]['brief_title'].values[0]
    except:
        data_dict['brief_title'] = ''
    try:
        data_dict['brief_summary'] = file_dict['brief_summaries'][file_dict['brief_summaries']['nct_id'] == nct_id]['description'].values[0]
    except:
        data_dict['brief_summary'] = ''
    try:
        data_dict['detailed_descriptions'] = file_dict['detailed_descriptions'][file_dict['detailed_descriptions']['nct_id'] == nct_id]['description'].values[0]
    except:
        data_dict['detailed_descriptions'] = ''
    try:
        data_dict['official_title'] = file_dict['studies'][file_dict['studies']['nct_id'] == nct_id]['official_title'].values[0]
    except:
        data_dict['official_title'] = ''
    # for conditions, concat all the conditions with ', '
    data_dict['conditions'] = ', '.join(file_dict['conditions'][file_dict['conditions']['nct_id'] == nct_id]['name'].values)

    # for interventions, follow the format: * intervention_type: name\n* intervention_type: name
    interventions = file_dict['interventions'][file_dict['interventions']['nct_id'] == nct_id]
    interventions_str = ''
    for index, row in interventions.iterrows():
        interventions_str += '* ' + row['intervention_type'] + ': ' + row['name'] + '\n'
    data_dict['interventions'] = interventions_str

    try:
        data_dict['eligibility_criteria'] = file_dict['eligibilities'][file_dict['eligibilities']['nct_id'] == nct_id]['criteria'].values[0]
    except:
        data_dict['eligibility_criteria'] = ''
    try:
        data_dict['minimum_age'] = file_dict['eligibilities'][file_dict['eligibilities']['nct_id'] == nct_id]['minimum_age'].values[0]
    except:
        data_dict['minimum_age'] = ''
    try:
        data_dict['maximum_age'] = file_dict['eligibilities'][file_dict['eligibilities']['nct_id'] == nct_id]['maximum_age'].values[0]
    except:
        data_dict['maximum_age'] = ''
    try:
        data_dict['gender'] = file_dict['eligibilities'][file_dict['eligibilities']['nct_id'] == nct_id]['gender'].values[0]
    except:
        data_dict['gender'] = ''
    try:
        data_dict['healthy_volunteers'] = file_dict['eligibilities'][file_dict['eligibilities']['nct_id'] == nct_id]['healthy_volunteers'].values[0]
    except:
        data_dict['healthy_volunteers'] = ''
    # design details
    # Primary Purpose: \n Allocation: \n Intervention Model: \n Interventional Model Description: \n Masking: \n
    # if the value is nan, then don't shown the prefix, like don't show 'Primary Purpose:' if the value is nan
    design = file_dict['designs'][file_dict['designs']['nct_id'] == nct_id]
    design_str = ''
    for index, row in design.iterrows():
        if not pd.isna(row['primary_purpose']):
            design_str += 'Primary Purpose: ' + row['primary_purpose'] + '\n'
        if not pd.isna(row['allocation']):
            design_str += 'Allocation: ' + row['allocation'] + '\n'
        if not pd.isna(row['intervention_model']):
            design_str += 'Intervention Model: ' + row['intervention_model'] + '\n'
        if not pd.isna(row['intervention_model_description']):
            design_str += 'Interventional Model Description: ' + row['intervention_model_description'] + '\n'
        if not pd.isna(row['masking']):
            design_str += 'Masking: ' + row['masking'] + '\n'
    data_dict['design_details'] = design_str

    # for primary outcome measures and secondary outcome measures, follow the table format
    # | Outcome Measure | Measure Description | Time Frame |
    # | --- | --- | --- |
    # |  |  |  |
    # |  |  |  |

    primary_outcome_measures = file_dict['design_outcomes'][(file_dict['design_outcomes']['nct_id'] == nct_id) & (file_dict['design_outcomes']['outcome_type'].str.lower() == 'primary')]
    primary_outcome_measures_str = '| Outcome Measure | Measure Description | Time Frame |\n| --- | --- | --- |\n'
    for index, row in primary_outcome_measures.iterrows():
        # if the value is nan, then don't show the prefix
        if not pd.isna(row['measure']):
            primary_outcome_measures_str += '| ' + row['measure'] + ' | '
        else:
            primary_outcome_measures_str += '|  | '
        if not pd.isna(row['description']):
            primary_outcome_measures_str += row['description'] + ' | '
        else:
            primary_outcome_measures_str += ' | '
        if not pd.isna(row['time_frame']):
            primary_outcome_measures_str += row['time_frame'] + ' | '
        else:
            primary_outcome_measures_str += ' | '
        primary_outcome_measures_str += '\n'
    
        
    data_dict['primary_outcome_measures'] = primary_outcome_measures_str

    secondary_outcome_measures = file_dict['design_outcomes'][(file_dict['design_outcomes']['nct_id'] == nct_id) & (file_dict['design_outcomes']['outcome_type'].str.lower() == 'secondary')]
    secondary_outcome_measures_str = '| Outcome Measure | Measure Description | Time Frame |\n| --- | --- | --- |\n' if len(secondary_outcome_measures) > 0 else ''
    for index, row in secondary_outcome_measures.iterrows():
        # if the value is nan, then don't show the prefix
        if not pd.isna(row['measure']):
            secondary_outcome_measures_str += '| ' + row['measure'] + ' | '
        else:
            secondary_outcome_measures_str += '|  | '
        if not pd.isna(row['description']):
            secondary_outcome_measures_str += row['description'] + ' | '
        else:
            secondary_outcome_measures_str += ' | '
        if not pd.isna(row['time_frame']):
            secondary_outcome_measures_str += row['time_frame'] + ' | '
        else:
            secondary_outcome_measures_str += ' | '
        secondary_outcome_measures_str += '\n'
    

    data_dict['secondary_outcome_measures'] = secondary_outcome_measures_str

    # keywords, follow the format keyword1, keyword2, keyword3
    keywords = file_dict['keywords'][file_dict['keywords']['nct_id'] == nct_id]
    keywords_str = ''
    for index, row in keywords.iterrows():
        keywords_str += row['name'] + ', '
    data_dict['keywords'] = keywords_str[:-2]

    # mesh terms, same as keywords
    mesh_terms = file_dict['browse_interventions'][file_dict['browse_interventions']['nct_id'] == nct_id]
    mesh_terms_str = ''
    for index, row in mesh_terms.iterrows():
        mesh_terms_str += row['mesh_term'] + ', '
    data_dict['mesh_terms'] = mesh_terms_str[:-2]
    if data_dict['mesh_terms'] == '':
        mesh_terms = file_dict['browse_conditions'][file_dict['browse_conditions']['nct_id'] == nct_id]
        mesh_terms_str = ''
        for index, row in mesh_terms.iterrows():
            mesh_terms_str += row['mesh_term'] + ', '
        data_dict['mesh_terms'] = mesh_terms_str[:-2]
        

    # get intervention dictionary
    # id as key, contains a dict, with keys: intervention_type, name, description
    # also load intervention_other_names, add new keys to above dict: intervention_other_names.name.
    intervention_dict = {}
    interventions = file_dict['interventions'][file_dict['interventions']['nct_id'] == nct_id]
    interventions_other_names = file_dict['intervention_other_names'][file_dict['intervention_other_names']['nct_id'] == nct_id]
    for index, row in interventions.iterrows():
        intervention_dict[row['id']] = {'intervention_type': row['intervention_type'], 'name': row['name'], 'description': row['description']}

    for index, row in interventions_other_names.iterrows():
        # update the intervention_dict
        intervention_dict[row['intervention_id']]['intervention_other_names'] = row['name']

    # for Arms and Interventions, follow the table format
    # | Participant Group/Arm | Intervention/Treatment |
    # | --- | --- |
    # | group_type: title\ndescription | intervention_type: name\n* description\n Other names:\n- intervention_other_names |

    design_groups = file_dict['design_groups'][file_dict['design_groups']['nct_id'] == nct_id]
    design_group_interventions = file_dict['design_group_interventions'][file_dict['design_group_interventions']['nct_id'] == nct_id]
    design_groups_str = '| Participant Group/Arm | Intervention/Treatment |\n| --- | --- |\n'
    for index, row in design_groups.iterrows():
        # design_groups_str += '| ' + row['group_type'] + ': ' + row['title'] + '\n' + row['description'] + ' | '
        # if the value is nan, then don't show the prefix
        if not pd.isna(row['group_type']):
            design_groups_str += '| ' + row['group_type'] + ': '
        else:
            design_groups_str += '| '
        if not pd.isna(row['title']):
            design_groups_str += row['title'] + '<br>'
        else:
            design_groups_str += ' <br> '
        if not pd.isna(row['description']):
            design_groups_str += row['description'] + ' | '
        else:
            design_groups_str += ' | '
        
        # get the intervention ids
        intervention_ids = design_group_interventions[design_group_interventions['design_group_id'] == row['id']]['intervention_id'].tolist()
        for intervention_id in intervention_ids:
            # design_groups_str += intervention_dict[intervention_id]['intervention_type'] + ': ' + intervention_dict[intervention_id]['name'] + '\n* ' + intervention_dict[intervention_id]['description'] + '\n'
            if not pd.isna(intervention_dict[intervention_id]['intervention_type']):
                design_groups_str += intervention_dict[intervention_id]['intervention_type'] + ': '
            else:
                design_groups_str += ' '
            if not pd.isna(intervention_dict[intervention_id]['name']):
                design_groups_str += intervention_dict[intervention_id]['name'] + '<br>'
            else:
                design_groups_str += ' <br> '
            if not pd.isna(intervention_dict[intervention_id]['description']):
                design_groups_str += '* ' + intervention_dict[intervention_id]['description'] + '<br>'
            else:
                design_groups_str += ' <br> '
            
            if 'intervention_other_names' in intervention_dict[intervention_id]:
                design_groups_str += '* Other names: ' + intervention_dict[intervention_id]['intervention_other_names'] + ';'
        design_groups_str += '|\n'

    data_dict['arms_and_interventions'] = design_groups_str

    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train or test')
    parser.add_argument('--num_process', type=int, default=32, help='number of process')
    args = parser.parse_args()
    
    file_dict = load_data()
    nct_ids = split_data(file_dict, args.split)

    # use multiprocessing to extract data
    # each process strore the dict into a file


    save_dir = 'data/downstream/design/raw/selected_step1'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    
    
    def extract_data(input):
        nct_ids, split, process_id = input
        for nct_id in tqdm(nct_ids):
            try:
                data_dict = extract_nct_data(nct_id, file_dict)
                with open(os.path.join(save_dir, split, f'ctgov_{process_id}.json'), 'a') as f:
                    f.write(json.dumps(data_dict) + '\n')
            except:
                continue
    
    num_process = args.num_process
    
    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]
    
    nct_ids_chunks = chunkify(nct_ids, num_process)
    
    # add args.split to the input
    input_data = [(nct_ids_chunks[i], args.split, i) for i in range(num_process)]
    
    with Pool(num_process) as p:
        p.map(extract_data, input_data)
