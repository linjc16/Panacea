import os
import glob
import pdb
import argparse
from tqdm import tqdm
import pandas as pd
import json

def process_markdown_table(markdown_text):
    rows = markdown_text.split('\n')
    processed_rows = []

    for row in rows:
        if '|' in row:
            cells = row.split('|')
            processed_cells = [cell.replace('\n', '<br>') for cell in cells]
            processed_row = '|'.join(processed_cells)
            processed_rows.append(processed_row)
        else:
            processed_rows.append(row)

    return '\n'.join(processed_rows)

def ctgov_dict_to_text(ctgov_dict):
    
    # Breif Title
    # \section{Study Overview}
    # \subsection{Brief Summary}
    # \subsection{Detailed Description}
    # \subsection{Official Title}
    # \subsection{Conditions}
    # \subsection{Intervention / Treatment}
    #
    # \section{Participation Criteria}
    # \subsection{Eligibility Criteria}
    # \subsection{Ages Eligible for Study}
    # \subsection{Sexes Eligible for Study}
    # \subsection{Accepts Healthy Volunteers}
    #
    # \section{Study Plan}
    # \subsection{How is the study designed?}
    # \subsubsection{Design Details}
    # \subsubsection{Arms and Interventions}
    # \subsection{What is the study measuring?}
    # \subsubsection{Primary Outcome Measures}
    # \subsubsection{Secondary Outcome Measures}
    #
    # \section{Terms related to the study}
    # \subsection{Keywords Provided by Centre Hospitalier Valida}
    # \subsection{Additional Relevant MeSH Terms}

    def check_value(value):
        if not pd.isna(value) and value != '':
            return True
        else:
            return False

    text = ""
    text += ctgov_dict['brief_title'] + "\n\n" if check_value(ctgov_dict['brief_title']) else ""
    text += "Study Overview\n=================\n"
    text += "Brief Summary\n-----------------\n" + ctgov_dict['brief_summary'] + "\n" if check_value(ctgov_dict['brief_summary']) else ""
    text += "\nDetailed Description\n-----------------\n" + ctgov_dict['detailed_descriptions'] + "\n" if check_value(ctgov_dict['detailed_descriptions']) else ""
    text += "\nOfficial Title\n-----------------\n" + ctgov_dict['official_title'] + "\n" if check_value(ctgov_dict['official_title']) else ""
    text += "\nConditions\n-----------------\n" + ctgov_dict['conditions'] + "\n" if check_value(ctgov_dict['conditions']) else ""
    text += "\nIntervention / Treatment\n-----------------\n" + ctgov_dict['interventions'] + "\n" if check_value(ctgov_dict['interventions']) else ""
    
    text += '\n'
    text += "Participation Criteria\n=================\n"
    text += "Eligibility Criteria\n-----------------\n" + ctgov_dict['eligibility_criteria'] + "\n" if check_value(ctgov_dict['eligibility_criteria']) else ""

    age_text = ""
    # check if is 'nan'
    if not pd.isna(ctgov_dict['minimum_age']):
        age_text += "Minimum Age: " + ctgov_dict['minimum_age'] + "\n"
    if not pd.isna(ctgov_dict['maximum_age']):
        age_text += "Maximum Age: " + ctgov_dict['maximum_age'] + "\n"
    
    text += "\nAges Eligible for Study\n-----------------\n" + age_text + "\n" if check_value(age_text) else ""
    text += "Sexes Eligible for Study\n-----------------\n" + ctgov_dict['gender'] + '\n' if check_value(ctgov_dict['gender']) else ""
    text += "\nAccepts Healthy Volunteers\n-----------------\n" + ctgov_dict['healthy_volunteers'] + "\n" if check_value(ctgov_dict['healthy_volunteers']) else ""

    text += "\n"
    text += "Study Plan\n=================\n"
    text += "How is the study designed?\n-----------------\n"
    text += "\nDesign Details\n\n" + ctgov_dict['design_details'] + "\n" if check_value(ctgov_dict['design_details']) else ""
    
    
    text += "\nArms and Interventions\n\n" + process_markdown_table(ctgov_dict['arms_and_interventions']) + "\n" if check_value(ctgov_dict['arms_and_interventions']) else ""
    # text_arms = ""
    # if check_value(ctgov_dict['arms_and_interventions']):
    #     """ if only == the following, skip
    #         | Participant Group/Arm | Intervention/Treatment |
    #         | --- | --- |
    #     """
    #     if ctgov_dict['arms_and_interventions'] != "| Participant Group/Arm | Intervention/Treatment |\n| --- | --- |\n":
    #         text_arms = "\\subsubsection{Arms and Interventions}\n" + ctgov_dict['arms_and_interventions'] + "\n"
    #     else:
    #         count[0] += 1
    # text += text_arms
    
    
    text += "What is the study measuring?\n-----------------\n"
    text += "Primary Outcome Measures\n\n" + process_markdown_table(ctgov_dict['primary_outcome_measures']) + "\n" if check_value(ctgov_dict['primary_outcome_measures']) else ""
    text += "Secondary Outcome Measures\n\n" + process_markdown_table(ctgov_dict['secondary_outcome_measures']) + "\n" if check_value(ctgov_dict['secondary_outcome_measures']) else ""

    text += " "
    text += "Terms related to the study\n=================\n"
    text += "Keywords Provided by Centre Hospitalier Valida\n-----------------\n" + ctgov_dict['keywords'] + "\n" if check_value(ctgov_dict['keywords']) else ""
    # text += "\\subsection{Additional Relevant MeSH Terms}\n" + ctgov_dict['mesh_terms'] + "\n" if ctgov_dict['mesh_terms'] else ""

    text = text.replace('~', ' ')

    return text


def load_ctgov(file_dir, split):
    filepaths = glob.glob(os.path.join(file_dir, split, '*.json'))
    # for each file, load the json
    # each row is a dict

    output_list = []
    for filepath in tqdm(filepaths[:]):
        with open(filepath, 'r') as f:
            # each row is a dict, read line by line
            for line in f:
                ctgov_dict = json.loads(line)
                output_list.append(ctgov_dict_to_text(ctgov_dict))
    
    

    return output_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, default='/data/linjc/trialfm/ctgov_fixed_md/merged')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    
    output_list = load_ctgov(args.file_dir, args.split)

    pdb.set_trace()