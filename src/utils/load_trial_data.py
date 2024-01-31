import pandas as pd
import os
import glob
import pdb
import re
from tqdm import tqdm

def load_dfs(filepaths: list):
    dfs = []
    for filepath in filepaths:
        dfs.append(pd.read_csv(filepath))
    df = pd.concat(dfs)
    return df

def load_aus_zealand(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'aus_zealand*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns = ['Public title', 'Scientific title', 'Description of intervention(s) / exposure',
                'Study type', 'Intervention code', 'Comparator / control treatment',
                'Control group', 'Key inclusion criteria', 'Key exclusion criteria',
                'Brief summary', 'Health condition(s) or problem(s) studied', 'Condition category',
                'Primary outcome', 'Secondary outcome']

    full_text_list = []

    # for each row, rearange the data as the following format
    # for each column name in columns, add "\section{column name} \n {content}" into text if column name is not empty 
    for index, row in tqdm(df.iterrows()):
        try:
            text = ''
            for column in columns:
                if not pd.isna(row[column]):
                    # if column == Primary outcome or Secondary outcome, remove "; Timepoint: "
                    if column == 'Primary outcome' or column == 'Secondary outcome':
                        row[column] = row[column].replace('; Timepoint: ', '')
                    if column == 'Key inclusion criteria':
                        # add Minimum age, Maximum age, and Sex behind Key inclusion criteria
                        # Format like: "Minimum age: , Maximum age: , Sex: "
                        # if it is empty, don't add it
                        if not pd.isna(row['Minimum age']):
                            row[column] += f"\nMinimum age: {row['Minimum age']}"
                        if not pd.isna(row['Maximum age']):
                            row[column] += f"\nMaximum age: {row['Maximum age']}"
                        if not pd.isna(row['Sex']):
                            row[column] += f"\nSex: {row['Sex']}"
                    
                    # if column content contains nil, skip this column
                    if 'nil' in row[column] or 'Nil' in row[column]:
                        continue
                    text += f'{column}\n=================\n{row[column]}\n\n'
            full_text_list.append(text.strip())
        except:
            print(f'skip this row {index}')
            continue
    
    return full_text_list

def load_brazil(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'brazil*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns_dict = {
        'trial.main.public_title': 'Public title',
        'trial.main.scientific_title': 'Scientific title',
        'trial.main.study_type': 'Study type',
        'trial.main.study_design': 'Study design',
        'trial.main.i_freetext': 'Description',
        'trial.criteria.inclusion_criteria': 'Inclusion criteria',
        'trial.criteria.exclusion_criteria': 'Exclusion criteria',
        # 'trial.criteria.agemin': 'Age min',
        # 'trial.criteria.agemax': 'Age max',
        # 'trial.criteria.gender': 'Gender',
        'trial.primary_outcome.prim_outcome': 'Primary outcome',
        'trial.secondary_outcome.sec_outcome': 'Secondary outcome',
    }

    full_text_list = []

    # for each row, rearange the data as the following format
    # for each column name in columns, add "\section{column name} \n {content}" into text if column name is not empty

    for index, row in tqdm(df.iterrows()):
        try:
            text = ''
            for column in columns_dict.keys():
                if not pd.isna(row[column]):
                    if column == 'trial.criteria.inclusion_criteria':
                        if not pd.isna(row['trial.criteria.agemin']):
                            row[column] += f"\nMinimum age: {row['trial.criteria.agemin']}"
                        if not pd.isna(row['trial.criteria.agemax']):
                            row[column] += f"\nMaximum age: {row['trial.criteria.agemax']}"
                        if not pd.isna(row['trial.criteria.gender']):
                            row[column] += f"\nGender: {row['trial.criteria.gender']}"
                    
                    text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
            full_text_list.append(text.strip())
        except:
            print(f'skip this row {index}')
            continue
    
    return full_text_list

def load_chictr(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'chictr*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns_dict = {
        'main.public_title': 'Public title',
        'main.scientific_title': 'Scientific title',
        'main.target_size': 'Target size',
        'main.study_type': 'Study type',
        'main.study_design': 'Study design',
        "criteria.inclusion_criteria": "Inclusion criteria",
        # "criteria.agemin": "Age min",
        # "criteria.agemax": "Age max",
        # "criteria.gender": "Gender",
        "criteria.exclusion_criteria": "Exclusion criteria",
        "primary_outcome.prim_outcome": "Primary outcome",
        "secondary_outcome.sec_outcome": "Secondary outcome",
    }

    full_text_list = []

    # for each row, rearange the data as the following format
    # for each column name in columns, add "\section{column name} \n {content}" into text if column name is not empty

    for index, row in tqdm(df.iterrows()):
        try:
            text = ''
            for column in columns_dict.keys():
                if not pd.isna(row[column]):
                    if column == 'criteria.inclusion_criteria':
                        if not pd.isna(row['criteria.agemin']):
                            row[column] += f"\nMinimum age: {row['criteria.agemin']}"
                        if not pd.isna(row['criteria.agemax']):
                            row[column] += f"\nMaximum age: {row['criteria.agemax']}"
                        if not pd.isna(row['criteria.gender']):
                            row[column] += f"\nGender: {row['criteria.gender']}"
                    
                    text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
            full_text_list.append(text.strip())
        except:
            print(f'skip this row {index}')
            continue
    
    return full_text_list

def load_dutch(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'dutch*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns_dict = {
        'titles.title.public_titles': 'Public title',
        'titles.title.scientific_title': 'Scientific title',
        'main.study_type': 'Study type',
        'criterias.criteria.inclusion_criteria': 'Inclusion criteria',
        'criterias.criteria.exclusion_criteria': 'Exclusion criteria',
        'primary_outcomes.primaryOutcome.prim_outcome': 'Primary outcome',
        'secondary_outcomes.secondaryOutcome.sec_outcome': 'Secondary outcome',
        "abstract": "Abstract",
        # "abstracts.abstract.background": "Background",
        # 'abstracts.abstract.objective': 'Objective',
        # 'abstracts.abstract.design': 'Design',
        # 'abstracts.abstract.intervention': 'Intervention',
    }

    full_text_list = []

    for index, row in tqdm(df.iterrows()):
        try:
            text = ''
            for column in columns_dict.keys():
                if column == 'abstract':
                    abs_text = ""
                    if not pd.isna(row['abstracts.abstract.background']):
                        abs_text += f"\nBackground\n-----------------\n{row['abstracts.abstract.background']}\n"
                    if not pd.isna(row['abstracts.abstract.objective']):
                        abs_text += f"\nObjective\n-----------------\n{row['abstracts.abstract.objective']}\n"
                    if not pd.isna(row['abstracts.abstract.design']):
                        abs_text += f"\nDesign\n-----------------\n{row['abstracts.abstract.design']}\n"
                    if not pd.isna(row['abstracts.abstract.intervention']):
                        abs_text += f"\nIntervention\n-----------------\n{row['abstracts.abstract.intervention']}\n"
                    text += f'{columns_dict[column]}\n=================\n{abs_text}\n\n'
                elif not pd.isna(row[column]):
                    text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
            
            # replace <br /> with \n
            text = text.replace('<br />', '')
            full_text_list.append(text.strip())
        except:
            print(f'skip this row {index}')
            continue
    
    return full_text_list

def load_euctr(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'euctr*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns = ['Main text']

    full_text_list = []
    for index, row in tqdm(df.iterrows()):
        # try:
            text = ''
            for column in columns:
                if not pd.isna(row[column]):
                    text += row[column]
            # replace 'Sponsor 1' with ""
            text = text.replace('Sponsor 1', '')
            full_text_list.append(text.strip())
        # except:
        #     print(f'skip this row {index}')
        #     continue

    return full_text_list

def load_german(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'german*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns = [
        'Brief summary in lay language',
        'Brief summary in scientific language',
        'Health condition or problem studied',
        'Interventions, Observational Groups',
        'Endpoints',
        'Study Design',
        'Recruitment',
    ]

    full_text_list = []

    for index, row in tqdm(df.iterrows()):
        try:
            text = ''
            for column in columns:
                if not pd.isna(row[column]):
                    text += f'{column}\n=================\n{row[column]}\n\n'
            full_text_list.append(text.strip())
        except:
            print(f'skip this row {index}')
            continue

    return full_text_list

def load_iran(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'iran*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns_dict = {
        'public_title': 'Public title',
        'scientific_title': 'Scientific title',
        'study_type': 'Study type',
        'study_design': 'Study design',
        'Intervention': 'Intervention',
        'target_size': 'Target size',
        'inclusion_criteria': 'Inclusion criteria',
        # 'agemin': 'Age min',
        # 'agemax': 'Age max',
        # 'gender': 'Gender',
        'exclusion_criteria': 'Exclusion criteria',
        'prim_outcome': 'Primary outcome',
        'sec_outcome': 'Secondary outcome',
    }

    full_text_list = []

    for index, row in tqdm(df.iterrows()):
        text = ''
        for column in columns_dict.keys():
            if not pd.isna(row[column]):
                if column == 'inclusion_criteria':
                    # row[column] = str(row[column])
                    # add agemin, agemax, gender before exclusion criteria if any, else add it to the end

                    # first find 'exclusion criteria' in the text
                    exclusion_criteria_index = row[column].find('exclusion criteria')
                    if exclusion_criteria_index == -1:
                        exclusion_criteria_index = row[column].find('Exclusion criteria')
                    if exclusion_criteria_index == -1:
                        exclusion_criteria_index = row[column].find('Exclusion Criteria')

                    text_before = ""
                    if not pd.isna(row['agemin']):
                        text_before += f"\nMinimum age: {str(row['agemin'])}"
                    if not pd.isna(row['agemax']):
                        text_before += f"\nMaximum age: {str(row['agemax'])}"
                    if not pd.isna(row['gender']):
                        text_before += f"\nGender: {row['gender']}"

                    if exclusion_criteria_index != -1:
                        # if any, add before exclusion criteria
                        text_before = row[column][:exclusion_criteria_index] + text_before
                        # merge text_before and text_after
                        text_after = row[column][exclusion_criteria_index:]

                        row[column] = text_before + '\n\n' + text_after
                    else:
                        # add to the end
                        row[column] += text_before
                        

                text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
        full_text_list.append(text.strip())

    return full_text_list


def load_isrctn(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'isrctn*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns = [
        'Title',
        'Scientific title',
        'Plain English Summary',
        'Study hypothesis',
        'Study design',
        'Primary study design',
        'Secondary study design',
        'Study setting(s)',
        'Study type',
        'Condition',
        'Intervention',
        'Intervention type',
        'Primary outcome measure',
        'Secondary outcome measures',
        'Participant inclusion criteria',
        # 'Participant type(s)',
        # 'Age group',
        # 'Sex',
        # 'Target number of participants',
        'Participant exclusion criteria',
    ]


    not_provided = [
        'Not provided at time of registration',
        'Not provided at time of registration or empty'
    ]

    def extract_points_for_cretiria(text):
        # Split the text by numbers followed by a period and a space, which seems to be the pattern for each point
        points = re.split(r'\d+\.\s', text)
        
        # The first split is always empty, so we remove it
        if points[0] == '':
            points = points[1:]

        # Prepend numbers to each point
        formatted_points = [f"{i+1}. {points[i].strip()}" for i in range(len(points))]
        
        return "\n".join(formatted_points)

    def extract_points_robust_v2(text):
        # Split the text into sections if they contain headings
        sections = re.split(r'(?<=\D)(?=\n[A-Z])', text)

        # Initialize a list to hold the formatted sections
        formatted_sections = []

        # Apply the point extraction for each section
        for section in sections:
            # Split the section by numbers followed by a period, without a space
            points = re.split(r'(?<=\D)(?=\d+\.)', section)
            # Join the points back with a newline separator
            formatted_section = "\n".join(point.strip() for point in points)
            # Add the formatted section to the list
            formatted_sections.append(formatted_section)

        # Join the formatted sections with two newlines as a separator
        return "\n\n".join(formatted_sections)
    
    full_text_list = []

    for index, row in tqdm(df.iterrows()):
        # try:
            text = ''
            for column in columns:
                if not pd.isna(row[column]) and not row[column] in not_provided:
                    if column == 'Participant inclusion criteria':
                        row[column] = extract_points_for_cretiria(row[column])
                        if not pd.isna(row['Participant type(s)']):
                            row[column] += f"\nParticipant type(s): {row['Participant type(s)']}"
                        if not pd.isna(row['Age group']):
                            row[column] += f"\nAge group: {row['Age group']}"
                        if not pd.isna(row['Sex']):
                            row[column] += f"\nSex: {row['Sex']}"
                        if not pd.isna(row['Target number of participants']):
                            row[column] += f"\nTarget number of participants: {str(row['Target number of participants'])}"
                        # split the inclusion criteria, such as artery2. Patients -> artery\n2. Patients, any number
                    
                    if column == 'Participant exclusion criteria':
                        row[column] = extract_points_for_cretiria(row[column])

                    if column == 'Secondary outcome measures' or column == 'Primary outcome measure':
                        row[column] = extract_points_robust_v2(row[column])
                    
                    text += f'{column}\n=================\n{row[column]}\n\n'
            full_text_list.append(text.strip())
        # except:
        #     print(f'skip this row {index}')
        #     continue
    
    return full_text_list


def load_japan(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'japan*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns = [
        'Health condition(s) or Problem(s) studied',
        'Intervention(s)',
        'Study type',
        'Include criteria',
        # 'Age minimum',
        # 'Age maximum',
        # 'Gender',
        'Exclude criteria',
        'Primary Outcome',
        'Secondary Outcome',
    ]

    full_text_list = []

    for index, row in tqdm(df.iterrows()):
        text = ''
        for column in columns:
            if not pd.isna(row[column]):
                if column == 'Include criteria':
                    # add age gender
                    if not pd.isna(row['Age minimum']):
                        row[column] += f"\nMinimum age: {row['Age minimum']}"
                    if not pd.isna(row['Age maximum']):
                        row[column] += f"\nMaximum age: {row['Age maximum']}"
                    if not pd.isna(row['Gender']):
                        row[column] += f"\nGender: {row['Gender']}"
                text += f'{column}\n=================\n{row[column]}\n\n'
        full_text_list.append(text.strip())
    
    return full_text_list

def load_korea(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'korea*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns_dict = {
        'Public/Brief Title': 'Public/Brief Title',
        'Scientific Title': 'Scientific Title',
        'Study Summary': 'Study Summary',
        'Study Design': 'Study Design',
        'Inclusion Criteria': 'Inclusion Criteria',
        'Exclusion Criteria': 'Exclusion Criteria',
        'PrimaryOutcome(s)': 'Primary Outcome(s)',
        'SecondaryOutcome(s)': 'Secondary Outcome(s)',
    }

    full_text_list = []

    for index, row in tqdm(df.iterrows()):
        text = ''
        for column in columns_dict.keys():
            if not pd.isna(row[column]):
                text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
        full_text_list.append(text.strip())
    
    return full_text_list

def load_pan_african(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'pan_african*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]
    
    intervention_dict = {
        'InterventionType': 'Intervention Type',
        'InterventionName': 'Intervention Name',
        'InterventionDose': 'Intervention Dose',
        'InterventionDuration': 'Intervention Duration',
        'InterventionDescription': 'Intervention Description',
        'InterventionGroupSize': 'Intervention Group Size',
        'InterventionControlNature': 'Intervention Control Nature',
    }

    outcome_dict = {
        'OutcomeType': 'Outcome Type',
        'OutCome': 'Outcome',
        'TimePoint': 'Time Point',
    }

    columns_dict = {
        'PublicTitle': 'Public Title',
        'OfficialScientificTitle': 'Official Scientific Title',
        'Description': 'Description',
        'TrialType': 'Trial Type',
        'TrialPhase': 'Trial Phase',
        'Purpose': 'Purpose',
        'Diseases': 'Diseases',
        'StudyDesignAssignment': 'Study Design Assignment',
        'StudyDesignAllocation': 'Study Design Allocation',
        'StudyDesignSequence': 'Study Design Sequence',
        'Intervention': intervention_dict,
        'Inclusion': 'Inclusion Criteria',
        # 'MinAge': 'Minimum Age', # 'MinAgeType'
        # 'MaxAge': 'Maximum Age', # 'MaxAgeType'
        # 'Gender': 'Gender',
        'Exclusion': 'Exclusion Criteria',
        'AgeGroup': 'Age Group',
        'Outcome_Measure': outcome_dict,
    }

    full_text_list = []

    for _, row in tqdm(df.iterrows()):
        text = ''
        for column in columns_dict.keys():
            if column == 'Intervention':
                text_iv = ''
                for iv_column in intervention_dict.keys():
                    if not pd.isna(row[iv_column]):
                        text_iv += f'{intervention_dict[iv_column]}\n-----------------\n{row[iv_column]}\n\n'
                text += f'{column}\n=================\n{text_iv}\n\n'
            elif column == 'Outcome_Measure':
                text_outcome = ''
                for outcome_column in outcome_dict.keys():
                    if not pd.isna(row[outcome_column]):
                        text_outcome += f'{outcome_dict[outcome_column]}\n-----------------\n{row[outcome_column]}\n\n'
                text += f'{column}\n=================\n{text_outcome}\n\n'

            elif column == 'Inclusion':
                # add MinAge + MinAgeType, MaxAge + MaxAgeType
                if not pd.isna(row['MinAge']):
                    row[column] += f'\nMinimum Age: {row["MinAge"]} {row["MinAgeType"]}'
                if not pd.isna(row['MaxAge']):
                    row[column] += f'\nMaximum Age: {row["MaxAge"]} {row["MaxAgeType"]}'
                if not pd.isna(row['Gender']):
                    row[column] += f'\nGender: {row["Gender"]}'
                text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
            elif not pd.isna(row[column]):
                text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
        full_text_list.append(text.strip())

    return full_text_list

def load_sri_lanka(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'sri_lanka*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns_dict = {
        'Scientific Title of Trial': 'Scientific Title of Trial',
        'Public Title of Trial': 'Public Title of Trial',
        'Disease or Health Condition(s) Studied': 'Disease or Health Condition(s) Studied',
        'What is the research question being addressed?': 'What is the research question being addressed?',
        'Type of study': 'Type of study',
        'Control': 'Control',
        'Assignment': 'Assignment',
        'Purpose': 'Purpose',
        'Study Phase': 'Study Phase',
        'Intervention(s) planned': 'Intervention(s) planned',
        'Inclusion criteria': 'Inclusion criteria',
        'Exclusion criteria': 'Exclusion criteria',
        'Primary outcome(s)': 'Primary outcome(s)',
        'Secondary outcome(s)': 'Secondary outcome(s)',
        'Target number/sample size': 'Target number/sample size',
    }

    full_text_list = []

    for index, row in tqdm(df.iterrows()):
        text = ''
        for column in columns_dict.keys():
            if not pd.isna(row[column]):
                text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
        full_text_list.append(text.strip())

    return full_text_list

def load_thai(file_dir, split='train'):
    filepaths = glob.glob(os.path.join(file_dir, 'thai*.csv'))
    df = load_dfs(filepaths)
    df = df[df['split'] == split]

    columns_dict = {
        'public_title': 'Public Title',
        'scientific_title': 'Scientific Title',
        'study_type': 'Study Type',
        'study_design': 'Study Design',
        'phase': 'Phase',
        'inclusion_criteria': 'Inclusion Criteria',
        # 'agemin': 'Age Min',
        # 'agemax': 'Age Max',
        # 'gender': 'Gender',
        'target_size': 'Target Size',
        'exclusion_criteria': 'Exclusion Criteria',
        'prim_outcome': 'Primary Outcome',
        'sec_outcome': 'Secondary Outcome',
    }

    full_text_list = []

    for index, row in tqdm(df.iterrows()):
        text = ''
        for column in columns_dict.keys():
            if not pd.isna(row[column]):
                if column == 'inclusion_criteria':
                    # add age gender
                    if not pd.isna(row['agemin']):
                        row[column] += f"\nMinimum age: {row['agemin']}"
                    if not pd.isna(row['agemax']):
                        row[column] += f"\nMaximum age: {row['agemax']}"
                    if not pd.isna(row['gender']):
                        row[column] += f"\nGender: {row['gender']}"
                text += f'{columns_dict[column]}\n=================\n{row[column]}\n\n'
        full_text_list.append(text.strip())

    return full_text_list

if __name__ == '__main__':
    file_dir = '/data/linjc/ctr_crawl/0_final_data/trials'

    # output = load_thai(file_dir)
    # output = load_sri_lanka(file_dir)
    # output = load_pan_african(file_dir)
    # output = load_korea(file_dir)
    # output = load_japan(file_dir)
    # output = load_isrctn(file_dir)
    # output = load_iran(file_dir)
    # output = load_german(file_dir)
    # output = load_euctr(file_dir)
    # output = load_dutch(file_dir)
    # output = load_chictr(file_dir)
    # output = load_brazil(file_dir)
    output = load_aus_zealand(file_dir)
    print(output[0])

    pdb.set_trace()