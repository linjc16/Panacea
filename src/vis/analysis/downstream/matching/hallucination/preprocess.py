from tqdm import tqdm
import json
import argparse
import pandas as pd
import os
import sys
sys.path.append('./')
from src.dataset.matching.patient2trial.utils import read_trec_qrels

import pdb


def format_input(df_notes, df_criteria, qrels, split='test'):
    inputs = {}
    for idx, sample in enumerate(tqdm((qrels[:]))):
        patient_id, nct_id, label = sample
        # print type of patient_id
        patient_note = df_notes[df_notes['Patient ID'] == int(patient_id)]['Description'].values[0]
        # patient_note_sentences = convert_patient_note_into_sentences(patient_note)
        patient_note_sentences = patient_note
        try:
            inclusion_criteria_curr = df_criteria[df_criteria['nct_id'] == nct_id]['inclusion_criteria'].values[0]
        except:
            inclusion_criteria_curr = ''
        try:
            exclusion_criteria_curr = df_criteria[df_criteria['nct_id'] == nct_id]['exclusion_criteria'].values[0]
        except:
            exclusion_criteria_curr = ''
        try:
            title_curr = df_criteria[df_criteria['nct_id'] == nct_id]['title'].values[0]
        except:
            title_curr = ''
        try:
            target_diseases_curr = "Target diseases: " + df_criteria[df_criteria['nct_id'] == nct_id]['target_diseases'].values[0]
        except:
            target_diseases_curr = ''
        try:
            interventions_curr = "Interventions: " + df_criteria[df_criteria['nct_id'] == nct_id]['interventions'].values[0]
        except:
            interventions_curr = ''
        try:
            summary_curr = "Summary: " + df_criteria[df_criteria['nct_id'] == nct_id]['summary'].values[0]
        except:
            summary_curr = ''
        
        input_clinical_trial = (
            f"Title: {title_curr}\n"
            f"{target_diseases_curr}\n"
            f"{interventions_curr}\n"
            f"{summary_curr}\n"
            f"Inclusion criteria: {inclusion_criteria_curr}\n"
            f"Exclusion criteria: {exclusion_criteria_curr}"
        )

        input_patient_note = (
            f"{patient_note_sentences}"
        )

        inputs[idx] = {
            'patient_notes': input_patient_note,
            'clinical_trial': input_clinical_trial,
        }
    
    return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    df_notes = pd.read_csv('data/downstream/matching/patient2trial/TREC2021/patient_notes.csv')
    df_criteria = pd.read_csv('data/downstream/matching/patient2trial/cohort/criteria.csv')
    if args.split == 'test':
        qrels = read_trec_qrels('data/downstream/matching/patient2trial/TREC2021/qrels-clinical_trials.txt')
    else:
        qrels = read_trec_qrels('data/downstream/matching/patient2trial/TREC2021/qrels-clinical_trials_train.txt')
    
    patient_note_tiral_dict = {}
    
    inputs = format_input(df_notes, df_criteria, qrels, args.split)

    # save to src/vis/analysis/downstream/matching/hallucination/inputs.json
    with open('src/vis/analysis/downstream/matching/hallucination/inputs.json', 'w') as f:
        json.dump(inputs, f, indent=4)
    