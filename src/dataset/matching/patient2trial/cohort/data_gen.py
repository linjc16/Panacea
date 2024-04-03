from tqdm import tqdm
import json
import pandas as pd
import pdb



one_shot_exmaple = (
"Here is an example patient note:\n"
"Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine complicated by severe lower "
"extremity weakness and urinary retention s/p Foley catheter, high-dose steroids, hypertension, and chronic pain. "
"The tumor is located in the T-L spine, unresectable anaplastic astrocytoma s/p radiation. "
"Complicated by progressive lower extremity weakness and urinary retention. "
"Patient initially presented with RLE weakness where his right knee gave out with difficulty walking and right anterior "
"thigh numbness. "
"MRI showed a spinal cord conus mass which was biopsied and found to be anaplastic astrocytoma. "
"Therapy included field radiation t10-l1 followed by 11 cycles of temozolomide 7 days on and 7 days off. "
"This was followed by CPT-11 Weekly x4 with Avastin Q2 weeks/ 2 weeks rest and repeat cycle.\n"
"Here is an example clinical trial:\n"
"Title: Is the Severity of Urinary Disorders Related to Falls in People With Multiple Sclerosis\n"
"Target diseases: Fall, Multiple Sclerosis, Lower Urinary Tract Symptoms\n"
"Interventions: Clinical tests\n"
"Summary: Falls are a common problem in people with multiple sclerosis (PwMS) and can lead to severe "
"consequences (trauma, fear of falling, reduction of social activities). Prevention of falls is one of the priority targets "
"of rehabilitation for PwMS and walking difficulties, which can result of different factors (motor impairment, ataxia, "
"sensitive disorders, fatigability…). Urinary incontinence has been evoked as predictive of falls. But lower urinary "
"tract symptoms (LUTSs) are frequent in PwMS, the prevalence of LUTSs is high (32-96.8%) and increases with MS "
"duration and severity of neurological deficiencies and disabilities. Overactive bladder (OAB) is the most common "
"symptom. Despite its high prevalence and impact on quality of life, the severity of LUTSs has never been studied as "
"specific risk factor of falling. However, urinary urgency and urinary incontinence could lead to precipitation and thus "
"could increase the risk of falling in these patients.~The aim of the study was to assess the relationship between "
"severity of LUTSs and risk of falling in PwMS.~Patients were asked about the number of falls in the past three "
"months and in the past year, and the circumstances in which they occurred (frequency, home, outdoors, going to "
"void, during urinary urgency, nocturia). Severity of LUTSs were assessed by the Urinary Symptoms Profile (USP) "
"Score and patient were classified as with or without urinary incontinence. Number of micturition by night were "
"specifically asked. To take into account motor difficulties and fear of falling, other clinical evaluations were done. "
"The impact of MS on walking was assessed by the 12-Item Multiple Sclerosis Walking Scale (MSWS12) "
"questionnaire, the Expanded Disability Status Scale score, and by clinical test with the Time to be Ready to Void "
"(TRV). Fear of falling was assessed by a simple question and with Falls Efficacy Scale-International (FES-I) "
"Questionnaire.~The primary aim was to assess the relationship between severity of LUTSs and occurrence of falls "
"during the past 3 months. The primary outcome was the importance of overactive bladder (OAB) symptoms with "
"OAB USP score. The secondary outcomes were the existence of urinary incontinence, the warning time (defined as "
"the time from the first sensation of urgency to voiding or incontinence), the importance of nocturia and the other "
"scores of USP questionnaire (low stream and stress urinary incontinence).~The secondary aims were to look for the "
"relationship between severity of LUTSs and occurrence of falls during the past year, and to assess the relationship "
"between falls and the classical risk factors of falls.\n"
"Inclusion criteria: inclusion criteria: "
"age ≥ 18 years, "
"Multiple sclerosis (MS) diagnosis, "
"Lower urinary tract symptoms with or without treatment, "
"Expanded Disability Status Scale score between 1 and 6.5\n"
"Example trial-level eligibility: 0) Would not refer this patient for this clinical trial.\n"
)

PROMPT = (
    "Hello. You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the "
    "inclusion criteria of a clinical trial to determine the patient's eligibility. "
    "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on "
    "characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other "
    "medical conditions."
    "\n\n"
    "The assessment of eligibility has a three-point scale: " 
    "0) Would not refer this patient for this clinical trial; "
    "1) Would consider referring this patient to this clinical trial upon further investigation; and "
    "2) Highly likely to refer this patient for this clinical trial. \n"
    "You should make a trial-level eligibility on each patient for the clinical trial, i.e., output the scale for the assessment of eligibility. "
    "\n\n```"
    f"{one_shot_exmaple}"
    "```\n\n"
)

input_patient_note_prefix = (
    "Here is the patient note:\n"
)
input_clinical_trial_prefix = (
    "Here is the clinical trial: \n"
)


def format_input(df_notes, df_criteria, qrels):
    inputs = {}
    for idx, sample in enumerate(tqdm((qrels[:]))):
        patient_id, nct_id, label = sample
        # print type of patient_id
        patient_note = df_notes[df_notes['Patient ID'] == int(patient_id)]['Description'].values[0]
        # patient_note_sentences = convert_patient_note_into_sentences(patient_note)
        patient_note_sentences = patient_note
        try:
            criteria_curr = df_criteria[df_criteria['nct_id'] == nct_id]['inclusion_criteria'].values[0]
            criteria_curr = criteria_curr.replace('~', '\n')
        except:
            criteria_curr = ''
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
            f"{input_clinical_trial_prefix}"
            f"Title: {title_curr}\n"
            f"{target_diseases_curr}\n"
            f"{interventions_curr}\n"
            f"{summary_curr}\n"
            f"Inclusion criteria: {criteria_curr}\n"
        )

        input_patient_note = (
            f"{input_patient_note_prefix}"
            f"{patient_note_sentences}"
        )

        # output_prefix = (
        #     "\n"
        #     "Let's think step by step. \n"
        #     "Finally, you should always repeat Trial-level eligibility in the last line by `Trial-level eligibility: `, e.g., `Trial-level eligibility: 2) Highly likely to refer this patient for this clinical trial.`.\n"
        # )

        output_prefix = (
            "\n"
            "Finally, you only need to output the Trial-level eligibility in the last line with `Trial-level eligibility: `, e.g., `Trial-level eligibility: 2) Highly likely to refer this patient for this clinical trial.`.\n"
        )

        input = PROMPT + input_patient_note + input_clinical_trial + output_prefix
        inputs[idx] = {
            'patient_id': patient_id,
            'nct_id': nct_id,
            'input': input,
            'label': label
        }
    
    with open('data/downstream/matching/patient2trial/cohort/test.json', 'w') as f:
        json.dump(inputs, f, indent=4, sort_keys=True)

    return inputs

if __name__ == '__main__':
    df_notes = pd.read_csv('data/downstream/matching/patient2trial/cohort/patient_notes.csv')
    df_criteria = pd.read_csv('data/downstream/matching/patient2trial/cohort/criteria.csv')

    with open('data/downstream/matching/patient2trial/cohort/qrels.json', 'r') as f:
        qrels = json.load(f)

    inputs = format_input(df_notes, df_criteria, qrels)