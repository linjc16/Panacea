import json

# Load the JSON file content
with open('data/analysis/icd10/chapter_dict.json', 'r') as file:
    data = json.load(file)

# Prepare data for DataFrame
rows = []
for chapter, sections_data in data.items():
    for section, diseases in sections_data['sections'].items():
        for disease, value in diseases.items():
            # upper case the first letter of the disease
            disease = disease[0].upper() + disease[1:]
            rows.append({
                'name': disease,
                'value': value,
                'chapter': chapter,
                'section': section
            })

# Create DataFrame
import pandas as pd
df = pd.DataFrame(rows)

# Save DataFrame to CSV
csv_path = 'data/analysis/icd10/diseases_data.csv'
df.to_csv(csv_path, index=False)
