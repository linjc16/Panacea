import json
import glob
from collections import Counter

# data/analysis/icd10/mesh/raw/embase

filepaths = glob.glob('data/analysis/icd10/mesh/raw/embase/*.json')

mesh_terms = []

for filepath in filepaths:
    with open(filepath, 'r') as f:
        for line in f:
            output_dict = json.loads(line)
            mesh_terms.extend(output_dict['mesh_terms']['mesh_terms'])

# transforme to dict for counting
mesh_terms = dict(Counter(mesh_terms))

# sort by count
mesh_terms = dict(sorted(mesh_terms.items(), key=lambda x: x[1], reverse=True))

# save to data/analysis/icd10/mesh
with open('data/analysis/icd10/mesh/embase_mesh_term_dict.json', 'w') as f:
    json.dump(mesh_terms, f, indent=4)