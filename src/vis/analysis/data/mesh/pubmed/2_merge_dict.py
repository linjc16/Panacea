import json
import glob

from collections import Counter

if __name__ == '__main__':

    # data/analysis/icd10/mesh/raw
    filepaths = glob.glob('data/analysis/icd10/mesh/raw/*.json')

    mesh_terms = []

    for filepath in filepaths:
        with open(filepath, 'r') as f:
            for line in f:
                output_dict = json.loads(line)
                mesh_terms.extend([term[1] for term in output_dict['mesh_terms']])
    
    # transforme to dict for counting
    mesh_terms = dict(Counter(mesh_terms))
    
    # sort by count
    mesh_terms = dict(sorted(mesh_terms.items(), key=lambda x: x[1], reverse=True))

    # save to data/analysis/icd10/mesh
    with open('data/analysis/icd10/mesh/pubmed_mesh_term_dict.json', 'w') as f:
        json.dump(mesh_terms, f, indent=4)