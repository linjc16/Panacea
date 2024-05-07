import json
import glob
import os
import pdb



if __name__ == '__main__':
    # data/analysis/icd10/icd10_counts.json
    with open('data/analysis/icd10/icd10_counts.json', 'r') as f:
        icd10_counts = json.load(f)
    
    