import json
import pdb
import pandas as pd
from collections import defaultdict

def find_icd10_matches(hierarchy, terms):
    results = {}
    for term in terms:
        term = term.lower()
        term_results = []
        # Traverse the hierarchy at chapter, section, and category levels
        for chapter_key, chapter_val in hierarchy.items():
            chapter_desc = chapter_val.get('description', '')
            if term.lower() in chapter_desc.lower():
                term_results.append({'level': 'chapter', 'key': chapter_key, 'description': chapter_desc})
            
            # Check sections within the chapter
            for section_key, section_val in chapter_val.get('codes', {}).items():
                section_desc = section_val.get('description', '')
                if term.lower() in section_desc.lower():
                    term_results.append({'level': 'section', 'key': section_key, 'description': section_desc})
                
                # Check categories within the section
                for category_key, category_val in section_val.get('codes', {}).items():
                    category_desc = category_val.get('description', '')
                    if term.lower() in category_desc.lower():
                        term_results.append({'level': 'category', 'key': category_key, 'description': category_desc})
                        # Since we only want the first three levels, stop after categories
        results[term] = term_results
    return results


with open('data/analysis/icd10/icd10_tree.json', 'r') as file:
    icd10_tree_refined = json.load(file)


# data/analysis/icd10/icd10_counts.json
with open('data/analysis/icd10/icd10_counts.json', 'r') as f:
    icd10_counts = json.load(f)

terms = list(icd10_counts.keys())

search_results = find_icd10_matches(icd10_tree_refined, terms)

# # save the search results
# with open('data/analysis/icd10/icd10_search_results.json', 'w') as f:
#     json.dump(search_results, f, indent=4)


# icd10_counts, lower case all keys
icd10_counts_new = {key.lower(): value for key, value in icd10_counts.items()}

# for the search results, find the icd code by value[i]['key']
# then keep the first three letters of the key as the icd code
# use the counts in icd10_counts, and merge the counts for the same icd code
# save to a new dict, key is the icd code, value is the count, the chapter and section in icd10_counts, and the description is icd10_code_to_desp_dict[icd_code]
icd10_matching_results = {}

for term, results in search_results.items():
    for result in results[:1]:
        key = result['key']
        icd_code = key[:3]
        if icd_code in icd10_matching_results:
            icd10_matching_results[icd_code]['count'] += icd10_counts[term]['count']
        else:
            icd10_matching_results[icd_code] = {
                'count': icd10_counts[term]['count'],
                'chapter': icd10_counts[term]['chapter'],
                'section': icd10_counts[term]['section'],
                'description': result.get('description', '')
            }
    
# save the icd10_matching_results
with open('data/analysis/icd10/icd10_matching_results.json', 'w') as f:
    json.dump(icd10_matching_results, f, indent=4)

# select the top 100 conditions, the first 100 conditions
top_100_conditions = {k: v for k, v in list(icd10_matching_results.items())[:100]}

chapter_dict = {}
for key, value in top_100_conditions.items():
    chapter = value['chapter']
    section = value['section']

    # chapter[section][key] = value['count']
    if chapter not in chapter_dict:
        chapter_dict[chapter] = {
            'sections': {}
        }
    if section not in chapter_dict[chapter]['sections']:
        chapter_dict[chapter]['sections'][section] = {}
    chapter_dict[chapter]['sections'][section][value['description']] = value['count']


# save chapter_dict and section_dict
with open('data/analysis/icd10/chapter_dict.json', 'w') as f:
    json.dump(chapter_dict, f, indent=4)
