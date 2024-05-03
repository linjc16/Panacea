"""
    https://caseolap.github.io/docs/mesh/mesh/#view-cvd-mesh-tree-online
"""

import json
from collections import Counter
import pdb

def load_mesh_tree(filepath):
    """
    Load the Mesh terms and their corresponding tree numbers from a file into a dictionary.
    
    Args:
    filepath (str): Path to the file containing Mesh terms and tree numbers.
    
    Returns:
    dict: Dictionary with Mesh terms as keys and tree numbers as values.
    """
    mesh_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if ';' in line:
                term, tree_num = line.strip().split(';')
                mesh_dict[term] = tree_num
    return mesh_dict

def get_tree_number(term, mesh_dict):
    """
    Retrieve the tree number for a given Mesh term from a dictionary.
    
    Args:
    term (str): The Mesh term to query.
    mesh_dict (dict): Dictionary with Mesh terms and tree numbers.
    
    Returns:
    str: The tree number if found, otherwise "Term not found".
    """
    return mesh_dict.get(term, "Term not found")


meshtree_file = "data/analysis/icd10/mesh/mtrees2023.bin"


# Load the mesh terms and their tree numbers into a dictionary
mesh_tree_dict = load_mesh_tree(meshtree_file)

# Example usage:
# Query for the tree numbers of some example terms
example_terms = ["Body Regions", "Breast", "Child", "Education, Dental"]
example_results = {term: get_tree_number(term, mesh_tree_dict) for term in example_terms}
print(example_results)

pdb.set_trace()

# Tree = []
# id2name = {}
# name2id = {}
# with open(meshtree_file, "r") as ftree:
#     for line in ftree:
#         term_tree = line.strip().split(";")
#         cur_term = term_tree[0]
#         cur_tree = term_tree[1]

#         id2name.update({cur_tree:cur_term})                        
#         name2id.update({cur_term:cur_tree})
#         Tree.append({'id':cur_tree ,'name':cur_term})
    

# CVDTree = []
# for name,ID in name2id.items():
#     if ID[0:3] == 'H02':
#             CVDTree.append({"name": name, "ID":ID})


# pdb.set_trace()