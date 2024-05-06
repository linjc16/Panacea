import sys
import os
import json

sys.path.append('./')

from src.vis.analysis.data.mesh.ui_to_tree_id import mesh_tree_dict, get_tree_number

if __name__ == '__main__':
    with open('data/analysis/icd10/mesh/pubmed_mesh_term_dict.json', 'r') as f:
        mesh_terms = json.load(f)

    
    mesh2tree = {} # key: mesh term, value: dict with {count, tree_num}

    for mesh_term in mesh_terms:
        tree_num = get_tree_number(mesh_term, mesh_tree_dict)
        mesh2tree[mesh_term] = {'count': mesh_terms[mesh_term], 'tree_num': tree_num}
    

    # only select the mesh terms starting with 'C'
    mesh2tree = {k: v for k, v in mesh2tree.items() if v['tree_num'].startswith('C')}
    mesh2tree = dict(sorted(mesh2tree.items(), key=lambda x: x[1]['count'], reverse=True))

    # sum the counts of all 
    total_count = sum([v['count'] for v in mesh2tree.values()])
    print(f'Total count: {total_count}')
    
    with open('data/analysis/icd10/mesh/pubmed_mesh2tree.json', 'w') as f:
        json.dump(mesh2tree, f, indent=4)
        

