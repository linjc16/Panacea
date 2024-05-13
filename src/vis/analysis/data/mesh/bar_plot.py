import json
import pdb
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


# set basic parameters
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({"ytick.color" : "black",
                     "xtick.color" : "black",
                     "axes.labelcolor" : "black",
                     "axes.edgecolor" : "black"})

mpl.rcParams.update({
    "pdf.use14corefonts": True
})

MEDIUM_SIZE = 14
SMALLER_SIZE = 12
plt.rc('font', size=SMALLER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
plt.rc('xtick', labelsize=MEDIUM_SIZE)	 # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)	 # fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('font', family='Helvetica')

hex_codes = [
    "#c16fc3", "#b67ac1", "#ac83c1", "#a38cc2",
    "#9997c2", "#8da4c1", "#86acc0", "#7ab6c1",
    '#6cc2c1', '#60cac1'
]

disease_categories = {
    "C01": "Infections",
    "C04": "Neoplasms",
    "C05": "Musculoskeletal Diseases",
    "C06": "Digestive System Diseases",
    "C07": "Stomatognathic Diseases",
    "C08": "Respiratory Tract Diseases",
    "C09": "Otorhinolaryngologic Diseases",
    "C10": "Nervous System Diseases",
    "C11": "Eye Diseases",
    "C12": "Urogenital Diseases",
    "C14": "Cardiovascular Diseases",
    "C15": "Hemic and Lymphatic Diseases",
    "C16": "Congenital, Hereditary, and Neonatal Diseases and Abnormalities",
    "C17": "Skin and Connective Tissue Diseases",
    "C18": "Nutritional and Metabolic Diseases",
    "C19": "Endocrine System Diseases",
    "C20": "Immune System Diseases",
    "C21": "Disorders of Environmental Origin",
    "C22": "Animal Diseases",
    "C23": "Pathological Conditions, Signs and Symptoms",
    "C24": "Occupational Diseases",
    "C25": "Chemically-Induced Disorders",
    "C26": "Wounds and Injuries"
}


if __name__ == '__main__':
    # data/analysis/icd10/mesh/embase_mesh2tree.json
    with open('data/analysis/icd10/mesh/embase_mesh2tree.json', 'r') as f:
        mesh2tree = json.load(f)

    # for each tree num, only extract the first 3 strigs
    tree2count = {}
    for mesh_term, v in mesh2tree.items():
        tree_num = v['tree_num']
        tree_num = tree_num.split('.')[0]
        tree2count[tree_num] = tree2count.get(tree_num, 0) + v['count']
    
    with open('data/analysis/icd10/mesh/pubmed_mesh2tree.json', 'r') as f:
        mesh2tree_pubmed = json.load(f)
    
    for mesh_term, v in mesh2tree_pubmed.items():
        tree_num = v['tree_num']
        tree_num = tree_num.split('.')[0]
        tree2count[tree_num] = tree2count.get(tree_num, 0) + v['count']
    
    # sort by count
    tree2count = dict(sorted(tree2count.items(), key=lambda x: x[1], reverse=True))

    # only select the first len(hex_codes) items
    tree2count = dict(list(tree2count.items())[:len(hex_codes)])

    # sum the counts of all
    total_count = sum(tree2count.values())
    print(f'Total count: {total_count}')
    

    with open('data/analysis/icd10/mesh/embase_tree2count.json', 'w') as f:
        json.dump(tree2count, f, indent=4)

    
    # plot
    disease_ids = list(tree2count.keys())
    data_counts = list(tree2count.values())

    disease_names = [disease_categories[disease_id] for disease_id in disease_ids]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(disease_names, data_counts, color=hex_codes, height=0.8)
    # set bar width
    plt.xscale('log')
    original_ticks = [1e4, 1e5, 5e5]
    ticks_between_1e4_and_1e5 = np.geomspace(1e4, 1e5, num=10)
    ticks_between_1e5_and_5e5 = np.geomspace(1e5, 5e5, num=10)
    all_ticks = np.unique(np.concatenate([ticks_between_1e4_and_1e5, ticks_between_1e5_and_5e5]))
    plt.xticks(all_ticks, ['']*len(all_ticks))  
    plt.xticks(original_ticks, [f'10$^{int(np.log10(x))}$' if x < 5e5 else '' for x in original_ticks], minor=False)
    # plt.xticks([1e4, 1e5, 5e5], ['10$^4$', '10$^5$', ''])  
    # set title left
    plt.title('Top 10 diseases by frequency in clinical trial publications', fontsize=16, loc='left')
    
    # remove the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.gca().invert_yaxis()


    for bar in bars:
        exponent = int(f'{bar.get_width():.0e}'.split('e+')[-1])  #
        mantissa = bar.get_width() / (10 ** exponent) 
        label = f'{mantissa:.2f} × 10$^{exponent}$'  
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, label, va='center')

    # tight layout
    plt.tight_layout()
    
    save_dir = 'visulization/data'
    os.makedirs(save_dir, exist_ok=True)

    # plt.savefig(f'{save_dir}/paper_disease_count.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/paper_disease_count.pdf', dpi=300, bbox_inches='tight')
