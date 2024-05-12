from pycirclize import Circos
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import random
import pdb
from collections import defaultdict
from Bio import Phylo
np.random.seed(0)


plt.rc('font', family='Helvetica')

def read_newick_file(file_path):
    trees = Phylo.read(file_path, 'newick')
    return trees

def split_trees_at_second_level(tree):
    second_level_trees = []
    if tree.root.clades:
        for clade in tree.root.clades:
            new_tree = Phylo.BaseTree.Tree(root=clade)
            second_level_trees.append(new_tree)
    return second_level_trees

wedge_color_dict = {
    'A00-B99': '#89d1a9',
    'C00-D49': "#f0c4c5",
    'E00-E89': "#eac6e9",
    'F01-F99': "#ffa090",
    'G00-G99': "#ffc865",
    'H00-H59': "#ffbb66",
    "I00-I99": "#ffe8bf",
    "J00-J99": "#d5ebff",
    "K00-K95": "#acd7ff",
    "M00-M99": "#93b7d5",
    "N00-N99": "#b3c9df",
    "R00-R99": "#cfdee9",
    "S00-T88": "#6cc6cb",
    "Z00-Z99": "#6cc6cb",
    "U00-U85": "#6cc6cb"
}

if __name__ == "__main__":

    tree_file = "data/analysis/icd10/icd10_hierarchy_newick_tree.nwk"
    tree = read_newick_file(tree_file)
    second_level_trees = split_trees_at_second_level(tree)

    sectors = {sub_tree.root: sub_tree.count_terminals() for sub_tree in second_level_trees}
    circos = Circos(sectors, space=5)


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # ********** plot the bar chart for each sector **********

    # data/analysis/icd10/diseases_data.csv
    df = pd.read_csv("data/analysis/icd10/diseases_data.csv")

    # group by chapter
    sectors = {}
    for chapter, group in df.groupby("chapter"):
        sectors[chapter] = len(group)
    
    circos = Circos(sectors, space=2, start=0, end=355)
    

    chapter_dict = defaultdict(dict) # key:chapter, value: dict{'section': {name: count, ...}, 'section': {name: count, ...}}
    for chapter, group in df.groupby("chapter"):
        for section, section_group in group.groupby("section"):
            # for each section, save the dict of {name: count, ...}, {name: count, ...}
            chapter_dict[chapter][section] = {name: count for name, count in zip(section_group["name"], section_group["value"])}
    
    

    # read data/analysis/icd10/chapter_name_dict.json
    with open("data/analysis/icd10/chapter_name_dict.json", "r") as file:
        chapter_name_dict = json.load(file)

    
    # data/analysis/icd10/condition_rename.json
    with open("data/analysis/icd10/condition_rename.json", "r") as file:
        condition_rename = json.load(file)
        

    def collect_values(data):
        values = []
        names = []
        for key, item in data.items():
            if isinstance(item, dict):
                item_values, item_names = collect_values(item)
                values.extend(item_values)
                names.extend(item_names)
            else:
                values.append(item)
                names.append(key)
        return values, names
    
    # data/analysis/icd10/icd10_matching_results.json
    with open("data/analysis/icd10/icd10_matching_results.json", "r") as file:
        icd10_matching_results = json.load(file)
    
    # top 100
    icd10_matching_results = {k: v for k, v in list(icd10_matching_results.items())[:100]}

    icd10_to_name = {}
    for key, value in icd10_matching_results.items():
        # upper case the first letter
        value['description'] = value['description'][0].upper() + value['description'][1:]
        if 'covid' in value['description'].lower():
            # replace covid with COVID
            value['description'] = value['description'].replace('covid', 'COVID')
        icd10_to_name[key] = value['description']


    counts_max = df["value"].max()

    # ********** plot the figure for each sector **********

    for sector, tree in zip(circos.sectors, second_level_trees):
        # Plot randomized tree
        tree_track = sector.add_track((18, 27.5))
        tree_track.axis(fc=wedge_color_dict[sector.name], alpha=0.2, lw=0)
        tv = tree_track.tree(tree, 
                             leaf_label_size=0, 
                             ignore_branch_length=True,
                             line_kws=dict(color="lightgray", lw=0.4)
        )
        
        leaf_labels_name = [icd10_to_name[label] for label in tv.leaf_labels]

        
        x_bar = np.arange(sector.start, sector.end) + 0.5
        y_bar, y_labels = collect_values(chapter_dict[sector.name])


        y_bar = (np.log10(np.array(y_bar)) - 2) * 2
        # pdb.set_trace()
        # reorganize the y_bar and y_labels according to the order of leaf_labels_name
        new_y_bar = []
        new_y_labels = []
        for name in leaf_labels_name:
            if name in y_labels:
                idx = y_labels.index(name)
                new_y_bar.append(y_bar[idx])
                new_y_labels.append(name)
            else:
                new_y_bar.append(0)
                new_y_labels.append(name)
        
        y_bar = new_y_bar
        y_labels = new_y_labels

        track3 = sector.add_track((28, 35), r_pad_ratio=0.1)
        track3.axis(lw=0.4, ec='#3f3f3f')
        track3.bar(x_bar, y_bar, vmin=0, vmax=4.5, bottom=0, color=wedge_color_dict[sector.name], alpha=0.8)
        track3.grid(y_grid_num=6, color="gray", alpha=0.5, linestyle="--")
        
        track1 = sector.add_track((35, 50), r_pad_ratio=0.1)
        track1.rect(sector.start, sector.end, fc=wedge_color_dict[sector.name])
        
        # add label for each wedge
        if len(y_bar) <= 10:
            if len(y_bar) > 1:
                if len(y_bar) >= 5 and len(y_bar) <= 10:
                    font_size = 5
                elif len(y_bar) >= 4 and len(y_bar) < 5:
                    font_size = 4
                else:
                    if sector.name in ['E00-E89', "G00-G99", "Z00-Z99"]:
                        font_size = 4
                    else:
                        font_size = 3
                # add line break in the middle of the chapter name
                new_chapter_name = chapter_name_dict[sector.name].replace(" ", "\n")
                sector.text(new_chapter_name, r=40, size=font_size, adjust_rotation=True)
        else:
            sector.text(chapter_name_dict[sector.name], r=40, size=6, adjust_rotation=True, orientation="horizontal")
        
        for idx, y_label in enumerate(y_labels):
            if y_label in condition_rename:
                y_label = condition_rename[y_label]
            sector.text(y_label, x=x_bar[idx], r=50, size=6, adjust_rotation=True, orientation="vertical")
        
        if sector.name == 'Z00-Z99':
            # yticks
            yticks = [0, 2, 4]
            yticklabels = ["$10^2$", "$10^3$", "$10^4$"]
            track3.yticks(yticks, yticklabels, label_size=4)
    
    save_dir = 'visulization/data'
    circos.plotfig(ax=ax)
    plt.savefig(os.path.join(save_dir, "circos_plot.pdf"), dpi=900)
    # plt.savefig(os.path.join(save_dir, "circos_plot.png"), dpi=900)