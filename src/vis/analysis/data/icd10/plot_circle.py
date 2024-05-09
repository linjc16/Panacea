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
    'A00-B99': '#2ba02d',
    'C00-D49': "#97e089",
    'E00-E89': "#ffbb78",
    'F01-F99': "#b0c6e8",
    'G00-G99': "#fd7f10",
    'H00-H59': "#c4b0d6",
    "I00-I99": "#f0c4c5",
    "J00-J99": "#efdcda",
    "K00-K95": "#9cd09a",
    "M00-M99": "#93b7d5",
    "N00-N99": "#b3c9df",
    "R00-R99": "#cfdee9",
    "S00-T88": "#6cc6cb",
    "Z00-Z99": "#6cc6cb",
    "U00-U85": "#6cc6cb"
}

if __name__ == "__main__":

    # {"projection": "polar"}
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, polar=True)

    tree_file = "data/analysis/icd10/icd10_hierarchy_newick_tree.nwk"
    tree = read_newick_file(tree_file)
    second_level_trees = split_trees_at_second_level(tree)

    sectors = {sub_tree.root: sub_tree.count_terminals() for sub_tree in second_level_trees}
    circos = Circos(sectors, space=5)



    # ********** plot the bar chart for each sector **********

    # data/analysis/icd10/diseases_data.csv
    df = pd.read_csv("data/analysis/icd10/diseases_data.csv")

    # group by chapter
    sectors = {}
    for chapter, group in df.groupby("chapter"):
        sectors[chapter] = len(group)
    
    circos = Circos(sectors, space=5)


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
    

    for sector, tree in zip(circos.sectors, second_level_trees):
        # Plot randomized tree
        tree_track = sector.add_track((10, 30))
        tree_track.axis()
        tree_track.tree(tree, leaf_label_size=0)

        x_bar = np.arange(sector.start, sector.end) + 0.5
        y_bar, y_labels = collect_values(chapter_dict[sector.name])
        track3 = sector.add_track((30, 40), r_pad_ratio=0.1)
        track3.axis()
        track3.bar(x_bar, y_bar, vmin=0, vmax=max(y_bar) + 2000, bottom=0, color=wedge_color_dict[sector.name])



    curr_dir = os.path.dirname(os.path.realpath(__file__))
    circos.savefig(os.path.join(curr_dir, 'temp', "circos_plot.png"), dpi=300)