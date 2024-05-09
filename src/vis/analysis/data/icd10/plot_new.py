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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    tree_file = "data/analysis/icd10/icd10_hierarchy_newick_tree.nwk"

    circos, tv = Circos.initialize_from_tree(
        tree_file,
        r_lim=(10, 30),
        # Set large margin to insert heatmap & bar track between tree and labels
        leaf_label_rmargin=32,
        leaf_label_size=0,
        ignore_branch_length=True,
        ladderize=True,
        line_kws=dict(lw=0.5),
    )
    
    # tree = Phylo.read(tree_file, "newick")
    circos.plotfig(ax=ax)

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
    leaf_labels_name = [icd10_to_name[label] for label in tv.leaf_labels]

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
            # according to the order of leaf_labels_name, get the dict
            section_dict = {}
            for idx, name in enumerate(leaf_labels_name):
                if name in section_group["name"].values:
                    count = section_group[section_group["name"] == name]["value"].values[0]
                    section_dict[name] = count

            chapter_dict[chapter][section] = section_dict
    


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
    
    for sector in circos.sectors:
        x_bar = np.arange(sector.start, sector.end) + 0.5
        y_bar, y_labels = collect_values(chapter_dict[sector.name])

        # sector.rect(sector.start, sector.end, r_lim=(52, 70))
        track1 = sector.add_track((40, 55), r_pad_ratio=0.1)
        track1.rect(sector.start, sector.end, fc=wedge_color_dict[sector.name])
        # if len(y_bar) >= 4:
        #     sector.text(f"{sector.name}", r=59, size=6, adjust_rotation=True)
        # else:

        # add label for each wedge
        if len(y_bar) <= 10:
            if len(y_bar) >= 5 and len(y_bar) <= 10:
                font_size = 5
            elif len(y_bar) >= 4 and len(y_bar) < 5:
                font_size = 4
            else:
                font_size = 3
            # add line break in the middle of the chapter name
            new_chapter_name = chapter_name_dict[sector.name].replace(" ", "\n")
            sector.text(new_chapter_name, r=45, size=font_size, adjust_rotation=True)
        else:
            sector.text(chapter_name_dict[sector.name], r=45, size=6, adjust_rotation=True, orientation="horizontal")
        
        track3 = sector.add_track((30, 40), r_pad_ratio=0.1)
        track3.axis()
        track3.bar(x_bar, y_bar, vmin=0, vmax=max(y_bar) + 2000, bottom=0, color=wedge_color_dict[sector.name])
        

        for idx, y_label in enumerate(y_labels):
            if y_label in condition_rename:
                y_label = condition_rename[y_label]
            sector.text(y_label, x=x_bar[idx], r=55, size=4, adjust_rotation=True, orientation="vertical")

        # track3.grid(y_grid_num=10, color="gray", alpha=0.5, linestyle="--")
        
        # y_ticks = [i for i in range(0, 8000, 1000)]
        # y_labels = [f"{int(y)}" for y in y_ticks]
        # track3.yticks(y_ticks, y_labels, label_size=4)


    curr_dir = os.path.dirname(os.path.realpath(__file__))
    circos.plotfig(ax=ax)
    # circos.savefig(os.path.join(curr_dir, 'temp', "circos_plot.png"), dpi=300)
    plt.savefig(os.path.join(curr_dir, 'temp', "circos_plot.png"), dpi=300, bbox_inches="tight")
    # pdb.set_trace()

# # Initialize Circos sectors
# sectors = {"A": 10, "B": 15, "C": 12, "D": 20, "E": 15}
# circos = Circos(sectors, space=5)

# for sector in circos.sectors:
#     # Plot sector name
#     # sector.text(f"Sector: {sector.name}", r=110, size=15)
#     # Create x positions & random y values
#     x = np.arange(sector.start, sector.end) + 0.5
#     y = np.random.randint(0, 100, len(x))

#     sector.rect(sector.start, sector.end, r_lim=(52, 70))


#     # # Plot lines
#     # track1 = sector.add_track((80, 100), r_pad_ratio=0.1)
#     # track1.xticks_by_interval(interval=1)
#     # track1.axis()
#     # track1.line(x, y)
#     # # Plot points 
#     # track2 = sector.add_track((55, 75), r_pad_ratio=0.1)
#     # track2.axis()
#     # track2.scatter(x, y)
#     # Plot bars
#     track3 = sector.add_track((30, 50), r_pad_ratio=0.1)
#     track3.axis()
#     track3.bar(x, y)

# # # Plot links 
# # circos.link(("A", 0, 3), ("B", 15, 12))
# # circos.link(("B", 0, 3), ("C", 7, 11), color="skyblue")
# # circos.link(("C", 2, 5), ("E", 15, 12), color="chocolate", direction=1)
# # circos.link(("D", 3, 5), ("D", 18, 15), color="lime", ec="black", lw=0.5, hatch="//", direction=2)
# # circos.link(("D", 8, 10), ("E", 2, 8), color="violet", ec="red", lw=1.0, ls="dashed")

# import os
# curr_dir = os.path.dirname(os.path.realpath(__file__))
# circos.savefig(os.path.join(curr_dir, 'temp', "circos_plot.png"), dpi=300)