from pycirclize import Circos
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import random
import pdb
from collections import defaultdict
np.random.seed(0)


if __name__ == "__main__":
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
            chapter_dict[chapter][section] = dict(zip(section_group["name"], section_group["value"]))



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
        track1 = sector.add_track((52, 70), r_pad_ratio=0.1)
        track1.rect(sector.start, sector.end)
        # if len(y_bar) >= 4:
        #     sector.text(f"{sector.name}", r=59, size=6, adjust_rotation=True)
        # else:
        sector.text(f"{sector.name}", r=56, size=6, adjust_rotation=True, orientation="vertical")
        
        track3 = sector.add_track((30, 50), r_pad_ratio=0.1)
        track3.axis()
        track3.bar(x_bar, y_bar, vmin=0, vmax=max(y_bar) + 2000, bottom=0)
        

        for idx, y_label in enumerate(y_labels):
            sector.text(y_label, x=x_bar[idx], r=73, size=4, adjust_rotation=True, orientation="vertical")

        track3.grid(y_grid_num=10, color="gray", alpha=0.5, linestyle="--")
        
        # y_ticks = [i for i in range(0, 8000, 1000)]
        # y_labels = [f"{int(y)}" for y in y_ticks]
        # track3.yticks(y_ticks, y_labels, label_size=4)


    curr_dir = os.path.dirname(os.path.realpath(__file__))
    circos.savefig(os.path.join(curr_dir, 'temp', "circos_plot.png"), dpi=300)
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