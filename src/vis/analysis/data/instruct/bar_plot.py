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

num_instruct_data = {
    'Study Arm Design': 53548,
    'Query Expansion': 51000,
    'Outcome Measure Design': 44809,
    'Criteria Design': 35951,
    'Patient-Trial Matching': 6368,
    'Single-Trial Summarization': 5000,
    'Query Generation': 2161,
    'Multi-Trial Summarization': 2029,
}

# hex_codes = [
#     "#7400B8", "#6930C3", "#5E60CE", "#5390D9", 
#     "#4EA8DE", "#48BFE3", "#56CFE1", "#64DFDF"
# ]

hex_codes = [
    "#c16fc3", "#b67ac1", "#ac83c1", "#a38cc2",
    "#9997c2", "#8da4c1", "#86acc0", "#7ab6c1"
]

task_names = list(num_instruct_data.keys())
data_counts = list(num_instruct_data.values())

plt.figure(figsize=(7, 6))
bars = plt.barh(task_names, data_counts, color=hex_codes, height=0.8)
# set bar width
plt.xscale('log')
plt.xticks([1e4, 1e5], ['10$^4$', '10$^5$'])  
# set title left
plt.title('Number of instruction data per task', fontsize=16, loc='left')

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

plt.savefig(f'{save_dir}/instruct_data_count.png', dpi=300, bbox_inches='tight')
