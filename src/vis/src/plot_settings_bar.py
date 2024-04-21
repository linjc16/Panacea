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
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('font', family='Helvetica')


 #, xtick.color='w', axes.labelcolor='w', axes.edge_color='w'
FIG_HEIGHT = 4
FIG_WIDTH = 4
# plt.style.use('dark_background')

# use white background
plt.style.use('default')

EVO_DEVO_MODELS = ['Sagittarius', 'linear', 'mean', 'mTAN', 'cvae', 'cpa',
                   'seq_by_seq_RNN', 'seq_by_seq_neuralODE']


def get_square_axis(FIG_WIDTH, FIG_HEIGHT):
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_wider_axis(FIG_WIDTH, FIG_HEIGHT, double=False):
    plt.figure(figsize=(int(FIG_WIDTH * (3/2 if not double else 5/2)), FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_double_square_axis():
    plt.figure(figsize=(2*FIG_WIDTH, 2*FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_model_ordering(actual_models):
    desired_ordering = ['Sagittarius', 'linear', 'mean',
                        'mTAN', 'cvae', 'cpa', 'seq_by_seq_RNN', 'seq_by_seq_neuralODE']
    return sorted(actual_models, key=lambda m: desired_ordering.index(m))


def get_metric_name(name, setting):
    return {
        'RMSE': 'RMSE',
        'spearman_rank_by_genes': 'Spearman correlation' if setting != 'EvoDevo' \
                else 'Spearman (rank genes)',
        'spearman_rank_by_times': 'T-Spearman correlation' if setting != 'EvoDevo' \
                else 'Spearman (rank time points)',
        'auroc': 'AUROC',
    }[name]


# def get_model_colors(mod):
#     return {
#         'cvae': '#543005',
#         'cpa': '#8c510a',
#         'seq_by_seq_RNN': '#bf812d',
#         'seq_by_seq_neuralODE': '#dfc27d',
#         'mTAN': '#003c30',
#         'mean': '#01665e',
#         'linear': '#35978f',
#         'Sagittarius': '#c7eae5'
#     }[mod]

def get_model_colors_ablation(mod):
    return {
        'Molecular-graph-based (w/o cl)': '#dfc27d',
        'SMILES-based (w/o cl)': '#bf812d',
        'Molecular graph and SMILES (w/o cl)': '#8c510a',
        'Molecular graph and SMILES (w/ cl)': '#35978f',
        # '': '#003c30',
        'Pisces': '#c7eae5',
    }[mod]


def get_model_colors(mod):
    return {
        'LLaMA-2-7B': '#543005',
        'LLaMA-2-13B': '#8c510a',
        'Mistral-7B': '#dfc27d',
        'OpenChat-7B': '#003c30',
        'Zephyr-7B': '#01665e',
        'BioMistral-7B': '#bf812d',  # Reallocated color from LLaMA-2-70B
        'MedAlpaca-7B': '#a1d76a',  # New gradient color
        'MedAlpaca-13B': '#4d9221',  # New gradient color
        'Meditron-7B': '#b8e186',  # New gradient color
        'GPT-3.5': '#35978f',
        'GPT-4': '#c7eae5',
        'Claude 3 Haiku': '#80cdc1',  # New gradient color
        'Claude 3 Sonnet': '#018571',  # New gradient color
        'Panacea-7B': '#a6dba0'
    }[mod]

# def get_model_colors(mod):
#     return {
#         'DeepSynergy': '#35978f',
#         'AuDNNsynergy':'#c7eae5',
#         'PRODeepSyn': '#bf812d',
#         'GraphSynergy': '#dfc27d',
#         'DeepDDS': '#003c30',
#         'Ours': '#01665e',
#     }[mod]

# def get_model_colors(mod):
#     return {
#         'DeepSynergy': 
#         'AuDNNsynergy': ,
#         'PRODeepSyn': '#b2df8a',
#         'GraphSynergy': '#33a02c',
#         'DeepDDS': '#1f78b4',
#         'Ours': '#a6cee3',
#     }[mod]

# light green: #b2df8a, dark green: #33a02c, light blue: #a6cee3, dark blue: #1f78b4

# def get_model_colors(mod):
#     return {
#         'GraphSynergy': 'light green',
#         'PRODeepSyn': 'dark green',
#         'DeepDDS': 'dark blue',
#         'Ours': 'light blue',
#     }[mod]

def get_sag_vs_baseline_colors(mod):
    if mod in {'Sagittarius'}:
        return '#c7eae5'
    else:
        return '#dfc27d'


def get_model_name_conventions(mname):
    if 'cvae' in mname:
        return 'cVAE'
    if 'cpa' in mname:
        return 'CPA'
    if 'seq_by_seq_RNN' in mname:
        return 'RNN'
    if 'seq_by_seq_neuralODE' in mname:
        return 'Neural ODE'
    if 'mTAN' in mname:
        return 'mTAN'
    if 'linear' in mname:
        return 'Linear'
    if 'mean' in mname:
        return 'Mean'
    if 'Sagittarius' in mname:
        return 'Sagittarius'
    
    print("Unrecognized:", mname)
    assert False
    
    
def get_LINCS_task_names(task, add_return=True):
    if task == 'continuousCombinatorialGeneration':
        return 'Complete{}Generation'.format('\n' if add_return else ' ')
    elif task == 'comboAndDosage':
        return 'Combination{}& Dosage'.format('\n' if add_return else ' ')
    elif task == 'comboAndTime':
        return 'Combination{}& Treatment Time'.format('\n' if add_return else ' ')
    print('Unrecognized', task)
    assert False
    
    
def get_species_axis_tick(species):
    if species == 'RhesusMacaque':
        return 'Rhesus\nMacaque'  # make this two lines
    return species


def get_organ_color_palette():
    return ['#01665e', '#5ab4ac', '#c7eae5', '#f6e8c3', '#dfc27d', '#bf812d', '#8c510a']


def get_TM_color_palette():
    return {
        'Sagittarius': '#80cdc1',
        'baseline': '#018571',
        'Heart': '#f5f5f5',
        'Kidney': '#dfc27d',
        'Liver': '#a6611a' }


def get_base_color():
    return '#dfc27d'


def get_line_style(mod):
    if mod in {'Sagittarius'}:
        return 'solid'
    elif mod == 'mean':
        return (0, (5, 1))
    elif mod == 'linear':
        return 'dotted'
    raise ValueError("Unknown model: {}".format(mod))