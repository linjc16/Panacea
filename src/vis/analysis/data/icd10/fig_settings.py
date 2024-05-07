import matplotlib as mpl
import matplotlib.pyplot as plt


# set basic parameters
mpl.rcParams['pdf.fonttype'] = 42

is_black_background = True
if is_black_background:
    plt.style.use('default')
    mpl.rcParams.update({"ytick.color" : "black",
                     "xtick.color" : "black",
                     "axes.labelcolor" : "black",
                     "axes.edgecolor" : "black"})

LARGE_SIZE = 16
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
mpl.rcParams.update({
    "pdf.use14corefonts": True
})
 #, xtick.color='w', axes.labelcolor='w', axes.edge_color='w'
FIG_HEIGHT = 4
FIG_WIDTH = 4

CANCER_TYPE = ['BRCA', 'PAAD', 'COAD']


def get_square_axis():
    fig, ax = plt.subplots(figsize=(1.2*FIG_WIDTH, 1.2*FIG_HEIGHT))
    return fig, ax

def get_wider_axis(double=False):
    plt.figure(figsize=(int(FIG_WIDTH * (3/2 if not double else 5/2)), FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax

def get_broken_line_axis():
    fig, ax = plt.subplots(figsize=(3*FIG_WIDTH, 1.2*FIG_HEIGHT))
    return fig, ax

def get_waterfall_axis():
    fig, ax = plt.subplots(figsize=(3.5*FIG_WIDTH, 2*FIG_HEIGHT))
    return fig, ax

def get_complex_heatmap_axis():
    fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 2*FIG_HEIGHT))
    return fig, ax

def get_narrow_axis():
    fig, ax = plt.subplots(figsize=(0.8*FIG_WIDTH, 1.5*FIG_HEIGHT))
    return fig, ax

def get_circular_plot_axis():
    fig, ax = plt.subplots(figsize=(2.1*FIG_WIDTH, 2.2*FIG_HEIGHT), subplot_kw={'projection': 'polar'})
    return fig, ax

def get_scatter_plot_axis():
    fig = plt.figure(figsize=(1.2*FIG_WIDTH, 1.1*FIG_HEIGHT))
    return fig

def get_umap_plot_axis():
    fig, ax = plt.subplots(figsize=(1.2*FIG_WIDTH, 1.0*FIG_HEIGHT))
    return fig, ax

def get_drug_combo_umap_plot_axis():
    fig, ax = plt.subplots(figsize=(1.4*FIG_WIDTH, 1.0*FIG_HEIGHT))
    return fig, ax

def get_double_square_axis():
    plt.figure(figsize=(2*FIG_WIDTH, 2*FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax

def get_three_square_axis():
    fig, axes = plt.subplots(figsize=(3.5*FIG_WIDTH, 1.5*FIG_HEIGHT), ncols=3)
    return fig, axes

def get_three_narrow_axis():
    fig, axes = plt.subplots(figsize=(2.5*FIG_WIDTH, 1.1*FIG_HEIGHT), ncols=3)
    return fig, axes

def get_four_square_axis():
    fig, axes = plt.subplots(figsize=(3.5*FIG_WIDTH, 1.5*FIG_HEIGHT), ncols=4)
    return fig, axes

def get_four_rows_axis():
    fig, axes = plt.subplots(figsize=(2.5*FIG_WIDTH, 2.5*FIG_HEIGHT), nrows=4)
    return fig, axes

def get_double_axis():
    fig, axes = plt.subplots(figsize=(2.7*FIG_WIDTH, 1.35*FIG_HEIGHT), ncols=2)
    return fig, axes

def get_model_colors(mod):
    return {
        'Pisces': '#c7eae5',
        'Pisces (transfer)': '#a6cee3',
        'DeepDDS': '#35978f',
        'GraphSynergy': '#01665e',
        'PRODeepSyn': '#8c510a',
        'AuDNNsynergy': '#bf812d',
        'DeepSynergy': '#dfc27d',
        'GAT-DDI': '#dfc27d',
        'MHCADDI': '#bf812d',
        'MR-GNN': '#8c510a',
        'SSI-DDI': '#01665e',
        'GMPNN-CS': '#35978f',
        'R2-DDI': '#80cdc1',
        
    }[mod]
    
def get_cancer_colors(cancer):
    return {
        'BRCA': '#c7eae5',
        'PAAD': '#a6cee3',
        'COAD': '#01665e',
    }[cancer]
    
def get_cluster_colors(cluster):
    colormap = plt.cm.tab20
    colors = {i:colormap(i) for i in range(21)}

    return colors[cluster]

'''def get_cluster_colors(cluster):
    colors = {
        1:'#543005',
        2:'#8c510a',
        3:'#bf812d',
        4:'#dfc27d',
        5:'#f6e8c3',
        6:'#c7eae5',
        7:'#80cdc1',
        8:'#35978f',
        9:'#01665e',
        10:'#003c30', 
        11:'#fddbc7',
        12:'#92c5de',
    }
    return colors[cluster]'''

def get_case_vs_control_colors(mod):
    if mod in {'Case'}:
        return '#35978f'
    else:
        return '#bf812d'
    
def get_er_analysis_colors():
    return ['#35978f', '#bf812d', '#2166ac']
    
def get_dot_size():
    return 10

def get_single_group_bar_plot_setting():
    return {
        'capsize': 3,
        'capthick': 1.5,
        'linewidth': 1.5,
    }
    
def get_box_plot_setting():
    return {
        'box alpha': 0.5,
        'box linewidth': 1.5,
        'cap linewidth': 1.5,
        'whisker linestyle': '--',
        'whisker linewidth': 1.5,
        'median color': 'w',
        'median linewidth': 1.5,
        'marker size': 50,
        'marker edge color': 'k',
    }
    
def get_scatter_plot_setting():
    return {
        'alpha': 0.6,
        's': 10,
        'alpha': 0.8,
        'cmap': 'gist_yarg',
        'linewidth': 1,
        'hist_bin':15,
        'hist_bin_width':0.3,
        'hist_color':'lightcyan',
    }
    
def get_umap_plot_setting():
    return {
        'alpha': 0.8,
        's': 20,
        'marker_linewidth': 0,
        'cmap': 'magma_r',
        'edgecolors': 'w',
    }
    
def get_umap_colors(cluster):
    colormap = plt.cm.tab20
    colors = {i:colormap(i) for i in range(30)}
    return colors[cluster]


def get_velocity_colors(cluster):
    colors = {
        0:'#543005',
        1:'#8c510a',
        2:'#bf812d',
        3:'#dfc27d',
        4:'#f6e8c3',
        5:'#c7eae5',
        6:'#80cdc1',
        7:'#35978f',
        8:'#01665e',
        9:'#003c30', 
        10:'#fddbc7',
        11:'#92c5de',
        12:'#00441b',
        13:'#662506'
    }
    return colors[cluster]

def get_velocity_color_markers(cluster):
    c_list = ['#80cdc1', '#018571', '#a6611a', '#dfc27d']
    m_list = ['o', '^', 'h', 'D']
    c_m_list = [(c, m) for c in c_list for m in m_list]
    return c_m_list[cluster]

def get_velocity_markers(idx):
    markers = {
        0:'o',
        1:'d',
        2:'H',
        3:'8',
        4:'X',
        5:'p',
        6:'P',
        7:'*',
        8:'v',
    }
    return markers[idx]

def get_broken_line_setting():
    return {
        'linewidth': 1.5,
        'marker': 'o',
        'color': '#5ab4ac',
        'markersize': 5,
        'capsize': 3,
        'markerfacecolor': '#5ab4ac',
        'markeredgecolor': '#5ab4ac',
        'markeredgewidth': 1.5,
        'error_bar_width': 0.5,
        'error_bar_color': 'w',
        'target_color': '#d8b365',
        'target_linewidth': 1,
        'target_linestyle': '--',
    }