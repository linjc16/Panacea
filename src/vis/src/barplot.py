import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

import plot_settings_bar
import plot_utils


trans_deepergcn = [
    [[0.6754, 0.4410, 0.4381, 0.4131]],
    [[0.6892, 0.4457, 0.4399, 0.4123]],
    [[0.7023, 0.4410, 0.4598, 0.4321]],
    [[0.6768, 0.4455, 0.4366, 0.4111]],
    [[0.6959, 0.4133, 0.4394, 0.4106]]]

trans_transformer = [
    [[0.6995, 0.4706, 0.4673, 0.4413]],
    [[0.6736, 0.4648, 0.4360, 0.4115]],
    [[0.6851, 0.4553, 0.4542, 0.4294]],
    [[0.6967, 0.4724, 0.4682, 0.4429]],
    [[0.6678, 0.4312, 0.4219, 0.3970]]
]

trans_dv = [
    [[0.6896, 0.4823, 0.4630, 0.4385]],
    [[0.6872, 0.4597, 0.4462, 0.4201]],
    [[0.6826, 0.4644, 0.4478, 0.4226]],
    [[0.6898, 0.4719, 0.4601, 0.4350]],
    [[0.6844, 0.4176, 0.4352, 0.4083]]
]

trans_dv_cl = [
    [[0.7117, 0.4684, 0.4780, 0.4512]],
    [[0.7051, 0.4503, 0.4592, 0.4313]],
    [[0.6995, 0.4527, 0.4516, 0.4234]],
    [[0.7102, 0.4606, 0.4665, 0.4385]],
    [[0.6916, 0.4180, 0.4350, 0.4065]]
]


trans_ours = [
    [[0.7168, 0.4874, 0.4865, 0.4599]],
    [[0.6961, 0.4718, 0.4692, 0.4445]],
    [[0.6994, 0.4749, 0.4721, 0.4467]],
    [[0.7034, 0.4919, 0.4816, 0.4569]],
    [[0.6879, 0.4316, 0.4538, 0.4288]]]

models_trans = ['Molecular-graph-based (w/o cl)', 'SMILES-based (w/o cl)', 'Molecular graph and SMILES (w/o cl)', 'Molecular graph and SMILES (w/ cl)', 'Pisces']
trans = (trans_deepergcn, trans_transformer, trans_dv, trans_dv_cl, trans_ours)


lc_deepergcn = [
    [[0.5494, 0.2443, 0.1699, 0.1534]],
    [[0.5379, 0.2463, 0.1369, 0.1261]],
    [[0.5641, 0.2051, 0.1993, 0.1736]]
]

lc_transformer = [
    [[0.6078, 0.2309, 0.2803, 0.2498]],
    [[0.6659, 0.2341, 0.3242, 0.2892]],
    [[0.6197, 0.2383, 0.2868, 0.2510]]
]

lc_dv = [
    [[0.6128, 0.2467, 0.2943, 0.2652]],
    [[0.6367, 0.2429, 0.3093, 0.2785]],
    [[0.5835, 0.2253, 0.2382, 0.2091]]
]

lc_dv_cl = [
    [[0.6078, 0.2413, 0.2748, 0.2428]],
    [[0.6513, 0.2608, 0.3174, 0.2844]],
    [[0.6232, 0.2398, 0.2970, 0.2624]]
]

lc_Ours = [
    [[0.6210, 0.2347, 0.2961, 0.2635]],
    [[0.6591, 0.2376, 0.3186, 0.2839]],
    [[0.6104, 0.2340, 0.2789, 0.2454]]
]



models_lc = models_trans
lc = (lc_deepergcn, lc_transformer, lc_dv, lc_dv_cl, lc_Ours)

ldc_deepergcn = [
    [[0.6732, 0.3157, 0.3803, 0.3481]],
    [[0.6741, 0.3731, 0.4041, 0.3727]],
    [[0.6124, 0.3179, 0.3061, 0.2831]]
]

ldc_transformer = [
    [[0.6426, 0.3037, 0.3420, 0.3111]],
    [[0.6549, 0.3697, 0.3796, 0.3492]],
    [[0.6474, 0.3224, 0.3566, 0.3309]]
]

ldc_dv = [
    [[0.6643, 0.3477, 0.3837, 0.3544]],
    [[0.6581, 0.3744, 0.3913, 0.3622]],
    [[0.6186, 0.3505, 0.3328, 0.3128]]
]

ldc_dv_cl = [
    [[0.6735, 0.3306, 0.3881, 0.3572]],
    [[0.6622, 0.3638, 0.3848, 0.3530]],
    [[0.6476, 0.3197, 0.3577, 0.3321]]
]

ldc_Ours = [
    [[0.6583, 0.3522, 0.3839, 0.3562]],
    [[0.6707, 0.3977, 0.4036, 0.3730]],
    [[0.6767, 0.3632, 0.4007, 0.3748]]
]

models_ldc = models_trans
ldc = (ldc_deepergcn, ldc_transformer, ldc_dv, ldc_dv_cl, ldc_Ours)


settings = ['Vanilla CV', 'Stratified CV for\ndrug combinations', 'Stratified CV for\ncell lines']



def bar_plot(dataset, models, labels, name, metrics_id, y_lim=None):

    
    ax = plot_settings_bar.get_wider_axis(3, 4)

    colors = [plot_settings_bar.get_model_colors_ablation(mod) for mod in models]
    
    means = []
    stderrs = []

    for data in dataset:
        task_vals = []
        for result in data:
            result = np.concatenate(result)
            task_vals.append((np.mean(result[:, metrics_id]), np.std(result[:, metrics_id]) / np.sqrt(result.shape[0])))
        means.append([v[0] for v in task_vals])
        stderrs.append([v[1] for v in task_vals])

    min_val = [0.5, 0.2, 0.15, 0.1]
    
    plot_utils.grouped_barplot(
        ax, means, 
        settings,
        xlabel='', ylabel=name, color_legend=labels,
        nested_color=colors, nested_errs=stderrs, tickloc_top=False, rotangle=45, anchorpoint='right',
        min_val=min_val[metrics_id], y_lim=y_lim)
    
    plot_utils.format_ax(ax)


    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', 
                            ncols=1)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    
    # plt.savefig(f'bar_{name}_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'visulization/bar_{name}_ablation.pdf', dpi=300, bbox_inches='tight')

dataset = (trans, ldc, lc)

bar_plot(dataset, models_trans, models_trans, 'BACC', 0, (0.5, 0.75))
bar_plot(dataset, models_trans, models_trans, 'AUPRC', 1, (0.2, 0.52))
bar_plot(dataset, models_trans, models_trans, r'$F_1$', 2, (0.15, 0.52))
bar_plot(dataset, models_trans, models_trans, 'KAPPA', 3, (0.1, 0.47))