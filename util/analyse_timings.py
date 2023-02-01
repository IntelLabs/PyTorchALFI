import numpy as np
import torch
import matplotlib.pyplot as plt
from util.analyse_results_fct_v2 import *
# import pandas as pd
# import seaborn as sns


# TODO: different averages by fault for neurons and weights?
# TODO: what is the worst case epoch (for weights)? what is the rmse?

# Note:  Ranger addition makes everything much slower (why?) ~0.2ms -> 8ms per image. Reference baseline is therefore network with trivial Ranger.
# Note: # 100 epochs measure the timings
# Note: timings include ranger activations saving, if not 8ms -> 0.2ms much faster :)

##
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario3_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario5_neurons_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario4_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario6_resnet50_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario7_resnet50_convfcc_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario8_resnet50_convfcc_neurons_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario9_neurons/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario9_weights/'
folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario12_timing/'

##
# results_raw = get_selection(folder_all, 'RawNet')
results_raw = None

results_noranger = get_selection(folder_all, 'NoRanger')
# results_noranger = None

# results_ranger = get_selection(folder_all, 'Ranger')
results_ranger = None

# results_clips = get_selection(folder_all, 'Clips')
results_clips= None

# results_backflip = get_selection(folder_all, 'Backflip')
results_backflip= None

# results_scale = get_selection(folder_all, 'Rescale')
results_scale = None

# results_fmap = get_selection(folder_all, 'Fmapavg')
results_fmap = None





## Plots -------------------------------------------------------------------------
# ######################################################################################
# plt.close('all')

# Same for all plots:
pic_folder = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/pics/'
zeta = 1.96 #95% CI
save_all = False
theo_errors = False

top_nr = 1




## Study relation of out-of-bounds (oob) and misclassification (mcl)

def time_per_im(dict_chosen, ind, zeta):
    tms = dict_chosen[ind]["fi_time_list"] #for 1 fault only. shape 100x50 (epochs and nr of batches)
    tms = flatten(tms)
    tms = tms[:100*50] #restrict to 100 epochs to be comparable
    m, std, n = get_acc_std_N(tms)
    err = std/np.sqrt(n)

    bs = dict_chosen[ind]["batch_size"]
    # n_epochs = dict_chosen[ind]["Num_epochs"]
    m_im, err_im = m/bs, err/bs*zeta
    # print('batchsize was', bs, 'epochs', n_epochs)
    return m_im, err_im


def collect_tms(ind, zeta):
    m_all = []
    err_all = []

    if results_raw is not None:
        m, err = time_per_im(results_raw, ind, zeta)
        m_all.append(m)
        err_all.append(err)
    # print(m, err)

    if results_noranger is not None:
        m, err = time_per_im(results_noranger, ind, zeta)
        m_all.append(m)
        err_all.append(err)
        # print(m, err)

    if results_ranger is not None:
        m, err = time_per_im(results_ranger, ind, zeta)
        m_all.append(m)
        err_all.append(err)
        # print(m, err)

    if results_clips is not None:
        m, err = time_per_im(results_clips, ind, zeta)
        m_all.append(m)
        err_all.append(err)
        # print(m, err)

    if results_backflip is not None:
        m, err = time_per_im(results_backflip, ind, zeta)
        m_all.append(m)
        err_all.append(err)
        # print(m, err)

    if results_scale is not None:
        m, err = time_per_im(results_scale, ind, zeta)
        m_all.append(m)
        err_all.append(err)
        # print(m, err)

    if results_fmap is not None:
        m, err = time_per_im(results_fmap, ind, zeta)
        m_all.append(m)
        err_all.append(err)
        # print(m, err)

    return m_all, err_all


# ind = 0 #for 1 fault, same results will be obtaine for multiple faults
m_all1, err_all1 = collect_tms(0, zeta)
m_all10, err_all10 = collect_tms(1, zeta)
m_all100, err_all100 = collect_tms(2, zeta)



## Plot timings
x = np.arange(len(m_all1))  # the label locations
y = np.array(m_all1)*1000 #in ms
y_err = np.array(err_all1)*1000 #in ms

z = np.array(m_all10)*1000 #in ms
z_err = np.array(err_all10)*1000 #in ms
k = np.array(m_all100)*1000 #in ms
k_err = np.array(err_all100)*1000 #in ms

labels = ["NoRanger", "Ranger", "Clipping", "Backflip", "Rescale", "Fmap avg"]



width = 0.2  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, y, yerr=y_err, width=width, align='center', alpha=0.5, ecolor='black', capsize=5, label='1')
rects2 = ax.bar(x, z, yerr=z_err, width=width, align='center', alpha=0.5, ecolor='black', capsize=5, label='10')
rects3 = ax.bar(x + width, k, yerr=k_err, width=width, align='center', alpha=0.5, ecolor='black', capsize=5, label='100')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xticklabels
ax.set_ylabel('Inf. time per image (ms)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(title= "Faults")
# ax.set_ylim([7, 9]) #7-9ms range :TODO:


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height. Put numbers on bars."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

fig.tight_layout()
plt.show()

# # Save the figure and show
plt.savefig('C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/pics/RangerMethods_timing.png')
