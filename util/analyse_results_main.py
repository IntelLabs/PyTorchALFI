import numpy as np
import torch
import matplotlib.pyplot as plt
from util.analyse_results_fct import *

# todo: max error in comparison?
# TODO: different averages by fault for neurons and weights?



##
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario3_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario5_neurons_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario4_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario6_resnet50_weights_top1/'
folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario7_resnet50_convfcc_weights_top1/'



##
results_ranger = get_selection(folder_all, 'Ranger')
# results_ranger = None

results_noranger = get_selection(folder_all, 'NoRanger')
# results_noranger = None

results_clips = get_selection(folder_all, 'Clips')
# results_clips= None

results_backflip = get_selection(folder_all, 'Backflip')
# results_backflip= None



## Plots -------------------------------------------------------------------------
# ######################################################################################
# plt.close('all')

# Same for all plots:
pic_folder = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/pics/'
zeta = 1.96 #95% CI
save_all = False
theo_errors = False



## Review which faults were injected:
# plt.close('all')
# toPlot = results_ranger[0]
#
# runset_sel = toPlot["runset"]
# inj_target = toPlot["injection_target"]
# runset_labels = []
# if inj_target == "weights":
#     runset_labels = ["layers", "filters", "channels", "depth", "height", "width", "flipped bit"]
# elif inj_target == "neurons":
#     runset_labels = ["batch_nr", "layers", "channels", "depth", "height", "width", "flipped bit"]
#
# fig, axs = plt.subplots(ncols=len(runset_sel))
# fig.suptitle(inj_target, fontsize=16)
#
# for n in range(len(runset_sel)):
#     # if n==3:
#     #     continue #skip depth info
#     # binwidth = 1. , bins=np.arange(min(runset_sel[n,:]), max(runset_sel[n,:]) + binwidth, binwidth)
#     axs[n].hist(runset_sel[n,:], bins= min(max(max(runset_sel[n,:])+1,1),100)) #max 100 bins for visibility
#     axs[n].set_title(runset_labels[n])
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
#
# # Save
# if save_all:
#     pic_name = pic_folder + "errors.png"
#     fig.savefig(pic_name, dpi=150)




## Timing tests
# lats = np.array([19.207409620285034, 20.05427384376526, 21.3982937335968, 21.613080263137817, 1936.2042512893677])/500*1000 #in ms
# labels = ['NoRanger', 'Ranger', 'Clipping', 'Backflip', 'Rescale']
#
# fig = plt.figure()
# ax = plt.gca()
#
# ax.bar(labels, lats)
# for i, v in enumerate(lats):
#     ax.text(i - 0.25, v + 4, str(np.round(v,2)), color='black') #, fontweight='bold')
#
# ax.set_ylabel('approx. inference time per image (ms)')
# plt.show()
# # Setup: one run (split on 2 gpu so number refers to 500 images, batchsize 20, 1 weight fault)






## Accuracies rate
# ----------------------------------------------------------------------------------------------------------------------
x_topic = "fixed_nr_faults"
y_topic = ["acc_unfiltered", "fi_acc_unfiltered"] #use acc_unfiltered topic as fault=0 entry
leg = []
# y_error_max = list(np.ones(len(y_error))*0.02)
# axs.errorbar(x_data, y_data, y_error_max, capsize=5)

fig, axs = plt.subplots(1,1)
plt.grid()

# Acc Ranger
if results_ranger is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_ranger, x_topic, y_topic, zeta)


    if theo_errors: # add theoretical error
        y_error_theo = get_y_error_theo(results_ranger, zeta) #theoretical worst-case error
        axs.errorbar(x_data, y_data, y_error_theo, capsize=5) # ls = ''
        # eb[-1][0].set_linestyle('--')
    else:
        axs.errorbar(x_data, y_data, y_error, capsize=5)

    leg.append("Ranger")
    print('acc ranger', x_data, y_data)


# Acc NoRanger
if results_noranger is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_noranger, x_topic, y_topic, zeta)

    if theo_errors: # add theoretical error
        y_error_theo = get_y_error_theo(results_noranger, zeta) #theoretical worst-case error
        axs.errorbar(x_data, y_data, y_error_theo, capsize=5)
        # eb[-1][0].set_linestyle('--')
    else:
        axs.errorbar(x_data, y_data, y_error, capsize=5)

    leg.append("NoRanger")
    print('acc no_ranger', x_data, y_data)

# Acc Clips
if results_clips is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_clips, x_topic, y_topic, zeta)

    if theo_errors: # add theoretical error
        y_error_theo = get_y_error_theo(results_clips, zeta) #theoretical worst-case error
        axs.errorbar(x_data, y_data, y_error_theo, capsize=5)
        # eb[-1][0].set_linestyle('--')
    else:
        axs.errorbar(x_data, y_data, y_error, capsize=5)

    leg.append("Clipping")
    print('acc clipping', x_data, y_data)


# Acc Backflip
if results_backflip is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_backflip, x_topic, y_topic, zeta)

    if theo_errors: # add theoretical error
        y_error_theo = get_y_error_theo(results_backflip, zeta) #theoretical worst-case error
        axs.errorbar(x_data, y_data, y_error_theo, capsize=5)
        # eb[-1][0].set_linestyle('--')
    else:
        axs.errorbar(x_data, y_data, y_error, capsize=5)

    leg.append("Backflip")
    print('acc backflip', x_data, y_data)


axs.set_xlabel('faults')
axs.set_ylabel('Top-1 accuracy')
axs.legend(leg)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Save
if save_all:
    pic_name = pic_folder + "acc.png"
    fig.savefig(pic_name, dpi=150)



## Distribution of accuracies across epochs
# ----------------------------------------------------------------------------------------------------------------------

# Ranger --------
plot_topic = "fi_correct_pred"
x_topic = "fixed_nr_faults"

if results_ranger is not None:
    dict_type = results_ranger
    title_name = 'Ranger'
    plotDistribution(dict_type, pic_folder, save_all, title_name, plot_topic, x_topic)

if results_noranger is not None:
    dict_type = results_noranger
    title_name = 'NoRanger'
    plotDistribution(dict_type, pic_folder, save_all, title_name, plot_topic, x_topic)

if results_clips is not None:
    dict_type = results_clips
    title_name = 'Clips'
    plotDistribution(dict_type, pic_folder, save_all, title_name, plot_topic, x_topic)

if results_backflip is not None:
    dict_type = results_backflip
    title_name = 'Backflip'
    plotDistribution(dict_type, pic_folder, save_all, title_name, plot_topic, x_topic)




## SDC rate
x_topic = "fixed_nr_faults"
y_topic = ['sdc', 'fi_sdc']
leg = []

fig, axs = plt.subplots(1,1)
plt.grid()

# Ranger
if results_ranger is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_ranger, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5)
    leg.append('Ranger')
    print('sdc ranger', x_data, y_data)

# # No Ranger
if results_noranger is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_noranger, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5)
    leg.append('NoRanger')
    print('sdc no_ranger', x_data, y_data)

# Clips
if results_clips is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_clips, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5)
    leg.append('Clipping')
    print('sdc clipping', x_data, y_data)

# Backflip
if results_backflip is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_backflip, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5)
    leg.append('Backflip')
    print('sdc backflip', x_data, y_data)

axs.set_xlabel('faults')
axs.set_ylabel('SDC rate')
axs.legend(leg)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Save
if save_all:
    pic_name = pic_folder + "sdc.png"
    fig.savefig(pic_name, dpi=150)









## Ranger coverage - how many ranger layers get activated

x_topic = "fixed_nr_faults"
y_topic = ["orig_ranger_act", "fi_ranger_act"]

fig, axs = plt.subplots(1,1)
plt.grid()

if results_ranger is not None:
    x_data, y_data, y_error = get_ranger_coverage(results_ranger, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5, label='Ranger')

if results_clips is not None:
    x_data, y_data, y_error = get_ranger_coverage(results_clips, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5, label='Clipping')

if results_backflip is not None:
    x_data, y_data, y_error = get_ranger_coverage(results_backflip, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5, label='Backflip')

# Theory:
p_large_error = 0.01
y_theor = [1 - (1 - p_large_error) ** i for i in x_data]
axs.plot(x_data, y_theor, label='p = ' + str(p_large_error), linestyle='dashed')
#
p_large_error = 0.02
y_theor = [1-(1-p_large_error)**i for i in x_data]
axs.plot(x_data, y_theor, label = 'p = ' + str(p_large_error), linestyle='dashed')
#
p_large_error = 0.03
y_theor = [1-(1-p_large_error)**i for i in x_data]
axs.plot(x_data, y_theor, label = 'p = ' + str(p_large_error), linestyle='dashed')
#
p_large_error = 0.04
y_theor = [1-(1-p_large_error)**i for i in x_data]
axs.plot(x_data, y_theor, label = 'p = ' + str(p_large_error), linestyle='dashed')

axs.set_xlabel('faults')
axs.set_ylabel('avg portion of activated protection layers') #it is the average (over all epochs) portion of images where at least one ranger layer was activated given the n faults per image.
# axs.set_title("Ranger")
axs.legend()
print('ranger coverage', x_data, y_data)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Save
if save_all:
    pic_name = pic_folder + "error_coverage.png"
    fig.savefig(pic_name, dpi=150)

##
# results_clips[0]["fi_ranger_act"]
# sum(flatten(results_clips[0]["fi_ranger_act"]))