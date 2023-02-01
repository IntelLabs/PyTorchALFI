import numpy as np
import torch
import matplotlib.pyplot as plt
from util.analyse_results_fct_v2 import *

# TODO: different averages by fault for neurons and weights?
# TODO: what is the worst case epoch (for weights)? what is the rmse?


##
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario3_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario5_neurons_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario4_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario6_resnet50_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario7_resnet50_convfcc_weights_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario8_resnet50_convfcc_neurons_top1/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario9_neurons/'
folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario9_weights/'


##

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



##
# results_noranger = results_noranger[0:3] #eliminate 100 faults
# results_ranger = results_ranger[0:3] #eliminate 100 faults


##
# cls = results_noranger[1]["class_pred"]
# # cls2 = flatten(cls)
# preds = results_noranger[1]["orig_correct_pred_top1"]
# # preds2 = flatten(preds)
#
# mcls_list = []
# for ep in range(len(cls)):
#     for btch in range(len(cls[0])):
#         for im in range(len(cls[0][0])):
#             if not preds[ep][btch][im]:
#                 mcls_list.append(cls[ep][btch][im][0])



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

# # choose top prediction, choose nr of faults, choose res technique? (none)
# # Note: split on 2 gpus means only 500 images have same fault in common. errors dont make sense?
#
#
# dict_chosen = results_noranger
# # dict_chosen = results_ranger
#
# ind = 0 #first item for 1 fault
# fi_ranger_act = dict_chosen[ind]["fi_ranger_act"] #for 1 fault only. shape 100x40x25
# fi_prediction = dict_chosen[ind]["fi_correct_pred_top1"] #shape 100x40x25
#
# ranger_act = dict_chosen[ind]["orig_ranger_act"] #no faults, for comparison. shape 100x40x25
# prediction = dict_chosen[ind]["orig_correct_pred_top1"] #shape 100x40x25
#
#
# oob_eps, oob_mcl_eps, ib_eps, ib_mcl_eps, oob_eps_filt, oob_mcl_eps_filt, ib_eps_filt, ib_mcl_eps_filt = analyse_miscl(fi_ranger_act, fi_prediction, ranger_act, prediction)
#
#
# m1, m2, mcond = get_cond_p(oob_eps, oob_mcl_eps, zeta)
# # print('cond oob', m1, m2, mcond)
# m1_filt, m2_filt, mcond_filt = get_cond_p(oob_eps_filt, oob_mcl_eps_filt, zeta)
# print('cond oob filt', m1_filt, m2_filt, mcond_filt)
# print('mcl', mcond_filt*m1_filt)
#
# m1, m2, mcond = get_cond_p(ib_eps, ib_mcl_eps, zeta)
# # print('cond ib', m1, m2, mcond)
# m1_filt, m2_filt, mcond_filt = get_cond_p(ib_eps_filt, ib_mcl_eps_filt, zeta)
# print('cond ib filt', m1_filt, m2_filt, mcond_filt)
# print('mcl', mcond_filt*m1_filt)







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
y_topic = ["acc_unfiltered" + "_top" + str(top_nr), "fi_acc_unfiltered" + "_top" + str(top_nr)] #use acc_unfiltered topic as fault=0 entry
leg = []

fig, axs = plt.subplots(1,1)
plt.grid()

# Acc NoRanger
plot_accuracy(results_noranger, x_topic, y_topic, zeta, theo_errors, axs, leg, "NoRanger")

# Acc Ranger
plot_accuracy(results_ranger, x_topic, y_topic, zeta, theo_errors, axs, leg, "Ranger")

# Acc Clips
plot_accuracy(results_clips, x_topic, y_topic, zeta, theo_errors, axs, leg, "Clipping")

# Acc Backflip
plot_accuracy(results_backflip, x_topic, y_topic, zeta, theo_errors, axs, leg, "Backflip")

# Acc Rescale
plot_accuracy(results_scale, x_topic, y_topic, zeta, theo_errors, axs, leg, "Rescale")

# Acc avg fmap
plot_accuracy(results_fmap, x_topic, y_topic, zeta, theo_errors, axs, leg, "Fmap_avg")


axs.set_xlabel('faults')
axs.set_ylabel('Top-' + str(top_nr) + ' accuracy')
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
plot_topic = "fi_correct_pred" + "_top" + str(top_nr)
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

if results_scale is not None:
    dict_type = results_scale
    title_name = 'Rescale'
    plotDistribution(dict_type, pic_folder, save_all, title_name, plot_topic, x_topic)

if results_fmap is not None:
    dict_type = results_fmap
    title_name = 'Fmap_avg'
    plotDistribution(dict_type, pic_folder, save_all, title_name, plot_topic, x_topic)


## SDC rate
x_topic = "fixed_nr_faults"
y_topic = ['sdc' + "_top" + str(top_nr), 'fi_sdc' + "_top" + str(top_nr)]
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

# Rescale
if results_scale is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_scale, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5)
    leg.append('Rescale')
    print('sdc rescale', x_data, y_data)

# Fmap avg
if results_fmap is not None:
    x_data, y_data, y_error = get_fault_nofault_combined(results_fmap, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5)
    leg.append('Fmap_avg')
    print('sdc fmap_avg', x_data, y_data)


axs.set_xlabel('faults')
axs.set_ylabel('Top-' + str(top_nr) + ' SDC rate')
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
    print('ranger coverage', x_data, y_data)

if results_clips is not None:
    x_data, y_data, y_error = get_ranger_coverage(results_clips, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5, label='Clipping')
    print('clipping coverage', x_data, y_data)

if results_backflip is not None:
    x_data, y_data, y_error = get_ranger_coverage(results_backflip, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5, label='Backflip')
    print('backflip coverage', x_data, y_data)

if results_scale is not None:
    x_data, y_data, y_error = get_ranger_coverage(results_scale, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5, label='Rescale')
    print('rescale coverage', x_data, y_data)

if results_fmap is not None:
    x_data, y_data, y_error = get_ranger_coverage(results_fmap, x_topic, y_topic, zeta)
    axs.errorbar(x_data, y_data, y_error, capsize=5, label='Fmap_avg')
    print('fmap_avg coverage', x_data, y_data)


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


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Save
if save_all:
    pic_name = pic_folder + "error_coverage.png"
    fig.savefig(pic_name, dpi=150)
