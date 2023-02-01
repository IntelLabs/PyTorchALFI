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
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario9_weights/'
# folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario10_oob_test_weights/'
folder_all = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario11_oob_test_neurons/'

##

# results_noranger_exp = get_selection(folder_all, 'NoRanger_exp')
results_noranger_exp = None

# results_noranger_mnt = get_selection(folder_all, 'NoRanger_mnt')
results_noranger_mant = None

results_noranger_all = get_selection(folder_all, 'NoRanger_all')
# results_noranger_all = None





## Plots -------------------------------------------------------------------------
# ######################################################################################
# plt.close('all')

# Same for all plots:
pic_folder = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/pics/'
zeta = 1.96 #95% CI
save_all = False
theo_errors = False

top_nr = 1


dict_chosen = results_noranger_all
# dict_chosen = results_noranger_exp
# dict_chosen = results_noranger_mnt




## Study relation of out-of-bounds (oob) and misclassification (mcl)

# choose top prediction, choose nr of faults, choose res technique? (none)
# Note: split on 2 gpus means only 500 images have same fault in common. errors dont make sense?
ind = 0 #first item for 1 fault
fi_ranger_act = dict_chosen[ind]["fi_ranger_act"] #for 1 fault only. shape 100x40x25
fi_prediction = dict_chosen[ind]["fi_correct_pred_top1"] #shape 100x40x25

ranger_act = dict_chosen[ind]["orig_ranger_act"] #no faults, for comparison. shape 100x40x25
prediction = dict_chosen[ind]["orig_correct_pred_top1"] #shape 100x40x25


oob_eps, oob_mcl_eps, ib_eps, ib_mcl_eps, oob_eps_filt, oob_mcl_eps_filt, ib_eps_filt, ib_mcl_eps_filt = analyse_miscl(fi_ranger_act, fi_prediction, ranger_act, prediction)


# m1, m2, mcond = get_cond_p(oob_eps, oob_mcl_eps, zeta)
# print('cond oob', m1, m2, mcond)
m1_filt, m2_filt, mcond_filt = get_cond_p(oob_eps_filt, oob_mcl_eps_filt, zeta) #probability oob, probability oob and mcl, conditional prob mcl given that oob
print('p (oob)', m1_filt, 'p (oob and mcl)', m2_filt, 'p (mcl | oob)', mcond_filt)
mcl_oob = mcond_filt*m1_filt
cl_oob = (1- mcond_filt)*m1_filt
print('mcl_oob', mcl_oob, 'cl_oob', cl_oob)

# m1, m2, mcond = get_cond_p(ib_eps, ib_mcl_eps, zeta)
# print('cond ib', m1, m2, mcond)
m1_filt, m2_filt, mcond_filt = get_cond_p(ib_eps_filt, ib_mcl_eps_filt, zeta)
# print('cond ib filt', m1_filt, m2_filt, mcond_filt)
print('p (ib)', m1_filt, 'p (ib and mcl)', m2_filt, 'p (mcl | ib)', mcond_filt)
mcl_ib = mcond_filt*m1_filt
cl_ib = (1- mcond_filt)*m1_filt
print('mcl_ib', mcl_ib)

print('mcl', mcl_oob + mcl_ib)
print('cl', cl_oob + cl_ib)
print('prec', mcl_oob/(mcl_oob + cl_oob))
print('rec', mcl_oob/(mcl_oob + mcl_ib))


# Get prediction lists for later:
cl_flat = np.array(flatten(prediction)) #all correct cls wo faults
fi_cl_flat = np.array(flatten(fi_prediction)) #all correct cls w faults. shape 500x1000 epoch x images
fi_mcl_flat_filt = cl_flat * (1 - fi_cl_flat) #orignal correct but fi incorrect



## Review which faults were injected:
runset_sel = dict_chosen[0]["runset"]
bs = dict_chosen[0]["batch_size"]
flip_pos = runset_sel[6,:] #all bit flip position is entry 6. shape 500 (1 fault per epoch)
# Note: in runset there are in general more faults than are used. E.g. only one weight fault is used per batch.


# For one weight per batch  -------------------------------------------------------------------------------------------
# # # If one new fault per each batch
# # Map misclassified images to used bit flips (one bitflip per batch only)
# nonzero = np.where(fi_mcl_flat_filt)[0] #indices of mcl images.
# btch_inds = np.floor(nonzero/bs) #respective batch nr
# btch_inds = np.array([int(n) for n in btch_inds]) #make int
#
# #if dataset is combined from n sets:
# # n = 2
# btch_inds[btch_inds >= 5000] +=  100000 - 5000 #need to skip unused faults in weights due to batchsize > 1
#
# flip_pos_btch = flip_pos[btch_inds]



# For 1 weight per epoch ----------------------------------------------------------------------------------------------

# # If only one weight flip per epoch:
# mcls = np.array(flatten(fi_mcl_flat_filt))
# bit_mcl_list = []
# for n in range(len(flip_pos)):
#     bit = flip_pos[n]
#     mcls = np.sum(fi_mcl_flat_filt[n, :]) #nr of mcl due to fault at that bit fault (one epoch)
#     bit_mcl_list.append([bit, mcls])

# If one bit flip every image
# mcls = np.array(flatten(fi_mcl_flat_filt))


# Neurons: ------------------------------------------------------------------------------------------------------------
nonzero = np.where(fi_mcl_flat_filt)[0]
flip_pos_btch = flip_pos[nonzero]
# print(flip_pos_btch[0:100])



## Analyse flipped bits
# x = flip_pos #bit_mcl_list[:,0]
# y = mcls #bit_mcl_list[:,1]

hist_x = list(range(min(flip_pos), max(flip_pos)+1)) #all possible bitflips
hist_y = []
for a in hist_x:
    # hist_y.append(np.sum(fi_mcl_flat_filt[flip_pos == a]))
    hist_y.append(np.sum(flip_pos_btch == a)/len(flip_pos_btch))

fig = plt.figure()
plt.bar(hist_x, hist_y, width = 1)


ax = fig.gca()
ax.set_xlabel('bit position')
ax.set_ylabel('rel. contribution to all misclassifications')

# Plot labels in correct scientific notation
round_to = 4
for i, v in enumerate(hist_y):
    if v != 0:
        if v > 0.01:
            ax.text(i + hist_x[0] - 0.25, v + .01, np.round(v, round_to)) #, color='blue', fontweight='bold')
        else:
            ax.text(i + hist_x[0] - 0.25, v + .01, "{:.2e}".format(np.round(v, round_to))) #, color='blue', fontweight='bold')
    # else:
        # ax.text(i + hist_x[0], v + .01, str(v))




if save_all:
    # pic_name = pic_folder + "errors_mnt.png"
    pic_name = pic_folder + "errors_all_neurons.png"
    # pic_name = pic_folder + "errors_exp.png"
    fig.savefig(pic_name, dpi=150)




# ## Analyse layers etc.
# ## Analyse group confusion etc.


