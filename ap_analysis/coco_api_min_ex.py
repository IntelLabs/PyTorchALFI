import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import metrics
import matplotlib

def generate_list(nr_samples, chance_tp, chance_fp, conf_low, conf_up):
    base = {'tpfpfn': [], 'nr_obj': 0, 'conf': []} #tpfpfn encoding: 0 fn, 1 tp, 2 fp
    tp_generator = ['1'] * int(chance_tp*100) + ['0'] * int(100-chance_tp*100)

    conf_list = []
    tp_list = []
    for x in range(nr_samples):
        tpfn = int(random.choice(tp_generator))
        if tpfn == 1:
            tp_list.append(tpfn)
            conf_list.append(random.uniform(conf_low, conf_up))
        # else:
        #     conf_list.append(0)

        # create a fp in between chance:
        if random.uniform(0, 1) < chance_fp:
            tp_list.append(2)
            conf_list.append(random.uniform(conf_low, conf_up))

    base['tpfpfn'] = tp_list
    base["nr_obj"] = nr_samples
    base['conf'] =  conf_list

    return base


def create_fns(nr_fn, base, conf='random'):
    tp_ind = np.where(np.array(base['tpfpfn'])==1)[0]
    if conf == 'random':
        to_cancel = random.sample(tp_ind.tolist(), nr_fn)
    if conf=='high':
        tp_scores = np.array(base['conf'])[tp_ind]
        top_ind = np.argsort(tp_scores)[::-1]
        to_cancel = top_ind[:nr_fn]
    if conf == 'low':
        tp_scores = np.array(base['conf'])[tp_ind]
        low_ind = np.argsort(tp_scores)
        to_cancel = low_ind[:nr_fn]


    tpfpfn_new = []
    conf_new = []
    for x in range(len(base['tpfpfn'])):
        if x not in to_cancel:
            tpfpfn_new.append(base['tpfpfn'][x])
            conf_new.append(base['conf'][x])
    base['tpfpfn'] = tpfpfn_new
    base['conf'] = conf_new
    return base


def sort_list(base):
    base_sorted = {'tpfpfn': [], 'nr_obj': 0, 'conf': []} #tpfpfn encoding: 0 fn, 1 tp, 2 fp
    ind = np.argsort(np.array(base['conf']))
    ind = ind[::-1] #inverse

    base_sorted['tpfpfn'] = np.array(base['tpfpfn'])[ind]
    base_sorted['nr_obj'] = base['nr_obj']
    base_sorted['conf'] = np.array(base['conf'])[ind]
    return base_sorted


def p_r_interpolate(pr_list, rc_list):
        
    # interpolation:
    nd = len(pr_list)
    p_thres = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)

    pr = deepcopy(pr_list)
    q = np.zeros((len(p_thres),))
    # ss = np.zeros((len(p_thres),))
    q = q.tolist()


    for i in range(nd-1, 0, -1):
        if pr[i] > pr[i-1]:
            pr[i-1] = pr[i]

    inds = np.searchsorted(rc_list, p_thres, side='left') #adds extra zeros
    try:
        for ri, pi in enumerate(inds):
            q[ri] = pr[pi]
            # ss[ri] = dtScoresSorted[pi]
    except:
        pass

    return q, p_thres

def add_fp(base, base_fp):
    base['tpfpfn'] = base['tpfpfn'] + base_fp['tpfpfn']
    base['conf'] = base['conf'] + base_fp['conf']
    # #nrobj stays the same as fp dont count
    return base

def get_P_R(base_sorted):
    pr_list = []
    rc_list = []
    for u in range(1, len(base_sorted['tpfpfn'])+1):
        lst = base_sorted['tpfpfn'][:u]
        tp_glob = np.sum(lst == 1)
        fp_glob = np.sum(lst == 2)
        # tp_and_fn = np.sum(lst == 0 or lst==1)
        pr = tp_glob/(tp_glob + fp_glob)
        pr_list.append(pr)
        rc = tp_glob/(base_sorted['nr_obj'])
        rc_list.append(rc)
        # print('pr', pr)
        # print('rc', rc)
    
    return pr_list, rc_list

def get_corr_from_config(base, nr_fp, conf_fp_low, conf_fp_up, nr_fn, conf):
    
    # add faults
    base_fp = generate_list(nr_fp, 0, 1, conf_fp_low, conf_fp_up) #two high conf fps
    base_corr = add_fp(deepcopy(base), base_fp)
    base_corr = create_fns(nr_fn, base_corr, conf=conf)
    # base_corr = create_fns(nr_fn, base_corr, conf='random')
    # base_corr = create_fns(nr_fn, base_corr, conf='high')
    # base_corr = create_fns(nr_fn, base_corr, conf='low')

    base_corr_sorted = sort_list(base_corr)

    pr_list_corr, rc_list_corr = get_P_R(base_corr_sorted)

    pr_list_int_corr, rc_list_int_corr = p_r_interpolate(pr_list_corr, rc_list_corr)

    ap_corr = metrics.auc(rc_list_int_corr, pr_list_int_corr)

    return rc_list_corr, pr_list_corr, rc_list_int_corr, pr_list_int_corr, ap_corr, base_corr_sorted

# Note: 
# - low confidence fp does not matter!
# - high confidence fp matters!
# - more fp is worse
# - 

# Original data ----------------------------------------------------

random.seed(9)
nr_samples = 5 #nr of tp objects
chance_tp = 0.7
chance_fp = 0.3
conf_low = 0.7
conf_up = 1

# Generate list 
base = generate_list(nr_samples, chance_tp, chance_fp, conf_low, conf_up)
print(base)
# base = {'tpfpfn': [1, 1, 1, 1], 'nr_obj': 5, 'conf': [0.8, 0.7, 0.8, 0.9]}


# sort list by conf 
base_sorted = sort_list(base)

# Get P,R 
pr_list, rc_list = get_P_R(base_sorted)

# Interpolation 
pr_list_int, rc_list_int = p_r_interpolate(pr_list, rc_list)

# Area under curve 
ap = metrics.auc(rc_list_int, pr_list_int)
print(ap)





# Add faults: ------------------------------------------------------------------
nr_fp = 30
conf_fp_low = 0.5
conf_fp_up = 0.7
nr_fn = 0

rc_list_corr, pr_list_corr, rc_list_int_corr, pr_list_int_corr, ap_corr, _ = get_corr_from_config(base, nr_fp, conf_fp_low, conf_fp_up, nr_fn, conf='random')
print('ap', ap, 'ap_corr', ap_corr)



nr_fp = 10
conf_fp_low = 1
conf_fp_up = 1
nr_fn = 0

rc_list_corr2, pr_list_corr2, rc_list_int_corr2, pr_list_int_corr2, ap_corr2, _ = get_corr_from_config(base, nr_fp, conf_fp_low, conf_fp_up, nr_fn, conf='random')
print('ap', ap, 'ap_corr', ap_corr2)


nr_fp = 0
conf_fp_low = 0.5
conf_fp_up = 0.7
nr_fn = 10

rc_list_corr3, pr_list_corr3, rc_list_int_corr3, pr_list_int_corr3, ap_corr3, base_corr_sorted = get_corr_from_config(base, nr_fp, conf_fp_low, conf_fp_up, nr_fn, conf='random')
print('ap', ap, 'ap_corr', ap_corr3)


# Plot -----------------------------------------------------------------------------
fnt_size = 22
msize = 10
text_left = 0.45
plt.rcParams.update({'font.size': fnt_size-2}) #title and legend


# n = 0
file_name = 'pr_curves_1.png'
fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(rc_list_int, pr_list_int, label='PR interpolated', color='k', zorder=1)
ax.scatter(rc_list, pr_list, marker='o', s=msize, label='PR samples', zorder=2)
ax.set_xlabel('Recall', fontsize=fnt_size)
ax.set_ylabel('Precision', fontsize=fnt_size)
ax.set_title('AP='+str(round(ap,3)))
ax.legend(loc='lower left')
ax.set_ylim([0,1.05])
ax.text(0.7, 1, 'Fault-free', fontsize=fnt_size-3)
ax.tick_params(axis='both', which='major', labelsize=fnt_size)
plt.tight_layout()
fig.savefig(file_name, dpi=300)
print('saved image', file_name)


# n = 1
file_name = 'pr_curves_2.png'
fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(rc_list_int_corr, pr_list_int_corr, label='PR interpolated', color='k', zorder=1)
ax.scatter(rc_list_corr, pr_list_corr, marker='o', s=msize, label='PR samples', color='red', zorder=2)
ax.set_xlabel('Recall', fontsize=fnt_size)
ax.set_ylabel('Precision', fontsize=fnt_size)
ax.set_title('AP='+str(round(ap_corr,3)))
ax.legend(loc='lower left')
ax.set_ylim([0,1.05])
ax.text(text_left, 1, 'Fault injection:', fontsize=fnt_size-3)
ax.text(text_left, 0.95, 'FP: 30, conf [0.5,0.7]', fontsize=fnt_size-3)
ax.text(text_left, 0.9, 'FN: 0', fontsize=fnt_size-3)
ax.tick_params(axis='both', which='major', labelsize=fnt_size)
plt.tight_layout()
fig.savefig(file_name, dpi=300)
print('saved image', file_name)


# n = 2
file_name = 'pr_curves_3.png'
fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(rc_list_int_corr2, pr_list_int_corr2, label='PR interpolated', color='k', zorder=1)
ax.scatter(rc_list_corr2, pr_list_corr2, marker='o', s=msize, label='PR samples', color='red', zorder=2)
ax.set_xlabel('Recall', fontsize=fnt_size)
ax.set_ylabel('Precision', fontsize=fnt_size)
ax.set_title('AP='+str(round(ap_corr2,3)))
ax.legend(loc='lower left')
ax.set_ylim([0,1.05])
# ax.text(0.6, 1, 'Fault injection:', fontsize=fnt_size-2)
# ax.text(0.6, 0.95, 'FP: 10, conf [1, 1]', fontsize=fnt_size-2)
# ax.text(0.6, 0.9, 'FN: 0', fontsize=fnt_size-2)
ax.text(text_left, 1, 'Fault injection:', fontsize=fnt_size-3)
ax.text(text_left, 0.95, 'FP: 10, conf [1, 1]', fontsize=fnt_size-3)
ax.text(text_left, 0.9, 'FN: 0', fontsize=fnt_size-3)

ax.tick_params(axis='both', which='major', labelsize=fnt_size)
plt.tight_layout()
fig.savefig(file_name, dpi=300)
print('saved image', file_name)


# n = 3
file_name = 'pr_curves_4.png'
fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(rc_list_int_corr3, pr_list_int_corr3, label='PR interpolated', color='k', zorder=1)
ax.scatter(rc_list_corr3, pr_list_corr3, marker='o', s=msize, label='PR samples', color='red', zorder=2)
ax.set_xlabel('Recall', fontsize=fnt_size)
ax.set_ylabel('Precision', fontsize=fnt_size)
ax.set_title('AP='+str(round(ap_corr3,3)))
ax.legend(loc='lower left')
ax.set_ylim([0,1.05])
# ax.text(0.6, 1, 'Fault injection:', fontsize=fnt_size)
# ax.text(0.6, 0.95, 'FP: 0, conf [0.5,0.7]', fontsize=fnt_size-2)
# ax.text(0.6, 0.9, 'FN: 10', fontsize=fnt_size-2)
ax.text(text_left, 1, 'Fault injection:', fontsize=fnt_size-3)
ax.text(text_left, 0.95, 'FP: 0, conf [0.5,0.7]', fontsize=fnt_size-3)
ax.text(text_left, 0.9, 'FN: 10', fontsize=fnt_size-3)

ax.tick_params(axis='both', which='major', labelsize=fnt_size-2)
plt.tight_layout()
fig.savefig(file_name, dpi=300)
print('saved image', file_name)





# n = 2
# ax[n].scatter(rc_list_corr2, pr_list_corr2, marker='o', s=6, label='PR samples')
# ax[n].plot(rc_list_int_corr2, pr_list_int_corr2, label='PR interpolated')
# ax[n].set_xlabel('Recall', fontsize=fnt_size)
# ax[n].set_ylabel('Precision', fontsize=fnt_size)
# ax[n].set_title('AP='+str(round(ap_corr2,3)))
# ax[n].legend(loc='lower left')
# ax[n].set_ylim([0,1.05])
# ax[n].text(0.6, 1, 'Fault injection:', fontsize=fnt_size-2)
# ax[n].text(0.6, 0.95, 'FP: 10, conf [1, 1]', fontsize=fnt_size-2)
# ax[n].text(0.6, 0.9, 'FN: 0', fontsize=fnt_size-2)
# ax[n].tick_params(axis='both', which='major', labelsize=fnt_size)


# n = 3
# ax[n].scatter(rc_list_corr3, pr_list_corr3, marker='o', s=6, label='PR samples')
# ax[n].plot(rc_list_int_corr3, pr_list_int_corr3, label='PR interpolated')
# ax[n].set_xlabel('Recall', fontsize=fnt_size)
# ax[n].set_ylabel('Precision', fontsize=fnt_size)
# ax[n].set_title('AP='+str(round(ap_corr3,3)))
# ax[n].legend(loc='lower left')
# ax[n].set_ylim([0,1.05])
# ax[n].text(0.6, 1, 'Fault injection:', fontsize=fnt_size)
# ax[n].text(0.6, 0.95, 'FP: 0, conf [0.5,0.7]', fontsize=fnt_size-2)
# ax[n].text(0.6, 0.9, 'FN: 10', fontsize=fnt_size-2)
# ax[n].tick_params(axis='both', which='major', labelsize=fnt_size-2)



# n = 2
# ax[n].scatter(rc_list_corr2, pr_list_corr2, marker='o', s=6, label='PR samples')
# ax[n].plot(rc_list_int_corr2, pr_list_int_corr2, label='PR interpolated')
# ax[n].set_xlabel('Recall', fontsize=fnt_size)
# ax[n].set_ylabel('Precision', fontsize=fnt_size)
# ax[n].set_title('AP='+str(round(ap_corr2,3)))
# ax[n].legend(loc='lower left')
# ax[n].set_ylim([0,1.05])
# ax[n].text(0.6, 1, 'Fault injection:', fontsize=fnt_size-2)
# ax[n].text(0.6, 0.95, 'FP: 10, conf [1, 1]', fontsize=fnt_size-2)
# ax[n].text(0.6, 0.9, 'FN: 0', fontsize=fnt_size-2)
# ax[n].tick_params(axis='both', which='major', labelsize=fnt_size)

# n = 3
# ax[n].scatter(rc_list_corr3, pr_list_corr3, marker='o', s=6, label='PR samples')
# ax[n].plot(rc_list_int_corr3, pr_list_int_corr3, label='PR interpolated')
# ax[n].set_xlabel('Recall', fontsize=fnt_size)
# ax[n].set_ylabel('Precision', fontsize=fnt_size)
# ax[n].set_title('AP='+str(round(ap_corr3,3)))
# ax[n].legend(loc='lower left')
# ax[n].set_ylim([0,1.05])
# ax[n].text(0.6, 1, 'Fault injection:', fontsize=fnt_size)
# ax[n].text(0.6, 0.95, 'FP: 0, conf [0.5,0.7]', fontsize=fnt_size-2)
# ax[n].text(0.6, 0.9, 'FN: 10', fontsize=fnt_size-2)
# ax[n].tick_params(axis='both', which='major', labelsize=fnt_size-2)


# plt.tight_layout()
# fig.savefig(file_name)
# print('saved image', file_name)




# Plot extra ----------------------
# file_name = 'ap_analysis/pr_curve.png'
# fig, ax = plt.subplots(1, 1)
# ax.scatter(rc_list, pr_list)
# ax.set_xlabel('recall')
# ax.set_ylabel('precision')
# fig.savefig(file_name)
# print('saved image', file_name)


