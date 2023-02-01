import os, sys
from pathlib import Path
import json
import numpy as np
from torch import is_tensor
from obj_det_analysis7 import evaluation, extract_sdc_due, get_fault_path, get_map_ap50_infnan, read_csv
from obj_det_evaluate_jsons import read_fault_file, load_json_indiv
# from obj_det_plot_metrics_all_models4 import get_m_err
from copy import deepcopy
sys.path.append("/home/fgeissle/fgeissle_ranger")
# from alficore.ptfiwrap_utils.evaluate import extract_ranger_bounds, get_ranger_bounds_quantiles
from alficore.ptfiwrap_utils.helper_functions import get_savedBounds_minmax
import matplotlib.pyplot as plt
import yaml
from alficore.ptfiwrap_utils.utils import read_yaml


def add_data(toplot_dict, json_path):

    # model_name = model_dict["model_name"]
    # dataset_name = model_dict["dataset_name"]
    # flt_type = model_dict["flt_type"]
    # suffix = model_dict["suffix"]
    # bits = model_dict["bits"]
    # label_name = model_dict["label_name"]
    
    # # Load from file saved in yolo_analysis3.py:
    # json_path = model_name + "_" + dataset_name + "_" + "results_1_" + flt_type + "_images" + suffix + ".json"


    results = load_json_indiv(json_path)
    print('Loaded:', json_path)
    # fltst = results["flts_filt"]
    # ids = results["img_ids"]

    orig_sdc, corr_sdc = results["metrics"]["sdc"]["orig_sdc"], results["metrics"]["sdc"]["corr_sdc"]
    orig_due, corr_due = results["metrics"]["due"]["orig_due"], results["metrics"]["due"]["corr_due"]
    orig_map, corr_map = results["metrics"]["map"]["orig_map"], results["metrics"]["map"]["corr_map"]
    orig_ap50, corr_ap50 = results["metrics"]["ap50"]["orig_ap50"], results["metrics"]["ap50"]["corr_ap50"]

    n_all, n_sdc, n_due =  results['nr_img_all'], results['nr_img_sdc'], results["nr_img_due"]
    
    sdc_rate = (np.array(n_sdc)/np.array(n_all)).tolist()
    due_rate = (np.array(n_due)/np.array(n_all)).tolist()
    # print('sdc rate (up to first 100)', sdc_rate[:100], '...', 'due rate (up to first 100)', due_rate[:100], '...')


    # SDC rate images
    m, err = get_m_err(sdc_rate)
    toplot_dict["sdc"]["mns"].append(m)
    toplot_dict["sdc"]["errs"].append(err)
    # DUE rate images
    m, err = get_m_err(due_rate)
    toplot_dict["due"]["mns_corr"].append(m)
    toplot_dict["due"]["errs_corr"].append(err)



    # # sdc weighted (commented for now)
    # diff_sdc = [np.array(corr_sdc[n]) - np.array(orig_sdc[n]) for n in range(len(orig_sdc))]
    # n_no_nan_inf = np.array(n_all) - np.array(n_due)
    # diff_sdc_ep_avg = [np.sum(diff_sdc[n])/n_no_nan_inf[n] for n in range(len(diff_sdc))]
    # # m_old= [np.sum(diff_sdc[n])/len(diff_sdc[n]) for n in range(len(diff_sdc))]
    # # diff_sdc_ep_avg = [np.mean(x) for x in diff_sdc]
    # m,err = get_m_err(diff_sdc_ep_avg)
    # toplot_dict["sdc_wgt"]["mns_diff"].append(m)
    # toplot_dict["sdc_wgt"]["errs_diff"].append(err)

    # # Orig sdc
    # orig_sdc_ep_avg = [np.mean(x) for x in orig_sdc]
    # m,err = get_m_err(orig_sdc_ep_avg)
    # toplot_dict["sdc_wgt"]["mns_orig"].append(m)
    # toplot_dict["sdc_wgt"]["errs_orig"].append(err)
    # # Corrupted sdc
    # corr_sdc_ep_avg = [np.mean(x) for x in corr_sdc]
    # m,err = get_m_err(corr_sdc_ep_avg)
    # toplot_dict["sdc_wgt"]["mns_corr"].append(m)
    # toplot_dict["sdc_wgt"]["errs_corr"].append(err)


    # orig map
    m,err = get_m_err([orig_map]) #make single number a list
    toplot_dict["map"]["mns_orig"].append(m)
    toplot_dict["map"]["errs_orig"].append(err)
    # Corr map
    m,err = get_m_err(corr_map)# TODO: fix get_m_err is nan if one is nan
    toplot_dict["map"]["mns_corr"].append(m)
    toplot_dict["map"]["errs_corr"].append(err)


    # orig ap50
    m,err = get_m_err([orig_ap50])
    toplot_dict["ap50"]["mns_orig"].append(m)
    toplot_dict["ap50"]["errs_orig"].append(err)
    # Corr ap50
    m,err = get_m_err(corr_ap50)
    toplot_dict["ap50"]["mns_corr"].append(m)
    toplot_dict["ap50"]["errs_corr"].append(err)


    # tp and bpos
    tpfpfn_orig = results['metrics']["sdc"]["tpfpfn_orig"]
    toplot_dict['tpfpfn']['orig']['tp'].append([n['tp'] for n in tpfpfn_orig])
    toplot_dict['tpfpfn']['orig']['fp'].append([n['fp'] for n in tpfpfn_orig])
    toplot_dict['tpfpfn']['orig']['fp_bbox'].append([n['fp_bbox'] for n in tpfpfn_orig])
    toplot_dict['tpfpfn']['orig']['fp_class'].append([n['fp_class'] for n in tpfpfn_orig])
    toplot_dict['tpfpfn']['orig']['fp_bbox_class'].append([n['fp_bbox_class'] for n in tpfpfn_orig])
    toplot_dict['tpfpfn']['orig']['fn'].append([n['fn'] for n in tpfpfn_orig])

    tpfpfn_corr = results['metrics']["sdc"]["tpfpfn_corr"]
    toplot_dict['tpfpfn']['corr']['tp'].append([n['tp'] for n in tpfpfn_corr])
    toplot_dict['tpfpfn']['corr']['fp'].append([n['fp'] for n in tpfpfn_corr])
    toplot_dict['tpfpfn']['corr']['fp_bbox'].append([n['fp_bbox'] for n in tpfpfn_corr])
    toplot_dict['tpfpfn']['corr']['fp_class'].append([n['fp_class'] for n in tpfpfn_corr])
    toplot_dict['tpfpfn']['corr']['fp_bbox_class'].append([n['fp_bbox_class'] for n in tpfpfn_corr])
    toplot_dict['tpfpfn']['corr']['fn'].append([n['fn'] for n in tpfpfn_corr])

    toplot_dict['tpfpfn']['bpos'].append(np.array(results["flts_sdc"])[6,:])

    # ax_leg.append(label_name)

    return toplot_dict

def ranger_string_add(x):
    return x[:-5] + '_ranger.json'

def get_m_err(list_to_plot):
    a = len(list_to_plot)
    list_to_plot = np.array(list_to_plot)[np.logical_not(np.isnan(list_to_plot))].tolist() #filter out nans
    if len(list_to_plot) < a:
        print('nans filter out for averaging in get_m_err:', a-len(list_to_plot))
    return np.mean(list_to_plot), np.std(list_to_plot)*1.96/np.sqrt(len(list_to_plot))

def load_json_indiv(gt_path):
    if gt_path == []:
        return None
    with open(gt_path) as f:
        coco_gt = json.load(f)
        f.close()
    return coco_gt

def get_index(el, ims):
    ep = int(el/ims)
    im = el % ims
    return ep, im


def plot_quants(qu_orig, qu_corr, folder, el, subfolder, title, target, bnds):
    fig, axes = plt.subplots(1,6, figsize=(35,6))
    layer_nrs = np.arange(len(bnds))
    # axes[0].plot(layer_nrs, qu_orig[:,0], '--', layer_nrs, qu_corr[:,0])
    # axes[1].plot(layer_nrs, qu_orig[:,1], '--', layer_nrs, qu_corr[:,1])
    # axes[2].plot(layer_nrs, qu_orig[:,2], '--', layer_nrs, qu_corr[:,2])
    # axes[3].plot(layer_nrs, qu_orig[:,3], '--', layer_nrs, qu_corr[:,3])
    # axes[4].plot(layer_nrs, qu_orig[:,4], '--', layer_nrs, qu_corr[:,4])

    # bnd_min_max = qu_orig[0]
    # bnd_qu_mns = qu_orig[1]
    # bnd_qu_errs = qu_orig[2]
    # bnd_qu_mins = qu_orig[3]
    # bnd_qu_maxs = qu_orig[4]
    # bnd_min_max = bnds
    # bnd_qu_mns = qu_orig[1]
    # bnd_qu_errs = qu_orig[2]
    bnd_qu_mins = [qu_orig['q10_min'], qu_orig['q25_min'], qu_orig['q50_min'], qu_orig['q75_min']]
    bnd_qu_maxs = [qu_orig['q10_max'], qu_orig['q25_max'], qu_orig['q50_max'], qu_orig['q75_max']]

    # Subtract mins
    # baseline_dim2 = np.array([bnd_min_max[:,0],]*2).T #min bound array for min, max
    # baseline_dim3 = np.array([bnd_min_max[:,0],]*3).T #min bound array for q25, q50, q75
    # baseline_dim5 = np.array([bnd_min_max[:,0],]*5).T #min bound array for min, q25, q50, q75, max
    # bnd_min_max = bnd_min_max - baseline_dim2
    # bnd_qu_mns = bnd_qu_mns - baseline_dim3
    # bnd_qu_mins = bnd_qu_mins - baseline_dim3
    # bnd_qu_maxs = bnd_qu_maxs - baseline_dim3
    # qu_corr = qu_corr - baseline_dim5

    normalizer = np.maximum(np.abs(bnd_qu_mins), np.abs(bnd_qu_maxs))
    lw = 3
    fs= 18
 
    # absolute
    axes[0].plot(layer_nrs, np.array(bnds)[:,0], 'grey', layer_nrs, qu_corr[:,0],'r', linewidth=lw) #first is special since neg is correct here, after above is indication of false
    axes[0].set_title('q0 (min)', fontsize=fs)
    axes[0].set_ylim([-1,1])
    axes[0].set_xlabel('Layer index', fontsize=fs)
    axes[0].set_ylabel('Activation magnitude (norm)', fontsize=fs)
    # axes[0].axhline(y=1., color='r', linestyle='--')


    n = 0
    axes[n+1].plot(layer_nrs, bnd_qu_mins[n]/normalizer[n], 'grey', layer_nrs, bnd_qu_maxs[n]/normalizer[n], 'grey', layer_nrs, qu_corr[:,n+1]/normalizer[n], 'r', linewidth=lw) #q10
    axes[n+1].set_title('q10', fontsize = fs)

    n = n+1
    axes[n+1].plot(layer_nrs, bnd_qu_mins[n]/normalizer[n], 'grey', layer_nrs, bnd_qu_maxs[n]/normalizer[n], 'grey', layer_nrs, qu_corr[:,n+1]/normalizer[n], 'r', linewidth=lw) #q25
    axes[n+1].set_title('q25', fontsize = fs)

    n = n+1
    axes[n+1].plot(layer_nrs, bnd_qu_mins[n]/normalizer[n], 'grey', layer_nrs, bnd_qu_maxs[n]/normalizer[n], 'grey', layer_nrs, qu_corr[:,n+1]/normalizer[n], 'r', linewidth=lw) #q25
    axes[n+1].set_title('q50', fontsize = fs)

    n = n+1
    axes[n+1].plot(layer_nrs, bnd_qu_mins[n]/normalizer[n], 'grey', layer_nrs, bnd_qu_maxs[n]/normalizer[n], 'grey', layer_nrs, qu_corr[:,n+1]/normalizer[n], 'r', linewidth=lw) #q25
    axes[n+1].set_title('q75', fontsize = fs)

    # axes[1].plot(layer_nrs, bnd_qu_mins[1]/normalizer[1], 'grey', layer_nrs, bnd_qu_maxs[1]/normalizer[1], 'grey', layer_nrs, qu_corr[:,1]/normalizer[1], 'r', linewidth=lw) #q25
    # # axes[1].axhline(y=1., color='r', linestyle='--')
    # axes[1].set_title('q25', fontsize = fs)

    # axes[2].plot(layer_nrs, bnd_qu_mins[2]/normalizer[:,1], 'grey', layer_nrs, bnd_qu_maxs[:,1]/normalizer[:,1], 'grey', layer_nrs, qu_corr[:,2]/normalizer[2], 'r', linewidth=lw) #q50
    # # axes[2].plot(layer_nrs, (bnd_qu_mns[:,1] - qu_corr[:,2]), layer_nrs, (bnd_qu_mins[:,1] - qu_corr[:,2]), layer_nrs, (bnd_qu_maxs[:,1] - qu_corr[:,2])) #q50
    # # axes[2].axhline(y=1., color='r', linestyle='--')
    # axes[2].set_title('q50 (median)', fontsize=fs)

    # axes[3].plot(layer_nrs, bnd_qu_mins[3]/normalizer[:,2], 'grey', layer_nrs, bnd_qu_maxs[:,2]/normalizer[:,2], 'grey', layer_nrs, qu_corr[:,3]/normalizer[:,2], 'r', linewidth=lw) #q75
    # # axes[3].plot(layer_nrs, (bnd_qu_mns[:,2] - qu_corr[:,3]), layer_nrs, (bnd_qu_mins[:,2] - qu_corr[:,3]), layer_nrs, (bnd_qu_maxs[:,2] - qu_corr[:,3])) #q75
    # # axes[3].axhline(y=1., color='r', linestyle='--')
    # axes[3].set_title('q75', fontsize=fs)

    n = n+1
    axes[n+1].plot(layer_nrs, np.array(bnds)[:,1], 'grey', layer_nrs, qu_corr[:,n+1], 'r', linewidth=lw)
    axes[n+1].set_title('q100 (max)', fontsize=fs)
    # axes[n+1].set_ylim([-100,100])

    # # Relative
    # axes[0].plot(layer_nrs, (bnd_min_max[:,0] - qu_corr[:,0])) #first is special since neg is correct here, after above is indication of false
    # axes[0].axhline(y=0., color='r', linestyle='--')
    # axes[1].plot(layer_nrs, (bnd_qu_mns[:,0] - qu_corr[:,1]), layer_nrs, (bnd_qu_mins[:,0] - qu_corr[:,1]), layer_nrs, (bnd_qu_maxs[:,0] - qu_corr[:,1])) #q25
    # axes[1].axhline(y=0., color='r', linestyle='--')
    # axes[2].plot(layer_nrs, (bnd_qu_mns[:,1] - qu_corr[:,2]), layer_nrs, (bnd_qu_mins[:,1] - qu_corr[:,2]), layer_nrs, (bnd_qu_maxs[:,1] - qu_corr[:,2])) #q50
    # axes[2].axhline(y=0., color='r', linestyle='--')
    # axes[3].plot(layer_nrs, (bnd_qu_mns[:,2] - qu_corr[:,3]), layer_nrs, (bnd_qu_mins[:,2] - qu_corr[:,3]), layer_nrs, (bnd_qu_maxs[:,2] - qu_corr[:,3])) #q75
    # axes[3].axhline(y=0., color='r', linestyle='--')
    # axes[4].plot(layer_nrs, (bnd_min_max[:,1] - qu_corr[:,4]))
    # axes[4].axhline(y=0., color='r', linestyle='--')

    # # Normalized
    # axes[0].plot(layer_nrs, (bnd_min_max[:,0] - qu_corr[:,0])/qu_orig[:,0]) #first is special since above is correct here, after above is indication of false
    # axes[0].axhline(y=0., color='r', linestyle='--')
    # axes[1].plot(layer_nrs, (qu_corr[:,1] - qu_orig[:,1])/qu_orig[:,1])
    # axes[1].axhline(y=0., color='r', linestyle='--')
    # axes[2].plot(layer_nrs, (qu_corr[:,2] - qu_orig[:,2])/qu_orig[:,2])
    # axes[2].axhline(y=0., color='r', linestyle='--')
    # axes[3].plot(layer_nrs, (qu_corr[:,3] - qu_orig[:,3])/qu_orig[:,3])
    # axes[3].axhline(y=0., color='r', linestyle='--')
    # axes[4].plot(layer_nrs, (qu_corr[:,4] - qu_orig[:,4])/qu_orig[:,4])
    # axes[4].axhline(y=0., color='r', linestyle='--')


    # axes[2].plot(layer_nrs, qu_orig[:,2], '--', layer_nrs, qu_corr[:,2])
    # axes[3].plot(layer_nrs, qu_orig[:,3], '--', layer_nrs, qu_corr[:,3])
    # axes[4].plot(layer_nrs, qu_orig[:,4], '--', layer_nrs, qu_corr[:,4])

    # # qu_orig[:,1], layer_nrs, qu_orig[:,2], layer_nrs, qu_orig[:,3], layer_nrs, qu_orig[:,4])
    # # axes[1].plot(layer_nrs, qu_corr[:,0], layer_nrs, qu_corr[:,1], layer_nrs, qu_corr[:,2], layer_nrs, qu_corr[:,3], layer_nrs, qu_corr[:,4])
    # axes[5].plot(layer_nrs, qu_orig[:,0]-qu_corr[:,0], layer_nrs, qu_corr[:,1]-qu_orig[:,1], layer_nrs, qu_corr[:,2]-qu_orig[:,2], layer_nrs, qu_corr[:,3]-qu_orig[:,3], layer_nrs, qu_corr[:,4]-qu_orig[:,4])
    # axes[5].legend(['0', '25', '50', '75', '100'])
    # # axes[1].legend(['0', '25', '50', '75', '100'])
    # # axes[2].legend(['0', '25', '50', '75', '100'])

    # axes[0].set_ylim([-100,100])
    # axes[1].set_ylim([-100,100])
    # axes[2].set_ylim([-100,100])

    fig.suptitle(title, fontsize=15)

    if not os.path.exists(folder + '/quant_plots/' + subfolder):
        os.makedirs(folder + '/quant_plots/' + subfolder)
    plt.savefig(folder + '/quant_plots/' + subfolder + str(el[0]) + '_' + str(el[1]) + '_' + target + '.png')
    print('saved:', folder + '/quant_plots/' + subfolder + str(el[0]) + '_' + str(el[1]) + '_' + target + '.png')
    plt.close()

def get_ep_im_sdc(dct_noranger):
    ids = dct_noranger['img_ids_sdc']
    nrs = dct_noranger['nr_img_sdc']
    ep_im = []
    
    for n in range(len(nrs)):
        to_add = []
        if nrs[n] > 0:
            if n==0:
                to_add = ids[:nrs[n]]
            else: 
                to_add = ids[np.sum(nrs[:n]):np.sum(nrs[:n])+nrs[n]]
        # if to_add != []:
        for x in to_add:
            ep_im.append([n, x])

    return ep_im

def get_ep_im_due(dct_noranger):
    ids = dct_noranger['img_ids_due']
    nrs = dct_noranger['nr_img_due']
    ep_im = []
    for n in range(len(nrs)):
        to_add = []
        if nrs[n] > 0:
            if n==0:
                to_add = ids[:nrs[n]]
            else: 
                to_add = ids[np.sum(nrs[:n]):np.sum(nrs[:n])+nrs[n]]
        for x in to_add:
            ep_im.append([n, x])

    return ep_im

def stack_custom(list_a, list_b, index):
    return np.vstack((np.array([n[index] for n in list_a]).T, np.array([n[index] for n in list_b]).T)).T

def get_layerwise_scaled_diffs(qu_corr, qu_orig):
    #Yields deviation from max divided by max bound

    # Reconstruct quantile values
    qu_keys = []
    for x in list(qu_orig.keys()):
        if ('min' not in x) and ('max' not in x) and ('avg' not in x) and ('ftraces' not in x) and ('N' not in x):
            qu_keys.append(x)

    assert len(qu_keys) > 0, "No quantile values found."
    nr_layers = len(qu_orig[qu_keys[0]+'_min'])

    for y in qu_keys: #assign all q-variables (q10, q20 etc.)
        globals()[f"q{y}_viol"] = np.zeros(nr_layers)

    def normalize_df(a,b):
        # Normalize feature values to -1 to 1. Values <0 means qu_corr does not exceed quantile max, >0 means max is exceeded
        # if a or b is zero: only -1, 0, 1 possible
        # if both nonzero:
        # if signs are the same, take max(abs(a), abs(b)) to normalize
        # if signs are different, take (abs(a)+abs(b))

        ret = np.zeros(a.shape)
        for n in range(a.shape[0]):
            for m in range(a.shape[1]):
                if a[n,m] == 0 or b[n,m] == 0:
                    ret[n,m] == np.sign(a[n,m]) - np.sign(b[n,m])
                else:
                    if np.sign(a[n,m]) != np.sign(b[n,m]):
                        ret[n,m] = (a[n,m]-b[n,m])/(abs(a[n,m])+abs(b[n,m]))
                    else:
                        ret[n,m] = (a[n,m]-b[n,m])/np.max([abs(a[n,m]), abs(b[n,m])]) 

        return ret

    # def min_max_scaling(x):
    #     # Scale values to range from 0 to 1, for explainability.
    #     # Protect against nan if elements of x are all the same.
    #     if np.min(x) != np.max(x):
    #         return (x-np.min(x))/(np.max(x)-np.min(x))
    #     else:
    #         if np.min(x) < 0:
    #             return np.zeros(x.shape)
    #         elif np.max(x) > 1:
    #             return np.ones(x.shape)
    #         else:
    #             return x

    # def get_mean(x):
    #     return [n[0] for n in x]

    qu_orig_max = np.vstack(([qu_orig[n+'_max'] for n in qu_keys])).T
    qu_orig_min = np.vstack(([qu_orig[n+'_min'] for n in qu_keys])).T
    # qu_orig_mean = np.vstack(([get_mean(qu_orig[n+'_avg']) for n in qu_keys])).T #for alternative trials
    
    # qu_df = normalize_df(qu_corr, qu_orig_max)

    # # qu_df1 = np.tanh((qu_corr - qu_orig_max)/(abs(qu_orig_max)+1e-8) + np.arctanh(0.5))
    # # qu_df2 = np.tanh((qu_orig_min - qu_corr)/(abs(qu_orig_min)+1e-8) + np.arctanh(0.5))
    # qu_df1 = np.tanh((qu_corr - qu_orig_max)/(abs(qu_orig_max)+1.e-8))
    # qu_df2 = -1. + np.heaviside(qu_orig_min - qu_corr, 0) * (1. + np.tanh((qu_orig_min - qu_corr)/(abs(qu_orig_min)+1.e-8)))
    # # qu_df2 = np.tanh((qu_orig_min - qu_corr)/(abs(qu_orig_min)+1e-8)*(-1e6 * np.sign(qu_orig_min - qu_corr))) #*(1/2)*(1+np.sign(qu_orig_min - qu_corr))
    # qu_df = np.maximum(qu_df1, qu_df2)

    qu_df1b = np.tanh((qu_corr - qu_orig_max)/(abs(qu_orig_max)+1.e-8))
    qu_df2b = np.tanh((qu_orig_min - qu_corr)/(abs(qu_orig_min)+1.e-8))
    qu_dfb = deepcopy(qu_df1b)
    qu_dfb[qu_corr < qu_orig_min] = qu_df2b[qu_corr < qu_orig_min] #if qu_corr < qu_orig_min take the negative version as it dominates over the positive and the other way round

    qu_viol = (qu_dfb + 1.)/2.
    # qu_viol = [min_max_scaling(qu_df[:, u]) for u in range(qu_df.shape[1])]


    return qu_viol, qu_keys, qu_orig_max, qu_orig_min

def get_layerwise_featuresum_dispersion(ftraces_dict, qu_orig, el):

        disp_corr_img = [np.std(ftraces_dict['corr'][el][lay]) for lay in range(len(ftraces_dict['corr'][el]))]
        disp_orig_img = [np.std(ftraces_dict['resil'][0][lay]) for lay in range(len(ftraces_dict['resil'][0]))]
        disp_orig_min = np.array(qu_orig["ftraces_fmap_disp"])[:,0].tolist()
        disp_orig_mean = np.array(qu_orig["ftraces_fmap_disp"])[:,1].tolist() #mean (over images) of the spread of activation sums, for a layer (one value as spread is over fmaps)
        disp_orig_max = np.array(qu_orig["ftraces_fmap_disp"])[:,2].tolist()
        disp_orig_std = np.array(qu_orig["ftraces_fmap_disp"])[:,3].tolist() #lists of 72


        # def norm_df(a,b):
        #     # all values are pos here because they are disps
        #     return abs(a-b)/(a+b)

        def norm_df_var(a,b,c):
            # all values are pos here because they are disps
            return abs(a-b)/c

        ft_unusual_disp = np.zeros(len(disp_corr_img))
        for u in range(len(disp_corr_img)):
            ft_unusual_disp[u] = norm_df_var(disp_orig_mean[u], disp_corr_img[u], disp_orig_std[u])

            # if disp_corr_img[u] > disp_orig_max[u]:
            #     ft_unusual_disp[u] = norm_df(disp_corr_img[u], disp_orig_max[u])
            # elif disp_corr_img[u] < disp_orig_min[u]:
            #     ft_unusual_disp[u] = norm_df(disp_orig_min[u], disp_corr_img[u])
            # else:
            #     ft_unusual_disp[u] = norm_df(disp_corr_img[u], disp_orig_max[u])
        # mask_above_max = np.array(disp_corr_img) - np.array(disp_orig_max) > 0
        # norm_disp(np.array(disp_corr_img)[mask_above_max], np.array(disp_orig_max)[mask_above_max])

        return ft_unusual_disp

def load_bounds_quantiles(ranger_file_name):
    # Load bounds ----------------------------------------------------------------------------------
    # bnds, bnds_min_max, bnds_qu = get_ranger_bounds_quantiles(None, None, ranger_file_name, False, False)
    bnds = get_savedBounds_minmax('./bounds/' + ranger_file_name + '.txt')
    # bnds = [[x[0], x[-1]] for x in bnds]
    print('Bounds loaded:', ranger_file_name)


    # Load original quantiles bounds ------------------------------------------------------------------------------
    try:
        qu_dict = load_json_indiv('./bounds/' + ranger_file_name + '_quantiles.json')
    except:
        try:
            qu_dict = load_json_indiv('./bounds/' + ranger_file_name + '_quantiles_ftraces.json')
        except:
            print('No file with saved quantiles found! Should be ', './bounds/' + ranger_file_name + '_quantiles.json')
            return bnds, None

    print('Quantiles loaded')

    return bnds, qu_dict

def update_feature_dict(feature_vector_dict, df_viols, is_sdc):

    # Length of dict will be total - due events.
    # nr_layers = len(df_viols[0])
    # q10_viol_df, q25_viol_df, q50_viol_df, q75_viol_df, q90_viol_df, max_viol_df = df_viols[0], df_viols[1], df_viols[2], df_viols[3], df_viols[4], df_viols[5]


    # # Define features -------------------
    # # Which layer locations are violating:
    # # list(np.where(min_viol_df > 0)[0]), 
    # det_layers = [list(np.where(q10_viol_df > 0)[0]), list(np.where(q25_viol_df > 0)[0]), list(np.where(q50_viol_df > 0)[0]), list(np.where(q75_viol_df > 0)[0]), list(np.where(max_viol_df > 0)[0])]
    # # By how much is it violated (compared to bounds):
    # # list(min_viol_df[min_viol_df>0]), 
    # det_deviation = [list(q10_viol_df[q10_viol_df>0]), list(q25_viol_df[q25_viol_df>0]), list(q50_viol_df[q50_viol_df>0]), list(q75_viol_df[q75_viol_df>0]), list(max_viol_df[max_viol_df>0])]


    # Save features: ------------------------
    # # feature_vector_dict = {'qu_glob_features': {'any_act': int(max(dets)>0), 'is_sdc': int(is_sdc)}, 'qu_features': {'nr_act_lay': dets, 'last_act_lay': [n[-1] if len(n)>0 else 0 for n in det_layers], 'max_dev_act': [max(n) if len(n)>0 else 0 for n in det_deviation] }}
    # # feature_vector_dict['qu_glob_features']['any_act'].append(int(max(dets)>0))
    feature_vector_dict['glob_features']['is_sdc'].append(int(is_sdc))

    # feature_vector_dict['features']['last_act_lay'].append([float(n[-1]/nr_layers) if len(n)>0 else 0 for n in det_layers])

    # # feature_vector_dict['qu_features']['nr_act_lay'].append([float(len(n)/nr_layers) for n in det_deviation]) #np.tanh(10*5/nr_layers)
    # feature_vector_dict['features']['nr_act_lay'].append([float(np.tanh(10*len(n)/nr_layers)) for n in det_deviation])

    # feature_vector_dict['features']['max_dev_act'].append([float(max(n)) if len(n)>0 else 0 for n in det_deviation])
    # # feature_vector_dict['qu_features']['max_dev_act'].append([float(np.mean(n)) if len(n)>0 else 0 for n in det_deviation])

    feature_vector_dict['features']['q_df_by_layer'].append(flatten_list(df_viols))
    # # feature_vector_dict['features']['q_df_by_layer']['q10'].append(q10_viol_df)

    # # if ft_unusual_disp is not None:
    # #     feature_vector_dict['features']['disp_max_sum'].append([np.max(ft_unusual_disp), np.sum(ft_unusual_disp)])
    
    return feature_vector_dict

def update_feature_dict_ftraces(feature_vector_dict_ftraces, f_act_corr_n, is_sdc):
    feature_vector_dict_ftraces['glob_features']['is_sdc'].append(int(is_sdc))
    feature_vector_dict_ftraces['features']['ftraces'].append(f_act_corr_n)
    return feature_vector_dict_ftraces

def plot_feature_distr(feature_vector_dict, feature_name, data_folder, target):
    # TODO: adjust for q10
    test1 = np.array(feature_vector_dict['qu_features'][feature_name])
    fig, ax = plt.subplots(1,6, figsize = (20,10))
    for n in range(6):
        ax[n].scatter(np.arange(test1.shape[0]), test1[:,n])
        ax[n].set_ylim([0,1])
        assert not (test1[:,n] > 1).any() and not (test1[:,n] < 0).any(), "Feature not scaled to 0-1."
    plt.savefig(data_folder + "/feature_test1_" + feature_name + "_" + target + ".png")

def flatten_list(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]

def norm_fact(f_act_orig, qu_orig): 
    f_act = [[] for n in range(len(f_act_orig))]
    for l in range(len(f_act)):
        f_act[l] = ( (np.array(f_act_orig[l]) - np.array(qu_orig['ftraces_mu'][l]))/(np.array(qu_orig['ftraces_sigma'][l])+1e-8) ).tolist()
    return f_act

def create_file_list(path):
    return_list = []

    for re in os.listdir(path):
        path1 = path + '/' + re
        for rep in os.listdir(path1):
            path2 = path1 + '/' + rep
            for repo in os.listdir(path2):
                path3 = path2 + '/' + repo
                for repos in os.listdir(path3):
                    path4 = path3 + '/' + repos
                    for ds in os.listdir(path4):
                        path5 = path4 + '/' + ds
                        if 'val' in os.listdir(path5) or 'train' in os.listdir(path5) or 'test' in os.listdir(path5): #for image classification use cases
                            for ds in os.listdir(path5):
                                path6 = path5 + '/' + ds
                                return_list.append(path6)
                        else:
                            return_list.append(path5)

    return return_list

def find_file_by_ending(folder_path, ending):
    return_list_all = []
    for n in folder_path:
        return_list = []
        for _, _, files in os.walk(n):
            for name in files:
                if name.endswith(ending):
                    return_list.append(n + '/' + name)
                    
        return_list_all.append(return_list)
    return return_list_all

def get_target_list(folder_list):
    scenario_list = find_file_by_ending(folder_list, "model_scenario.yml")
    
    target_list = []
    setup_list = []
    for sc in scenario_list:
        data = read_yaml(sc[0])[0]
        target_list.append(data['exp_type'])
        setup_list.append([data['model_name'], data['dataset']])
        
    return target_list, setup_list

def plot_v_line(q, qname, **kwargs):
    vshift = kwargs.get('vup', 0.)
    ax = kwargs.get('ax', None)
    if ax is not None:
        ax.axvline(x=q, color='b', lw=1, alpha=0.25)
    else:
        plt.axvline(x=q, color='b', lw=1, alpha=0.25)
    # plt.text(q, -0.03 +vshift, qname + '=' + str(round(q,2)), horizontalalignment='center', color='black', fontsize=14) #fontweight='bold'

def plot_h_line(q0, q10, perc_diff, **kwargs):
    # plt.axvline(x=q0, color='b')
    # print(perc_diff/(q10-q0), q0, q10)
    # plt.axhline(y=perc_diff/(q10-q0), xmin=q0, xmax=q10, color='g')
    ax = kwargs.get('ax', None)
    if ax is not None:
        ax.plot([q0, q10], [perc_diff/(q10-q0), perc_diff/(q10-q0)], color='g')
    else:
        plt.plot([q0, q10], [perc_diff/(q10-q0), perc_diff/(q10-q0)], color='g')
        
def plot_full_acts(ftraces_dict, im, lay, typ, is_sdc, target, flt_lay):
    dats = ftraces_dict[typ][im][lay]
    img_lay = flatten_list(flatten_list(dats)) # dims: images, layers
    q0, q10, q20, q30, q40, q50, q60, q70, q80, q90, q100 = np.quantile(img_lay, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.figure(figsize=(9, 5)) #figsize=(13, 10)
    plt.title('Layer: ' + str(lay*10) + ", N: " + str(len(img_lay))+ ', SDC: ' + str(is_sdc)+ ', Flt layer: ' + str(flt_lay))
    plot_v_line(q0, 'q0', vup=-0.03)
    plot_v_line(q10, 'q10', vup=-0.03)
    plot_v_line(q20, 'q20', vup=-0.03)
    plot_v_line(q30, 'q30', vup=-0.03)
    plot_v_line(q40, 'q40', vup=-0.03)
    plot_v_line(q50, 'q50', vup=-0.03)
    plot_v_line(q60, 'q60', vup=-0.03)
    plot_v_line(q70, 'q70', vup=-0.03)
    plot_v_line(q80, 'q80', vup=-0.03)
    plot_v_line(q90, 'q90', vup=-0.03)
    plot_v_line(q100, 'q100', vup=-0.03)
    plot_h_line(q0, q10, 0.1)
    plot_h_line(q10, q20, 0.1)
    plot_h_line(q20, q30, 0.1)
    plot_h_line(q30, q40, 0.1)
    plot_h_line(q40, q50, 0.1)
    plot_h_line(q50, q60, 0.1)
    plot_h_line(q60, q70, 0.1)
    plot_h_line(q70, q80, 0.1)
    plot_h_line(q80, q90, 0.1)
    plot_h_line(q90, q100, 0.1)
    #
    plt.hist(img_lay, bins=100, density=True) #density=True)
    plt.xlabel('Activation magnitude', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    txt = 'q0=' + str(round(q0,2))  + '\n' + 'q10=' + str(round(q10,2))  + '\n' + 'q20=' + str(round(q20,2))  + '\n' + 'q30=' + str(round(q30,2)) + '\n' + 'q40=' + str(round(q40,2))  + '\n' \
        + 'q50=' + str(round(q50,2))  + '\n' + 'q60=' + str(round(q60,2))  + '\n' + 'q70=' + str(round(q70,2))  + '\n' + 'q80=' + str(round(q80,2))  + '\n' \
            + 'q90=' + str(round(q90,2))  + '\n' + 'q100=' + str(round(q100,2))
    plt.text(1.01, 0.35, txt, 
         fontsize=9, fontfamily='Georgia', color='k',
         ha='left', va='bottom', transform=plt.gca().transAxes)
    
    sv_path = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/theory/Oct_2022/' + str(target) + '_' + str(im) + '_' + str(lay) + '_' + str(typ) + ".png"
    plt.savefig(sv_path, dpi=150)
    print('saved', sv_path)

def subplot_full_acts(ftraces_dict, im, lay, typ, figsz, axs):
    dats = ftraces_dict[typ][im][lay]
    img_lay = flatten_list(flatten_list(dats)) # dims: images, layers
    q0, q10, q20, q30, q40, q50, q60, q70, q80, q90, q100 = np.quantile(img_lay, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.figure(figsize=(9, 5)) #figsize=(13, 10)
    # axs[lay].title('Layer: ' + str(lay*10) + ", N: " + str(len(img_lay))+ ', SDC: ' + str(is_sdc)+ ', Flt layer: ' + str(flt_lay))
    plot_v_line(q0, 'q0', vup=-0.03, ax=axs[lay])
    plot_v_line(q10, 'q10', vup=-0.03, ax=axs[lay])
    plot_v_line(q20, 'q20', vup=-0.03, ax=axs[lay])
    plot_v_line(q30, 'q30', vup=-0.03, ax=axs[lay])
    plot_v_line(q40, 'q40', vup=-0.03, ax=axs[lay])
    plot_v_line(q50, 'q50', vup=-0.03, ax=axs[lay])
    plot_v_line(q60, 'q60', vup=-0.03, ax=axs[lay])
    plot_v_line(q70, 'q70', vup=-0.03, ax=axs[lay])
    plot_v_line(q80, 'q80', vup=-0.03,ax=axs[lay])
    plot_v_line(q90, 'q90', vup=-0.03,ax=axs[lay])
    plot_v_line(q100, 'q100', vup=-0.03,ax=axs[lay])
    plot_h_line(q0, q10, 0.1, ax=axs[lay])
    plot_h_line(q10, q20, 0.1, ax=axs[lay])
    plot_h_line(q20, q30, 0.1, ax=axs[lay])
    plot_h_line(q30, q40, 0.1, ax=axs[lay])
    plot_h_line(q40, q50, 0.1, ax=axs[lay])
    plot_h_line(q50, q60, 0.1, ax=axs[lay])
    plot_h_line(q60, q70, 0.1, ax=axs[lay])
    plot_h_line(q70, q80, 0.1, ax=axs[lay])
    plot_h_line(q80, q90, 0.1, ax=axs[lay])
    plot_h_line(q90, q100, 0.1, ax=axs[lay])
    #
    axs[lay].hist(img_lay, bins=100, density=True)
    if lay == len(axs)-1:
        axs[lay].set_xlabel('Activation magnitude', fontsize=14)
    axs[lay].set_ylabel('Density (Lay ' + str(lay*10) + ')', fontsize=14)
    txt = 'q0=' + str(round(q0,2))  + '\n' + 'q10=' + str(round(q10,2))  + '\n' + 'q20=' + str(round(q20,2))  + '\n' + 'q30=' + str(round(q30,2)) + '\n' + 'q40=' + str(round(q40,2))  + '\n' \
        + 'q50=' + str(round(q50,2))  + '\n' + 'q60=' + str(round(q60,2))  + '\n' + 'q70=' + str(round(q70,2))  + '\n' + 'q80=' + str(round(q80,2))  + '\n' \
            + 'q90=' + str(round(q90,2))  + '\n' + 'q100=' + str(round(q100,2))
    axs[lay].text(1.01, 0.0 - 0.001*lay*figsz[1]/len(axs), txt, 
         fontsize=10, fontfamily='Georgia', color='k', transform=axs[lay].transAxes) #, ha='left', va='bottom'

def get_quant_ftrace(folder):
    filelist = list(Path(folder).glob('**/*quantiles.json'))
    if len(filelist) > 0:
        quant_dict = load_json_indiv(str(filelist[0])) #keys: corr, resil, resil_corr; dims: epochs*images, layers (72), quants (6). resil has only 1 epoch!
    else: 
        quant_dict = None

    filelist = list(Path(folder).glob('**/*ftraces.json'))
    # filelist = [] #TODO TODO: suppressed for now
    if len(filelist) > 0:
        print('Loading ftraces ...')
        ftraces_dict = load_json_indiv(str(filelist[0])) #keys: corr, resil, resil_corr; dims: epochs*images, layers (72), ftraces (32, 64, etc, varies!). resil has only 1 epoch!
                
        # # Extract (normalized) f_act
        # f_act_corr = [flatten_list(norm_fact(ftraces_dict['corr'][n], qu_orig)) for n in range(len(ftraces_dict['corr']))] #each entry is f_act vector for one img.       
    else:
        ftraces_dict = None
        
    return quant_dict, ftraces_dict

def plot_acts_lays_separate(ftraces_dict, nr_ims, image_ids, ep_im_sdc, target, sdc_lays):
    typ = 'corr'
    do_separate_plots(ftraces_dict, typ, nr_ims, image_ids, ep_im_sdc, target, sdc_lays)
        
    typ = 'resil' #surrogate for original here if resil=no_ranger
    do_separate_plots(ftraces_dict, typ, nr_ims, image_ids, ep_im_sdc, target, sdc_lays)

def plot_acts_lays_in_one_plot(ftraces_dict, nr_ims, image_ids, ep_im_sdc, target, sdc_lays):
    
    typ = 'corr'
    do_subplots(ftraces_dict, typ, nr_ims, image_ids, ep_im_sdc, target, sdc_lays)
        
    typ = 'resil' #=orig here since no ranger
    do_subplots(ftraces_dict, typ, nr_ims, image_ids, ep_im_sdc, target, sdc_lays)

def do_subplots(ftraces_dict, typ, nr_ims, image_ids, ep_im_sdc, target, sdc_lays):
    for el in range(len(ftraces_dict[typ])):

        ep, im = get_index(el, nr_ims)

        if 'corr' in typ:
            is_sdc = [ep, image_ids[im]] in ep_im_sdc
            if is_sdc:
                sdc_pos = ep_im_sdc.index([ep, image_ids[im]])
            else:
                sdc_pos = None
                
            if 'hw' in target and sdc_pos is not None:
                flt_info = sdc_lays[sdc_pos]
            else:
                flt_info = '-'
        else:
            is_sdc = False
            sdc_pos = None
            flt_info = '-'
            target = 'orig'
        # print('ep', ep, 'im', im, 'is_sdc', is_sdc, 'el', el)

        figsz = (10, 22)
        nr_plots = len(ftraces_dict[typ][el])
        fig, axs = plt.subplots(nr_plots, figsize=figsz)
        fig.suptitle(str(target) + ", " + 'SDC: ' + str(is_sdc)+ ', Flt layer: ' + str(flt_info), fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        
        try:
            for lay in range(len(ftraces_dict[typ][el])):
                subplot_full_acts(ftraces_dict, el, lay, typ, figsz, axs)
                
            sv_path = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/theory/Oct_2022/combined_' + str(target) + '_' + str(ep) + '_' + str(im) + '_' + str(typ) + ".png"
            plt.savefig(sv_path, dpi=150, bbox_inches='tight')
            print('saved', sv_path)
        except:
            print('Could not plot activation distribution, probably nan values?')

def do_separate_plots(ftraces_dict, typ, nr_ims, image_ids, ep_im_sdc, target, sdc_lays):
    for el in range(len(ftraces_dict[typ])):
        # is_sdc = len(dct_noranger['flts_sdc'][el]) > 0

        ep, im = get_index(el, nr_ims)
        is_sdc = [ep, image_ids[im]] in ep_im_sdc
        if is_sdc:
            sdc_pos = ep_im_sdc.index([ep, image_ids[im]])
        else:
            sdc_pos = None
            
        if 'hw' in target and sdc_pos is not None:
            flt_info = sdc_lays[sdc_pos]
        else:
            flt_info = '-'
        for lay in range(len(ftraces_dict[typ][el])):
            plot_full_acts(ftraces_dict, el, lay, typ, is_sdc, target, flt_info)

def create_fake_ftrace():
    # for testing
    import random 
    fake_list = []
    for img in range(10):
        d_list = []
        for lay in range(8):
            c_list = []
            for kern in range(32):    
                b_list = []
                for x in range(10):
                    a_list = []        
                    for y in range(10):
                        a_list.append(3*random.random())
                    b_list.append(a_list)
                c_list.append(b_list)
            d_list.append(c_list)
        fake_list.append(d_list)

    ftraces_dict = {'corr': fake_list, 'resil': fake_list}
    quant_dict = None
    
    return quant_dict, ftraces_dict

def normalize_ftraces(ftrace_corr, qu_orig):
    res = []
    for x in range(len(ftrace_corr)):
        res.append(list((np.array(ftrace_corr[x]) - np.array(qu_orig['ftraces_mu'][x]))/(np.array(qu_orig['ftraces_sigma'][x])+1e-8) ))
    
    return res

def map_layers_fi_monitoring(no_layers):

    fi_lay_no = list(range(no_layers))
    mon_hook_lay_no = deepcopy(fi_lay_no)

    def swap_layer_index(lay_list, a, b):
        repl = deepcopy(lay_list[a])
        lay_list[a] = deepcopy(lay_list[b])
        lay_list[b] = repl
        return lay_list

    mon_hook_lay_no = swap_layer_index(mon_hook_lay_no, 71 -1, 72 -1) #-1 because here we start at 0, in beyondcompare from 1
    mon_hook_lay_no = swap_layer_index(mon_hook_lay_no, 72 -1, 74 -1) #-1 because here we start at 0, in beyondcompare from 1
    mon_hook_lay_no = swap_layer_index(mon_hook_lay_no, 73 -1, 74 -1) #-1 because here we start at 0, in beyondcompare from 1

    fi_mon_mapping = dict(zip(fi_lay_no, mon_hook_lay_no))
    return fi_mon_mapping


def main(argv):
    # 1. Select folder after min_test has finished
    # (2. run min_eval not needed)
    # 3. run this script (creates feature data files for detector)
    # 4. run train_detection_model_LR or similar

    """
    This script takes the raw data and creates jsons with quantile data features used for SDC prediction.
    In particular:
    - Extracts features and saves a json to './object_detection/quantile_detection_data/exp_1_' + target + '.json' for training.
    - Plots the layerwise distribution of activations if plot_full_act is true (skips the rest then).

    # - Can also plot the course of activations in experiment folder /quant_plots/... if plot_or_not tag is True.
    # - Further, plots the feature distribution in '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/feature_scale_analysis' if plot_or_not tag is True.
    """
    
    save_result = False #True #False only for testing, put back later TODO
    plot_full_act = False #Plot all act fcts 
    save_dfviol_stats = True
    # plot_or_not = False
    hwonly = False
    
    # Get data location and metadata -----------------------------------------------
    #########################################################

    # exp_list = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti']
    exp_list = ['yolo_kitti']

    for df in exp_list:

        # Typical convention: 1) hw 100 0-31 bits, input faults, hw 500 1-3 bits
        # New 100 all, 500 hw
        if df == 'yolo_coco':
            lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-14-09-43', '/nwstore/florian/LR_detector_data_auto/2022-12-14-12-32', '/nwstore/florian/LR_detector_data_auto/2022-12-16-04-20'] # noise, blur; rest of 100; 500hw - done
            # lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-16-04-20']
            # lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-01-08-34', '/nwstore/florian/LR_detector_data_auto/2022-11-11-22-59', '/nwstore/florian/LR_detector_data_auto/2022-11-29-09-31']
        elif df == "yolo_kitti":
            lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-14-16-18', '/nwstore/florian/LR_detector_data_auto/2022-12-20-23-09'] #100all, 500hw neurons, - done
            # lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-01-10-05/', '/nwstore/florian/LR_detector_data_auto/2022-11-12-16-14', '/nwstore/florian/LR_detector_data_auto/2022-11-30-12-23/']
        elif df == 'ssd_coco':
            lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-13-21-22', '/nwstore/florian/LR_detector_data_auto/2022-12-16-18-00']  #all 100; 500 hw - done
            # lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-01-11-29', '/nwstore/florian/LR_detector_data_auto/2022-11-13-13-46', '/nwstore/florian/LR_detector_data_auto/2022-12-05-19-39']
        elif df == 'ssd_kitti':
            lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-14-22-43', '/nwstore/florian/LR_detector_data_auto/2022-12-20-12-09'] #100all, 500hw neurons, - done
            # lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-14-21-00']
            # lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-01-12-50', '/nwstore/florian/LR_detector_data_auto/2022-11-13-17-13', '/nwstore/florian/LR_detector_data_auto/2022-12-05-11-12']
        elif df == 'retina_coco':
            lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-15-03-23', '/nwstore/florian/LR_detector_data_auto/2022-12-21-14-55'] #100 all, 500 hw - done
            # lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-01-14-04', '/nwstore/florian/LR_detector_data_auto/2022-11-13-20-49', '/nwstore/florian/LR_detector_data_auto/2022-12-06-10-27']
        elif df == 'retina_kitti':
            lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-15-13-51', '/nwstore/florian/LR_detector_data_auto/2022-12-22-08-28'] #100 all, 500 hw - done
            # lpaths = ['/nwstore/florian/LR_detector_data_auto/2022-12-01-16-36', '/nwstore/florian/LR_detector_data_auto/2022-12-08-09-54', '/nwstore/florian/LR_detector_data_auto/2022-12-07-10-06']


        # # TEST feature trace monitoring a la Schorn: 100 epochs for everything (inlc hw 32) 
        # lpaths = ['/nwstore/florian/LR_detector_data_auto/2023-01-19-11-22/']


        # Ranger bounds:
        ranger_file_name = df + '_v3_presum'
        
        # Save folder:
        save_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/' + df + '_presum/' #check
        
        for loading_path in lpaths:
            #########################################################
            
            # Get all subfolders and setup: ----------------------
            folder_list = create_file_list(loading_path)
            target_list, setup_list = get_target_list(folder_list)
            print('setup was:', setup_list)
            print('experiments were:', target_list)
            

            for u in range(len(folder_list)):
                folder = folder_list[u]
                target = target_list[u]
                if hwonly and "hw" not in target:
                    print('skipping:', target)
                    continue
                else:
                    print('Evaluating...', target)
                # adjust target
                if 'hwfault' in target:
                    bit_nr = folder.split('/')[-2][-7:-5].replace('_', '')
                    # bit_nr = folder.split('/')[-2][-9:-5].replace('_', '').replace(']','').replace(',','').replace('0', '').replace('[', '')

                    if 'weights' in folder:
                        target = target[:-1] + "weights" + bit_nr
                        print('It is weights.')
                    elif "neurons" in folder:
                        target = target[:-1] + "neurons" + bit_nr
                        print('It is neurons.')
                    else:
                        print('Neither neuron nor fault injection recognized... please check fault mode.')

                ##### 1. Load quantiles/ftraces
                # # Get quantiles/ftraces from inference images -------------------------------------------------------------------------------
                quant_dict, ftraces_dict = get_quant_ftrace(folder)
                # quant_dict, ftraces_dict = create_fake_ftrace() #for testing


                is_due = [np.isnan(n).any() or np.isinf(n).any() for n in quant_dict['corr']]
                print('DUE quantiles rate', np.sum(is_due)/len(is_due), "(out of)", np.sum(is_due), len(is_due))
                
                ##### 2. Get SDC info, faults
                # Analyze all data for fp, fn, tp ----------------------------------------------------------
                save_name = folder + "/results.json"
                if not os.path.exists(save_name):
                    evaluation(folder, save_name) 
                else:
                    print('File results.json already exists, not evaluating again.')
                results = load_json_indiv(save_name) #epochs, images


                # Get fault files ---------------------------------------------------------------------------
                fault_file_noranger = get_fault_path(folder, typ="no_ranger")
                fault_file_ranger = get_fault_path(folder, typ="ranger")
                ff_noranger = read_fault_file(fault_file_noranger) #tests #epochs*images, files are different for now
                ff_ranger = read_fault_file(fault_file_ranger)

                # # skip later
                # if "neurons" in target:
                #     all_lays = ff_noranger[1,:] #for neurons [0,:] for weights
                # elif "weights" in target:
                #     all_lays = ff_noranger[0,:] #for neurons [0,:] for weights
                # fig = plt.figure()
                # plt.hist(all_lays)
                # save_name_extra = "test_" + target + ".png"
                # fig.savefig(save_name_extra, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
                # print('saved as', save_name_extra)

                # ff_normal = read_fault_file('/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_10_trials/neurons_injs/per_image/objDet_20220325-100646_1_faults_[0_32]_bits/coco2017/val/yolov3_ultra_test_random_sbf_neurons_inj_1_10_1bs_coco2017_fault_locs.bin')
                if not (ff_noranger[-1,0] is None or ff_ranger[-1,0] is None):
                    assert np.sum(ff_noranger[-1,:]>0) == np.sum(ff_ranger[-1,:]>0), 'Different number of bnd violations for ranger and no ranger?' #binary detection should be the same


                # Extract and evaluate only SDC, DUE cases ------------------------------------------------------
                flt_type = "weights" if "weights" in target else "neurons"
                new_save_name = folder + "/results_sdc_due.json"
                dct_noranger = extract_sdc_due(save_name, flt_type, folder, fault_file_noranger, new_save_name, typ='no_ranger') 
                # if os.path.isdir(folder + '/ranger_model/'):
                #     dct_ranger = extract_sdc_due(save_name, flt_type, folder, fault_file_ranger, ranger_string_add(new_save_name), typ='ranger')


                # Show SDC bits and layers
                sdc_lays = []
                sdc_lays_all = []
                if 'weights' in target:
                    sdc_lays = [int(n) for n in np.array(dct_noranger['flts_sdc'])[0,:]]
                    sdc_lays_all = list(ff_noranger[0]) #whether sdc or not!
                elif 'neurons' in target:
                    # continue
                    sdc_lays = [int(n) for n in np.array(dct_noranger['flts_sdc'])[1,:]]
                    sdc_lays_all = list(ff_noranger[1])
                sdc_bpos = [int(n) for n in np.array(dct_noranger['flts_sdc'])[6,:]]
                sdc_vals = [n for n in np.array(dct_noranger['flts_sdc'])[-1,:]]
                print('check sdc bits count:', [np.sum(np.array(sdc_bpos) == n) for n in range(32)])
                if len(sdc_lays)>0:
                    print('sdc layers:', 'min', np.min(sdc_lays) if len(sdc_lays)>0 else [], 'max', np.max(sdc_lays) if len(sdc_lays)>0 else [], 'all', sdc_lays)

                # Reorder sdc_all (sdc or not (any) fault locations)
                if "yolo" in df:
                    no_yolo = 75
                    fi_mon_mapping = map_layers_fi_monitoring(no_yolo)
                    if len(sdc_lays_all)>0:
                        for n in range(len(sdc_lays_all)):
                            sdc_lays_all[n] = fi_mon_mapping[sdc_lays_all[n]] #replace layer with index according to monitored order (can be different!)
                        print('Yolo layer order adjusted from FI to Monitoring.')


                sdc_rates_per_epoch = np.array(dct_noranger['nr_img_sdc'])/np.array(dct_noranger['nr_img_all'])
                sdc_mean = np.mean(sdc_rates_per_epoch)
                sdc_err = 1.96*np.std(sdc_rates_per_epoch)/np.sqrt(len(sdc_rates_per_epoch))
                print('SDC rate:', sdc_mean, 'pm', sdc_err, '(out of)', np.sum(dct_noranger['nr_img_sdc']), np.sum(dct_noranger['nr_img_all']))

                due_rates_per_epoch = np.array(dct_noranger['nr_img_due'])/np.array(dct_noranger['nr_img_all'])
                due_mean = np.mean(due_rates_per_epoch)
                due_err = 1.96*np.std(due_rates_per_epoch)/np.sqrt(len(due_rates_per_epoch))
                print('DUE rate:', due_mean, 'pm', due_err, '(out of)', np.sum(dct_noranger['nr_img_due']), np.sum(dct_noranger['nr_img_all']))


                # Get sdc ids ------------------------------------------------------------------------------------
                ep_im_sdc = get_ep_im_sdc(dct_noranger)
                ep_im_due = get_ep_im_due(dct_noranger)

                # Comparison
                nr_eps = len(results['corr'])
                nr_ims = len(results['corr'][0])
                image_ids = [results['gt'][0][n]['id'] for n in range(len(results['gt'][0]))]
                
                
                ##### 3. Plot full act distributions (optional)
                # -------------------------------------------------
                # Plot quantile numbers, plot fault info
                if plot_full_act:
                    plot_acts_lays_in_one_plot(ftraces_dict, nr_ims, image_ids, ep_im_sdc, target, sdc_lays)
                    # plot_acts_lays_separate(ftraces_dict, nr_ims, image_ids, ep_im_sdc, target, sdc_lays)
                    continue
                
                
                
                ##### 4. Get saved reference bounds
                # Get min max and quantiles bounds/ftraces bounds  ---------------------------------------------------------------------------
                bnds, qu_orig = load_bounds_quantiles(ranger_file_name)
                # nr_layers = len(bnds)


                
                ### 5. Create feature dictionary -------------------------------------------------------

                lp = nr_eps*nr_ims #50
                due_count = 0
                sdc_count = 0
                sdc_viols_demo = []
                nonsdc_viols_demo = []
                feature_vector_dict = {'glob_features': {'is_sdc': [], 'sdc_rate': [], 'due_rate': [], 'qu_monitored': [], 'sdc_lays': [], 'sdc_bpos': [], 'sdc_vals': []}, 'features': {'nr_act_lay': [], 'last_act_lay': [], 'max_dev_act': [], 'disp_max_sum': [], 'q_df_by_layer':[] }}
                feature_vector_dict_ftraces = {'glob_features': {'is_sdc': [], 'sdc_rate': [], 'due_rate': [], 'sdc_lays': [], 'sdc_bpos': [], 'sdc_vals': []}, 'features': {'ftraces': []}}

                feature_vector_dict['glob_features']['sdc_rate'] = [sdc_mean, sdc_err]
                feature_vector_dict['glob_features']['due_rate'] = [due_mean, due_err]
                feature_vector_dict_ftraces['glob_features']['sdc_rate'] = [sdc_mean, sdc_err]
                feature_vector_dict_ftraces['glob_features']['due_rate'] = [due_mean, due_err]

                feature_vector_dict['glob_features']['sdc_lays'] = sdc_lays
                feature_vector_dict['glob_features']['sdc_bpos'] = sdc_bpos
                feature_vector_dict['glob_features']['sdc_vals'] = sdc_vals
                feature_vector_dict_ftraces['glob_features']['sdc_lays'] = sdc_lays
                feature_vector_dict_ftraces['glob_features']['sdc_bpos'] = sdc_bpos
                feature_vector_dict_ftraces['glob_features']['sdc_vals'] = sdc_vals


                for el in range(lp): #lp
                    if el/lp*100 % 10 == 0:
                        print(round(el/lp*100,2),'% progress')
                    ep, im = get_index(el, nr_ims)
                    
                    if [ep, image_ids[im]] in ep_im_due: #Exclude all due events
                        assert [ep, image_ids[im]] not in ep_im_sdc, "All due events should have been discarded from sdc?"
                        due_count += 1
                        # print('due found, skipping')
                        continue
                    

                    # #results
                    # res_orig = results['orig'][ep][im]
                    # res_corr = results['corr'][ep][im]
                    # res_orig_resil = results['orig_resil'][ep][im]
                    # res_corr_resil = results['corr_resil'][ep][im]

                    # SDC check
                    is_sdc = [ep, image_ids[im]] in ep_im_sdc
                    # print('index', el, ep, im, is_sdc)

                    # Quantile monitoring
                    qu_corr = np.array(quant_dict['corr'][el]) #size 72,5
                    # qu_orig_resil = np.array(quant_dict['resil'][el]) #resil has only one epoch
                    # qu_corr_resil = np.array(quant_dict['resil_corr'][el])
                    
                    


                    df_viols, qu_keys, qu_orig_max, qu_orig_min = get_layerwise_scaled_diffs(qu_corr, qu_orig)
                    feature_vector_dict = update_feature_dict(feature_vector_dict, df_viols, is_sdc)

                    
                    # ------------------ (optional, only for yolo now) -------------------------------------
                    if save_dfviol_stats:
                        lys_checked = 5
                        # For info of fault propagation (optional, plot shifts). TODO only implemented for yolo for now (mapping):
                        if len(sdc_lays_all) == 0: #input faults
                            ly = 0
                        else: #hw faults
                            ly = sdc_lays_all[el]
                            # print('lay', sdc_lays[sdc_count], 'bit:', sdc_bpos[sdc_count], 'val:', sdc_vals[sdc_count])
                            # if np.max(abs(sdc_vals[sdc_count])) > 5000*np.max(abs(qu_corr[ly])) and ly not in [71,73]:
                            #     print('sth wrong!', np.max(abs(sdc_vals[sdc_count])), np.max(abs(qu_corr[ly])))
                            #     sys.exit()

                        df_viols_demo = df_viols[ly:min(ly+lys_checked,df_viols.shape[0]), :]

                        if is_sdc:
                            if not np.isnan(df_viols_demo).any():
                                sdc_viols_demo.append(df_viols_demo)
                            else:
                                print('Nan due to detector encountered.')

                            # Extra check:
                            if len(sdc_lays) > 0:
                                ly2 = fi_mon_mapping[sdc_lays[sdc_count]]
                                assert ly2 == ly, "order not correct?"
                            sdc_count += 1
                        else:
                            if not np.isnan(df_viols_demo).any():
                                nonsdc_viols_demo.append(df_viols_demo)
                    # -------------------------------------------------------------------------

  
                    # if ftraces_dict is not None:
                    #     ft_unusual_disp = get_layerwise_featuresum_dispersion(ftraces_dict, qu_orig, el)
                    # else:
                    #     ft_unusual_disp = None
                    

                    # Make ftrace monitoring -------------------------
                    if ftraces_dict is not None:
                        ftrace_corr_norm = normalize_ftraces(ftraces_dict['corr'][el], qu_orig)
                        feature_vector_dict_ftraces = update_feature_dict_ftraces(feature_vector_dict_ftraces, ftrace_corr_norm, is_sdc)



                    # # plot sdc corr
                    # if plot_or_not:
                    #     flt = ff_noranger[:,el]
                    #     title = 'lay:' + str(flt[1]) + ', bit:' + str(flt[6]) + ', values:' + str(round(flt[8],2)) + ',' + str("{:.2e}".format(flt[9])) + ', det:' + str(flt[10])

                    #     if [ep, image_ids[im]] in ep_im_sdc:
                    #         plot_quants(qu_orig, qu_corr, folder, [ep, image_ids[im]], subfolder='sdc/', title=title, target=target, bnds=bnds)
                    #     else:
                    #         plot_quants(qu_orig, qu_corr, folder, [ep, image_ids[im]], subfolder='no_sdc/', title=title, target=target, bnds=bnds)


                # # plt.subplots()
                # # sdc_arr = ['blue' if n==True else 'red' for n in test_sdc]
                # # # label_arr = [None for n in test_sdc]
                # # # label_arr[np.where(np.array(sdc_arr) == 'blue')[0][0]] = 'SDC'
                # # # label_arr[np.where(np.array(sdc_arr) == 'red')[0][0]] = 'No SDC'
                # # plt.scatter(np.arange(0, len(test_sdc)), np.array(test_sum)[:,0], color=sdc_arr)
                # # plt.savefig('test.png')

                # # plt.subplots()
                # # # sdc_arr = ['blue' if n==True else 'red' for n in test_sdc]
                # # plt.scatter(np.arange(0, len(test_sdc)), test_max, color=sdc_arr)
                # # plt.savefig('test2.png')
                

                # # # Test feature validity: -----------------------------------------------------------------------------------
                # # if plot_or_not:
                # #     data_folder = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/feature_scale_analysis'

                # #     plot_feature_distr(feature_vector_dict, "nr_act_lay", data_folder, target)
                # #     plot_feature_distr(feature_vector_dict, "last_act_lay", data_folder, target)
                # #     plot_feature_distr(feature_vector_dict, "max_dev_act", data_folder, target)


                # 6. Save features ---------------------------------------------------------------------------------------------------------
                # save_folder = './object_detection/quantile_detection_data/'
                # save_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/'

                # Quantiles
                if save_result:
                    if not os.path.exists(os.path.dirname(save_folder)):
                        os.makedirs(os.path.dirname(save_folder))
                    json_name = save_folder + 'exp_' + target + '.json' #

                    with open(json_name, "w") as outfile: 
                        json.dump(feature_vector_dict, outfile)
                    print('saved:', json_name)


                # Ftraces
                if save_result and (ftraces_dict is not None):
                    json_name = save_folder + 'exp_ftraces_' + target + '.json' #

                    with open(json_name, "w") as outfile: 
                        json.dump(feature_vector_dict_ftraces, outfile)
                    print('saved:', json_name)


                # Df_viols_demo (optional):
                if save_dfviol_stats and len(sdc_viols_demo) > 0:
                    save_folder = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/'
                    save_name = save_folder + "quantile_shift_stats_" + target + ".json"
                    dct = {'target': target, 'sdc': [x.tolist() for x in sdc_viols_demo], 'nonsdc': [x.tolist() for x in nonsdc_viols_demo]}

                    with open(save_name, "w") as outfile: 
                        json.dump(dct, outfile)
                    print('saved:', save_name)




if __name__ == "__main__":
    main(sys.argv)