import os, sys
from pathlib import Path
import json
import numpy as np
from torch import is_tensor
from obj_det_analysis7 import evaluation, extract_sdc_due, get_fault_path, get_map_ap50_infnan, read_csv
from obj_det_evaluate_jsons import read_fault_file, load_json_indiv
# from obj_det_plot_metrics_all_models4 import get_m_err
from copy import deepcopy
from alficore.ptfiwrap_utils.evaluate import extract_ranger_bounds, extract_ranger_bounds_auto2, get_ranger_bounds_quantiles
from alficore.ptfiwrap_utils.helper_functions import get_savedBounds_minmax
import matplotlib.pyplot as plt
import torch

from attic.quantiles_extract_features_plot2 import get_layerwise_scaled_diffs, flatten_list
from train_detection_model_LR3 import get_tp_fp_fn_multiclass, get_p_r, create_feature_labels, get_tp_fp_fn_multiclass_all_classes


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
    bnds = [[x[0], x[-1]] for x in bnds]
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


def update_feature_dict(feature_vector_dict, df_viols, is_sdc, ft_unusual_disp):

    # Length of dict will be total - due events.
    nr_layers = len(df_viols[0])
    q10_viol_df, q25_viol_df, q50_viol_df, q75_viol_df, max_viol_df = df_viols[0], df_viols[1], df_viols[2], df_viols[3], df_viols[4] #, df_viols[5]


    # Define features -------------------
    # Which layer locations are violating:
    # list(np.where(min_viol_df > 0)[0]), 
    det_layers = [list(np.where(q10_viol_df > 0)[0]), list(np.where(q25_viol_df > 0)[0]), list(np.where(q50_viol_df > 0)[0]), list(np.where(q75_viol_df > 0)[0]), list(np.where(max_viol_df > 0)[0])]
    # By how much is it violated (compared to bounds):
    # list(min_viol_df[min_viol_df>0]), 
    det_deviation = [list(q10_viol_df[q10_viol_df>0]), list(q25_viol_df[q25_viol_df>0]), list(q50_viol_df[q50_viol_df>0]), list(q75_viol_df[q75_viol_df>0]), list(max_viol_df[max_viol_df>0])]


    # Save features: ------------------------
    # feature_vector_dict = {'qu_glob_features': {'any_act': int(max(dets)>0), 'is_sdc': int(is_sdc)}, 'qu_features': {'nr_act_lay': dets, 'last_act_lay': [n[-1] if len(n)>0 else 0 for n in det_layers], 'max_dev_act': [max(n) if len(n)>0 else 0 for n in det_deviation] }}
    # feature_vector_dict['qu_glob_features']['any_act'].append(int(max(dets)>0))
    feature_vector_dict['glob_features']['is_sdc'].append(int(is_sdc))

    feature_vector_dict['features']['last_act_lay'].append([float(n[-1]/nr_layers) if len(n)>0 else 0 for n in det_layers])

    # feature_vector_dict['qu_features']['nr_act_lay'].append([float(len(n)/nr_layers) for n in det_deviation]) #np.tanh(10*5/nr_layers)
    feature_vector_dict['features']['nr_act_lay'].append([float(np.tanh(10*len(n)/nr_layers)) for n in det_deviation])

    feature_vector_dict['features']['max_dev_act'].append([float(max(n)) if len(n)>0 else 0 for n in det_deviation])
    # feature_vector_dict['qu_features']['max_dev_act'].append([float(np.mean(n)) if len(n)>0 else 0 for n in det_deviation])

    feature_vector_dict['features']['q_df_by_layer'].append(flatten_list(df_viols))
    # feature_vector_dict['features']['q_df_by_layer']['q10'].append(q10_viol_df)

    if ft_unusual_disp is not None:
        feature_vector_dict['features']['disp_max_sum'].append([np.max(ft_unusual_disp), np.sum(ft_unusual_disp)])
    
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



def get_data(folder, ranger_file_name):


    ##### Get quantile/ftrace info
    # Get min max and quantiles bounds/ftraces bounds  ---------------------------------------------------------------------------
    bnds, qu_orig = load_bounds_quantiles(ranger_file_name)
    nr_layers = len(bnds)


    # # Get quantiles/ftraces from inference images -------------------------------------------------------------------------------
    filelist = list(Path(folder).glob('**/*quantiles.json'))
    assert len(filelist)==1, "Too many files with similar name found."
    quant_dict = load_json_indiv(str(filelist[0])) #keys: corr, resil, resil_corr; dims: epochs*images, layers (72), quants (6). resil has only 1 epoch!


    ##### Get SDC info
    # Analyze all data for fp, fn, tp ----------------------------------------------------------
    save_name = folder + "/results.json"
    # evaluation(folder, save_name) #TODO: activate if needed
    if not os.path.exists(save_name):
        evaluation(folder, save_name) #TODO: activate if needed
    else:
        print('File results.json already exists, not evaluating again.')
    results = load_json_indiv(save_name) #epochs, images

    # Get fault files ---------------------------------------------------------------------------
    fault_file_noranger = get_fault_path(folder, typ="no_ranger")
    fault_file_ranger = get_fault_path(folder, typ="ranger")
    ff_noranger = read_fault_file(fault_file_noranger) #tests #epochs*images, files are different for now
    ff_ranger = read_fault_file(fault_file_ranger)
    # ff_normal = read_fault_file('/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_10_trials/neurons_injs/per_image/objDet_20220325-100646_1_faults_[0_32]_bits/coco2017/val/yolov3_ultra_test_random_sbf_neurons_inj_1_10_1bs_coco2017_fault_locs.bin')

    assert np.sum(ff_noranger[-1,:]>0) == np.sum(ff_ranger[-1,:]>0), 'Different number of bnd violations for ranger and no ranger?' #binary detection should be the same


    # Extract and evaluate only SDC, DUE cases ------------------------------------------------------
    flt_type = "weights" if "weights" in folder else "neurons"

    new_save_name = folder + "/results_sdc_due.json"
    dct_noranger = extract_sdc_due(save_name, flt_type, folder, fault_file_noranger, new_save_name, typ='no_ranger') 
    # print(np.array(dct_noranger['flts_sdc'])[6,:])
    if os.path.isdir(folder + '/ranger_model/'):
        dct_ranger = extract_sdc_due(save_name, flt_type, folder, fault_file_ranger, ranger_string_add(new_save_name), typ='ranger')
    # np.array(dct_noranger['flts_sdc'])[6,:]

    if 'weights' in folder:
        sdc_lays = [int(n) for n in np.array(dct_noranger['flts_sdc'])[0,:]]
    else:
        sdc_lays = [int(n) for n in np.array(dct_noranger['flts_sdc'])[1,:]]
    sdc_bpos = [int(n) for n in np.array(dct_noranger['flts_sdc'])[6,:]]
    print('check sdc bits count:', [np.sum(np.array(sdc_bpos) == n) for n in range(32)])
    print('sdc layers:', 'min', np.min(sdc_lays), 'max', np.max(sdc_lays), 'all', sdc_lays)

    # Get sdc ids ------------------------------------------------------------------------------------
    ep_im_sdc = get_ep_im_sdc(dct_noranger)
    ep_im_due = get_ep_im_due(dct_noranger)

    # Comparison
    nr_eps = len(results['corr'])
    nr_ims = len(results['corr'][0])
    image_ids = [results['gt'][0][n]['id'] for n in range(len(results['gt'][0]))]

    return quant_dict, qu_orig, bnds, [ep_im_sdc, ep_im_due], [nr_eps, nr_ims, image_ids], sdc_lays, sdc_bpos


def plot_explanation(qu_orig, qu_corr, folder, el, subfolder, title, target, bnds):
    fig, axes = plt.subplots()
    layer_nrs = np.arange(len(bnds))

    def normalize_df(a,b):
        return (a-b)/(abs(a) + abs(b))
    
    def min_max_scaling(x):
        # Protect against nan if elements of x are all the same.
        if np.min(x) != np.max(x):
            return (x-np.min(x))/(np.max(x)-np.min(x))
        else:
            if np.min(x) < 0:
                return np.zeros(x.shape)
            elif np.max(x) > 1:
                return np.ones(x.shape)
            else:
                return x
    
    qu_orig_max = np.vstack((qu_orig['q10_max'], qu_orig['q25_max'], qu_orig['q50_max'], qu_orig['q75_max'], qu_orig['q90_max'], np.array(bnds)[:,1])).T
    qu_df = normalize_df(qu_corr[:,1:], qu_orig_max)

    # q10_viol = min_max_scaling(qu_df[:,0])
    # q25_viol = min_max_scaling(qu_df[:,1])
    # q50_viol = min_max_scaling(qu_df[:,2])
    # q75_viol = min_max_scaling(qu_df[:,3])
    # max_viol = min_max_scaling(qu_df[:,4])



    # normalizer = np.maximum(np.abs(bnd_qu_mins), np.abs(bnd_qu_maxs))
    lw = 2
    fs= 18
    
    axes.plot(layer_nrs, qu_df[:,0],'orange', linewidth=lw, label='q10')
    axes.plot(layer_nrs, qu_df[:,1],'grey', linewidth=lw, label='q25')
    axes.plot(layer_nrs, qu_df[:,2],'blue', linewidth=lw, label='q50')
    axes.plot(layer_nrs, qu_df[:,3],'green', linewidth=lw, label='q75')
    axes.plot(layer_nrs, qu_df[:,4],'black', linewidth=lw, label='q90')
    axes.plot(layer_nrs, qu_df[:,5],'black', linewidth=lw, label='q100')
    # axes.plot(layer_nrs, q10_viol,'orange', linewidth=lw, label='q10')
    # axes.plot(layer_nrs, q25_viol,'grey', linewidth=lw, label='q25')
    # axes.plot(layer_nrs, q50_viol,'blue', linewidth=lw, label='q50')
    # axes.plot(layer_nrs, q75_viol,'green', linewidth=lw, label='q75')
    # axes.plot(layer_nrs, max_viol,'black', linewidth=lw, label='q100')

    # # absolute
    # axes[0].plot(layer_nrs, np.array(bnds)[:,0], 'grey', layer_nrs, qu_corr[:,0],'r', linewidth=lw) #first is special since neg is correct here, after above is indication of false
    # axes[0].set_title('q0 (min)', fontsize=fs)
    # axes[0].set_ylim([-1,1])
    axes.set_xlabel('Layer index', fontsize=fs)
    axes.set_ylabel('Activation magnitude (norm)', fontsize=fs)
    axes.axhline(y=0., color='r', linestyle='--')

    axes.legend()
    fig.suptitle(title, fontsize=fs)

    if not os.path.exists(folder + '/' + subfolder):
        os.makedirs(folder + '/' + subfolder)
    plt.savefig(folder + '/' + subfolder + str(el[0]) + '_' + str(el[1]) + '_' + target + '.png')
    print('saved:', folder + '/' + subfolder + str(el[0]) + '_' + str(el[1]) + '_' + target + '.png')
    plt.close()



def plot_explanation2(qu_orig, qu_corr, folder, el, subfolder, title, target, bnds):
    fig, axes = plt.subplots()
    layer_nrs = np.arange(len(bnds))

    def normalize_df(a,b):
        return (a-b)/(abs(a) + abs(b))
    
    def min_max_scaling(x):
        # Protect against nan if elements of x are all the same.
        if np.min(x) != np.max(x):
            return (x-np.min(x))/(np.max(x)-np.min(x))
        else:
            if np.min(x) < 0:
                return np.zeros(x.shape)
            elif np.max(x) > 1:
                return np.ones(x.shape)
            else:
                return x
    
    # 
    def mean_plus_err(x):
        return [n[0] + n[1] for n in x]

    qu_orig_max = np.vstack((qu_orig['q10_max'], qu_orig['q25_max'], qu_orig['q50_max'], qu_orig['q75_max'], qu_orig['q90_max'], np.array(bnds)[:,1])).T
    qu_orig_mean = np.vstack((mean_plus_err(qu_orig['q10_avg']), mean_plus_err(qu_orig['q25_avg']), mean_plus_err(qu_orig['q50_avg']), mean_plus_err(qu_orig['q75_avg']), mean_plus_err(qu_orig['q90_avg']), mean_plus_err(qu_orig['q100_avg']))).T
    qu_df = normalize_df(qu_corr[:,1:], qu_orig_max)

    # q10_viol = min_max_scaling(qu_df[:,0])
    # q25_viol = min_max_scaling(qu_df[:,1])
    # q50_viol = min_max_scaling(qu_df[:,2])
    # q75_viol = min_max_scaling(qu_df[:,3])
    # max_viol = min_max_scaling(qu_df[:,4])



    # normalizer = np.maximum(np.abs(bnd_qu_mins), np.abs(bnd_qu_maxs))
    lw = 2
    fs= 18
    
    axes.plot(layer_nrs, qu_df[:,0],'orange', linewidth=lw, label='q10')
    axes.plot(layer_nrs, qu_df[:,1],'grey', linewidth=lw, label='q25')
    axes.plot(layer_nrs, qu_df[:,2],'blue', linewidth=lw, label='q50')
    axes.plot(layer_nrs, qu_df[:,3],'green', linewidth=lw, label='q75')
    axes.plot(layer_nrs, qu_df[:,4],'green', linewidth=lw, label='q90')
    axes.plot(layer_nrs, qu_df[:,5],'black', linewidth=lw, label='q100')
    axes.plot(layer_nrs, np.sum(qu_df,axis=1),'red', linewidth=lw, label='sum')
    # # axes.plot(layer_nrs, q10_viol,'orange', linewidth=lw, label='q10')
    # # axes.plot(layer_nrs, q25_viol,'grey', linewidth=lw, label='q25')
    # # axes.plot(layer_nrs, q50_viol,'blue', linewidth=lw, label='q50')
    # # axes.plot(layer_nrs, q75_viol,'green', linewidth=lw, label='q75')
    # # axes.plot(layer_nrs, max_viol,'black', linewidth=lw, label='q100')
    # axes.plot(layer_nrs, qu_corr[:,1],'orange', linewidth=lw, label='q10')
    # axes.plot(layer_nrs, qu_corr[:,2],'grey', linewidth=lw, label='q25')
    # axes.plot(layer_nrs, qu_corr[:,3],'blue', linewidth=lw, label='q50')
    # axes.plot(layer_nrs, qu_corr[:,4],'green', linewidth=lw, label='q75')
    # axes.plot(layer_nrs, qu_corr[:,5],'black', linewidth=lw, label='q100')

    # # absolute
    # axes[0].plot(layer_nrs, np.array(bnds)[:,0], 'grey', layer_nrs, qu_corr[:,0],'r', linewidth=lw) #first is special since neg is correct here, after above is indication of false
    # axes[0].set_title('q0 (min)', fontsize=fs)
    # axes[0].set_ylim([-1,1])
    axes.set_xlabel('Layer index', fontsize=fs)
    axes.set_ylabel('Activation magnitude (norm)', fontsize=fs)
    axes.axhline(y=0., color='r', linestyle='--')

    axes.legend()
    fig.suptitle(title, fontsize=fs)

    if not os.path.exists(folder + '/' + subfolder):
        os.makedirs(folder + '/' + subfolder)
    plt.savefig(folder + '/' + subfolder + str(el[0]) + '_' + str(el[1]) + '_' + target + '.png')
    print('saved:', folder + '/' + subfolder + str(el[0]) + '_' + str(el[1]) + '_' + target + '.png')
    plt.close()


def get_ylist_max(target):
    if ("neurons" in target):
        return 1
    elif ("weights" in target):
        return 2
    elif ("blur" in target): 
        return 3
    elif ("noise" in target):
        return 4


def main(argv):

    """
    This script tests a given detector model with a given set of data (can also be from different models). 

    """

    # Parameters ########################################################
   
    # data to test:
    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220801-184837_1_faults_[0_32]_bits/coco2017/val' # noise lvl1
    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220801-232521_1_faults_[0_32]_bits/coco2017/val' #noise lvl5
    folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220802-100025_1_faults_[0_32]_bits/coco2017/val' #noise lvl10

    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220802-132848_1_faults_[0_32]_bits/coco2017/val' # blur lvl1
    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220802-141159_1_faults_[0_32]_bits/coco2017/val' # blur lvl2
    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220802-151009_1_faults_[0_32]_bits/coco2017/val' # blur lvl3

    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220801-165617_1_faults_[0_32]_bits/coco2017/val' # weights 0-32
    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220801-145644_1_faults_[0,8]_bits/coco2017/val' #weights 0-8
    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220801-105919_1_faults_[0_32]_bits/coco2017/val' #neurons 0-32
    # folder = '/nwstore/florian/LR_detector_data/new_90/objDet_20220801-124703_1_faults_[0,8]_bits/coco2017/val' #neurons 0-8

    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220719-134728_1_faults_[0,8]_bits/kitti/val'

    # ranger_file_name = 'yolov3_bounds_CoCo_train20p_act_v8'
    ranger_file_name = 'yolov3_bounds_CoCo_train20p_act_v9'
    # ranger_file_name = 'yolov3_bounds_kitti_train20p_act_v3'
    # ranger_file_name = 'yolov3_bounds_alexnet_Imagenet_act_v1'

    # plot_or_not = False
    # save_result = True
    target = 'noise_lvl10' #noise, blur, neurons, weights

    # model_load_path = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/exp_1_' + 'all' + '.pt'
    model_load_path = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/exp_1_all.pt'
    
    #########################################################

    # For comparison:
    data_folder = '/nwstore/florian/LR_detector_data/quantile_detection_data/maxs_new_norm/'
    
    from train_detection_model_LR3 import load_data_from_quantile_data
    x_list, y_list = load_data_from_quantile_data(data_folder, target)


    
#   tp: [336, 628] fp: [29, 128] fn: [6, 6]
#   P: 0.8668363019508057, R: 0.988394584139265, | or P1: 0.9205479452054794, R1: 0.9824561403508771, P2: 0.8306878306878307, R2: 0.9905362776025236, conf_rate: 0.05609284332688588


    # Load info
    quant_dict, qu_orig, bnds, sdc_due, indices, sdc_lays, sdc_bpos = get_data(folder, ranger_file_name)
    nr_eps, nr_ims, image_ids = indices
    ep_im_sdc, ep_im_due = sdc_due

    # Load model metadata
    info_path = model_load_path[:-3] + '_meta.json'
    info = load_json_indiv(info_path)
    input_dim = info['input_size']
    output_dim = info['output_size']

    # Load model
    from train_detection_model_LR3 import LogisticRegression
    model = LogisticRegression(input_dim, output_dim)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    print(model.state_dict()['linear.weight'])
    print('Loaded', model_load_path)
    # tensor([[ 0.6924,  0.0087, -0.0121,  ..., -0.6591,  0.8253,  2.5193],
    #     [ 0.0643,  0.0242,  0.0238,  ...,  2.0293,  2.0327,  0.4729],
    #     [-1.1990, -0.0988,  0.0387,  ...,  0.6887,  1.1980, -0.1368]])


    # Model analysis:
    import math
    b = model.state_dict()["linear.bias"].numpy()
    w = model.state_dict()["linear.weight"].numpy()[0]

    # Sort
    feature_labels = create_feature_labels(x_list)
    # nr_layers = len(quant_dict['corr'][0])
    # feature_labels = []
    # q_names = ['q10', 'q25', 'q50', 'q75', 'q90', 'q100']
    # for u in range(len(q_names)):
    #     feature_labels.extend([q_names[u] + '_lay' + str(n) for n in range(nr_layers)])


    # w_sorted = [x for x,_ in sorted(zip((math.e**w).tolist(),feature_labels))][::-1]
    # feature_labels_sorted = [x for _,x in sorted(zip((math.e**w).tolist(),feature_labels))][::-1]
    w_sorted = [x for x,_ in sorted(zip((w).tolist(),feature_labels))][::-1]
    feature_labels_sorted = [x for _,x in sorted(zip((w).tolist(),feature_labels))][::-1]
    print(target)
    print('ALL: feature importance', w_sorted[:10], 'bias:', (b).astype(float))
    print('ALL: feature labels', feature_labels_sorted[:10])

    #maxs
    # feature importance [inf, inf, inf, inf, inf, 1.1774169328872711e+36, 2.903866279682679e+33, 1.5611256410065065e+33, 2.3369822390222852e+32, 1.674420167713021e+31] bias: [4.27512277e-05 2.36689453e+04 2.08333209e-02]
    # feature labels ['q75_lay8', 'q75_lay32', 'q50_lay69', 'q50_lay28', 'q100_lay65', 'q50_lay63', 'q100_lay5', 'q50_lay70', 'q75_lay28', 'q75_lay52']
    #means  
    # feature importance [3.7343033832311665e+33, 3.283222501574163e+30, 4.6939056655211054e+27, 2.3150432121399921e+27, 5.902918327795148e+24, 9.536158740334688e+22, 8.151514424820672e+22, 8.922850211420063e+21, 3.2969496074630726e+19, 7.531000440847073e+18] bias: [1.24164726e-05 6.96677683e+09 1.51581501e+10]
    # feature labels ['q100_lay58', 'q100_lay65', 'q50_lay70', 'q100_lay34', 'q100_lay45', 'q75_lay28', 'q100_lay37', 'q100_lay5', 'q75_lay48', 'q75_lay45']


    # For expl plots:
    save_folder = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/explanations'
    if "noise" in target:
        subfolder = "noise/"
    elif "blur" in target:
        subfolder = "blur/"
    elif "neurons" in target:
        subfolder = "neurons/"
    elif "weights" in target:
        subfolder = "weights/"
    else:
        subfolder = ""

    nr_example_plots = 10


    # Extract feature dict -------------------------------------------------------

    lp = nr_eps*nr_ims #50

    tp_count = np.zeros(model.linear.out_features-1)
    fp_count = np.zeros(model.linear.out_features-1)
    fn_count = np.zeros(model.linear.out_features-1)

    sdc_conf_count = 0
    sdc_count = 0
    due_count = 0
    plt_cnt_no_sdc = 0
    plt_cnt_sdc = 0

    for el in range(lp): #lp
        if el/lp*100 % 10 == 0:
            print(round(el/lp*100,2),'% progress')
        ep, im = get_index(el, nr_ims)

        if [ep, image_ids[im]] in ep_im_due: #Exclude all due events
            due_count += 1
            continue

        # #results
        # res_orig = results['orig'][ep][im]
        # res_corr = results['corr'][ep][im]
        # res_orig_resil = results['orig_resil'][ep][im]
        # res_corr_resil = results['corr_resil'][ep][im]

        # SDC check
        is_sdc = [ep, image_ids[im]] in ep_im_sdc
        

        # Quantile monitoring
        qu_corr = np.array(quant_dict['corr'][el]) #size 72,6
        # qu_orig_resil = np.array(quant_dict['resil'][el]) #resil has only one epoch
        # qu_corr_resil = np.array(quant_dict['resil_corr'][el])
        # assert not np.isnan(qu_corr).any() and not np.isinf(qu_corr).any(), "x values contain inf or nans!"
        # assert not np.isnan(qu_corr).any() and not np.isinf(qu_corr).any(), "x values contain inf or nans!"
        

        # Calculate deviations  ----------------------
        df_viols = get_layerwise_scaled_diffs(qu_corr, qu_orig, bnds)

        # # Make features and update feature vector dicts -------------------------
        # feature_vector_dict = update_feature_dict(feature_vector_dict, df_viols, is_sdc, None)



        # Try: rule-based detection simple:
        # TEST metric ############################# 
        
        x = torch.tensor(flatten_list(df_viols)).to(torch.float)
        # print('precision diffs', max(x-x_list[el]))

        if ("neurons" in target) and is_sdc:
            is_sdc = 1
        elif ("weights" in target) and is_sdc:
            is_sdc = 2
        elif ("blur" in target) and is_sdc: 
            is_sdc = 3
        elif ("noise" in target) and is_sdc: 
            is_sdc = 4
        else:
            is_sdc = 0


        # print(is_sdc, y_list[el])
        # print('check', is_sdc, y_list[el], max(x-x_list[el]))
        assert is_sdc==y_list[el], max(x-x_list[el])<1e-6

        # tp, fp, fn, tn, sdc, sdc_conf, pred = get_tp_fp_fn_multiclass(np.array([x_list[el]]), [y_list[el]], model, None)
        tp, fp, fn, tn, sdc, sdc_conf, pred = get_tp_fp_fn_multiclass_all_classes(np.array([x.numpy()]), [int(is_sdc)], model, None)
        # if is_sdc > 0:
        #     print(is_sdc, pred)

        sdc_count += np.sign(is_sdc) #binary 0 or 1
        sdc_conf_count += sdc_conf
        tp_count += np.array(tp)
        fp_count += np.array(fp)
        fn_count += np.array(fn)
        # tp_count_2 += tp[1]
        # fp_count_1 += fp[0]
        # fp_count_2 += fp[1]
        # fn_count_1 += fn[0]
        # fn_count_2 += fn[1]

        # Visualize an explanation:
        if (is_sdc > 0 and plt_cnt_sdc >= nr_example_plots) or (is_sdc == 0 and plt_cnt_no_sdc >= nr_example_plots):
            continue
        else:
            if is_sdc > 0:
                subfolder_to_save = subfolder + 'sdc/'
                plt_cnt_sdc += 1
                ttl = 'sdc gt: ' + str(is_sdc) + " , sdc pred: " + str(int(pred)) + "; flt: lay " + str(sdc_lays[plt_cnt_sdc]) + " , bpos " + str(sdc_bpos[plt_cnt_sdc])
            else:
                subfolder_to_save = subfolder + 'no_sdc/'#
                plt_cnt_no_sdc += 1
                ttl = 'sdc gt: ' + str(is_sdc) + " , sdc pred: " + str(int(pred)) + "; flt: lay " + str(sdc_lays[plt_cnt_no_sdc]) + " , bpos " + str(sdc_bpos[plt_cnt_no_sdc])


            plot_explanation2(qu_orig, qu_corr, save_folder, [ep, image_ids[im]], subfolder_to_save, ttl, target, bnds)

            # focus layers
            b = model.state_dict()["linear.bias"].numpy()
            w = model.state_dict()["linear.weight"].numpy() #has no 3 dims
            # mult = np.array(w)*np.array(x)
            

            # feature_labels = []
            # q_names = ['q10', 'q25', 'q50', 'q75', 'q90', 'q100']
            # for u in range(len(q_names)):
            #     feature_labels.extend([q_names[u] + '_lay' + str(n) for n in range(len(bnds))])

            # mult_sorted = [x for x,_ in sorted(zip(mult.tolist(),feature_labels))][::-1]
            # feature_labels_sorted = [x for _,x in sorted(zip(mult.tolist(),feature_labels))][::-1]
            # print('features', feature_labels_sorted[:10], 'values', mult_sorted[:10])
            # for 
            pred0 = torch.sigmoid(torch.tensor(np.sum(np.array(w)*np.array(x), 1) + b))
            # np.sort((np.array(w)*np.array(x))[pred])[::-1]

            # features that support this decision:
            sm = (np.array(w)*np.array(x))[pred]
            sm_pos = sm[sm>0]
            fl_pos = np.array(feature_labels)[sm>0]
            sm_sorted = [x for x,_ in sorted(zip(sm_pos.tolist(),fl_pos))][::-1]
            feature_labels_sorted = [x for _,x in sorted(zip(sm_pos.tolist(),fl_pos))][::-1]
            print(sm_sorted[:10])
            print(feature_labels_sorted[:10])
            # print()
            # print(torch.sigmoid(torch.tensor(np.sum(np.array(w)[pos]*np.array(x_sdc[0,pos])) + b)))




    print('Simplistic test detection', 'TP:', tp_count, 'FP:', fp_count, 'FN:', fn_count, 'SDC conf:', sdc_conf_count, 'SDC:', sdc_count, 'DUE', due_count, 'total', lp, 'total effective', lp-due_count)
    tp = tp_count
    fp = fp_count
    fn = fn_count
    sdc = sdc_count
    sdc_conf = sdc_conf_count
    p_1, r_1 = get_p_r(tp[0], fp[0], fn[0])
    p_2, r_2 = get_p_r(tp[1], fp[1], fn[1])
    p_3, r_3 = get_p_r(tp[2], fp[2], fn[2])
    p_4, r_4 = get_p_r(tp[3], fp[3], fn[3])
    p, r = get_p_r(np.sum(tp), np.sum(fp), np.sum(fn)) # +sdc_conf
    # print('tp:', tp, 'fp:', fp, 'fn:', fn)
    print(f"P: {p}, R: {r}, | or P1: {p_1}, R1: {r_1} | P2: {p_2}, R2: {r_2} | or P3: {p_3}, R3: {r_3} | P4: {p_4}, R4: {r_4}, conf_rate: {sdc_conf/sdc}")





if __name__ == "__main__":
    main(sys.argv)