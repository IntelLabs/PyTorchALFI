# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import json
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import pathlib
import pickle
from pathlib import Path
import yaml

def  read_yaml(file):
    with open(file, 'r') as f:
        data = list(yaml.load_all(f, Loader=yaml.Loader))
    return data

def load_json_indiv(gt_path):
    with open(gt_path) as f:
        coco_gt = json.load(f)
        f.close()
    return coco_gt

def read_faultbin_file(file):
    _file = open(file, 'rb')
    return pickle.load(_file)

def add_data(toplot_dict, ax_leg, model_dict):

    path = model_dict["path"]
    label_name = model_dict["label_name"]
    typ = model_dict["typ"]
    model_name = [split for split in path.split('/') if '_trials' in split][0]
    model_name = "_".join(model_name.split('_')[:-2])
    dataset_name = path.split('/')[-2]

    flt_type = model_dict["flt_type"]
    suffix = model_dict["suffix"]
    bits = model_dict["bits"]
    
    
    # Load from file saved in yolo_analysis3.py:
    try:
        json_path = os.path.join(path, model_name + "_" + dataset_name + "_" + "results_1_" + flt_type + "_images" + '_' + suffix + ".json")
        results = load_json_indiv(json_path)
        print('Loaded:', json_path)
    except:
        print("File path != valid")
        ax_leg.append(label_name)
        return toplot_dict, ax_leg

    fault_file = str([a for a in list(pathlib.Path(os.path.dirname(path)).glob('*.bin')) if 'corr' not in str(a)][0])
    faults = read_faultbin_file(fault_file)
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
    toplot_dict["sdc"]["corr_mns"].append(m)
    toplot_dict["sdc"]["corr_errs"].append(err)
    # DUE rate images
    m, err = get_m_err(due_rate)
    toplot_dict["due"]["corr_mns"].append(m)
    toplot_dict["due"]["corr_errs"].append(err)

    # orig map
    m,err = get_m_err([orig_map]) #make single number a list
    toplot_dict["map"]["orig_mns"].append(m)
    toplot_dict["map"]["orig_errs"].append(err)
    # Corr map
    m,err = get_m_err(corr_map)# TODO: fix get_m_err is nan if one is nan
    toplot_dict["map"]["corr_mns"].append(m)
    toplot_dict["map"]["corr_errs"].append(err)


    # orig ap50
    m,err = get_m_err([orig_ap50])
    toplot_dict["ap50"]["orig_mns"].append(m)
    toplot_dict["ap50"]["orig_errs"].append(err)
    # Corr ap50
    m,err = get_m_err(corr_ap50)
    toplot_dict["ap50"]["corr_mns"].append(m)
    toplot_dict["ap50"]["corr_errs"].append(err)


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

    toplot_dict['tpfpfn']['corr']['bpos'].append(np.array(results["flts_sdc"])[6,:])

    if typ != "no_resil":
        resil_orig_sdc, resil_corr_sdc = results["metrics"]["sdc"]["resil_sdc"], results["metrics"]["sdc"]["resil_corr_sdc"]
        resil_orig_due, resil_corr_due = results["metrics"]["due"]["resil_due"], results["metrics"]["due"]["resil_corr_due"]
        resil_orig_map, resil_corr_map = results["metrics"]["map"]["resil_map"], results["metrics"]["map"]["resil_corr_map"]
        resil_orig_ap50, resil_corr_ap50 = results["metrics"]["ap50"]["resil_ap50"], results["metrics"]["ap50"]["resil_corr_ap50"]

        resil_n_all, resil_n_sdc, resil_n_due =  results['nr_img_all'], results['resil_nr_img_sdc'], results["resil_nr_img_due"]

        resil_sdc_rate = (np.array(resil_n_sdc)/np.array(n_all)).tolist()
        resil_due_rate = (np.array(resil_n_due)/np.array(n_all)).tolist()

        # SDC rate images
        m, err = get_m_err(resil_sdc_rate)
        toplot_dict["sdc"]["resil_corr_mns"].append(m)
        toplot_dict["sdc"]["resil_corr_errs"].append(err)
        # DUE rate images
        m, err = get_m_err(resil_due_rate)
        toplot_dict["due"]["resil_corr_mns"].append(m)
        toplot_dict["due"]["resil_corr_errs"].append(err)

        # orig map
        m,err = get_m_err([resil_orig_map]) #make single number a list
        toplot_dict["map"]["resil_orig_mns"].append(m)
        toplot_dict["map"]["resil_orig_errs"].append(err)
        # Corr map
        m,err = get_m_err(resil_corr_map)# TODO: fix get_m_err is nan if one is nan
        toplot_dict["map"]["resil_corr_mns"].append(m)
        toplot_dict["map"]["resil_corr_errs"].append(err)


        # orig ap50
        m,err = get_m_err([resil_orig_ap50])
        toplot_dict["ap50"]["resil_orig_mns"].append(m)
        toplot_dict["ap50"]["resil_orig_errs"].append(err)
        # Corr ap50
        m,err = get_m_err(resil_corr_ap50)
        toplot_dict["ap50"]["resil_corr_mns"].append(m)
        toplot_dict["ap50"]["resil_corr_errs"].append(err)


        # tp and bpos
        tpfpfn_orig = results['metrics']["sdc"]["tpfpfn_resil_orig"]
        toplot_dict['tpfpfn']['resil_orig']['tp'].append([n['tp'] for n in tpfpfn_orig])
        toplot_dict['tpfpfn']['resil_orig']['fp'].append([n['fp'] for n in tpfpfn_orig])
        toplot_dict['tpfpfn']['resil_orig']['fp_bbox'].append([n['fp_bbox'] for n in tpfpfn_orig])
        toplot_dict['tpfpfn']['resil_orig']['fp_class'].append([n['fp_class'] for n in tpfpfn_orig])
        toplot_dict['tpfpfn']['resil_orig']['fp_bbox_class'].append([n['fp_bbox_class'] for n in tpfpfn_orig])
        toplot_dict['tpfpfn']['resil_orig']['fn'].append([n['fn'] for n in tpfpfn_orig])

        tpfpfn_corr = results['metrics']["sdc"]["tpfpfn_resil_corr"]
        toplot_dict['tpfpfn']['resil_corr']['tp'].append([n['tp'] for n in tpfpfn_corr])
        toplot_dict['tpfpfn']['resil_corr']['fp'].append([n['fp'] for n in tpfpfn_corr])
        toplot_dict['tpfpfn']['resil_corr']['fp_bbox'].append([n['fp_bbox'] for n in tpfpfn_corr])
        toplot_dict['tpfpfn']['resil_corr']['fp_class'].append([n['fp_class'] for n in tpfpfn_corr])
        toplot_dict['tpfpfn']['resil_corr']['fp_bbox_class'].append([n['fp_bbox_class'] for n in tpfpfn_corr])
        toplot_dict['tpfpfn']['resil_corr']['fn'].append([n['fn'] for n in tpfpfn_corr])

        toplot_dict['tpfpfn']['resil_corr']['bpos'].append(np.array(results["resil_flts_sdc"])[6,:])
    ax_leg.append(label_name)

    return toplot_dict, ax_leg

def plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, legend_text, yname, sv_name, ax_leg, cols = None, scale_to_perc = None):

    ind = np.arange(len(mns_orig))  # the x locations for the groups
    fig, ax = plt.subplots()
    width = 0.35  # the width of the bars
    if scale_to_perc:
        mns_orig = np.array(mns_orig)*100
        mns_corr = np.array(mns_corr)*100
        errs_orig = np.array(errs_orig)*100
        errs_corr = np.array(errs_corr)*100

    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    viridis = cm.get_cmap('copper')
    newcolors = viridis(np.linspace(0, 1, 2))
    # newcmp = ListedColormap(newcolors)
    colors = newcolors

    if cols is None:
        if mns_corr != None:
            # All original ones:
            ax.bar(ind - width/2, mns_orig, width, yerr=errs_orig, label=legend_text[0])
            ax.errorbar(ind - width/2, mns_orig, yerr=errs_orig, ecolor='black', capsize=10, label='', ls = 'none') #, alpha=0.5,  

            # All corrupted ones:
            ax.bar(ind + width/2, mns_corr, width, yerr=errs_corr, label=legend_text[1])
            ax.errorbar(ind + width/2, mns_corr, yerr=errs_corr, ecolor='black', capsize=10, label='', ls = 'none')

            ax.legend(loc="upper right") #, bbox_to_anchor=(0.8,0.8)
        else:
            ax.bar(ind , mns_orig, width, yerr=errs_orig, label='', color=colors[0])
            ax.errorbar(ind, mns_orig, yerr=errs_orig, ecolor='black', capsize=10, label='', ls = 'none') #, alpha=0.5, 
    else:
        if mns_corr != None:
            # All original ones:
            ax.bar(ind - width/2, mns_orig, width, yerr=errs_orig, label=legend_text[0], color=cols[0])
            ax.errorbar(ind - width/2, mns_orig, yerr=errs_orig, ecolor='black', capsize=10, label='', ls = 'none') #, alpha=0.5,  

            # All corrupted ones:
            ax.bar(ind + width/2, mns_corr, width, yerr=errs_corr, label=legend_text[1], color=cols[1])
            ax.errorbar(ind + width/2, mns_corr, yerr=errs_corr, ecolor='black', capsize=10, label='', ls = 'none')

            ax.legend(loc="upper right") #bbox_to_anchor=(1.2,0.8)
        else:
            ax.bar(ind , mns_orig, width, yerr=errs_orig, label='', color=cols)
            ax.errorbar(ind, mns_orig, yerr=errs_orig, ecolor='black', capsize=10, label='', ls = 'none') #, alpha=0.5, 

    # Add some text for labels, title and custom x-axis tick labels, etc.
    fnt_size = 13
    ax.set_ylabel(yname, fontsize=fnt_size)
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(ind)
    ax.set_xticklabels(ax_leg, fontsize=fnt_size, rotation=45)
    # ax.set_xticklabels(ax_leg)
    ax.set_ylim([0, max(max(mns_orig), max(mns_corr))*1.1+0.1])

    for i, p in enumerate(ax.patches):
        number_height = 1
        if p.get_height() > 0:
            number_height = p.get_height()
        else:
            continue
        ax.annotate("{:.1f}".format(p.get_height(), '.1f'), 
                    (p.get_x() + p.get_width() / 2., number_height), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 10), 
                    size=14,
                    rotation=0,
                    textcoords = 'offset points')

    fig.tight_layout()
    os.makedirs(os.path.dirname(sv_name), exist_ok=True)
    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)

    # plt.show()

def get_m_err(list_to_plot):
    a = len(list_to_plot)
    list_to_plot = np.array(list_to_plot)[np.logical_not(np.isnan(list_to_plot))].tolist() #filter out nans
    if len(list_to_plot) < a:
        print('nans filter out for averaging in get_m_err:', a-len(list_to_plot))
    return np.mean(list_to_plot), np.std(list_to_plot)*1.96/np.sqrt(len(list_to_plot))

def eval_n_w(tpl, plothow, ax_leg):
    
    res_n_w = []
    for n in range(len(tpl)):
        toplot_dict = tpl[n] #0 is neurons, 1 is weights

        # # Filter only by those where tp, fp or fn changes ------------------------
        tps_orig = toplot_dict['tpfpfn']['orig']['tp']
        tps_corr = toplot_dict['tpfpfn']['corr']['tp']

        if plothow == 'fp' or plothow == 'fn':
            fps_orig = toplot_dict['tpfpfn']['orig']['fp']
            fps_corr = toplot_dict['tpfpfn']['corr']['fp']
        elif plothow == 'fp_bbox':
            fps_orig = toplot_dict['tpfpfn']['orig']['fp_bbox']
            fps_corr = toplot_dict['tpfpfn']['corr']['fp_bbox']
        elif plothow == 'fp_class':
            fps_orig = toplot_dict['tpfpfn']['orig']['fp_class']
            fps_corr = toplot_dict['tpfpfn']['corr']['fp_class']
        elif plothow == 'fp_bbox_class':
            fps_orig = toplot_dict['tpfpfn']['orig']['fp_bbox_class']
            fps_corr = toplot_dict['tpfpfn']['corr']['fp_bbox_class']
        else:
            print('unknown parameter plothow')
            return

        fns_orig = toplot_dict['tpfpfn']['orig']['fn']
        fns_corr = toplot_dict['tpfpfn']['corr']['fn']

        tps_all = np.array([np.array(x) for x in tps_corr]) - np.array([np.array(x) for x in tps_orig])
        fps_all = np.array([np.array(x) for x in fps_corr]) - np.array([np.array(x) for x in fps_orig])
        fns_all = np.array([np.array(x) for x in fns_corr]) - np.array([np.array(x) for x in fns_orig])
        bpos_all = toplot_dict['tpfpfn']['corr']['bpos']
        baseline = np.array([np.array(x) for x in tps_orig])

        if len(bpos_all) == 0:
            print('No information for plotting available.')
            return

        # Averaging
        m_tps_all, err_tps_all = [], [] #tps per bit
        m_fps_all, err_fps_all = [], []
        m_fns_all, err_fns_all = [], []
        
        # Print overall averages:
        if 'fp' in plothow:
            fp_means = [np.mean(x) for x in fps_all]
            fp_errs = [np.std(x)*1.96/np.sqrt(len(x)) for x in fps_all]
            if n==0:
                pr = 'neurons'
            elif n==1:
                pr = "weights"
            # print('FP (', pr, '):', plothow, 'means:', fp_means, 'errs:', fp_errs)

        if plothow == 'fn':
            fns_all_n = []
            for i in range(len(fns_all)):
                fns_add = []
                for j in range(len(fns_all[i])):
                    b = baseline[i][j]
                    fn = fns_all[i][j]
                    if b > 0:
                        fns_add.append(fn/b)
                    elif b == 0 and fn == 0:
                        fns_add.append(0.)
                    elif b == 0 and fn > 0:
                        fns_add.append(1)
                    elif b == 0 and fn < 0:
                        fns_add.append(0)
                fns_all_n.append(fns_add)

            # fns_all_n = fns_all/baseline
            fn_means = [np.mean(x) for x in fns_all_n]
            fn_errs = [np.std(x)*1.96/np.sqrt(len(x)) for x in fns_all_n]
            if n==0:
                pr = 'neurons'
            elif n==1:
                pr = "weights"
            # print('FN (', pr, '):', 'fn', 'means:', fn_means, 'errs:', fn_errs)

        for x in range(len(bpos_all)):
            m_tps, err_tps = [], [] #tps per bit
            m_fps, err_fps = [], []
            m_fns, err_fns = [], []

            bpos = bpos_all[x]
            tps = tps_all[x]
            base = baseline[x]
            fps = fps_all[x]
            fns = fns_all[x]
            for i in range(0, 31+1):
                try:
                    lst = tps[bpos == i]
                except:
                    x=0
                if len(lst) == 0:
                    m_tps.append(np.nan)
                    err_tps.append(np.nan)
                else:
                    m_tps.append(np.mean(lst))
                    err_tps.append(np.std(lst)*1.96/np.sqrt(len(lst)))

                lst = fps[bpos == i]
                if len(lst) == 0:
                    m_fps.append(np.nan)
                    err_fps.append(np.nan)
                else:
                    m_fps.append(np.mean(lst))
                    err_fps.append(np.std(lst)*1.96/np.sqrt(len(lst)))

                # make it a ratio
                fns_new = []
                for x in range(len(base)):
                    if base[x] > 0:
                        fns_new.append(fns[x]/base[x])
                    elif base[x] == 0 and fns[x] == 0:
                        fns_new.append(0.)
                    elif base[x] == 0 and fns[x] > 0:
                        fns_new.append(1)
                    elif base[x] == 0 and fns[x] < 0:
                        fns_new.append(0)
                fns_new = np.array(fns_new)

                # lst = fns[bpos == i]
                lst = fns_new[bpos == i]
                if len(lst) == 0:
                    m_fns.append(np.nan)
                    err_fns.append(np.nan)
                else:
                    m_fns.append(np.mean(lst))
                    err_fns.append(np.std(lst)*1.96/np.sqrt(len(lst)))
            
            m_tps_all.append(m_tps)
            m_fps_all.append(m_fps)
            m_fns_all.append(m_fns)
            err_tps_all.append(err_tps)
            err_fps_all.append(err_fps)
            err_fns_all.append(err_fns)

        res_n_w.append({'m_tps': m_tps_all, 'err_tps': err_tps_all, 'm_fps': m_fps_all, 'err_fps': err_fps_all, 'm_fns': m_fns_all, 'err_fns': err_fns_all, 'ax_leg': ax_leg[n]})

    return res_n_w, ax_leg, bpos_all

def eval_n_w_resil(tpl, plothow, ax_leg):
    
    res_n_w = []
    for n in range(len(tpl)):
        toplot_dict = tpl[n] #0 is neurons, 1 is weights

        # # Filter only by those where tp, fp or fn changes ------------------------
        tps_orig = toplot_dict['tpfpfn']['resil_orig']['tp']
        tps_corr = toplot_dict['tpfpfn']['resil_corr']['tp']

        if plothow == 'fp' or plothow == 'fn':
            fps_orig = toplot_dict['tpfpfn']['resil_orig']['fp']
            fps_corr = toplot_dict['tpfpfn']['resil_corr']['fp']
        elif plothow == 'fp_bbox':
            fps_orig = toplot_dict['tpfpfn']['resil_orig']['fp_bbox']
            fps_corr = toplot_dict['tpfpfn']['resil_corr']['fp_bbox']
        elif plothow == 'fp_class':
            fps_orig = toplot_dict['tpfpfn']['resil_orig']['fp_class']
            fps_corr = toplot_dict['tpfpfn']['resil_corr']['fp_class']
        elif plothow == 'fp_bbox_class':
            fps_orig = toplot_dict['tpfpfn']['resil_orig']['fp_bbox_class']
            fps_corr = toplot_dict['tpfpfn']['resil_corr']['fp_bbox_class']
        else:
            print('unknown parameter plothow')
            return

        fns_orig = toplot_dict['tpfpfn']['resil_orig']['fn']
        fns_corr = toplot_dict['tpfpfn']['resil_corr']['fn']

        tps_all = np.array([np.array(x) for x in tps_corr]) - np.array([np.array(x) for x in tps_orig])
        fps_all = np.array([np.array(x) for x in fps_corr]) - np.array([np.array(x) for x in fps_orig])
        fns_all = np.array([np.array(x) for x in fns_corr]) - np.array([np.array(x) for x in fns_orig])
        bpos_all = toplot_dict['tpfpfn']['resil_corr']['bpos']
        baseline = np.array([np.array(x) for x in tps_orig])

        if len(bpos_all) == 0:
            print('No information for plotting available.')
            return

        # Averaging
        m_tps_all, err_tps_all = [], [] #tps per bit
        m_fps_all, err_fps_all = [], []
        m_fns_all, err_fns_all = [], []
        
        # Print overall averages:
        if 'fp' in plothow:
            fp_means = [np.mean(x) for x in fps_all]
            fp_errs = [np.std(x)*1.96/np.sqrt(len(x)) for x in fps_all]
            if n==0:
                pr = 'neurons'
            elif n==1:
                pr = "weights"
            # print('FP (', pr, '):', plothow, 'means:', fp_means, 'errs:', fp_errs)

        if plothow == 'fn':
            fns_all_n = []
            for i in range(len(fns_all)):
                fns_add = []
                for j in range(len(fns_all[i])):
                    b = baseline[i][j]
                    fn = fns_all[i][j]
                    if b > 0:
                        fns_add.append(fn/b)
                    elif b == 0 and fn == 0:
                        fns_add.append(0.)
                    elif b == 0 and fn > 0:
                        fns_add.append(1)
                    elif b == 0 and fn < 0:
                        fns_add.append(0)
                fns_all_n.append(fns_add)

            # fns_all_n = fns_all/baseline
            fn_means = [np.mean(x) for x in fns_all_n]
            fn_errs = [np.std(x)*1.96/np.sqrt(len(x)) for x in fns_all_n]
            if n==0:
                pr = 'neurons'
            elif n==1:
                pr = "weights"
            # print('FN (', pr, '):', 'fn', 'means:', fn_means, 'errs:', fn_errs)

        for x in range(len(bpos_all)):
            m_tps, err_tps = [], [] #tps per bit
            m_fps, err_fps = [], []
            m_fns, err_fns = [], []

            bpos = bpos_all[x]
            tps = tps_all[x]
            base = baseline[x]
            fps = fps_all[x]
            fns = fns_all[x]
            for i in range(0, 31+1):
                lst = tps[bpos == i]
                if len(lst) == 0:
                    m_tps.append(np.nan)
                    err_tps.append(np.nan)
                else:
                    m_tps.append(np.mean(lst))
                    err_tps.append(np.std(lst)*1.96/np.sqrt(len(lst)))

                lst = fps[bpos == i]
                if len(lst) == 0:
                    m_fps.append(np.nan)
                    err_fps.append(np.nan)
                else:
                    m_fps.append(np.mean(lst))
                    err_fps.append(np.std(lst)*1.96/np.sqrt(len(lst)))

                # make it a ratio
                fns_new = []
                for x in range(len(base)):
                    if base[x] > 0:
                        fns_new.append(fns[x]/base[x])
                    elif base[x] == 0 and fns[x] == 0:
                        fns_new.append(0.)
                    elif base[x] == 0 and fns[x] > 0:
                        fns_new.append(1)
                    elif base[x] == 0 and fns[x] < 0:
                        fns_new.append(0)
                fns_new = np.array(fns_new)

                # lst = fns[bpos == i]
                lst = fns_new[bpos == i]
                if len(lst) == 0:
                    m_fns.append(np.nan)
                    err_fns.append(np.nan)
                else:
                    m_fns.append(np.mean(lst))
                    err_fns.append(np.std(lst)*1.96/np.sqrt(len(lst)))
            
            m_tps_all.append(m_tps)
            m_fps_all.append(m_fps)
            m_fns_all.append(m_fns)
            err_tps_all.append(err_tps)
            err_fps_all.append(err_fps)
            err_fns_all.append(err_fns)

        res_n_w.append({'m_tps': m_tps_all, 'err_tps': err_tps_all, 'm_fps': m_fps_all, 'err_fps': err_fps_all, 'm_fns': m_fns_all, 'err_fns': err_fns_all, 'ax_leg': ax_leg[n]})

    return res_n_w, ax_leg, bpos_all

def eval_n_w_hist(tpl, plothow, ax_leg):
    
    res_n_w = []
    for n in range(len(tpl)):
        toplot_dict = tpl[n] #0 is neurons, 1 is weights

        # # Filter only by those where tp, fp or fn changes ------------------------
        orig_fps = toplot_dict['tpfpfn']['orig']['fp']
        corr_fps = toplot_dict['tpfpfn']['corr']['fp']

        orig_fns = toplot_dict['tpfpfn']['orig']['fn']
        corr_fns = toplot_dict['tpfpfn']['corr']['fn']
        bit_positions = toplot_dict['tpfpfn']['corr']['bpos']

        fps_all = np.array([np.array(x) for x in corr_fps]) - np.array([np.array(x) for x in orig_fps])
        fns_all = np.array([np.array(x) for x in corr_fns]) - np.array([np.array(x) for x in orig_fns])

        # Averaging
        m_fps_all = []
        m_fns_all = []

        for x in range(len(bit_positions)):
            m_fps = []
            m_fns = []
            fps = np.array(fps_all[x])
            fns = np.array(fns_all[x])
            bpos = bit_positions[x]

            for i in range(0, 31+1):
                lst = fps[bpos == i]
                lst = np.nonzero(lst)[0]
                m_fps.append(len(lst))

                lst = fns[bpos == i]
                lst = np.nonzero(lst)[0]
                m_fns.append(len(lst))

            m_fps_all.append(m_fps)
            m_fns_all.append(m_fns)

        res_n_w.append({'m_fps': m_fps_all, 'm_fns': m_fns_all, 'ax_leg': ax_leg[n]})

    return res_n_w, ax_leg, bit_positions

def eval_n_w_resil_hist(tpl, plothow, ax_leg):

    res_n_w = []
    for n in range(len(tpl)):
        toplot_dict = tpl[n] #0 is neurons, 1 is weights

        # # Filter only by those where tp, fp or fn changes ------------------------
        orig_fps = toplot_dict['tpfpfn']['resil_orig']['fp']
        corr_fps = toplot_dict['tpfpfn']['resil_corr']['fp']

        orig_fns = toplot_dict['tpfpfn']['resil_orig']['fn']
        corr_fns = toplot_dict['tpfpfn']['resil_corr']['fn']
        bit_positions = toplot_dict['tpfpfn']['resil_corr']['bpos']

        fps_all = np.array([np.array(x) for x in corr_fps]) - np.array([np.array(x) for x in orig_fps])
        fns_all = np.array([np.array(x) for x in corr_fns]) - np.array([np.array(x) for x in orig_fns])

        # Averaging
        m_fps_all = []
        m_fns_all = []

        for x in range(len(bit_positions)):
            m_fps = []
            m_fns = []
            fps = np.array(fps_all[x])
            fns = np.array(fns_all[x])
            bpos = bit_positions[x]

            for i in range(0, 31+1):
                lst = fps[bpos == i]
                lst = np.nonzero(lst)[0]
                m_fps.append(len(lst))

                lst = fns[bpos == i]
                lst = np.nonzero(lst)[0]
                m_fns.append(len(lst))

            m_fps_all.append(m_fps)
            m_fns_all.append(m_fns)

        res_n_w.append({'m_fps': m_fps_all, 'm_fns': m_fns_all, 'ax_leg': ax_leg[n]})

    return res_n_w, ax_leg, bit_positions

def plot_avg_tp_bpos_old(tpl, ax_leg, sv_name, plothow='fp', n_w='None', typ="no_resil"):
    """
    plothow: switches between fp and fn
    n_w: switches between "neurons", "weights" or both "None"
    """
    function_eva_n_w = eval_n_w_resil if typ != "no_resil" else eval_n_w
    res_n_w, ax_leg, bpos_all = function_eva_n_w(tpl, plothow, ax_leg)

    if n_w == 'neurons':
        res_n_w = [res_n_w[0]]
    elif n_w == 'weights':
        res_n_w = [res_n_w[1]]

    fig, ax = plt.subplots()  
    ll = np.arange(0, 31+1)

    colors_fp = ['b', 'g', 'r', 'k', 'orange', 'purple'] 
    if 'fp' in plothow:
        for m in range(len(res_n_w)): #neurons, weights
            res = res_n_w[m]

            m_fps_all = res['m_fps']
            err_fps_all = res['err_fps']
            ax_leg = res['ax_leg']

            for u in range(len(m_fps_all)):
                m_pl = m_fps_all[u]
                mask = np.logical_not(np.isnan(m_pl))
                m_pl = np.array(m_pl)[mask]
                err_pl = np.array(err_fps_all[u])[mask]
                ll_pl = np.array(ll)[mask]
                # print(ax_leg[u], ' (m=' + str(m) + '): Sdc event adds an avg number of fps', np.mean(m_pl), np.std(m_pl)*1.96/np.sqrt(len(m_pl))) #, len(m_pl), m_pl)
                if n_w == 'None' and m == 0:
                    fmt_get='-o'
                    add_leg = 'neurons'
                elif n_w == 'None' and m == 1:
                    fmt_get=':o'
                    add_leg = 'weights'
                else:
                    add_leg = n_w
                    fmt_get = '_'
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fp[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u] + ": " + add_leg, linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fp[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u], linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                ax.bar(ll_pl, m_pl, yerr=err_pl, color=colors_fp[u], label=ax_leg[u], width=0.15)
                ax.errorbar(ll_pl, m_pl, yerr=err_pl, color=colors_fp[u], ecolor='k', markersize=5, capsize=5, label='', elinewidth=0.5, markeredgewidth=0.5, ls='none')
                
                # plt.ylabel(r"$FP_{ad}$") #$avg(FP_{corr} - FP_{orig})$ objects")
                plt.ylabel(r"$bitavg(\Delta FP)$")

    colors_fn = ['b', 'g', 'r', 'k', 'orange', 'purple'] 
    if plothow == 'fn':
        for m in range(len(res_n_w)):
            res = res_n_w[m]

            m_fns_all = res['m_fns']
            err_fns_all = res['err_fns']
            ax_leg = res['ax_leg']

            for u in range(len(m_fns_all)):
                # fns
                m_pl = m_fns_all[u]
                mask = np.logical_not(np.isnan(m_pl))
                m_pl = np.array(m_pl)[mask]
                err_pl = np.array(err_fns_all[u])[mask]
                ll_pl = np.array(ll)[mask]
                # print(ax_leg[u], ' (m=' + str(m) + '): Sdc event adds an avg number of fns', np.mean(m_pl), np.std(m_pl)*1.96/np.sqrt(len(m_pl)))
                if n_w == 'None' and m == 0:
                    fmt_get='-o'
                    add_leg = 'neurons'
                elif n_w == 'None' and m == 1:
                    fmt_get=':o'
                    add_leg = 'weights'
                else:
                    add_leg = n_w
                    fmt_get = 'o'
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fn[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u]+ ": " + add_leg, linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fn[u], markersize=3, ecolor='k', capsize=5, \
                    label=ax_leg[u], linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                plt.ylabel(r"$bitavg(\Delta FN_{n})$")


    max_all = np.max([np.max(n) for n in bpos_all])
    ax.set_xlim([0, max_all+1])

    plt.legend(loc="upper right")
    plt.xlabel("Bit position")

    os.makedirs(os.path.dirname(sv_name), exist_ok=True)
    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)
    # plt.show()

def plot_avg_tp_bpos(tpl, ax_leg, sv_name, plothow='fp', n_w='None', typ = "no_resil"):
    """
    plothow: switches between fp and fn
    n_w: switches between "neurons", "weights" or both "None"
    """
    function_eva_n_w = eval_n_w_resil if typ != "no_resil" else eval_n_w
    res_n_w, ax_leg, bpos_all = function_eva_n_w(tpl, plothow, ax_leg)

    if n_w == 'neurons':
        res_n_w = [res_n_w[0]]
    elif n_w == 'weights':
        res_n_w = [res_n_w[1]]

    fig, ax = plt.subplots()
    ll = np.arange(0, 31+1)
    # ll = np.arange(0, 15+1)

    colors_fp_ = ['b', 'g', 'r', 'k', 'orange', 'purple']

    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    colors_fp = np.linspace(0, 0.9, num=len(colors_fp_))
    colors_fp = [str(color) for color in colors_fp]
    viridis = cm.get_cmap('copper')
    newcolors = viridis(np.linspace(0, 1, len(colors_fp_)))
    # newcmp = ListedColormap(newcolors)
    colors_fp = newcolors
    if 'fp' in plothow:
        for m in range(len(res_n_w)): #neurons, weights
            res = res_n_w[m]

            m_fps_all = res['m_fps']
            err_fps_all = res['err_fps']
            ax_leg = res['ax_leg']

            
            shifts = np.linspace(-0.3,0.3, num=len(m_fps_all))
            wid = shifts[1]-shifts[0] if len(shifts) > 1 else 0.2
            for u in range(len(m_fps_all)):
                m_pl = m_fps_all[u]
                mask = np.logical_not(np.isnan(m_pl))
                m_pl = np.array(m_pl)[mask]
                err_pl = np.array(err_fps_all[u])[mask]
                ll_pl = np.array(ll)[mask]
                try:
                    print(ax_leg[u], ' (', n_w, '): Sdc event adds an avg number of fps', 'mean', np.mean(m_pl), 'range', np.min(m_pl), np.max(m_pl), 'err', np.std(m_pl)*1.96/np.sqrt(len(m_pl))) #, len(m_pl), m_pl)
                except:
                    x=0
                if n_w == 'None' and m == 0:
                    fmt_get='-o'
                    add_leg = 'neurons'
                elif n_w == 'None' and m == 1:
                    fmt_get=':o'
                    add_leg = 'weights'
                else:
                    add_leg = n_w
                    fmt_get = '_'
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fp[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u] + ": " + add_leg, linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fp[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u], linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                ax.bar(ll_pl+shifts[u], m_pl, yerr=err_pl, color=colors_fp[u], label=ax_leg[u], width=wid, align='center', edgecolor='white', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2, label='', elinewidth=1, markeredgewidth=0.7, ls='none'))
                # ax.errorbar(ll_pl+shifts[u], m_pl, yerr=err_pl, ecolor='gray', capsize=3, label='', elinewidth=0.01, markeredgewidth=0.7, markeredgecolor='gray', ls='none')
                
                # plt.ylabel(r"$FP_{ad}$") #$avg(FP_{corr} - FP_{orig})$ objects")
            ax.set_xticklabels([31, 30, 29, 28, 27, 26, 25, 24, 23])
            plt.ylabel(r"$bitavg(\Delta FP)$")
            ax.set_ylim([-2, 1000])

    colors_fn = colors_fp
    if plothow == 'fn':
        for m in range(len(res_n_w)):
            res = res_n_w[m]

            m_fns_all = res['m_fns']
            err_fns_all = res['err_fns']
            ax_leg = res['ax_leg']

            shifts = np.linspace(-0.3,0.3, num=len(m_fns_all))
            wid = shifts[1]-shifts[0] if len(shifts) > 1 else 0.2
            for u in range(len(m_fns_all)):
                # fns
                m_pl = m_fns_all[u]
                mask = np.logical_not(np.isnan(m_pl))
                m_pl = np.array(m_pl)[mask]
                err_pl = np.array(err_fns_all[u])[mask]
                ll_pl = np.array(ll)[mask]
                try:
                    print(ax_leg[u], ' (' + n_w + '): Sdc event adds an avg number of fns', 'mean', np.mean(m_pl), 'range', np.min(m_pl), np.max(m_pl), 'err',  np.std(m_pl)*1.96/np.sqrt(len(m_pl)))
                except:
                    x = 0
                # ax_leg[u] = 0 #TODO: add avg here?
                if n_w == 'None' and m == 0:
                    fmt_get='-o'
                    add_leg = 'neurons'
                elif n_w == 'None' and m == 1:
                    fmt_get=':o'
                    add_leg = 'weights'
                else:
                    add_leg = n_w
                    fmt_get = 'o'
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fn[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u]+ ": " + add_leg, linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fn[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u], linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                # ax.bar(ll_pl+shifts[u], m_pl, yerr=err_pl, color=colors_fn[u], label=ax_leg[u], width=wid, align='center')
                # ax.errorbar(ll_pl+shifts[u], m_pl, yerr=err_pl, color=colors_fn[u], ecolor='k', capsize=3, label='', elinewidth=0.01, markeredgewidth=0.7, ls='none')
                ax.bar(ll_pl+shifts[u], m_pl*100, yerr=err_pl*100, color=colors_fn[u], label=ax_leg[u], edgecolor='white', width=wid, align='center', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2, label='', elinewidth=1, markeredgewidth=0.7, ls='none'))
            
            plt.ylabel(r"$bitavg(\Delta FN_{n})(\%)$")
            ax.set_ylim([-30, 100])
            ax.set_xticklabels([31, 30, 29, 28, 27, 26, 25, 24, 23])
            
    max_all = np.max([np.max(n) for n in bpos_all])
    # ax.set_xlim([0, max_all+1])
    x_lim_new = 8
    ax.set_xlim([-0.5, x_lim_new])
    ax.set_xticks(np.arange(0, x_lim_new+1, step=1))

    plt.legend(loc="upper right")
    plt.xlabel("Bit position")
    plt.tight_layout()

    os.makedirs(os.path.dirname(sv_name), exist_ok=True)
    plt.savefig(sv_name, dpi=300, bbox_inches='tight',pad_inches = 0)
    print('saved as ', sv_name)
    # plt.show()

def plot_hist_tp_bpos(tpl, ax_leg, sv_name, plothow='fp', n_w='None', typ = "no_resil"):
    """
    plothow: switches between fp and fn
    n_w: switches between "neurons", "weights" or both "None"
    """
    function_eva_n_w = eval_n_w_resil_hist if typ != "no_resil" else eval_n_w_hist
    res_n_w, ax_leg, bpos_all = function_eva_n_w(tpl, plothow, ax_leg)

    if n_w == 'neurons':
        res_n_w = [res_n_w[0]]
    elif n_w == 'weights':
        res_n_w = [res_n_w[1]]

    fig, ax = plt.subplots()
    ll = np.arange(0, 31+1)
    # ll = np.arange(0, 15+1)

    colors_fp_ = ['b', 'g', 'r', 'k', 'orange', 'purple']

    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    colors_fp = np.linspace(0, 0.9, num=len(colors_fp_))
    colors_fp = [str(color) for color in colors_fp]
    viridis = cm.get_cmap('copper')
    newcolors = viridis(np.linspace(0, 1, len(colors_fp_)))
    # newcmp = ListedColormap(newcolors)
    colors_fp = newcolors
    if 'fp' in plothow:
        for m in range(len(res_n_w)): #neurons, weights
            res = res_n_w[m]

            m_fps_all = res['m_fps']
            ax_leg = res['ax_leg']

            
            shifts = np.linspace(-0.3,0.3, num=len(m_fps_all))
            wid = shifts[1]-shifts[0] if len(shifts) > 1 else 0.2
            for u in range(len(m_fps_all)):
                m_pl = m_fps_all[u]
                mask = np.logical_not(np.isnan(m_pl))
                m_pl = np.array(m_pl)[mask]
                ll_pl = np.array(ll)[mask]
                try:
                    print(ax_leg[u], ' (', n_w, '): Sdc event adds an avg number of fps', 'mean', np.mean(m_pl), 'range', np.min(m_pl), np.max(m_pl)) #, len(m_pl), m_pl)
                except:
                    x=0
                if n_w == 'None' and m == 0:
                    fmt_get='-o'
                    add_leg = 'neurons'
                elif n_w == 'None' and m == 1:
                    fmt_get=':o'
                    add_leg = 'weights'
                else:
                    add_leg = n_w
                    fmt_get = '_'
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fp[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u] + ": " + add_leg, linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fp[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u], linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                ax.bar(ll_pl+shifts[u], m_pl, color=colors_fp[u], label=ax_leg[u], width=wid, align='center', edgecolor='white')
                # ax.errorbar(ll_pl+shifts[u], m_pl, yerr=err_pl, ecolor='gray', capsize=3, label='', elinewidth=0.01, markeredgewidth=0.7, markeredgecolor='gray', ls='none')
                
                # plt.ylabel(r"$FP_{ad}$") #$avg(FP_{corr} - FP_{orig})$ objects")
            ax.set_xticklabels([31, 30, 29, 28, 27, 26, 25, 24, 23])
            plt.ylabel(r"$bitavg(\Delta FP)$")
            if n_w == 'neurons':
                ax.set_ylim([0, 1000])
            elif n_w == 'weights':
                ax.set_ylim([-2, 1600])

    colors_fn = colors_fp
    if plothow == 'fn':
        for m in range(len(res_n_w)):
            res = res_n_w[m]

            m_fns_all = res['m_fns']
            ax_leg = res['ax_leg']

            shifts = np.linspace(-0.3,0.3, num=len(m_fns_all))
            wid = shifts[1]-shifts[0] if len(shifts) > 1 else 0.3
            for u in range(len(m_fns_all)):
                # fns
                m_pl = m_fns_all[u]
                mask = np.logical_not(np.isnan(m_pl))
                m_pl = np.array(m_pl)[mask]
                ll_pl = np.array(ll)[mask]
                try:
                    print(ax_leg[u], ' (' + n_w + '): Sdc event adds an avg number of fns', 'mean', np.mean(m_pl), 'range', np.min(m_pl), np.max(m_pl))
                except:
                    x = 0
                # ax_leg[u] = 0 #TODO: add avg here?
                if n_w == 'None' and m == 0:
                    fmt_get='-o'
                    add_leg = 'neurons'
                elif n_w == 'None' and m == 1:
                    fmt_get=':o'
                    add_leg = 'weights'
                else:
                    add_leg = n_w
                    fmt_get = 'o'
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fn[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u]+ ": " + add_leg, linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                # ax.errorbar(ll_pl, m_pl, yerr=err_pl, fmt=fmt_get, color=colors_fn[u], markersize=3, ecolor='k', capsize=5, \
                #     label=ax_leg[u], linewidth=2, elinewidth=0.5, markeredgewidth=0.5)
                # ax.bar(ll_pl+shifts[u], m_pl, yerr=err_pl, color=colors_fn[u], label=ax_leg[u], width=wid, align='center')
                # ax.errorbar(ll_pl+shifts[u], m_pl, yerr=err_pl, color=colors_fn[u], ecolor='k', capsize=3, label='', elinewidth=0.01, markeredgewidth=0.7, ls='none')
                ax.bar(ll_pl+shifts[u], m_pl, color=colors_fn[u], label=ax_leg[u], width=wid, align='center', edgecolor='white')
            
            plt.ylabel(r"$bitavg(\Delta FN)$")
            if n_w == 'neurons':
                ax.set_ylim([-2, 700])
            elif n_w == 'weights':
                ax.set_ylim([0, 1600])
            ax.set_xticklabels([31, 30, 29, 28, 27, 26, 25, 24, 23])

    for i, p in enumerate(ax.patches):
        ax.annotate("{:d}".format(p.get_height(), ':d'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10), 
                size=8,
                rotation=90,
                textcoords = 'offset points')
    max_all = np.max([np.max(n) for n in bpos_all])
    # ax.set_xlim([0, max_all+1])
    x_lim_new = 8
    ax.set_xlim([-0.5, x_lim_new])
    plt.tight_layout()
    ax.set_xticks(np.arange(0, x_lim_new+1, step=1))

    plt.legend(loc="upper right")
    plt.xlabel("Bit position")


    os.makedirs(os.path.dirname(sv_name), exist_ok=True)
    plt.savefig(sv_name, dpi=300, bbox_inches='tight',pad_inches = 0)
    print('saved as ', sv_name)
    # plt.show()

toplot_dict_template = {'sdc': {'orig_mns': [], 'orig_errs': [], 'corr_mns': [], 'corr_errs': [],
                                'resil_orig_mns': [], 'resil_orig_errs': [], 'resil_corr_mns': [], 'resil_corr_errs': []}, \
    'due': {'orig_mns': [], 'orig_errs': [], 'corr_mns': [], 'corr_errs': [],
            'resil_orig_mns': [], 'resil_orig_errs': [], 'resil_corr_mns': [], 'resil_corr_errs': []}, \
    'sdc_wgt': {'orig_mns': [], 'orig_errs': [], 'corr_mns': [], 'corr_errs': [], 'mns_diff': [], 'errs_diff': [],
                'resil_orig_mns': [], 'resil_orig_errs': [], 'resil_corr_mns': [], 'resil_corr_errs': [], 'resil_mns_diff': [], 'resil_errs_diff': []}, \
    'due_wgt': {'orig_mns': [], 'orig_errs': [], 'corr_mns': [], 'corr_errs': [],
                'resil_orig_mns': [], 'resil_orig_errs': [], 'resil_corr_mns': [], 'resil_corr_errs': []}, \
    'ap50': {'orig_mns': [], 'orig_errs': [], 'corr_mns': [], 'corr_errs': [],
             'resil_orig_mns': [], 'resil_orig_errs': [], 'resil_corr_mns': [], 'resil_corr_errs': []}, \
    'map': {'orig_mns': [], 'orig_errs': [], 'corr_mns': [], 'corr_errs': [],
            'resil_orig_mns': [], 'resil_orig_errs': [], 'resil_corr_mns': [], 'resil_corr_errs': []}, \
    'tpfpfn': {'orig': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': []}, 'corr': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': [], 'bpos': []},
              'resil_orig': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': []}, 'resil_corr': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': [], 'bpos': []}}}




####################################################################################
flts = ['neurons', 'weights'] #['neurons', 'weights'] #'neurons', 'weights'
suffix = "no_resil" 
eval_mode = "iou+class_labels" # iou+class_labels , iou


if eval_mode == "iou":
    suffix =  suffix + "_iou"
toplot_dict_n_w = []
ax_leg_n_w = []


"""
## experiment on clipper and DUE correction.
"""
def obj_det_plot_metrics(exp_folder_paths):

    ax_leg_template = []
    paths = exp_folder_paths
    flts_valid = {'neurons':False, 'weights':False}
    for flt_type in flts:
        toplot_dict = deepcopy(toplot_dict_template)
        ax_leg = deepcopy(ax_leg_template)

        plot=False
        for key in paths.keys():
            path = paths[key]
            model_dict = {"flt_type": flt_type, "suffix": suffix, 'bits': 1}
            model_dict['label_name'] = key
            try:
                model_dict["path"] = os.path.join(path[flt_type]['path'], "sdc_eval")
                model_dict["typ"] = path[flt_type]["typ"]
                toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)
                plot=True
            except:
                print("parsing of data failed for combination {}: {}".format(flt_type, model_dict))

        flts_valid[flt_type] = plot
        if plot:
            toplot_dict_n_w.append(toplot_dict)
            ax_leg_n_w.append(ax_leg)

            """
            CORR PLOTS
            """
            # Plot the images with all models: ----------------------------------------------------------------
            # mAP: 
            sv_name = "plots/evaluation/corr_metrics/" + "map_all_" + flt_type + "_corr.png"
            yname = "mAP"
            leg = ['orig', 'corr']

            mns_orig, errs_orig = toplot_dict['map']['orig_mns'], toplot_dict['map']['orig_errs']
            mns_corr, errs_corr = toplot_dict['map']['corr_mns'], toplot_dict['map']['corr_errs']
            plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name, ax_leg)

            # AP50:
            sv_name = "plots/evaluation/corr_metrics/" + "ap50_all_" + flt_type + "_corr.png"
            yname = "AP50"
            leg = ['orig', 'corr']

            mns_orig, errs_orig = toplot_dict['ap50']['orig_mns'], toplot_dict['ap50']['orig_errs']
            mns_corr, errs_corr = toplot_dict['ap50']['corr_mns'], toplot_dict['ap50']['corr_errs']
            plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name, ax_leg)

            # SDC rates:
            sv_name = "plots/evaluation/corr_metrics/" + "sdc_all_" + flt_type + "_corr.png"
            yname = "Error rates (%)"
            leg = ['$IVMOD_{corr\_sdc}$', '$IVMOD_{corr\_due}}$']

            mns_orig, errs_orig = toplot_dict['sdc']['corr_mns'], toplot_dict['sdc']['corr_errs']
            mns_corr, errs_corr = toplot_dict['due']['corr_mns'], toplot_dict['due']['corr_errs']
            plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name, ax_leg, cols=['indianred', 'lightgreen'], scale_to_perc=True)

            """
            ## RESIL Plots##
            """
            if suffix != 'no_resil':
                # Plot the images with all models: ----------------------------------------------------------------
                # mAP: 
                sv_name = "plots/evaluation/resil_metrics/" + "map_all_" + flt_type + "_resil.png"
                yname = "mAP"
                leg = ['resil_orig', 'resil_corr']

                mns_orig, errs_orig = toplot_dict['map']['resil_orig_mns'], toplot_dict['map']['resil_orig_errs']
                mns_corr, errs_corr = toplot_dict['map']['resil_corr_mns'], toplot_dict['map']['resil_corr_errs']
                plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name, ax_leg)

                # AP50:
                sv_name = "plots/evaluation/resil_metrics/" + "ap50_all_" + flt_type + "_resil.png"
                yname = "AP50"
                leg = ['resil_orig', 'resil_corr']

                mns_orig, errs_orig = toplot_dict['ap50']['resil_orig_mns'], toplot_dict['ap50']['resil_orig_errs']
                mns_corr, errs_corr = toplot_dict['ap50']['resil_corr_mns'], toplot_dict['ap50']['resil_corr_errs']
                plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name, ax_leg)

                # SDC rates:
                sv_name = "plots/evaluation/resil_metrics/" + "sdc_all_" + flt_type + suffix + "_resil.png"
                yname = "Error rates (%)"
                leg = ['$IVMOD_{resil\_sdc}$', '$IVMOD_{resil\_due}$']

                mns_orig, errs_orig = toplot_dict['sdc']['resil_corr_mns'], toplot_dict['sdc']['resil_corr_errs']
                mns_corr, errs_corr = toplot_dict['due']['resil_corr_mns'], toplot_dict['due']['resil_corr_errs']
                plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name, ax_leg, cols=['indianred', 'lightgreen'], scale_to_perc=True)


    # Verify that there are more faults in weights:
    # len(toplot_dict_n_w[0]['tpfpfn']['corr']['tp'][1]) #length of tps (neurons)
    # len(toplot_dict_n_w[1]['tpfpfn']['corr']['tp'][1]) #length of tps (weights)


    """
    ## CORR Plots##
    """
    # TP FP FN vs bpos
    if flts_valid['neurons']:
        n_w = 'neurons'
        fpfn = 'fp'
        sv_name = "plots/evaluation/corr_metrics/" + fpfn + "_diff_bpos_" + 'all' + '_' + n_w + ".png"
        plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

        fpfn = 'fn'
        sv_name = "plots/evaluation/corr_metrics/" + fpfn + "_diff_bpos_" + 'all'+ '_' + n_w + ".png"
        plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

        # TP FP FN vs bpos histogram
        fpfn = 'fp'
        sv_name = "plots/evaluation/corr_metrics/" + fpfn + "_hist_diff_bpos_" + 'all'  + '_' + n_w + ".png"
        plot_hist_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

        fpfn = 'fn'
        sv_name = "plots/evaluation/corr_metrics/" + fpfn + "_hist_diff_bpos_" + 'all' + '_' + n_w + ".png"
        plot_hist_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

    if flts_valid['weights']:
        n_w = 'weights'
        fpfn = 'fp'
        sv_name = "plots/evaluation/corr_metrics/" + fpfn + "_diff_bpos_" + 'all' + '_' + n_w + ".png"
        plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

        fpfn = 'fn'
        sv_name = "plots/evaluation/corr_metrics/" + fpfn + "_diff_bpos_" + 'all'+ '_' + n_w + ".png"
        plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

        fpfn = 'fp'
        sv_name = "plots/evaluation/corr_metrics/" + fpfn + "_hist_diff_bpos_" + 'all' + '_' + n_w + ".png"
        plot_hist_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

        fpfn = 'fn'
        sv_name = "plots/evaluation/corr_metrics/" + fpfn + "_hist_diff_bpos_" + 'all' + '_' + n_w + ".png"
        plot_hist_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

    """
    ## RESIL Plots##
    """
    if suffix != 'no_resil':
        if flts_valid['neurons']:
        # TP FP FN vs bpos
            n_w = 'neurons'
            fpfn = 'fp'
            sv_name = "plots/evaluation/resil_metrics/" + fpfn + "_diff_bpos_" + 'all' + suffix + '_' + n_w + "_resil.png"
            plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w, typ=suffix)

            fpfn = 'fn'
            sv_name = "plots/evaluation/resil_metrics/" + fpfn + "_diff_bpos_" + 'all' + suffix + '_' + n_w + "_resil.png"
            plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w, typ=suffix)

            fpfn = 'fp'
            sv_name = "plots/evaluation/resil_metrics/" + fpfn + "_hist_diff_bpos_" + 'all' + suffix + '_' + n_w + "_resil.png"
            plot_hist_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w, typ=suffix)

            fpfn = 'fn'
            sv_name = "plots/evaluation/resil_metrics/" + fpfn + "_hist_diff_bpos_" + 'all' + suffix + '_' + n_w + "_resil.png"
            plot_hist_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w, typ=suffix)


        if flts_valid['weights']:
            # TP FP FN vs bpos histogram
            n_w = 'weights'
            fpfn = 'fp'
            sv_name = "plots/evaluation/resil_metrics/" + fpfn + "_diff_bpos_" + 'all' + suffix + '_' + n_w + "_resil.png"
            plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w, typ=suffix)

            fpfn = 'fn'
            sv_name = "plots/evaluation/resil_metrics/" + fpfn + "_diff_bpos_" + 'all' + suffix + '_' + n_w + "_resil.png"
            plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w, typ=suffix)
            
            fpfn = 'fp'
            sv_name = "plots/evaluation/resil_metrics/" + fpfn + "_hist_diff_bpos_" + 'all' + suffix + '_' + n_w + "_resil.png"
            plot_hist_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w, typ=suffix)

            fpfn = 'fn'
            sv_name = "plots/evaluation/resil_metrics/" + fpfn + "_hist_diff_bpos_" + 'all' + suffix + '_' + n_w + "_resil.png"
            plot_hist_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w, typ=suffix)