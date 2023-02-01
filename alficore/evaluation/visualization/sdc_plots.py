import json
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os

from os.path import dirname as up
sys.path.append(up(up(up(os.getcwd()))))

def load_json_indiv(gt_path):
    with open(gt_path) as f:
        coco_gt = json.load(f)
        f.close()
    return coco_gt


def add_data(toplot_dict, ax_leg, model_dict):

    model_name = model_dict["model_name"]
    dataset_name = model_dict["dataset_name"]
    flt_type = model_dict["flt_type"]
    suffix = model_dict["suffix"]
    bits = model_dict["bits"]
    label_name = model_dict["label_name"]
    exp_path = model_dict["exp_path"][flt_type]
    
    # Load from file saved in yolo_analysis3.py:
    json_path = os.path.join(exp_path, "sdc_plots/json_files", model_name + "_" + dataset_name + "_" + "results_1_" + flt_type + "_images" + suffix + ".json")

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
    toplot_dict["sdc"]["mns_orig"].append(m)
    toplot_dict["sdc"]["errs_orig"].append(err)
    # DUE rate images
    m, err = get_m_err(due_rate)
    toplot_dict["due"]["mns_orig"].append(m)
    toplot_dict["due"]["errs_orig"].append(err)



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


    ax_leg.append(label_name)

    return toplot_dict, ax_leg


def plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, legend_text, yname, sv_name, cols = None, scale_to_perc = None):

    ind = np.arange(len(mns_orig))  # the x locations for the groups
    fig, ax = plt.subplots()
    width = 0.35  # the width of the bars
    if scale_to_perc:
        mns_orig = np.array(mns_orig)*100
        mns_corr = np.array(mns_corr)*100
        errs_orig = np.array(errs_orig)*100
        errs_corr = np.array(errs_corr)*100


    if cols == None:
        if mns_corr is not None:
            # All original ones:
            ax.bar(ind - width/2, mns_orig, width, yerr=errs_orig, label=legend_text[0], edgecolor='black')
            ax.errorbar(ind - width/2, mns_orig, yerr=errs_orig, ecolor='black', capsize=10, label='', ls = 'none') #, alpha=0.5,  

            # All corrupted ones:
            ax.bar(ind + width/2, mns_corr, width, yerr=errs_corr, label=legend_text[1], edgecolor='black')
            ax.errorbar(ind + width/2, mns_corr, yerr=errs_corr, ecolor='black', capsize=10, label='', ls = 'none')

            ax.legend(loc="upper right") #, bbox_to_anchor=(0.8,0.8)
        else:
            ax.bar(ind , mns_orig, width, yerr=errs_orig, label='', edgecolor='black')
            ax.errorbar(ind, mns_orig, yerr=errs_orig, ecolor='black', capsize=10, label='', ls = 'none') #, alpha=0.5, 
    else:
        if mns_corr is not None:
            # All original ones:
            ax.bar(ind - width/2, mns_orig, width, yerr=errs_orig, label=legend_text[0], color=cols[0], edgecolor='black')
            ax.errorbar(ind - width/2, mns_orig, yerr=errs_orig, ecolor='black', capsize=10, label='', ls = 'none') #, alpha=0.5,  

            # All corrupted ones:
            ax.bar(ind + width/2, mns_corr, width, yerr=errs_corr, label=legend_text[1], color=cols[1], edgecolor='black')
            ax.errorbar(ind + width/2, mns_corr, yerr=errs_corr, ecolor='black', capsize=10, label='', ls = 'none')

            ax.legend(loc="upper right") #bbox_to_anchor=(1.2,0.8)
        else:
            ax.bar(ind , mns_orig, width, yerr=errs_orig, label='', color=cols, edgecolor='black')
            ax.errorbar(ind, mns_orig, yerr=errs_orig, ecolor='black', capsize=10, label='', ls = 'none') #, alpha=0.5, 

     

    # Add some text for labels, title and custom x-axis tick labels, etc.
    fnt_size = 13
    
    ax.set_ylabel(yname, fontsize=fnt_size)
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(ind)
    ax.set_xticklabels(ax_leg, fontsize=fnt_size, rotation=45)
    # ax.set_xticklabels(ax_leg)
    ax.set_ylim([0, max(mns_orig)*1.1+0.1])

    
    # Plot labels in correct scientific notation
    round_to = 1
    if mns_corr is not None:
        for i, v in enumerate(mns_orig):
            ax.text(i - 0.40, v + errs_orig[i] +0.05 + 0.01*(v+errs_orig[i]), np.round(v, round_to), fontsize=fnt_size-1)  # , color='blue', fontweight='bold')
        for i, v in enumerate(mns_corr):
            ax.text(i + 0.05, v + errs_corr[i] +0.05 + 0.01*(v+errs_corr[i]), np.round(v, round_to), fontsize=fnt_size-1)  # , color='blue', fontweight='bold')
    else:
        for i, v in enumerate(mns_orig):
            ax.text(i - 0.1, v + errs_orig[i] + +0.05 + 0.01*(v+errs_orig[i]), np.round(v, round_to), fontsize=fnt_size-1)  # , color='blue', fontweight='bold')


    fig.tight_layout()
    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)

    plt.show()


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
        bpos_all = toplot_dict['tpfpfn']['bpos']
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



def plot_avg_tp_bpos_old(tpl, ax_leg, sv_name, plothow='fp', n_w='None'):
    """
    plothow: switches between fp and fn
    n_w: switches between "neurons", "weights" or both "None"
    """

    res_n_w, ax_leg, bpos_all = eval_n_w(tpl, plothow, ax_leg)

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

    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)
    plt.show()


def plot_avg_tp_bpos(tpl, ax_leg, sv_name, plothow='fp', n_w='None'):
    """
    plothow: switches between fp and fn
    n_w: switches between "neurons", "weights" or both "None"
    """

    res_n_w, ax_leg, bpos_all = eval_n_w(tpl, plothow, ax_leg)

    if n_w == 'neurons':
        res_n_w = [res_n_w[0]]
    elif n_w == 'weights':
        res_n_w = [res_n_w[1]]

    fig, ax = plt.subplots()  
    ll = np.arange(0, 31+1)
    # ll = np.arange(0, 15+1)

    colors_fp = ['b', 'g', 'r', 'k', 'orange', 'purple'] 
    if 'fp' in plothow:
        for m in range(len(res_n_w)): #neurons, weights
            res = res_n_w[m]

            m_fps_all = res['m_fps']
            err_fps_all = res['err_fps']
            ax_leg = res['ax_leg']

            
            shifts = np.linspace(-0.4,0.4, num=len(m_fps_all))
            wid = shifts[1]-shifts[0]
            for u in range(len(m_fps_all)):
                m_pl = m_fps_all[u]
                mask = np.logical_not(np.isnan(m_pl))
                m_pl = np.array(m_pl)[mask]
                err_pl = np.array(err_fps_all[u])[mask]
                ll_pl = np.array(ll)[mask]
                print(ax_leg[u], ' (', n_w, '): Sdc event adds an avg number of fps', 'mean', np.mean(m_pl), 'range', np.min(m_pl), np.max(m_pl), 'err', np.std(m_pl)*1.96/np.sqrt(len(m_pl))) #, len(m_pl), m_pl)
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
                ax.bar(ll_pl+shifts[u], m_pl, yerr=err_pl, color=colors_fp[u], label=ax_leg[u], width=wid, align='center', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2, label='', elinewidth=1, markeredgewidth=0.7, ls='none'))
                # ax.errorbar(ll_pl+shifts[u], m_pl, yerr=err_pl, ecolor='gray', capsize=3, label='', elinewidth=0.01, markeredgewidth=0.7, markeredgecolor='gray', ls='none')
                
                # plt.ylabel(r"$FP_{ad}$") #$avg(FP_{corr} - FP_{orig})$ objects")

            plt.ylabel(r"$bitavg(\Delta FP)$")
            ax.set_ylim([-2, 1000])

    colors_fn = colors_fp
    if plothow == 'fn':
        for m in range(len(res_n_w)):
            res = res_n_w[m]

            m_fns_all = res['m_fns']
            err_fns_all = res['err_fns']
            ax_leg = res['ax_leg']

            shifts = np.linspace(-0.4,0.4, num=len(m_fns_all))
            wid = shifts[1]-shifts[0]
            for u in range(len(m_fns_all)):
                # fns
                m_pl = m_fns_all[u]
                mask = np.logical_not(np.isnan(m_pl))
                m_pl = np.array(m_pl)[mask]
                err_pl = np.array(err_fns_all[u])[mask]
                ll_pl = np.array(ll)[mask]
                print(ax_leg[u], ' (' + n_w + '): Sdc event adds an avg number of fns', 'mean', np.mean(m_pl), 'range', np.min(m_pl), np.max(m_pl), 'err',  np.std(m_pl)*1.96/np.sqrt(len(m_pl)))
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
                ax.bar(ll_pl+shifts[u], m_pl*100, yerr=err_pl*100, color=colors_fn[u], label=ax_leg[u], width=wid, align='center', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2, label='', elinewidth=1, markeredgewidth=0.7, ls='none'))
            
            plt.ylabel(r"$bitavg(\Delta FN_{n})(\%)$")
            ax.set_ylim([-30, 100])

    max_all = np.max([np.max(n) for n in bpos_all])
    # ax.set_xlim([0, max_all+1])
    x_lim_new = 8
    ax.set_xlim([-0.5, x_lim_new])
    ax.set_xticks(np.arange(0, x_lim_new+1, step=1))

    plt.legend(loc="upper right")
    plt.xlabel("Bit position")

    plt.savefig(sv_name, dpi=300, bbox_inches='tight',pad_inches = 0)
    print('saved as ', sv_name)
    plt.show()




toplot_dict_template = {'sdc': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    'due': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    'sdc_wgt': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': [], 'mns_diff': [], 'errs_diff': []}, \
    'due_wgt': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    'ap50': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    'map': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    'tpfpfn': {'orig': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': []}, 'corr': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': []}, 'bpos': []}}
ax_leg_template = []



####################################################################################
flts = ['neurons', 'weights'] #['neurons', 'weights'] #'neurons', 'weights'
suffix = "_none"
eval_mode = "iou+class_labels" # iou+class_labels , iou


if eval_mode == "iou":
    suffix =  suffix + "_iou"
toplot_dict_n_w = []
ax_leg_n_w = []

exp_path = [{'neurons': "path/to/neurons_weight experiment", 'weights': "..."}, \
                {'neurons': "...", 'weights': "..."}, \
                    {'neurons': "...", 'weights': "..."}, \
                        {'neurons': "...", 'weights': "..."}, \
                            {'neurons': "...", 'weights': "..."},\
                                {'neurons': "...", 'weights': "..."}]

for flt_type in flts:
    
    toplot_dict = deepcopy(toplot_dict_template)
    ax_leg = deepcopy(ax_leg_template)

    model_dict = {"model_name": 'yolov3', "dataset_name": 'coco2017', "flt_type": flt_type, "suffix": suffix, 'bits': 32, "label_name": "Yolo+Coco", "exp_path":exp_path[0]}
    toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)

    model_dict = {"model_name": 'yolov3', "dataset_name": 'kitti', "flt_type": flt_type, "suffix": suffix, 'bits': 32, "label_name": "Yolo+kitti", "exp_path":exp_path[1]}
    toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)

    model_dict = {"model_name": 'yolov3u_silu', "dataset_name": 'lyft', "flt_type": flt_type, "suffix": suffix, 'bits': 32, "label_name": "Yolo+Lyft", "exp_path":exp_path[2]}
    toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)

    model_dict = {"model_name": 'retina005', "dataset_name": 'coco2017', "flt_type": flt_type, "suffix": suffix, 'bits': 32, "label_name": "RetinaNet+Coco", "exp_path":exp_path[3]}
    toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)

    model_dict = {"model_name": 'detectron', "dataset_name": 'coco2017', "flt_type": flt_type, "suffix": suffix, 'bits': 32, "label_name": "FRCNN+Coco", "exp_path":exp_path[4]}
    toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)

    model_dict = {"model_name": 'detectron', "dataset_name": 'kitti', "flt_type": flt_type, "suffix": suffix, 'bits': 32, "label_name": "FRCNN+Kitti", "exp_path":exp_path[5]}
    toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)


    # # not needed for now:
    # model_dict = {"model_name": 'yolov3u_leaky', "dataset_name": 'lyft', "flt_type": flt_type, "suffix": suffix, 'bits': 32, "label_name": "Yolo+Lyft+Leaky"}
    # toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)

    # model_dict = {"model_name": 'yolov3u_relu', "dataset_name": 'lyft', "flt_type": flt_type, "suffix": suffix, 'bits': 32, "label_name": "Yolo+Lyft+Relu"}
    # toplot_dict, ax_leg = add_data(toplot_dict, ax_leg, model_dict)


    toplot_dict_n_w.append(toplot_dict)
    ax_leg_n_w.append(ax_leg)

    # Plot the images with all models: ----------------------------------------------------------------
    # mAP: 
    sv_name = "plots/evaluation/metrics/" + "map_all_" + flt_type + ".png"
    yname = "mAP"
    leg = ['orig', 'corr']

    mns_orig, errs_orig = toplot_dict['map']['mns_orig'], toplot_dict['map']['errs_orig']
    mns_corr, errs_corr = toplot_dict['map']['mns_corr'], toplot_dict['map']['errs_corr']
    plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name)


    # AP50: 
    sv_name = "plots/evaluation/metrics/" + "ap50_all_" + flt_type + ".png"
    yname = "AP50"
    leg = ['orig', 'corr']

    mns_orig, errs_orig = toplot_dict['ap50']['mns_orig'], toplot_dict['ap50']['errs_orig']
    mns_corr, errs_corr = toplot_dict['ap50']['mns_corr'], toplot_dict['ap50']['errs_corr']
    plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name)


    # SDC rates: 
    sv_name = "plots/evaluation/metrics/" + "sdc_all_" + flt_type + suffix + ".png"
    yname = "Error rates (%)"
    leg = ['sdc', 'due']

    mns_orig, errs_orig = toplot_dict['sdc']['mns_orig'], toplot_dict['sdc']['errs_orig']
    mns_corr, errs_corr = toplot_dict['due']['mns_orig'], toplot_dict['due']['errs_orig']
    plot_metric(mns_orig, errs_orig, mns_corr, errs_corr, leg, yname, sv_name, cols=['indianred', 'lightgreen'], scale_to_perc=True)
    


# Verify that there are more faults in weights:
len(toplot_dict_n_w[0]['tpfpfn']['corr']['tp'][1]) #length of tps (neurons)
len(toplot_dict_n_w[1]['tpfpfn']['corr']['tp'][1]) #length of tps (weights)



# TP FP FN vs bpos
fpfn = 'fp'
n_w = 'neurons'
sv_name = "plots/evaluation/metrics/" + fpfn + "_diff_bpos_" + 'all' + suffix + '_' + n_w + ".png"
plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)


fpfn = 'fp'
n_w = 'weights'
sv_name = "plots/evaluation/metrics/" + fpfn + "_diff_bpos_" + 'all' + suffix + '_' + n_w + ".png"
plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

fpfn = 'fn'
n_w = 'neurons'
sv_name = "plots/evaluation/metrics/" + fpfn + "_diff_bpos_" + 'all' + suffix + '_' + n_w + ".png"
plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)

fpfn = 'fn'
n_w = 'weights'
sv_name = "plots/evaluation/metrics/" + fpfn + "_diff_bpos_" + 'all' + suffix + '_' + n_w + ".png"
plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = fpfn, n_w=n_w)


# # # # plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = 'fp_bbox')
# # # # plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = 'fp_class')

# # # # sv_name = "plots/evaluation/metrics/" + "fn_diff_bpos_" + 'all' + suffix + ".png"
# # # # plot_avg_tp_bpos(toplot_dict_n_w, ax_leg_n_w, sv_name, plothow = 'fn')

