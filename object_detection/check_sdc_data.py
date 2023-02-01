from pathlib import Path
from train_detection_model_LR3 import load_json_indiv
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def load_general_data(data_folder):
    files = list(Path(data_folder).glob('**/*.json'))
    file_list =[]
    for u in range(len(files)):
        if "ftraces" in str(files[u]) or 'feature' in str(files[u]): #Here NO ftraces used
            continue

        if "13" in str(files[u]): #Here NO ftraces used or "contrast" in str(files[u])
            print('Skipping hw faults 13.')
            continue

        # if target == 'all':
        file_list.append(files[u])
        print('added', files[u])
        # continue #(no double add)

    result_list = []
    for u in range(len(file_list)):

        dt = load_json_indiv(file_list[u]) #first file
        flt_type = str(file_list[u]).split("/")[-1][4:-5]
        dt2 = {"flt_type": flt_type, 'glob_features': dt["glob_features"]}
        result_list.append(dt2)

    return result_list

def plot_sdc_due(x_pos, sdc_m, sdc_err, save_name):
    fig = plt.figure()
    ax = fig.gca()

    ax.bar(x_pos, sdc_m)
    ax.errorbar(x_pos, sdc_m, yerr=sdc_err, fmt="o", capsize=3) #color='k', fmt="o", markersize='3'
    ax.set_xticks(ticks=x_pos, labels=flt_names, rotation=90, fontsize=14)

    # save_name = pth + "sdc_rates.png"
    fig.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
    print('Saved as', save_name)

def plot_sdc_due_same_graph(fig, ax, x_pos, sdc_m, sdc_err, cnt, flt_names, lbl):
    w = 0.5
    ax.bar(np.array(x_pos) - 4*w +0.5*w + cnt*w, sdc_m, width = w, label = lbl)
    ax.errorbar(np.array(x_pos) - 4*w +0.5*w + cnt*w, sdc_m, yerr=sdc_err, fmt="o", capsize=3, color='k', markersize='1', label=None) #color='k', fmt="o", markersize='3'
    ax.set_xticks(ticks=x_pos, labels=flt_names, rotation=90)

def merge_neurons_weights(res):
    flt_names = [n["flt_type"] for n in res]
    mem_ind = ["neurons" in x or "weights" in x for x in flt_names]
    res_mem = list(np.array(res)[mem_ind])
    sdc_rate_mean = [np.mean([x['glob_features']['sdc_rate'][0] for x in res_mem]), np.sum([x['glob_features']['sdc_rate'][1] for x in res_mem])]
    due_rate_mean = [np.mean([x['glob_features']['due_rate'][0] for x in res_mem]), np.sum([x['glob_features']['due_rate'][1] for x in res_mem])]
    if "sdc_lays" in res_mem[0]['glob_features'].keys():
        sdc_lays = flatten_list([n["glob_features"]["sdc_lays"] for n in res_mem])
    else:
        sdc_lays = []
    if "sdc_bpos" in res_mem[0]['glob_features'].keys():
        sdc_bpos = flatten_list([n["glob_features"]["sdc_bpos"] for n in res_mem])
    else:
        sdc_bpos = []
    if "sdc_vals" in res_mem[0]['glob_features'].keys():
        sdc_vals = flatten_list([n["glob_features"]["sdc_vals"] for n in res_mem])
    else:
        sdc_vals = []
    # sdc_bpos = flatten_list([n["glob_features"]["sdc_bpos"] for n in res_mem])
    # sdc_vals= flatten_list([n["glob_features"]["sdc_vals"] for n in res_mem])
    res_mem_new = [{'flt_type': 'memory_faults', 'glob_features': {'sdc_rate': sdc_rate_mean, 'due_rate': due_rate_mean, 'sdc_lays': sdc_lays, 'sdc_vals':sdc_vals, 'sdc_bpos': sdc_bpos}}]
    return list(np.array(res)[np.logical_not(mem_ind)]) + res_mem_new

def flatten_list(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]

########################

df = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti', 'resnet_imagenet', 'alexnet_imagenet']
# df = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'resnet_imagenet', 'alexnet_imagenet']
# df = ['resnet_imagenet']
########################

pth = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/'


fig_sdc = plt.figure()
ax_sdc = fig_sdc.gca()
fig_due = plt.figure()
ax_due = fig_due.gca()

cnt = 0
for exp in df:
    # Load extracted quantiles: -------------
    data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/' + exp + '_presum/'

    # fault_class_dict= {'no_sdc': 0, 'neurons': 1, 'weights': 2, 'blur': 3, 'noise': 4, 'contrast':5}
    # classes = list(fault_class_dict.keys())
    # output_dim = len(classes)
    print('Loading...', exp)
    res = load_general_data(data_folder)
    


    # Consolidate neurons/weights to memory
    res = merge_neurons_weights(res)

    stretch = 5
    x_pos = list(np.array(range(len(res)))*stretch)
    
    flt_names = [n["flt_type"] for n in res]
    sdc_rates = [n["glob_features"]["sdc_rate"] for n in res]
    sdc_m, sdc_err = [x[0] for x in sdc_rates], [x[1] for x in sdc_rates]

    due_rates = [n["glob_features"]["due_rate"] for n in res]
    due_m, due_err = [x[0] for x in due_rates], [x[1] for x in due_rates]

    # # # Plot sdc lays, bpos info (optional) -----------------------------
    if "sdc_lays" in res[0]["glob_features"].keys():
        sdc_lays = [n["glob_features"]["sdc_lays"] for n in res]
    else:
        sdc_lays = []
        # print('sdc_lays', sdc_lays)
    if "sdc_bpos" in res[0]["glob_features"].keys():
        sdc_bpos = [n["glob_features"]["sdc_bpos"] for n in res]
    else:
        sdc_bpos = []
    if "sdc_vals" in res[0]["glob_features"].keys():
        sdc_vals = [n["glob_features"]["sdc_vals"] for n in res]
    else:
        sdc_vals = []

    # # optional: plot sdc lays and bpos
    # for l in range(len(flt_names)):
    #     nm = flt_names[l]
    #     if "memory" not in nm:
    #         continue

    #     fig = plt.figure()
    #     plt.hist(sdc_lays[l], bins=75)
    #     save_name = pth + "sdc_lays_" + nm + ".png"
    #     fig.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
    #     print('saved as', save_name)

    #     fig = plt.figure()
    #     plt.hist(sdc_bpos[l], bins=32)
    #     save_name = pth + "sdc_bpos_" + nm + ".png"
    #     fig.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
    #     print('saved as', save_name)


    # Round
    sdc_m = [round(x*100,1) for x in sdc_m]
    sdc_err = [round(x*100,1) for x in sdc_err]
    due_m = [round(x*100,1) for x in due_m]
    due_err = [round(x*100,1) for x in due_err]

    # sort alphabetically
    sdc_m_sorted = [i for _,i in sorted(zip(flt_names, sdc_m))]
    sdc_err_sorted = [i for _,i in sorted(zip(flt_names, sdc_err))]
    flt_names_sorted = sorted(flt_names)
    print('flt', flt_names_sorted, 'sdc', sdc_m_sorted)
    due_m_sorted = [i for _,i in sorted(zip(flt_names, due_m))]
    due_err_sorted = [i for _,i in sorted(zip(flt_names, due_err))]
    print('flt', flt_names_sorted, 'due', due_m_sorted)

    plot_sdc_due_same_graph(fig_sdc, ax_sdc, x_pos, sdc_m_sorted, sdc_err_sorted, cnt, flt_names_sorted, lbl=exp)
    plot_sdc_due_same_graph(fig_due, ax_due, x_pos, due_m_sorted, due_err_sorted, cnt, flt_names_sorted, lbl=exp)
    # plot_sdc_due(x_pos, sdc_m, sdc_err, pth + "sdc_rates.png")
    # plot_sdc_due(x_pos, due_m, due_err, pth + "due_rates.png")
    # print([[flt_names[x], sdc_m[x], sdc_err[x], due_m[x], due_err[x]] for x in range(len(flt_names))])
    # print()

    cnt += 1

save_name = pth + "sdc_rates_all.png"
# fig_sdc.legend(bbox_to_anchor=(0.8, 1.05), loc='upper right', ncol=3, fontsize=9)
fig_sdc.legend(bbox_to_anchor=(1.25, 0.7), loc='upper right', ncol=1, fontsize=12)
ax_sdc.set_ylabel('SDC rates (%)', fontsize=14)
# ax_sdc.set_xlim([-0.5, np.max(x_pos)+2.5]) #adjusted
fig_sdc.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
print('Saved as', save_name)

save_name = pth + "due_rates_all.png"
# fig_due.legend(bbox_to_anchor=(0.8, 1.05), loc='upper right', ncol=3, fontsize=9)
fig_due.legend(bbox_to_anchor=(1.25, 0.7), loc='upper right', ncol=1, fontsize=12)
ax_due.set_ylabel('DUE rates (%)', fontsize=14)
fig_due.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
print('Saved as', save_name)


