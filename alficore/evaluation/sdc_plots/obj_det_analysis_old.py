import argparse
import json
import logging.config
import os
import sys
import numpy as np
import torch
from torch.multiprocessing import Process
from torchvision import transforms, models
from tqdm import tqdm
from os.path import dirname as up
from collections import namedtuple

## comment this for debug
sys.path.append(up(up(up(os.getcwd()))))
from alficore.dataloader.coco_loader import coco_loader
from alficore.wrapper.ptfiwrap import ptfiwrap
# from util.helper_functions import getdict_ranger, get_savedBounds_minmax
# from util.evaluate import extract_ranger_bounds
# from util.ranger_automation import get_Ranger_protection
import pandas as pd
from typing import Dict, List, Optional, Tuple
from alficore.models.yolov3.darknet import Darknet
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize


# from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg
from alficore.dataloader.objdet_baseClasses.boxes import Boxes, BoxMode
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.models.yolov3.utils import non_max_suppression, rescale_boxes
from alficore.evaluation.visualization.visualization import Visualizer, ColorMode


from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
from alficore.dataloader.ppp_loader import PPP_obj_det_dataloader
# from alficore.dataloader.kitti2D_loader import Kitti2D_dataloader
import pickle
# from darknet_Ranger import Darknet_Ranger

from obj_det_evaluate_jsons import *
import csv
import json 
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")



torch.cuda.empty_cache()
logging.config.fileConfig('fi.conf')
log = logging.getLogger()
#NOTE: cannot save numpy variables as json, need to convert to lists, int, etc.

def read_csv(file):
    import csv
    result = []
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            result.append(row)
    return result
    

def extract(corr_ap, corr_ap50, nan_inf_flags, typ, nr_epochs, folder_path):
    # typ = 'corr', 'ranger_corr'

    for n in range(nr_epochs):
        corr = load_json_indiv(folder_path + '/' + typ + '_model/epochs/' + str(n) +'/coco_results_' + typ + '_model_' +str(n) + '_epoch.json') 
        corr_ap.append(corr["AP"])
        corr_ap50.append(corr["AP50"])

        if typ == 'corr':
            typ2 = 'corr'
        elif typ == 'ranger_corr':
            typ2 = 'resil_corr'
        infnan = read_csv(folder_path + '/' + typ + '_model/epochs/' + str(n) +'/inf_nan_' + typ2 + '.csv') 
        infnan = infnan[1:] #remover header
        nans = [n[0] == 'True' for n in infnan]
        nan_infs = [n[1] == 'True' for n in infnan]
        # print('Nans', np.sum(nans), 'Nans or infs', np.sum(nan_infs))
        nan_inf_flags.append(nan_infs)

    return corr_ap, corr_ap50, nan_inf_flags


def get_map_ap50_infnan(folder_path):

    list_subfolders_with_paths = [f.path for f in os.scandir(folder_path + '/' + 'corr' + '_model/epochs/') if f.is_dir()]
    nr_epochs = len(list_subfolders_with_paths)

    orig = load_json_indiv(folder_path + '/orig_model/epochs/0/coco_results_orig_model_0_epoch.json') 
    orig_ap = orig["AP"]
    orig_ap50 = orig["AP50"]

    corr_ap = []
    corr_ap50 = []
    corr_nan_inf_flags = []
    corr_ap, corr_ap50, corr_nan_inf_flags = extract(corr_ap, corr_ap50, corr_nan_inf_flags, 'corr', nr_epochs, folder_path)

    # print('map', orig_ap)
    # print('ap50', orig_ap50)
    # print('map corr', np.mean(corr_ap), np.std(corr_ap)*1.96/np.sqrt(len(corr_ap)))
    # print('ap50 corr', np.mean(corr_ap50), np.std(corr_ap50)*1.96/np.sqrt(len(corr_ap50)))

    if os.path.isdir(folder_path + '/ranger_model/'): 
        orig = load_json_indiv(folder_path + '/ranger_model/epochs/0/coco_results_ranger_model_0_epoch.json') 
        orig_ranger_ap = orig["AP"]
        orig_ranger_ap50 = orig["AP50"]

        ranger_corr_ap = []
        ranger_corr_ap50 = []
        ranger_corr_nan_inf_flags = []
        ranger_corr_ap, ranger_corr_ap50, ranger_corr_nan_inf_flags = extract(ranger_corr_ap, ranger_corr_ap50, ranger_corr_nan_inf_flags, 'ranger_corr', nr_epochs, folder_path)

        # print('map ranger', orig_ranger_ap)
        # print('ap50 ranger', orig_ranger_ap50)
        # print('map corr ranger', np.mean(ranger_corr_ap), np.std(ranger_corr_ap)*1.96/np.sqrt(len(ranger_corr_ap)))
        # print('ap50 corr ranger', np.mean(ranger_corr_ap50), np.std(ranger_corr_ap50)*1.96/np.sqrt(len(ranger_corr_ap50)))
        # # print('percentage of nan/infs encountered:', np.sum(np.sum(nan_inf_flags))/np.size(nan_inf_flags))
    else:
        orig_ranger_ap, orig_ranger_ap50, ranger_corr_ap, ranger_corr_ap50, ranger_corr_nan_inf_flags = None, None, None, None, None

    return orig_ap, orig_ap50, orig_ranger_ap, orig_ranger_ap50, corr_ap, corr_ap50, ranger_corr_ap, ranger_corr_ap50, nr_epochs, corr_nan_inf_flags, ranger_corr_nan_inf_flags



# def get_map_ap50_infnan(folder_path):
#
#
#     list_subfolders_with_paths = [f.path for f in os.scandir(folder_path + '/corr_model/epochs/') if f.is_dir()]
#     nr_epochs = len(list_subfolders_with_paths)
#
#     orig = load_json_indiv(folder_path + '/orig_model/epochs/0/coco_results_orig_model_0_epoch.json') 
#     orig_ap = orig["AP"]
#     orig_ap50 = orig["AP50"]
#
#     corr_ap = []
#     corr_ap50 = []
#     nan_inf_flags = []
#
#     for n in range(nr_epochs):
#         corr = load_json_indiv(folder_path + '/corr_model/epochs/' + str(n) +'/coco_results_corr_model_' +str(n) + '_epoch.json') 
#         corr_ap.append(corr["AP"])
#         corr_ap50.append(corr["AP50"])
#
#         infnan = read_csv(folder_path + '/corr_model/epochs/' + str(n) +'/inf_nan_corr.csv') 
#         infnan = infnan[1:] #remover header
#         nans = [n[0] == 'True' for n in infnan]
#         nan_infs = [n[1] == 'True' for n in infnan]
#         # print('Nans', np.sum(nans), 'Nans or infs', np.sum(nan_infs))
#         nan_inf_flags.append(nan_infs)
#
#     print('map', orig_ap)
#     print('ap50', orig_ap50)
#     print('map corr', np.mean(corr_ap), np.std(corr_ap)*1.96/np.sqrt(len(corr_ap)))
#     print('ap50 corr', np.mean(corr_ap50), np.std(corr_ap50)*1.96/np.sqrt(len(corr_ap50)))
#     # print('percentage of nan/infs encountered:', np.sum(np.sum(nan_inf_flags))/np.size(nan_inf_flags))
#
#     return orig_ap, orig_ap50, corr_ap, corr_ap50, nr_epochs, nan_inf_flags


def plot_hist(list1, list2, yname, sv_name):
    fig, ax = plt.subplots()

    list_to_plot = list1 #[orig_ap]
    ax.bar([0], np.mean(list_to_plot), yerr=np.std(list_to_plot)*1.96/np.sqrt(len(list_to_plot)), align='center', alpha=0.5, ecolor='black', capsize=10, label='orig')
    ax.errorbar([0], np.mean(list_to_plot), yerr=np.std(list_to_plot)*1.96/np.sqrt(len(list_to_plot)), alpha=0.5, ecolor='black', capsize=10, label='')

    list_to_plot = list2 #corr_ap
    ax.bar([1], np.mean(list_to_plot), yerr=np.std(list_to_plot)*1.96/np.sqrt(len(list_to_plot)), align='center', alpha=0.5, ecolor='black', capsize=10, label='corr')
    ax.errorbar([1], np.mean(list_to_plot), yerr=np.std(list_to_plot)*1.96/np.sqrt(len(list_to_plot)), alpha=0.5, ecolor='black', capsize=10, label='')

    plt.ylabel(yname) #"mAP") #'AP50'

    # Plot labels in correct scientific notation
    round_to = 3
    for i, v in enumerate([np.mean(list1), np.mean(list2)]):
        ax.text(i + - 0.25, v + .01, np.round(v, round_to))  # , color='blue', fontweight='bold')
    # Remove axis
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.legend()

    # sv_name = "plots/evaluation/retina_coco_map_" + flt_type + "_b" + str(bits) + suffix + ".png"
    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)
    plt.show()


def plot_hist_no_avg(orig_sdc, sv_name):
    fig, ax = plt.subplots()
    ax.hist(orig_sdc)

    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    plt.xlabel("sdc")
    plt.ylabel("frequency")
    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)
    plt.show()


def get_mask_any_change_orig_corr(dct_sdc, typ = 'no_ranger'):
    # Filter only by those where tp, fp or fn changes ------------------------
    if typ == 'no_ranger':
        tps_orig = [n["orig"]["tp"] for n in dct_sdc]
        fps_orig = [n["orig"]["fp"] for n in dct_sdc]
        fns_orig = [n["orig"]["fn"] for n in dct_sdc]

        tps_corr = [n["corr"]["tp"] for n in dct_sdc]
        fps_corr = [n["corr"]["fp"] for n in dct_sdc]
        fns_corr = [n["corr"]["fn"] for n in dct_sdc]
    elif typ == 'ranger': #TODO
        tps_orig = [n["orig_resil"]["tp"] for n in dct_sdc]
        fps_orig = [n["orig_resil"]["fp"] for n in dct_sdc]
        fns_orig = [n["orig_resil"]["fn"] for n in dct_sdc]

        tps_corr = [n["corr_resil"]["tp"] for n in dct_sdc]
        fps_corr = [n["corr_resil"]["fp"] for n in dct_sdc]
        fns_corr = [n["corr_resil"]["fn"] for n in dct_sdc]


    tps = np.array(tps_corr) - np.array(tps_orig)
    fps = np.array(fps_corr) - np.array(fps_orig)
    fns = np.array(fns_corr) - np.array(fns_orig)

    return np.logical_or(np.logical_or(fns != 0, tps != 0), fps != 0)


def plot_tpfpfn(dct_sdc, flts_del, sv_name):

    # # Filter only by those where tp, fp or fn changes ------------------------
    tps_orig = [n["orig"]["tp"] for n in dct_sdc]
    fps_orig = [n["orig"]["fp"] for n in dct_sdc]
    fns_orig = [n["orig"]["fn"] for n in dct_sdc]

    tps_corr = [n["corr"]["tp"] for n in dct_sdc]
    fps_corr = [n["corr"]["fp"] for n in dct_sdc]
    fns_corr = [n["corr"]["fn"] for n in dct_sdc]

    tps = np.array(tps_corr) - np.array(tps_orig)
    fps = np.array(fps_corr) - np.array(fps_orig)
    fns = np.array(fns_corr) - np.array(fns_orig)


    # # mask_any_change = np.logical_or(np.logical_or(fns != 0, tps != 0), fps != 0)
    # # print('In', sv_name, ': Filtered only changing samples from', len(tps_orig), 'to', np.sum(mask_any_change))
    # # # mask_any_change = get_mask_any_change(dct_sdc)
    # # tps = tps[mask_any_change]
    # # fps = fps[mask_any_change]
    # # fns = fns[mask_any_change]
    # # faults = flts_del[:, mask_any_change]

    faults = flts_del
    bpos = faults[6,:]
    

    x_ind = np.arange(len(tps))

    # Get plotting boundaries: -----------------------------------------    
    # tps_pos = [n if n > 0 else 0 for n in tps]
    # # if np.sum(tps_pos) > 0:
    # #     print('stop')
    # #     a = np.where(np.array(tps_pos) > 0)[0]
    # #     dct_sdc_filt = np.array(dct_sdc)[mask_any_change].tolist()

    # tps_neg = [n if n < 0 else 0 for n in tps]
    tps_pos = [max(n,0) for n in tps]
    tps_neg = [min(n,0) for n in tps]
    # bottom1 = [tps_pos[n] if fps[n] >= 0 else tps_neg[n] for n in range(len(tps))] 
    bottom1 = [tps_pos[n] if fns[n] >= 0 else tps_neg[n] for n in range(len(tps))] 

    # tps_fps_pos = [bottom1[n] + fps[n] if fps[n] >= 0 else 0 for n in range(len(fps))]
    # tps_fps_neg = [bottom1[n] + fps[n] if fps[n] < 0 else 0 for n in range(len(fps))] 
    tps_fns_pos = [max([tps_pos[n], bottom1[n] + fns[n], 0]) for n in range(len(fns))]
    tps_fns_neg = [min([tps_neg[n], bottom1[n] + fns[n], 0]) for n in range(len(fns))] 
    # bottom2 = [tps_fps_pos[n] if fns[n] >= 0 else tps_fps_neg[n] for n in range(len(fns))] 
    bottom2 = [tps_fns_pos[n] if fps[n] >= 0 else tps_fns_neg[n] for n in range(len(fps))] 

    width = 1.

    # Plot corrupted images vs tps, fps, fns change
    fig, ax = plt.subplots()    
    plt.bar(x_ind, tps, width, color='r') #tps starting from 0
    plt.bar(x_ind, fns, width, bottom=np.array(bottom1), color='g')
    plt.bar(x_ind, fps, width, bottom=np.array(bottom2), color='b')

    plt.legend(['tp', 'fn', 'fp'], loc="upper right", bbox_to_anchor=(1.12,1))
    plt.xlabel("Index of corrupted image")
    plt.ylabel("Difference in tp, fp, fn")

    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)
    plt.show()


def plot_lay_tpfpfn(dct_sdc, flts_del, sv_name, injection_type, plot_ind = False):
    """"
    Plots only those data where a change happens! Can be smaller than input list.
    """

    # # Filter only by those where tp, fp or fn changes ------------------------
    tps_orig = [n["orig"]["tp"] for n in dct_sdc]
    fps_orig = [n["orig"]["fp"] for n in dct_sdc]
    fns_orig = [n["orig"]["fn"] for n in dct_sdc]

    tps_corr = [n["corr"]["tp"] for n in dct_sdc]
    fps_corr = [n["corr"]["fp"] for n in dct_sdc]
    fns_corr = [n["corr"]["fn"] for n in dct_sdc]

    tps = np.array(tps_corr) - np.array(tps_orig)
    fps = np.array(fps_corr) - np.array(fps_orig)
    fns = np.array(fns_corr) - np.array(fns_orig)


    # # mask_any_change = np.logical_or(np.logical_or(fns != 0, tps != 0), fps != 0)
    # # # mask_any_change = get_mask_any_change(dct_sdc)
    # # tps = tps[mask_any_change]
    # # fps = fps[mask_any_change]
    # # fns = fns[mask_any_change]
    # # faults = flts_del[:, mask_any_change]

    faults = flts_del
    bpos = faults[6,:]
    if injection_type == "neurons":
        lays = faults[1,:]
    elif injection_type == "weights":
        lays = faults[0,:]


    m_tps, err_tps = [], []
    m_fps, err_fps = [], []
    m_fns, err_fns = [], []

    if len(lays) == 0:
        print('No information for plotting available.')
        return

    for i in range(0, max(lays)+1):
        lst = tps[lays == i]
        m_tps.append(np.mean(lst))
        err_tps.append(np.std(lst)*1.96/np.sqrt(len(lst)))

        lst = fps[lays == i]
        m_fps.append(np.mean(lst))
        err_fps.append(np.std(lst)*1.96/np.sqrt(len(lst)))

        lst = fns[lays == i]
        m_fns.append(np.mean(lst))
        err_fns.append(np.std(lst)*1.96/np.sqrt(len(lst)))


    fig, ax = plt.subplots()
    if plot_ind:    
        plt.scatter(lays, tps, color='r', s=2.2)
        plt.scatter(lays, fps, color='b', s=2.1)
        plt.scatter(lays, fns, color='g', s=2.)
    else:
        ll = np.arange(0, max(lays)+1)
        ax.errorbar(ll, m_tps, yerr=err_tps, fmt='-o', color='r', markersize=3)
        ax.errorbar(ll, m_fps, yerr=err_fps, fmt='-o', color='b', markersize=3)
        ax.errorbar(ll, m_fns, yerr=err_fns, fmt='-o', color='g', markersize=3)

    plt.legend(['tp', 'fp', 'fn'], loc="upper right")
    plt.xlabel("Layer index")
    plt.ylabel("Avg difference in tp, fp, fn")

    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)
    plt.show()


def plot_bpos_tpfpfn(dct_sdc, flts_del, sv_name, injection_type, plot_ind = False):

    # # Filter only by those where tp, fp or fn changes ------------------------
    tps_orig = [n["orig"]["tp"] for n in dct_sdc]
    fps_orig = [n["orig"]["fp"] for n in dct_sdc]
    fns_orig = [n["orig"]["fn"] for n in dct_sdc]

    tps_corr = [n["corr"]["tp"] for n in dct_sdc]
    fps_corr = [n["corr"]["fp"] for n in dct_sdc]
    fns_corr = [n["corr"]["fn"] for n in dct_sdc]

    tps = np.array(tps_corr) - np.array(tps_orig)
    fps = np.array(fps_corr) - np.array(fps_orig)
    fns = np.array(fns_corr) - np.array(fns_orig)


    # mask_any_change = np.logical_or(np.logical_or(fns != 0, tps != 0), fps != 0)
    # # mask_any_change = get_mask_any_change(dct_sdc)
    # tps = tps[mask_any_change]
    # fps = fps[mask_any_change]
    # fns = fns[mask_any_change]
    # faults = flts_del[:, mask_any_change]

    faults =flts_del
    bpos = faults[6,:]
    # if injection_type == "neurons":
    #     lays = faults[1,:]
    # elif injection_type == "weights":
    #     lays = faults[0,:]


    m_tps, err_tps = [], []
    m_fps, err_fps = [], []
    m_fns, err_fns = [], []
    if len(bpos) == 0:
        print('No information for plotting available.')
        return

    for i in range(0, max(bpos)+1):
        lst = tps[bpos == i]
        m_tps.append(np.mean(lst))
        err_tps.append(np.std(lst)*1.96/np.sqrt(len(lst)))

        lst = fps[bpos == i]
        m_fps.append(np.mean(lst))
        err_fps.append(np.std(lst)*1.96/np.sqrt(len(lst)))

        lst = fns[bpos == i]
        m_fns.append(np.mean(lst))
        err_fns.append(np.std(lst)*1.96/np.sqrt(len(lst)))


    fig, ax = plt.subplots()
    if plot_ind:    
        plt.scatter(bpos, tps, color='r', s=2.2)
        plt.scatter(bpos, fps, color='b', s=2.1)
        plt.scatter(bpos, fns, color='g', s=2.)
    else:
        ll = np.arange(0, max(bpos)+1)
        ax.errorbar(ll, m_tps, yerr=err_tps, fmt='-o', color='r', markersize=3)
        ax.errorbar(ll, m_fps, yerr=err_fps, fmt='-o', color='b', markersize=3)
        ax.errorbar(ll, m_fns, yerr=err_fns, fmt='-o', color='g', markersize=3)

    plt.legend(['tp', 'fp', 'fn'], loc="upper right")
    plt.xlabel("Bit position")
    plt.ylabel("Avg difference in tp, fp, fn")

    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)
    plt.show()



def plot_lay_bpos_tpfpfn(dct_sdc, flts_del, sv_name, injection_type):

    # # Filter only by those where tp, fp or fn changes ------------------------
    tps_orig = [n["orig"]["tp"] for n in dct_sdc]
    fps_orig = [n["orig"]["fp"] for n in dct_sdc]
    fns_orig = [n["orig"]["fn"] for n in dct_sdc]

    tps_corr = [n["corr"]["tp"] for n in dct_sdc]
    fps_corr = [n["corr"]["fp"] for n in dct_sdc]
    fns_corr = [n["corr"]["fn"] for n in dct_sdc]

    tps = np.array(tps_corr) - np.array(tps_orig)
    fps = np.array(fps_corr) - np.array(fps_orig)
    fns = np.array(fns_corr) - np.array(fns_orig)


    mask_any_change = np.logical_or(np.logical_or(fns != 0, tps != 0), fps != 0)
    # mask_any_change = get_mask_any_change(dct_sdc)
    tps = tps[mask_any_change]
    fps = fps[mask_any_change]
    fns = fns[mask_any_change]
    faults = flts_del[:, mask_any_change]

    bpos = faults[6,:]
    if injection_type == "neurons":
        lays = faults[1,:]
    elif injection_type == "weights":
        lays = faults[0,:]

    if len(bpos) == 0 or len(lays) == 0:
        print('No information for plotting available.')
        return

    tps_new = np.empty((32,max(lays)+1))
    tps_new[:] = np.nan
    fps_new = np.empty((32,max(lays)+1))
    fps_new[:] = np.nan
    fns_new = np.empty((32,max(lays)+1))
    fns_new[:] = np.nan

    for x in range(0, tps_new.shape[0]):
        for y in range(tps_new.shape[1]):

            sel = np.logical_and(bpos == x, lays==y)
            if sel.any():
                # if y > 100:
                #     x= 0
                # print('bit', x, 'lay', y, np.sum(bpos == x), " ", np.sum(lays == y), " ", np.sum(sel), len(tps[sel]), len(fps[sel]), len(fns[sel]))
                tps_new[x,y] = np.mean(tps[sel]) #avg of diff in tp diff
                fps_new[x,y] = np.mean(fps[sel]) #avg of diff in fp diff
                fns_new[x,y] = np.mean(fns[sel]) #avg of diff in fn diff

    mn, mx = min([min(tps), min(fps), min(fns)]), max([max(tps), max(fps), max(fns)])

    fig, ax = plt.subplots(3,1)
    # a = np.random.random((16, 16))
    p1 = ax[0].imshow(tps_new, vmin=mn, vmax=mx, cmap='hot', interpolation='nearest')
    ax[0].set_title("tp")
    fig.colorbar(p1, ax=ax[0])

    p2 = ax[1].imshow(fps_new, vmin=mn, vmax=mx, cmap='hot', interpolation='nearest')
    ax[1].set_title("fp")
    fig.colorbar(p2, ax=ax[1])

    p3 = ax[2].imshow(fns_new, vmin=mn, vmax=mx, cmap='hot', interpolation='nearest')
    ax[2].set_title("fn")
    fig.colorbar(p3, ax=ax[2])

    # plt.legend(['tp', 'fp', 'fn'], loc="upper right")
    plt.xlabel("Layers")
    plt.ylabel("Bits")

    plt.savefig(sv_name, dpi=300)
    print('saved as ', sv_name)
    plt.show()




    # fig, ax = plt.subplots()
    # if plot_ind:    
    #     plt.scatter(lays, tps, color='r', s=2.2)
    #     plt.scatter(lays, fps, color='b', s=2.1)
    #     plt.scatter(lays, fns, color='g', s=2.)
    # else:
    #     ll = np.arange(0, max(lays)+1)
    #     ax.errorbar(ll, m_tps, yerr=err_tps, fmt='-o', color='r', markersize=3)
    #     ax.errorbar(ll, m_fps, yerr=err_fps, fmt='-o', color='b', markersize=3)
    #     ax.errorbar(ll, m_fns, yerr=err_fns, fmt='-o', color='g', markersize=3)

    
def plot_sdc_diff_vs_bpos(sv_name, orig_sdc, corr_sdc, bpos, bits):
    fig, ax = plt.subplots(1,1)

    orig_av = []
    orig_err = []
    corr_av = []
    corr_err = []
    for b in np.arange(0,bits+1):
        orig_av.append(np.mean(np.array(orig_sdc)[np.array(bpos)==b]))
        orig_err.append(np.std(np.array(orig_sdc)[np.array(bpos)==b])*1.96/np.sqrt(len(np.array(bpos)==b)))
        corr_av.append(np.mean(np.array(corr_sdc)[np.array(bpos)==b]))
        corr_err.append(np.std(np.array(corr_sdc)[np.array(bpos)==b])*1.96/np.sqrt(len(np.array(bpos)==b)))


    ms = np.array(corr_av) - np.array(orig_av)
    errms = np.sqrt(np.array(orig_err)**2 + np.array(corr_err)**2)
    ax.plot(np.arange(0,bits+1), ms)
    ax.errorbar(np.arange(0,bits+1), ms, yerr=errms, fmt='o', markersize=2)
    
    ax.set_xlabel('bit position')
    ax.set_ylabel('SDC rate corr - orig')
    ax.set_ylim((0., 1))

    fig.tight_layout()
    
    plt.savefig(sv_name, dpi=300)
    print('saved as', sv_name)
    fig.show()


def plot_sdc_diff_vs_lay(sv_name, model_name, orig_sdc, corr_sdc, lays):
        if model_name == "retina":
            nr_layers = 71
        elif model_name == "yolov3":
            nr_layers = 75
        else:
            nr_layers = 71
        fig, ax = plt.subplots(1,1)


        orig_av = []
        orig_err = []
        corr_av = []
        corr_err = []
        for b in np.arange(0,nr_layers+1):
            orig_av.append(np.mean(np.array(orig_sdc)[np.array(lays)==b]))
            orig_err.append(np.std(np.array(orig_sdc)[np.array(lays)==b])*1.96/np.sqrt(len(np.array(lays)==b)))
            corr_av.append(np.mean(np.array(corr_sdc)[np.array(lays)==b]))
            corr_err.append(np.std(np.array(corr_sdc)[np.array(lays)==b])*1.96/np.sqrt(len(np.array(lays)==b)))


        ms = np.array(corr_av) - np.array(orig_av)
        errms = np.sqrt(np.array(orig_err)**2 + np.array(corr_err)**2)
        ax.plot(np.arange(0,nr_layers+1), ms)
        ax.errorbar(np.arange(0,nr_layers+1), ms, yerr=errms, fmt='o', markersize=2)

        ax.set_xlabel('layer')
        ax.set_ylabel('SDC rate corr - orig')
        ax.set_ylim((0., 1.1))
        fig.tight_layout()

        
        plt.savefig(sv_name, dpi=300)
        print('saved as', sv_name)
        fig.show()


def split_in_pieces_of_n(llist, nr_samples):
    # Get numbers for due, sdc in epoch-wise pieces 
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    return list(chunks(llist, nr_samples))


def filter_with_sublists(llist, mask):
    return [np.array(llist[u])[mask[u]].tolist() for u in range(len(llist))]


def evaluation(folder_path, save_name, iou_thresh=0.5, filter_ooI=True, eval_mode="iou+class_labels", suffix=''):
    # iou_thresh = 0.5
    # filter_ooI = True #drop bounding boxes that are out of the image

    # # Extract map, ap50 results + plot ------------------------------------------
    _, _, _, _, _, _, _, _, nr_epochs, nan_infs, ranger_nan_infs = get_map_ap50_infnan(folder_path)

    # # Extract + save fp, fn, tp data to json -----------------------------------------------
    if eval_mode == "iou":
        suffix = suffix + "_iou"

    # ------------------------------------------------------ #
    # EVAL ------------------------------------------------- #
    eval_experiment(nr_epochs, iou_thresh, folder_path, save_name, None, [nan_infs, ranger_nan_infs], filter_ooI, eval_mode=eval_mode) #Note: Saves file with relevant results to save_name
    # -------------------------------------------------------#
    # -------------------------------------------------------#
    print('saved as', save_name)





def extract_sdc_due(json_path, flt_type, folder_path, faults_path, save_as_name, typ='no_ranger'):

    # # Load + analyse data ----------------------------------------------------------------------------------
    # Takes care of images that are not annotated

    orig_map, orig_ap50, orig_ranger_map, orig_ranger_ap50, corr_map, corr_ap50, ranger_corr_map, ranger_corr_ap50, nr_epochs, _, _ = get_map_ap50_infnan(folder_path)
    if flt_type == "weights":
        # Weights: --------------------------------
        print('Loaded: ', json_path)
        # res_all = load_json_indiv(json_path) #individual results
        # orig_sdc, corr_sdc, resil_orig_sdc, resil_corr_sdc, orig_due, corr_due, resil_orig_due, resil_corr_due, flts_all, bpos, lays, dct_all = load_weights(json_path, faults_path) #sdc (averaged per epoch already for weights)
        orig_sdc, corr_sdc, resil_orig_sdc, resil_corr_sdc, orig_due, corr_due, resil_orig_due, resil_corr_due, flts_all, bpos, lays, dct_all = load_neurons(json_path, faults_path)
    elif flt_type == "neurons":
        # # Neurons: --------------------------------------
        print('Loaded: ', json_path)
        # orig_sdc, corr_sdc, resil_orig_sdc, resil_corr_sdc, flts_del, bpos, lays, dct_sdc = load_neurons(json_path, faults_path)
        orig_sdc, corr_sdc, resil_orig_sdc, resil_corr_sdc, orig_due, corr_due, resil_orig_due, resil_corr_due, flts_all, bpos, lays, dct_all = load_neurons(json_path, faults_path)


    if typ == 'ranger':
        orig_map = orig_ranger_map
        orig_ap50 = orig_ranger_ap50
        corr_map = ranger_corr_map
        corr_ap50 = ranger_corr_ap50
        orig_sdc = resil_orig_sdc
        corr_sdc = resil_corr_sdc
        orig_due = resil_orig_due
        corr_due = resil_corr_due


    # Basic analysis  + save -------------------------------------------------------------------------------------------
    # counting due, sdc numbers
    nr_all = len(dct_all)
    nr_samples = int(nr_all/nr_epochs)
    
    flts_all = flts_all[:,:nr_samples*nr_epochs] #reduce fault number if needed
    bpos = bpos[:nr_samples*nr_epochs]
    lays = lays[:nr_samples*nr_epochs]
    

    # Extend faults (only needed for per_epoch mode!)
    # if flt_type == "weights":
    #     print('Extending faults...')
    #     ext_array = np.zeros((7, nr_all)).astype(int)
    #     for ep in range(nr_epochs): #go through all epochs
    #         for n in range(nr_samples):
    #             # ext_array = np.append(ext_array, flts_all[:,ep], 1)
    #             ext_array[:,ep*nr_samples+n] = flts_all[:,ep]
    #             # count up epochs

    #     flts_all = ext_array
    #     bpos = ext_array[6,:].astype(int).tolist()
    #     lays = ext_array[0,:].astype(int).tolist() #for weights layer is 0, for neurons 1
    

    mask_nan_inf, mask_nan_inf_flat, mask_any_change_and_no_naninf, mask_any_change_and_no_naninf_flat = get_nan_inf_masks(corr_due, dct_all, nr_samples, nr_all, typ)
    nr_imgs_due = [int(np.sum(x)) for x in corr_due]
    nr_imgs_sdc = [int(np.sum(x)) for x in mask_any_change_and_no_naninf]



    # DUE: -------------------------------------------
    dct_due_filt = np.array(dct_all)[mask_nan_inf_flat].tolist() #only those from list that have due
    flts_due_filt = flts_all[:, mask_nan_inf_flat]
    bpos_due_filt = np.array(bpos)[mask_nan_inf_flat].tolist()
    lays_due_filt = np.array(lays)[mask_nan_inf_flat].tolist()

    if any(orig_due):
        orig_due_filt = filter_with_sublists(orig_due, mask_nan_inf)
    else:
        orig_due_filt = orig_due
    corr_due_filt = filter_with_sublists(corr_due, mask_nan_inf)

    img_ids_due_filt = [u["id"] for u in dct_due_filt]


    # SDC: -------------------------------------------
    dct_sdc_filt2 = np.array(dct_all)[mask_any_change_and_no_naninf_flat].tolist()
    flts_sdc_filt2 = flts_all[:, mask_any_change_and_no_naninf_flat]
    bpos_sdc_filt2 = np.array(bpos)[mask_any_change_and_no_naninf_flat].tolist()
    lays_sdc_filt2 = np.array(lays)[mask_any_change_and_no_naninf_flat].tolist()

    orig_sdc_filt2 = filter_with_sublists(orig_sdc, mask_any_change_and_no_naninf)
    corr_sdc_filt2 = filter_with_sublists(corr_sdc, mask_any_change_and_no_naninf)
    
    img_ids_sdc_filt2 = [u["id"] for u in dct_sdc_filt2]
    # BS > 1 inf nan order not working? fixed: nan inf tags the whole batch currently so also other bits seem to get flagged.


    # save filtered data for plotting:
    tpfpfn_sdc_orig = [n ["orig"] for n in dct_sdc_filt2]
    tpfpfn_sdc_corr= [n ["corr"] for n in dct_sdc_filt2]
    tpfpfn_due_orig = [n ["orig"] for n in dct_due_filt]
    tpfpfn_due_corr = [n ["corr"] for n in dct_due_filt]
    metrics = {"sdc": {"orig_sdc": orig_sdc_filt2, "corr_sdc": corr_sdc_filt2, "tpfpfn_orig": tpfpfn_sdc_orig, "tpfpfn_corr": tpfpfn_sdc_corr}, "due": {"orig_due": orig_due_filt, "corr_due": corr_due_filt, "tpfpfn_orig": tpfpfn_due_orig, "tpfpfn_corr": tpfpfn_due_corr}, "map": {"orig_map": orig_map, "corr_map": corr_map}, "ap50": {"orig_ap50": orig_ap50, "corr_ap50": corr_ap50}}
    ddct = {"flts_sdc": flts_sdc_filt2.tolist(), "flts_due": flts_due_filt.tolist(), "img_ids_sdc": img_ids_sdc_filt2, "img_ids_due": img_ids_due_filt, 'nr_img_all': [nr_samples for n in range(nr_epochs)], 'nr_img_sdc': nr_imgs_sdc, "nr_img_due": nr_imgs_due, 'metrics': metrics}
    # json_file = model_name + "_" + dataset_name + "_" + "results_1_" + flt_type + "_images" + suffix + ".json"

    with open(save_as_name, "w") as outfile: 
        json.dump(ddct, outfile)
    print('saved:', save_as_name)

    return ddct



def get_nan_inf_masks(corr_due, dct_all, nr_samples, nr_all, typ='no_ranger'):
    # Filter by naninfs and sdc:
    mask_no_nan_inf = [np.logical_not(np.array(n)) for n in corr_due]
    mask_no_nan_inf_flat = np.logical_not(np.array(flatten_list(corr_due)))

    mask_nan_inf = [np.logical_not(x) for x in mask_no_nan_inf]
    mask_nan_inf_flat = np.logical_not(mask_no_nan_inf_flat)
    print('Ratio of images affected by due: ', np.sum(mask_nan_inf_flat)/nr_all, np.sum(mask_nan_inf_flat), nr_all)

    mask_any_change_raw_flat = get_mask_any_change_orig_corr(dct_all, typ) #true if there are changes in tp, fp, fn between orig and sdc
    mask_any_change_raw = split_in_pieces_of_n(mask_any_change_raw_flat, nr_samples)

    mask_any_change_and_no_naninf_flat = np.logical_and(mask_no_nan_inf_flat, mask_any_change_raw_flat)
    mask_any_change_and_no_naninf = split_in_pieces_of_n(mask_any_change_and_no_naninf_flat, nr_samples)

    print('Ratio of images affected by sdc: ', np.sum(mask_any_change_and_no_naninf_flat)/nr_all, np.sum(mask_any_change_and_no_naninf_flat), nr_all)


    return mask_nan_inf, mask_nan_inf_flat, mask_any_change_and_no_naninf, mask_any_change_and_no_naninf_flat



def get_fault_path(folder_path):
    filelist = list(Path(folder_path).glob('**/*updated_rs_fault_locs.bin') )
    # 'ranger' not in

    for n in range(len(filelist)):
        if 'inj_ranger' in str(filelist[n]):
            return filelist[n]
            
    return filelist[0]

    
    



def main(argv):
    
    # # Evaluation --------------------------------------------------------------------------------------------------------------------------------
    # # Part I: extraction -----

    # # Neurons -----------------------------------
    # # Neurons (yolo, coco, no resil, Y, extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220212-013556_1_faults_[0_32]_bits/coco2017/val'
    # faults_path = get_fault_path(folder_path)
    # flt_type = "neurons"
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "yolov3" #yolov3 #retina
    # dataset_name = "coco2017" #coco2017
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"



    # # Neurons (yolo, kitti, no resil, Y, extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220213-134550_1_faults_[0_32]_bits/kitti/val'
    # faults_path = get_fault_path(folder_path)
    # flt_type = "neurons"
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "yolov3" #yolov3 #retina
    # dataset_name = "kitti"
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"


    # # # (retina, coco, no resil, conf=0.05, Y, extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/retina_50_trials/neurons_injs/per_image/objDet_20220113-184245_1_faults_[0_32]_bits/coco2017/val'
    # faults_path = get_fault_path(folder_path)
    # flt_type = "neurons" #"weights"
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "retina" #yolov3 #retina
    # dataset_name = "coco2017" #coco2017
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"


    # # # (detectron2, coco, no resil neurons, Y, extr extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/det2_fasterRCNN_50_trials_Conv/neuron_injs/per_image/objDet_20220214-002924_1_faults_[0, 31]_bits/coco2017/val'
    # faults_path = get_fault_path(folder_path)
    # flt_type = "neurons" #"weights"
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "detectron" #yolov3 #retina
    # dataset_name = "coco2017" 
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"



    # # (detectron2, kitti, no resil neurons, Y, extr, extr, extr, final)
    folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/det2_fasterRCNN_50_trials_Conv/neuron_injs/per_image/objDet_20220215-1501248_1_faults_[0, 31]_bits/kitti/val'
    faults_path = get_fault_path(folder_path)
    flt_type = "neurons" #"weights"
    suffix = '_none' #'_avg' # '_ranger'
    model_name = "detectron" #yolov3 #retina
    dataset_name = "kitti" 
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"
    


    # # # (yolov3u, lyft, no resil neurons, double checked, SiLU, Y, extr new, final)
    # # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_50_trials/neurons_injs/per_image/objDet_20220116-201033_1_faults_[0_32]_bits/lyft/val' #silu
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_50_trials/neurons_injs/per_image/objDet_20220206-165457_1_faults_[0_32]_bits/lyft/val'
    # faults_path = get_fault_path(folder_path)
    # flt_type = "neurons" #"weights"
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "yolov3u_silu" #yolov3 #retina
    # dataset_name = "lyft" 
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"



    # # Weights -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # # (yolo, coco, extr, extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/weights_injs/per_batch/objDet_20220214-095032_1_faults_[0_32]_bits/coco2017/val'
    # faults_path = get_fault_path(folder_path)
    # flt_type = "weights" #"weights", 'neurons
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "yolov3" #yolov3 #retina #detectron
    # dataset_name = "coco2017" 
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"


    # # # (yolo, kitti, extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/weights_injs/per_batch/objDet_20220213-202846_1_faults_[0_32]_bits/kitti/val'
    # faults_path = get_fault_path(folder_path)
    # flt_type = "weights" #"weights", 'neurons
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "yolov3" #yolov3 #retina #detectron
    # dataset_name = "kitti"
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"


    # # # (retina, coco, weights Y, extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/retina_50_trials/weights_injs/per_batch/objDet_20220113-235113_1_faults_[0_32]_bits/coco2017/val'
    # faults_path = get_fault_path(folder_path)
    # flt_type = "weights" #"weights", 'neurons
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "retina" #yolov3 #retina #detectron
    # dataset_name = "coco2017" 
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"


    # # # (detectron2, coco, no resil weights, Y, extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/det2_fasterRCNN_50_trials_Conv/weights_injs/objDet_20220214-003013_1_faults_[0, 31]_bits/coco2017/val'
    # faults_path = faults_path = get_fault_path(folder_path)
    # flt_type = "weights" #"weights"
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "detectron" #yolov3 #retina
    # dataset_name = "coco2017" 
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"



    # # (detectron2, kitti, no resil neurons, Y, extr, extr, extr, final)
    folder_path = '/home/qutub/PhD/git_repos/intel_git_repos/pfd_uet/result_files/result_files_paper/retina_50_trials/weights_injs/per_batch/objDet_20211208-194836_1_faults_[0_32]_bits/coco2017/val'
    faults_path = get_fault_path(folder_path)
    flt_type = "weights" #"weights"
    suffix = '_none' #'_avg' # '_ranger'
    model_name = "detectron" #yolov3 #retina
    dataset_name = "kitti" 
    bits = 32
    eval_mode = "iou+class_labels" # "iou+class_labels", "iou"



    # # (yolov3u, lyft, no resil weights, double checked, SiLU, Y, extr, extr, extr, final)
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_50_trials/weights_injs/per_batch/objDet_20220206-024503_1_faults_[0_32]_bits/lyft/val' #silu
    # faults_path = get_fault_path(folder_path)
    # flt_type = "weights" #"weights"
    # suffix = '_none' #'_avg' # '_ranger'
    # model_name = "yolov3u_silu" #yolov3 #retina
    # dataset_name = "lyft" 
    # bits = 32
    # eval_mode = "iou+class_labels" # "iou+class_labels", "iou"




    # Evaluation --------------
    save_name = model_name + "_" + dataset_name + "_" + "results_1_" + flt_type + "_backup" + suffix + ".json"
    evaluation(folder_path, save_name)


    save_as_name = model_name + "_" + dataset_name + "_" + "results_1_" + flt_type + "images" + suffix + ".json"
    json_file = extract_sdc_due(save_name, flt_type, folder_path, faults_path, save_as_name)

    # ddct = load_json_indiv(json_file)
    # ddct2 = load_json_indiv(save_name)

    print()


    


if __name__ == "__main__":
    main(sys.argv)
