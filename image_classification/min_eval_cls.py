import os, sys
from pathlib import Path
import json
from turtle import position
import numpy as np
from alficore.evaluation.sdc_plots.obj_det_analysis import evaluation, extract_sdc_due, get_fault_path, get_map_ap50_infnan, read_csv
from alficore.evaluation.sdc_plots.obj_det_evaluate_jsons import read_fault_file, load_json_indiv
import matplotlib.pyplot as plt
import imageio as iio
from colorama import init, Fore, Back, Style
from colorama import init as colorama_init
from termcolor import colored


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

def get_col_nr_x(corr_res, y):
    ret = []
    for u in range(len(corr_res)):
        if corr_res[u][y] == 'als':
            print()
        ret.append(corr_res[u][y])
    return ret

def get_gnd_orig(orig_res):
    gnd = np.array(orig_res)[:,1]
    gnd = [int(gnd[x]) for x in range(1,len(gnd))]

    orig = np.array(orig_res)[:,2]
    orig = [list(map(int,orig[x][1:-1].split())) for x in range(1,len(orig))]
    
    orig_resil = np.array(orig_res)[:,3]
    orig_resil = [list(map(int,orig_resil[x][1:-1].split())) for x in range(1,len(orig_resil))]

    return gnd, orig, orig_resil

def get_label_info(filename):
    f = open(filename, "r")
    bounds = []
    if f.mode == 'r':
        contents = f.read().splitlines()
        bounds = [u.split(',') for u in contents]
    f.close()
    return bounds

def get_class_mapping():
    info = get_label_info('/home/fgeissle/aircrafts/MMAL-Net/datasets/FGVC-aircraft/data/images_variant_test.txt')
    info_fam = get_label_info('/home/fgeissle/aircrafts/MMAL-Net/datasets/FGVC-aircraft/data/images_manufacturer_test.txt')
    info_scores = get_label_info('/home/fgeissle/aircrafts/MMAL-Net/datasets/FGVC-aircraft/data/test.txt')

    scs = []
    nms = []
    for i in range(len(info)):
        var = info[i][0][8:]
        man = info_fam[i][0][8:]
        aircraft_name = (man + " " + var, info[i][0][:7])[0]

        score = int(info_scores[i][0][12:])
        if score not in scs:
            scs.append(score)
            nms.append(aircraft_name)

    scs = list(np.array(scs) -1)
    dct = {scs[i]: nms[i] for i in range(len(scs))}

    return dct

def filter_by_mask(ls, mask):
    return np.array(ls)[mask].tolist()

def filter_by_mask_all(gnd, orig, corr, orig_resil, corr_resil, due_corr, due_corr_resil, dets, fpths, mask):
    if gnd is not None:
        gnd = filter_by_mask(gnd, mask)
    if orig is not None: 
        orig = filter_by_mask(orig, mask)
    if orig_resil is not None:
        orig_resil = filter_by_mask(orig_resil, mask)
    if corr is not None:
        corr = filter_by_mask(corr, mask)
    if corr_resil is not None:
        corr_resil = filter_by_mask(corr_resil, mask)
    if due_corr is not None:
        due_corr = filter_by_mask(due_corr, mask)
    if due_corr_resil is not None:
        due_corr_resil = filter_by_mask(due_corr_resil, mask)
    if dets is not None:
        dets = filter_by_mask(dets, mask)
    if fpths is not None:
        fpths = filter_by_mask(fpths, mask)

    return gnd, orig, corr, orig_resil, corr_resil, due_corr, due_corr_resil, dets, fpths


def main(argv):


    #######################################################
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_1_trials/neurons_injs/per_image/objDet_20220404-171008_1_faults_[0,8]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/alexnet_2_trials/neurons_injs/per_image/objDet_20220401-114406_1_faults_[0,8]_bits/imagenet/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_1_trials/neurons_injs/per_image/objDet_20220404-190123_1_faults_[1,3]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_1_trials/neurons_injs/per_image/objDet_20220405-113601_1_faults_[0,8]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_1_trials/neurons_injs/per_image/objDet_20220405-144527_1_faults_[0,8]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_2_trials/neurons_injs/per_image/objDet_20220405-151028_1_faults_[0,8]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/alexnet_2_trials/neurons_injs/per_image/objDet_20220405-175606_1_faults_[0,8]_bits/imagenet/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_1_trials/neurons_injs/per_image/objDet_20220406-111134_1_faults_[0,8]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/alexnet_2_trials/neurons_injs/per_image/objDet_20220406-115506_1_faults_[0,8]_bits/imagenet/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_2_trials/neurons_injs/per_image/objDet_20220412-174913_1_faults_[0,8]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_10_trials/neurons_injs/per_image/objDet_20220413-185913_1_faults_[0,8]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_100_trials/neurons_injs/per_image/objDet_20220414-165700_1_faults_[0_32]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_100_trials/neurons_injs/per_image/objDet_20220427-155605_1_faults_[0_32]_bits/fgvc/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_100_trials/neurons_injs/per_image/objDet_20220428-112538_1_faults_[0_32]_bits/fgvc/val' #neurons
    folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_100_trials/weights_injs/per_batch/objDet_20220428-131317_1_faults_[0_32]_bits/fgvc/val' #weights

    #######################################################


    fault_file = get_fault_path(folder, 'ranger')
    fault_file_noranger = get_fault_path(folder, 'no_ranger')
    ff = read_fault_file(fault_file) #tests
    ff_noranger = read_fault_file(fault_file_noranger) #tests
    

    det_path = list(Path(folder).glob('**/*_detections.bin') ) #dets without faults?
    dets = read_fault_file(str(det_path[0])) #tests #orig ranger, should be zero with proper bounds
    print('nr of ranger detections', 'no faults', np.sum(np.array(dets)>0)/len(dets), np.sum(np.array(dets)>0), len(dets))
    print('nr of ranger detections', 'faults', 'ranger applied', np.sum(ff[10,:]>0), '; no ranger applied', np.sum(ff_noranger[10,:]>0), '(should be the same)')

    # --------------------------------------------------------------------
    orig_path = list(Path(folder).glob('**/*_golden.csv') ) 
    orig_res = read_csv(str(orig_path[0]))

    corr_path = list(Path(folder).glob('**/*_corr.csv') ) 
    corr_res = read_csv(str(corr_path[0]))

    top_nr = 1


    # Extract full data --------------------------------------------------
    assert get_col_nr_x(orig_res, 0) == get_col_nr_x(corr_res, 0)
    fpths = get_col_nr_x(corr_res, 0)
    fpths = [int(fpths[x]) for x in range(1,len(fpths))]

    # orig
    gnd, orig, orig_resil = get_gnd_orig(orig_res)

    # corr
    corr = get_col_nr_x(corr_res, 2)
    corr = [list(map(int,corr[x][1:-1].split())) for x in range(1,len(corr))]

    corr_resil = get_col_nr_x(corr_res, 3)
    corr_resil = [list(map(int,corr_resil[x][1:-1].split())) for x in range(1,len(corr_resil))]

    # due
    due_corr= get_col_nr_x(corr_res, 16)
    due_corr = np.array([due_corr[x]=='True' for x in range(1,len(due_corr))])

    due_corr_resil = get_col_nr_x(corr_res, 19)
    due_corr_resil = np.array([due_corr_resil[x]=='True' for x in range(1,len(due_corr_resil))])


    # Filter by data that does correct top N predictions --------------------------------------------
    mask_orig_correct = [orig[n][:top_nr][0] == gnd[n] for n in range(len(gnd))]
    print('filtered out incorrect orig preds:', len(orig)-np.sum(mask_orig_correct))

    gnd, orig, corr, orig_resil, corr_resil, due_corr, due_corr_resil, dets, fpths = filter_by_mask_all(gnd, orig, corr, orig_resil, corr_resil, due_corr, due_corr_resil, dets, fpths, mask_orig_correct)

    ff = np.array([ff.T[row] for row in range(len(mask_orig_correct)) if mask_orig_correct[row]]).T
    ff_noranger = np.array([ff_noranger.T[row] for row in range(len(mask_orig_correct)) if mask_orig_correct[row]]).T
    print('nr of ranger detections (after filtering incorrect pred)', 'faults', 'ranger applied', np.sum(ff[10,:]>0), '; no ranger applied', np.sum(ff_noranger[10,:]>0), '(should be the same)')
    print('due rate', np.sum(due_corr)/len(due_corr), np.sum(due_corr), len(due_corr))
    print('due resil rate', np.sum(due_corr_resil)/len(due_corr_resil), np.sum(due_corr_resil), len(due_corr_resil))


    # filter out due, sdc --------------------------------------------------------
    # No Ranger
    gnd_no_due, orig_no_due, corr_no_due, orig_resil_forsdc_no_due, corr_resil_forsdc_no_due, _, _, dets_no_due, fpths_no_due = filter_by_mask_all(gnd, orig, corr, orig_resil, corr_resil, None, None, dets, fpths, np.logical_not(due_corr))

    mask_sdc = np.array(corr_no_due)[:,0] != np.array(orig_no_due)[:,0]
    print('result sdc rate corr-orig:', np.sum(mask_sdc), len(corr_no_due), np.sum(mask_sdc)/len(corr_no_due))
    
    gnd_sdc, orig_sdc, corr_sdc, orig_resil_forsdc, corr_resil_forsdc, _, _, dets_sdc, fpths_sdc = filter_by_mask_all(gnd_no_due, orig_no_due, corr_no_due, orig_resil_forsdc_no_due, corr_resil_forsdc_no_due, None, None, dets_no_due, fpths_no_due, mask_sdc)


    ff_no_due = np.array([ff_noranger.T[row] for row in range(len(due_corr)) if np.logical_not(due_corr)[row]]).T
    ff_sdc = np.array([ff_no_due.T[row] for row in range(len(mask_sdc)) if mask_sdc[row]]).T

    print('nr of ranger detections', 'faults', 'no ranger applied', 'no due', np.sum(ff_no_due[10,:]>0) if len(ff_no_due > 0) else 0, '(should be <= resil as there it can prevent DUE and have those additional diff cases for ranger det)')
    print('nr of ranger detections', 'faults', 'no ranger applied', 'sdc', np.sum(ff_sdc[10,:]>0) if len(ff_sdc > 0) else 0)


    # Ranger
    _, orig_resil_no_due, corr_resil_no_due, _, _, _, _, dets_no_due_resil, fpths_no_due_resil = filter_by_mask_all(None, orig_resil, corr_resil, None, None, None, None, dets, fpths, np.logical_not(due_corr_resil))
    # corr_resil_no_due = np.array(corr_resil)[np.logical_not(due_corr_resil)]
    # orig_resil_no_due = np.array(orig_resil)[np.logical_not(due_corr_resil)]

    mask_sdc_resil = np.array(corr_resil_no_due)[:,0] != np.array(orig_resil_no_due)[:,0]
    print('result sdc rate resil corr-orig:', np.sum(mask_sdc_resil), len(corr_resil_no_due), np.sum(mask_sdc_resil)/len(corr_resil_no_due))

    _, orig_sdc_resil, corr_sdc_resil, _, _, _, _, dets_sdc_resil, fpths_sdc_resil = filter_by_mask_all(None, orig_resil_no_due, corr_resil_no_due, None, None, None, None, dets_no_due_resil, fpths_no_due_resil, mask_sdc_resil)

    ff_no_due_resil = np.array([ff.T[row] for row in range(len(due_corr_resil)) if np.logical_not(due_corr_resil)[row]]).T
    ff_sdc_resil = np.array([ff_no_due_resil.T[row] for row in range(len(mask_sdc_resil)) if mask_sdc_resil[row]]).T


    print('nr of ranger detections', 'faults', 'ranger applied', 'no due', np.sum(ff_no_due_resil[10,:]>0) if len(ff_no_due_resil > 0) else 0)
    print('nr of ranger detections', 'faults', 'ranger applied', 'sdc', np.sum(ff_sdc_resil[10,:]>0) if len(ff_sdc_resil > 0) else 0)


    # FP rates: -------------------------------------------
    nf_ff_no_due_det = np.sum(ff_no_due[-1,:]>0)
    nr_ff_no_due = ff_no_due.shape[1]
    print('overall detection rate (TP+FP)', nf_ff_no_due_det/nr_ff_no_due, nf_ff_no_due_det, nr_ff_no_due)

    nr_ff_sdc_det = np.sum(ff_sdc[-1,:]>0)
    nr_ff_sdc = ff_sdc.shape[1]
    print('sdc detection rate (TP)', nr_ff_sdc_det/nr_ff_sdc, nr_ff_sdc_det, nr_ff_sdc)

    print('FP detection rate (FP)', (nf_ff_no_due_det - nr_ff_sdc_det)/nr_ff_no_due, (nf_ff_no_due_det - nr_ff_sdc_det), nr_ff_no_due)


    # # Visualization ----------------------------------------

    # # orig_resil_forsdc = np.array(orig_resil_no_due)[mask_sdc].tolist()
    # # corr_resil_forsdc = np.array(corr_resil_no_due)[mask_sdc].tolist()

    # fp_file = list(Path(folder).glob('**/*_fp.csv') ) 
    # paths = read_csv(str(fp_file[0]))[1:]

    # for n in range(len(fpths_sdc)):
    #     # n = 0
    #     assert int(paths[fpths_sdc[n]][0]) == fpths_sdc[n]
    #     map_dct = get_class_mapping()

    #     gnd_pred = map_dct[gnd_sdc[n]]
    #     orig_pred = map_dct[orig_sdc[n][0]]
    #     corr_pred = map_dct[corr_sdc[n][0]]
    #     orig_resil_pred = map_dct[orig_resil_forsdc[n][0]]
    #     corr_resil_pred = map_dct[corr_resil_forsdc[n][0]]

        
    #     # read an image
    #     img = iio.imread(paths[fpths_sdc[n]][1])

    #     fig, ax = plt.subplots(1,1, figsize = (10,8))
    #     plt.imshow(img)



    #     # colorama_init(autoreset=True)
    #     # colors = [x for x in dir(Fore) if x[0] != "_"]
    #     # colors = [i for i in colors if i not in ["BLACK", "RESET"] and "LIGHT" not in i] 

    #     # if gnd_pred == orig_pred:
    #     #     color_orig = colors[2].lower() #Green
    #     # else:
    #     #     color_orig = colors[4].lower() #Red

    #     # if corr_pred == orig_pred:
    #     #     color_corr = colors[2].lower() #Green
    #     # else:
    #     #     color_corr = colors[4].lower() #Red

    #     # if orig_resil_pred == orig_pred:
    #     #     color_orig_resil = colors[2].lower() #Green
    #     # else:
    #     #     color_orig_resil = colors[4].lower() #Red

    #     # if corr_resil_pred == orig_pred:
    #     #     color_corr_resil = colors[2].lower() #Green
    #     # else:
    #     #     color_corr_resil = colors[4].lower() #Red

    #     # lbl = colored('ORIG=' + str(orig_pred), color_orig) + ',\n' + colored('CORR=' + str(corr_pred), color_corr) + ',\n' + colored('ORIG_resil=' + str(orig_resil_pred), color_orig_resil) + ',\n' + colored('CORR_resil=' + str(corr_resil_pred), color_corr_resil)
    #     # ax.set_title(lbl, fontsize=10)

    #     ax.set_title('GT=' + str(gnd_pred) + ',\n ORIG=' + str(orig_pred) + ',\n CORR=' + str(corr_pred) + ',\n ORIG_resil=' + str(orig_resil_pred) + ',\n CORR_resil=' + str(corr_resil_pred), fontsize=10)
    #     plt.axis('off')
    #     if not os.path.exists(folder + '/vis/images_sdc/'):
    #         os.makedirs(os.path.dirname(folder + '/vis/images_sdc/'))

    #     img_name = folder + '/vis/images_sdc/' + "im_" + str(n) + ".png"
    #     fig.savefig(img_name, dpi=300)
    #     print('save img ', n)


if __name__ == "__main__":
    main(sys.argv)