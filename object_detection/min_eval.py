import pickle
import os, sys
from pathlib import Path
import json
import numpy as np
from obj_det_analysis7 import evaluation, extract_sdc_due, get_fault_path, get_map_ap50_infnan, read_csv
from obj_det_evaluate_jsons import read_fault_file, load_json_indiv
# from obj_det_plot_metrics_all_models4 import get_m_err
from copy import deepcopy


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



def main(argv):
    # min_eval.py
    # analys_images_videos.py
    # obj_det_test_fmap_plotauto_no_rep.py (2x for ranger and no ranger)

    #######################################################
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_500_trials/neurons_injs/per_image/objDet_20220322-152949_1_faults_[0_32]_bits/robo/val'
    # folder = "/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220330-201223_1_faults_[1]_bits/robo/val"
    # '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_3_trials/neurons_injs/per_image/objDet_20220322-001237_1_faults_[0,8]_bits/robo/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220331-125720_1_faults_[1,4]_bits/coco2017/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220331-143056_1_faults_[1,4]_bits/robo/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220331-151035_1_faults_[1,4]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_100_trials/neurons_injs/per_image/objDet_20220407-110545_1_faults_[0,8]_bits/robo/val' #robo
    # # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_10_trials/neurons_injs/per_epoch/objDet_20220408-161123_1_faults_[0_32]_bits/ppp/val' #ppp
    # # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_3_trials/neurons_injs/per_epoch/objDet_20220408-163312_1_faults_[0,8]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_3_trials/neurons_injs/per_epoch/objDet_20220411-124828_1_faults_[0,8]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_3_trials/weights_injs/per_epoch/objDet_20220411-125320_1_faults_[0,8]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_3_trials/neurons_injs/per_epoch/objDet_20220411-165454_1_faults_[0_32]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_3_trials/neurons_injs/per_epoch/objDet_20220408-235125_1_faults_[0,8]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_1000_trials/weights_injs/per_epoch/objDet_20220411-191405_1_faults_[0_32]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_1000_trials/neurons_injs/per_epoch/objDet_20220412-160634_1_faults_[0_32]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220419-151228_1_faults_[0,8]_bits/coco2017/val' #test platform faults
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220421-173738_1_faults_[31]_bits/coco2017/val' #image corr set
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220502-190646_1_faults_[0,8]_bits/coco2017/val' #test
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220502-190646_1_faults_[0,8]_bits/coco2017/val' #test2
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220505-111814_1_faults_[0_32]_bits/coco2017/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220513-101024_1_faults_[31]_bits/coco2017/val' #gaussian noise test

    folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220509-195335_1_faults_[0_32]_bits/coco2017/val' #platform fault set neurons
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/weights_injs/per_batch/objDet_20220511-174708_1_faults_[0_32]_bits/coco2017/val' #platform faults sets weights
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220509-163146_1_faults_[31]_bits/coco2017/val' #corrupted images blur
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220510-115056_1_faults_[31]_bits/coco2017/val' #corrupted images noise (gaussian)

    folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220519-103625_1_faults_[0_32]_bits/coco2017/val'

    flt_type = 'neurons' # neurons, weights
    #######################################################

    fault_file = get_fault_path(folder, 'ranger')
    fault_file_noranger = get_fault_path(folder, 'no_ranger')
    ff = read_fault_file(fault_file) #tests
    ff_noranger = read_fault_file(fault_file_noranger) #tests
    # print('bit distribution' [np.sum(ff[6,:] == n) for n in range(32)])
    assert np.sum(ff_noranger[-1,:]>0) == np.sum(ff[-1,:]>0), 'Different number of bnd violations for ranger and no ranger?' #binary detection should be the same


    # ff_orig = read_fault_file(folder + '/yolov3_ultra_test_random_sbf_neurons_inj_1_3_1bs_ppp_fault_locs.bin')

    # Analyze all data for fp, fn, tp ------------------------------------------------------
    save_name = folder + "/results.json"
    evaluation(folder, save_name)


    # Extract and evaluate only SDC, DUE cases ------------------------------------------------------
    new_save_name = folder + "/results_sdc_due.json"
    dct_noranger = extract_sdc_due(save_name, flt_type, folder, fault_file_noranger, new_save_name, typ='no_ranger') 

    if os.path.isdir(folder + '/ranger_model/'):
        dct_ranger = extract_sdc_due(save_name, flt_type, folder, fault_file, ranger_string_add(new_save_name), typ='ranger')



    # Summarize overall SDC, DUE rates etc. ------------------------------------------------------
    toplot_dict_template = {'sdc': {'mns': [], 'errs': []}, \
        'due': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
        'sdc_wgt': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': [], 'mns_diff': [], 'errs_diff': []}, \
        'due_wgt': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
        'ap50': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
        'map': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
        'tpfpfn': {'orig': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': []}, 'corr': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': []}, 'bpos': []}}

    toplot_dict_noranger = add_data(deepcopy(toplot_dict_template), new_save_name)
    print('sdc no ranger', toplot_dict_noranger['sdc']['mns'], toplot_dict_noranger['sdc']['errs'])

    if os.path.isdir(folder + '/ranger_model/'):
        toplot_dict_ranger = add_data(deepcopy(toplot_dict_template), ranger_string_add(new_save_name))
        print('sdc ranger', toplot_dict_ranger['sdc']['mns'], toplot_dict_ranger['sdc']['errs'])

    print()


    # # FP rates: -------------------------------------------
    # nf_ff_no_due_det = np.sum(ff_no_due[-1,:]>0)
    # nr_ff_no_due = ff_no_due.shape[1]
    # print('overall detection rate (TP+FP)', nf_ff_no_due_det/nr_ff_no_due, nf_ff_no_due_det, nr_ff_no_due)

    # nr_ff_sdc_det = np.sum(ff_sdc[-1,:]>0)
    # nr_ff_sdc = ff_sdc.shape[1]
    # print('sdc detection rate (TP)', nr_ff_sdc_det/nr_ff_sdc, nr_ff_sdc_det, nr_ff_sdc)

    # print('FP detection rate (FP)', (nf_ff_no_due_det - nr_ff_sdc_det)/nr_ff_no_due, (nf_ff_no_due_det - nr_ff_sdc_det), nr_ff_no_due)





if __name__ == "__main__":
    main(sys.argv)