import pickle
import os, sys
from pathlib import Path
import json
import numpy as np
from alficore.evaluation.sdc_plots.obj_det_analysis import evaluation, extract_sdc_due, get_fault_path, get_map_ap50_infnan, read_csv
from alficore.evaluation.sdc_plots.obj_det_evaluate_jsons import read_fault_file, load_json_indiv
from copy import deepcopy
import glob

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
    folder = '/home/fgeissle/ranger_repo/ranger/result_files/mmal_10_trials/neurons_injs/per_image/objDet_20220413-185913_1_faults_[0,8]_bits/fgvc/val'
    
    #######################################################


    fault_file = get_fault_path(folder, 'ranger')
    fault_file_noranger = get_fault_path(folder, 'no_ranger')
    ff = read_fault_file(fault_file) #tests
    ff_noranger = read_fault_file(fault_file_noranger) #tests
    

    det_path = list(Path(folder).glob('**/*_detections.bin') ) #dets without faults?
    dets = read_fault_file(str(det_path[0])) #tests #orig ranger, should be zero with proper bounds
    print('nr of ranger detections', 'no faults', np.sum(np.array(dets)>0))
    print('nr of ranger detections', 'faults', 'ranger applied', np.sum(ff[10,:]>0), '; no ranger applied', np.sum(ff_noranger[10,:]>0), '(should be the same)')

    # --------------------------------------------------------------------
    orig_path = list(Path(folder).glob('**/*_golden.csv') ) 
    orig_res = read_csv(str(orig_path[0]))

    corr_path = list(Path(folder).glob('**/*_corr.csv') ) 
    corr_res = read_csv(str(corr_path[0]))

    top_nr = 1


    # Extract full data --------------------------------------------------
    # orig
    gnd, orig, orig_resil = get_gnd_orig(orig_res)
    corr = get_col_nr_x(corr_res, 2)
    corr = [list(map(int,corr[x][1:-1].split())) for x in range(1,len(corr))]

    # corr
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

    gnd = np.array(gnd)[mask_orig_correct].tolist()
    orig = np.array(orig)[mask_orig_correct].tolist()
    orig_resil = np.array(orig_resil)[mask_orig_correct].tolist()
    corr = np.array(corr)[mask_orig_correct].tolist()
    corr_resil = np.array(corr_resil)[mask_orig_correct].tolist()
    due_corr = np.array(due_corr)[mask_orig_correct].tolist()
    due_corr_resil = np.array(due_corr_resil)[mask_orig_correct].tolist()

    ff = np.array([ff.T[row] for row in range(len(mask_orig_correct)) if mask_orig_correct[row]]).T
    ff_noranger = np.array([ff_noranger.T[row] for row in range(len(mask_orig_correct)) if mask_orig_correct[row]]).T
    dets = np.array(dets)[mask_orig_correct].tolist()

    print('nr of ranger detections (after filtering incorrect pred)', 'faults', 'ranger applied', np.sum(ff[10,:]>0), '; no ranger applied', np.sum(ff_noranger[10,:]>0), '(should be the same)')
    print('due rate', np.sum(due_corr)/len(due_corr), np.sum(due_corr), len(due_corr))
    print('due resil rate', np.sum(due_corr_resil)/len(due_corr_resil), np.sum(due_corr_resil), len(due_corr_resil))


    # filter out due, sdc --------------------------------------------------------
    # No Ranger
    corr_no_due = np.array(corr)[np.logical_not(due_corr)]
    orig_no_due = np.array(orig)[np.logical_not(due_corr)]

    mask_sdc = corr_no_due[:,0] != orig_no_due[:,0]
    print('result sdc rate corr-orig:', np.sum(mask_sdc), len(corr_no_due), np.sum(mask_sdc)/len(corr_no_due))
    
    ff_no_due = np.array([ff_noranger.T[row] for row in range(len(due_corr)) if np.logical_not(due_corr)[row]]).T
    dets_no_due = np.array(dets)[np.logical_not(due_corr)].tolist()
    ff_sdc = np.array([ff_no_due.T[row] for row in range(len(mask_sdc)) if mask_sdc[row]]).T
    dets_sdc = np.array(dets_no_due)[mask_sdc].tolist()
    print('nr of ranger detections', 'faults', 'no ranger applied', 'no due', np.sum(ff_no_due[10,:]>0) if len(ff_no_due > 0) else 0, '(should be <= resil as there it can prevent DUE and have those additional diff cases for ranger det)')
    print('nr of ranger detections', 'faults', 'no ranger applied', 'sdc', np.sum(ff_sdc[10,:]>0) if len(ff_sdc > 0) else 0)

    # Ranger
    corr_resil_no_due = np.array(corr_resil)[np.logical_not(due_corr_resil)]
    orig_resil_no_due = np.array(orig_resil)[np.logical_not(due_corr_resil)]

    mask_sdc_resil = corr_resil_no_due[:,0] != orig_resil_no_due[:,0]
    print('result sdc rate resil corr-orig:', np.sum(mask_sdc_resil), len(corr_resil_no_due), np.sum(mask_sdc_resil)/len(corr_resil_no_due))

    ff_no_due_resil = np.array([ff.T[row] for row in range(len(due_corr_resil)) if np.logical_not(due_corr_resil)[row]]).T
    dets_no_due_resil = np.array(dets)[np.logical_not(due_corr_resil)].tolist()
    ff_sdc_resil = np.array([ff_no_due_resil.T[row] for row in range(len(mask_sdc_resil)) if mask_sdc_resil[row]]).T
    dets_sdc_resil = np.array(dets_no_due_resil)[mask_sdc_resil].tolist()
    print('nr of ranger detections', 'faults', 'ranger applied', 'no due', np.sum(ff_no_due_resil[10,:]>0) if len(ff_no_due_resil > 0) else 0)
    print('nr of ranger detections', 'faults', 'ranger applied', 'sdc', np.sum(ff_sdc_resil[10,:]>0) if len(ff_sdc_resil > 0) else 0)

    print()
    # TODO: why so many ranger dets??


    # # Analyze all data for fp, fn, tp ------------------------------------------------------
    # save_name = folder + "/results.json"
    # evaluation(folder, save_name)



    # Extract and evaluate only SDC, DUE cases ------------------------------------------------------
    # flt_type = 'neurons'

    # new_save_name = folder + "/results_sdc_due.json"
    # dct_noranger = extract_sdc_due(save_name, flt_type, folder, fault_file, new_save_name, typ='no_ranger') 

    # if os.path.isdir(folder + '/ranger_model/'):
    #     dct_ranger = extract_sdc_due(save_name, flt_type, folder, fault_file, ranger_string_add(new_save_name), typ='ranger')



    # # Summarize overall SDC, DUE rates etc. ------------------------------------------------------
    # toplot_dict_template = {'sdc': {'mns': [], 'errs': []}, \
    #     'due': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    #     'sdc_wgt': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': [], 'mns_diff': [], 'errs_diff': []}, \
    #     'due_wgt': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    #     'ap50': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    #     'map': {'mns_orig': [], 'errs_orig': [], 'mns_corr': [], 'errs_corr': []}, \
    #     'tpfpfn': {'orig': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': []}, 'corr': {'tp': [], 'fp': [], 'fp_bbox': [], 'fp_class':[], 'fp_bbox_class':[], 'fn': []}, 'bpos': []}}

    # toplot_dict_noranger = add_data(deepcopy(toplot_dict_template), new_save_name)
    # print('sdc no ranger', toplot_dict_noranger['sdc']['mns'], toplot_dict_noranger['sdc']['errs'])

    # if os.path.isdir(folder + '/ranger_model/'):
    #     toplot_dict_ranger = add_data(deepcopy(toplot_dict_template), ranger_string_add(new_save_name))
    #     print('sdc ranger', toplot_dict_ranger['sdc']['mns'], toplot_dict_ranger['sdc']['errs'])

    # print()




# # Extract faults: -----------------

# fault_file = get_fault_path(folder)
# ff = read_fault_file(fault_file) #get fault file with ranger detections
# # len(ff[6,:])

# # get predictions: --------------------
# orig_path = folder + '/orig_model/epochs/0/coco_instances_results_0_epoch.json'
# orig = load_json_indiv(orig_path)
# corr_path = folder + '/corr_model/epochs/0/coco_instances_results_0_epoch.json'
# corr = load_json_indiv(corr_path)

# get quantiles ----------------
# filelist = list(Path(folder).glob('**/*quantiles.json') )
# quant_dict = load_json_indiv(str(filelist[0]))

# # # get aps and nan inf: --------------
# # orig_ap, orig_ap50, orig_ranger_ap, orig_ranger_ap50, corr_ap, corr_ap50, ranger_corr_ap, ranger_corr_ap50, nr_epochs, corr_nan_inf_flags, ranger_corr_nan_inf_flags = get_map_ap50_infnan(folder) #not batch-sensitive

# # # get ranger detections --------
# # dets = read_fault_file(list(Path(folder).glob('**/*ranger_detections.bin'))[0])
# # print()


if __name__ == "__main__":
    main(sys.argv)