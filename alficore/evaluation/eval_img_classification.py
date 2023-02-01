import sys
from pathlib import Path
import numpy as np
sys.path.append("/home/fgeissle/fgeissle_ranger")
from alficore.evaluation.sdc_plots.obj_det_analysis import get_fault_path, read_csv
from alficore.evaluation.sdc_plots.obj_det_evaluate_jsons import read_fault_file, load_json_indiv


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
        # if corr_res[u][y-1] == 'als':
        #     print()
        ret.append(corr_res[u][y])
    return ret

def get_gnd_orig(orig_res):
    gnd = np.array(orig_res)[:,1]
    gnd = [int(gnd[x]) for x in range(1,len(gnd))]

    orig = np.array(orig_res)[:,2]
    orig = [list(map(int,orig[x][1:-1].split())) for x in range(1,len(orig))]
    
    if 'resil' in np.array(orig_res)[:,3][0]:
        orig_resil = np.array(orig_res)[:,3]
        orig_resil = [list(map(int,orig_resil[x][1:-1].split())) for x in range(1,len(orig_resil))]
    else:
        orig_resil = None

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


def extract_data(folder):
    # Fault file ----------------------
    fault_file_noranger = get_fault_path(folder)
    if fault_file_noranger is None or fault_file_noranger == []:
        print('No fault file found:')
        ff_noranger = None
    else:
        print('Loading from fault file:', fault_file_noranger)
        ff_noranger = read_fault_file(fault_file_noranger) #tests
    
    # Ranger detection file ---------------
    # det_path = list(Path(folder).glob('**/*_detections.bin') ) #dets without faults?
    # if det_path != []:
    #     dets = read_fault_file(str(det_path[0])) #tests #orig ranger, should be zero with proper bounds
    #     print('nr of ranger detections', 'no faults', np.sum(np.array(dets)>0)/len(dets), np.sum(np.array(dets)>0), len(dets))
    # # print('nr of ranger detections', 'faults', 'ranger applied', np.sum(ff[10,:]>0), '; no ranger applied', np.sum(ff_noranger[10,:]>0), '(should be the same)')

    # orig and corr csv files --------------
    orig_path = list(Path(folder).glob('**/*_golden.csv') ) 
    orig_res = read_csv(str(orig_path[0]))

    corr_path = list(Path(folder).glob('**/*_corr.csv') )
    if len(corr_path) > 0:
        corr_res = read_csv(str(corr_path[0]))
    else:
        corr_res = None

    return ff_noranger, orig_res, corr_res

def extract_predictions(orig_res, corr_res):
    
    # orig ----------------------
    gnd, orig, orig_resil = get_gnd_orig(orig_res)
    corr, corr_resil, fpths = None, None, None

    if corr_res is not None:
        # Extract fault paths --------------------------------------------------
        assert get_col_nr_x(orig_res, 0) == get_col_nr_x(corr_res, 0)
        fpths = get_col_nr_x(corr_res, 0)
        fpths = [int(fpths[x]) for x in range(1,len(fpths))]


        # corr ---------------------
        corr_ind = corr_res[0].index('corr output index - top5')
        corr = get_col_nr_x(corr_res, corr_ind)
        corr = [list(map(int,corr[x][1:-1].split())) for x in range(1,len(corr))]

        try:
            corr_resil_ind = corr_res[0].index('resil corr output index - top5')
            corr_resil = get_col_nr_x(corr_res, corr_resil_ind)
            corr_resil = np.array([corr_resil[x]=='True' for x in range(1,len(corr_resil))])
        except:
            print('resil corr output column not found')
            corr_resil = None

    return gnd, orig, orig_resil, corr, corr_resil, fpths


def extract_predictions_prob(orig_res, corr_res):
    
    # orig ---------------------
    orig_ind = orig_res[0].index('orig output prob - top5')
    orig = get_col_nr_x(orig_res, orig_ind)
    orig = [list(map(float,orig[x][1:-1].split())) for x in range(1,len(orig))]

    if corr_res is not None:
        # corr ---------------------
        corr_ind = corr_res[0].index('corr output prob - top5')
        corr = get_col_nr_x(corr_res, corr_ind)
        corr = [list(map(float,corr[x][1:-1].split())) for x in range(1,len(corr))]
    else:
        corr = []

    return orig, corr



def get_due_masks(corr_res):
    # due
    try:
        due_corr_ind = corr_res[0].index('nan_or_inf_flag_corr_model')
        due_corr= get_col_nr_x(corr_res, due_corr_ind)
        due_corr = np.array([due_corr[x]=='True' for x in range(1,len(due_corr))])
    except:
        due_corr = None

    try:
        due_corr_resil_ind = corr_res[0].index('nan_or_inf_flag_corr_resil_model')
        due_corr_resil = get_col_nr_x(corr_res, due_corr_resil_ind)
        due_corr_resil = np.array([due_corr_resil[x]=='True' for x in range(1,len(due_corr_resil))])
    except:
        due_corr_resil = None

    return due_corr, due_corr_resil



def main(argv):

    #######################################################
    # folder = '/home/fgeissle/hdfit_pytorch/pytorch/personal.squtub.pytorchalfi/result_files/LeNet_orig_1_trials/_injs/per_batch/objDet_20221020-100725_0_faults_[0,8]_bits/mnist/val'
    # folder = '/home/fgeissle/hdfit_pytorch/pytorch/personal.squtub.pytorchalfi/result_files/LeNet_orig_2_trials/_injs/per_batch/objDet_20221014-124446_0_faults_[0,8]_bits/mnist/val'
    # folder = '/home/fgeissle/hdfit_pytorch/pytorch/personal.squtub.pytorchalfi/result_files/output_imagenet/imagenet/val'
    # folder = '/home/fgeissle/hdfit_pytorch/pytorch/personal.squtub.pytorchalfi/result_files/output_imagenet/imagenet/val'
    folder = 'result_files/lenet_1_trials/neurons_injs/per_image/objDet_20221116-145213_1_faults_[0,31]_bits/mnist/val'
    #######################################################



    fault_file, orig_result, corr_result = extract_data(folder)

    gnd, orig, orig_resil, corr, corr_resil, fpths = extract_predictions(orig_result, corr_result)



    # Accuracy orig vs gnd, corr vs gnd -------------------------------------
    top_nr = 1

    mask_orig_correct = [orig[n][:top_nr][0] == gnd[n] for n in range(len(gnd))]
    print('Accuracy orig:', np.sum(mask_orig_correct)/len(mask_orig_correct)*100, ' %', '(', np.sum(mask_orig_correct), ' out of ', len(mask_orig_correct), ')')
    mask_corr_correct = [corr[n][:top_nr][0] == gnd[n] for n in range(len(gnd))]
    print('Accuracy corr:', np.sum(mask_corr_correct)/len(mask_corr_correct)*100, ' %', '(', np.sum(mask_corr_correct), ' out of ', len(mask_corr_correct), ')')

    # DUE rate -----------------------------------------------------
    due_corr_mask, due_corr_resil_mask = get_due_masks(corr_result)
    print('DUE rate:', np.sum(due_corr_mask)/len(due_corr_mask)*100, '%', '(', np.sum(due_corr_mask), ' out of ', len(due_corr_mask), ')')


    # Filter by due
    gnd, orig, corr, orig_resil, corr_resil, mask_orig_correct, mask_corr_correct, _, fpths = filter_by_mask_all(gnd, orig, corr, orig_resil, corr_resil, mask_orig_correct, mask_corr_correct, None, fpths, np.logical_not(due_corr_mask))
    fault_file = np.array([fault_file.T[row] for row in range(len(np.logical_not(due_corr_mask))) if np.logical_not(due_corr_mask)[row]]).T
    print('filtering out due events:', np.sum(due_corr_mask))
    
    # Filter by correct
    gnd, orig, corr, orig_resil, corr_resil, _, _, _, fpths = filter_by_mask_all(gnd, orig, corr, orig_resil, corr_resil, None, None, None, fpths, mask_orig_correct)
    fault_file = np.array([fault_file.T[row] for row in range(len(mask_orig_correct)) if mask_orig_correct[row]]).T
    print('filtering out incorrect orig preds:', np.sum(np.logical_not(due_corr_mask))-len(gnd))


    # filter out sdc --------------------------------------------------------
    if len(corr) > 0:
        mask_sdc = np.array(corr)[:,0] != np.array(orig)[:,0]
        print('SDC rate:', np.sum(mask_sdc)/len(mask_sdc)*100, '%', '(', np.sum(mask_sdc), ' out of ', len(mask_sdc), ')')
        gnd, orig, corr, orig_resil, corr_resil, _, _, _, fpths = filter_by_mask_all(gnd, orig, corr, orig_resil, corr_resil, None, None, None, fpths, mask_sdc)
        fault_file = np.array([fault_file.T[row] for row in range(len(mask_sdc)) if mask_sdc[row]]).T
        print('Filter out non-SDC events:', np.sum(mask_sdc))
    else:
        mask_sdc = np.array([])
        print('SDC rate:', np.sum(mask_sdc), ' %', '(', np.sum(mask_sdc), ' out of ', len(mask_sdc), ')')
    
    


if __name__ == "__main__":
    main(sys.argv)
