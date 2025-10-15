import sys
from pathlib import Path
import numpy as np
from alficore.evaluation.sdc_plots.obj_det_analysis import get_fault_path, read_csv
from alficore.evaluation.sdc_plots.obj_det_evaluate_jsons import read_fault_file

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

def img_class_eval(exp_folder_path):

    fault_file, orig_result, corr_result = extract_data(exp_folder_path)

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