# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import json
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from alficore.dataloader import coco_loader

# from visualization_florian import *
from tqdm import tqdm
import pickle
from copy import deepcopy
import os

def read_fault_file(file):
        file = open(file, 'rb')
        return pickle.load(file)


def load_json_indiv(gt_path):
    with open(gt_path) as f:
        coco_gt = json.load(f)
        f.close()
    return coco_gt

def flatten_list(list):
    return [element for sublist in list for element in sublist]


def load_neurons(json_path, faults_path):
    """
    No average since neuron fault different for each image here (assumption).
    Due is list of 0 or 1 depending on whether nan or inf was found during this inference.
    """
    res_all = load_json_indiv(json_path)
    # print('check', len(res_all["gt"][0]))
    orig_sdc, corr_sdc, resil_orig_sdc, resil_corr_sdc, orig_due, corr_due, resil_orig_due, resil_corr_due, dict_sdc = get_sdc_all(res_all)

    # compare to: res_all["faults"]
    flts = read_fault_file(faults_path)
    bpos = [flts[:,n][6] for n in range(flts.shape[1])] #
    lays = [flts[:,n][1] for n in range(flts.shape[1])] #index 0 is batch index, 1 is layer! 

    
    # Correct unannotated images in faults: -------------------
    unann_img_ind_ext = res_all["unannotated_images"]

    nr_samples = len(bpos) #before foul images are eliminated
    nr_epochs = len(res_all["gt"])
    used_upto = nr_epochs*nr_samples 

    bpos = bpos[:used_upto]
    lays = lays[:used_upto]
    flts_del = flts
    if unann_img_ind_ext:
        shift = 0
        for i in unann_img_ind_ext:
            bpos.pop(i-shift) #pops index i
            lays.pop(i-shift)
            flts_del = np.delete(flts_del, i-shift, 1) #object, axis
            shift += 1

    return orig_sdc, corr_sdc, resil_orig_sdc, resil_corr_sdc, orig_due, corr_due, resil_orig_due, resil_corr_due, flts_del, bpos, lays, dict_sdc




def group_by_image_id(coco_labels, coco_orig, coco_corr):
    """
    Splits list of result dicts into sublits that belong to the same image_id.
    """

    img_ids_gt = [n["image_id"] for n in coco_labels]
    img_ids_orig = [n["image_id"] for n in coco_orig]
    img_ids_corr = [n["image_id"] for n in coco_corr]

    coco_orig_grouped = []
    coco_corr_grouped = []
    coco_labels_grouped = []
    for x in list(np.sort(list(set(img_ids_gt)))):

        #orig 
        coco_orig_1img = []
        for u in range(len(img_ids_orig)):
            if img_ids_orig[u] == x:
                coco_orig_1img.append(coco_orig[u])
        coco_orig_grouped.append(coco_orig_1img)

        #corr 
        coco_corr_1img = []
        for u in range(len(img_ids_corr)):
            if img_ids_corr[u] == x:
                coco_corr_1img.append(coco_corr[u])
        coco_corr_grouped.append(coco_corr_1img)

        #lbls 
        coco_labels_1img = []
        for u in range(len(img_ids_gt)):
            if img_ids_gt[u] == x:
                coco_labels_1img.append(coco_labels[u])
        coco_labels_grouped.append(coco_labels_1img)

    return coco_labels_grouped, coco_orig_grouped, coco_corr_grouped



def group_by_image_id_indiv(coco_labels, img_ids_ref):
    """
    Splits list of result dicts into sublits that belong to the same image_id.
    """

    img_ids_gt = [n["image_id"] for n in coco_labels] #possible that some ids do not show up
    # img_ids_sorted = list(np.sort(list(set(img_ids_gt))))

    coco_labels_grouped = []

    for x in img_ids_ref:
        coco_labels_1img = []
        for u in range(len(img_ids_gt)):
            if img_ids_gt[u] == x:
                coco_labels_1img.append(coco_labels[u])
        coco_labels_grouped.append(coco_labels_1img)

    return coco_labels_grouped


def get_iou(bb1_dict, bb2_dict):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters:
    New: Input is a dict of form {'image_id': 0, 'category_id': 2, 'bbox': [...], 'score': 1, 'bbox_mode': 0}, including bbox format.
    bb1, bb2 will be converted in the process to x1,x2,y1,y2, where:
    -The (x1, y1) position is at the top left corner,
    -the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float in [0, 1]
    """
    # As default, choose 1 if data comes from json, 0 if it comes from network prediction directly
    if "bbox_mode" in bb1_dict.keys():
        bm1 = bb1_dict["bbox_mode"]
    else:
        bm1 = 1
    if "bbox_mode" in bb2_dict.keys():
        bm2 = bb2_dict["bbox_mode"]
    else:
        bm2 = 1
    # bm1, bm2 = bb1_dict["bbox_mode"], bb2_dict["bbox_mode"] #0 is XYXY_ABS, 1 is XYWH_ABS
    bb1, bb2 = bb1_dict["bbox"], bb2_dict["bbox"]

    # Incoming boxes are (x1, y1, x2, y2) when coming from network prediction directly
    # Incoming boxes are xywh when coming from json files or annos!
    # Transform to right format for iou calculation (xxyy):
    if bm1 == 0: #xyxy -> xxyy
        bb1 = {'x1': bb1[0], 'x2': bb1[2], 'y1': bb1[1] , 'y2': bb1[3]}
    elif bm1 == 1: #xywh -> xxyy
        bb1 = {'x1': bb1[0], 'x2': bb1[0] + bb1[2], 'y1': bb1[1] , 'y2': bb1[1] + bb1[3]}

    if bm2 == 0: #xyxy -> xxyy
        bb2 = {'x1': bb2[0], 'x2': bb2[2], 'y1': bb2[1] , 'y2': bb2[3]}
    elif bm2 == 1: #xywh -> xxyy
        bb2 = {'x1': bb2[0], 'x2': bb2[0] + bb2[2], 'y1': bb2[1] , 'y2': bb2[1] + bb2[3]}

    if True in np.isnan(list(bb1.values())) or True in np.isnan(list(bb2.values())):
        return 0.0

    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if float(bb1_area + bb2_area - intersection_area)>0:
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    else:
        iou = 0.
    # print('check', iou, intersection_area, bb1_area, bb2_area, x_right, x_left, y_bottom, y_top, bb1, bb2)

    if np.isnan(iou):
        return 0.0
    assert iou >= 0.0
    assert iou <= 1.0
    return iou




def assign_hungarian_trial(source, target, iou_thresh, **kwargs):
    """
    source, target are lists of detected-object dictionaries.
    """
    
    cost = np.zeros((len(source), len(target))) #rows: source, cols: targets
    cost_iou = np.zeros((len(source), len(target)))
    cost_cls = np.zeros((len(source), len(target)))
    for n in range(len(source)):
        for m in range(len(target)):
            iou = get_iou(source[n], target[m])
            cost[n][m] = iou if (source[n]["category_id"]==target[m]["category_id"]) else 0 
            cost_iou[n][m] = iou
            cost_cls[n][m] = 1 if (source[n]["category_id"]==target[m]["category_id"]) else 0 #1 if classes match otherwise 0

    cost = cost*(-1) #invert because large iou is better. If class does not match or iou < thresh then cost is 0 already.
    row_ind,col_ind=linear_sum_assignment(cost)
    cost = cost*(-1) #invert again to make costs positive (=iou)
    # print('costs', cost[row_ind,col_ind])# Extract the element where the optimal assigned column index of each row index is located to form an array

    all_rows = np.arange(cost.shape[0])
    all_cols = np.arange(cost.shape[1])
    fp_bbox = 0
    fp_class = 0
    fp_bbox_class = len(list(set(all_rows) - set(row_ind))) #fps where there is no gt match

    # All possible rows and columns have been assigned: Some of them can be invalid and should be eliminated (iou still too low - can happen if no better choice available): 
    assert len(row_ind) == min(len(target), len(source)), len(col_ind) == min(len(target), len(source))
    elim = cost[row_ind, col_ind] < iou_thresh #elimination based on combined iou + cls criterion
    wrong_class = cost_cls[row_ind, col_ind] == 0 #are elements to be eliminated of the wrong classes?
    wrong_iou = cost_iou[row_ind, col_ind] < iou_thresh #are elements to be eliminated based on bbox iou only?
    if elim.any():
        row_ind = row_ind[np.logical_not(elim)] #cancel invalid ones
        col_ind = col_ind[np.logical_not(elim)]
        fp_bbox += np.sum(np.logical_and(np.logical_not(wrong_class), wrong_iou))
        fp_class += np.sum(np.logical_and(wrong_class, np.logical_not(wrong_iou)))
        fp_bbox_class += np.sum(np.logical_and(wrong_class, wrong_iou))

    # Derive unassigned rows and cols
    row_ind_unassigned = list(set(all_rows) - set(row_ind))
    col_ind_unassigned = list(set(all_cols) - set(col_ind))

    fp_flags = {"bbox_mismatch": int(fp_bbox), "class_mismatch": int(fp_class), "bbox_class_mismatch": int(fp_bbox_class)}
    iou_list = cost[row_ind,col_ind] #list of chosen iou pairings, in order of source list
    return row_ind, col_ind, row_ind_unassigned, col_ind_unassigned, fp_flags, iou_list



def assign_hungarian(source, target, iou_thresh, **kwargs):
    """
    source, target are lists of detected-object dictionaries.
    """
    metadata = kwargs.get("metadata", None)
    check_class_labels= kwargs.get("check_class_labels", False)
    check_class_groups= kwargs.get("check_class_groups", False)

    cost = np.zeros((len(source), len(target))) #rows: source, cols: targets
    for n in range(len(source)):
        for m in range(len(target)):
            cost[n][m] = get_iou(source[n], target[m]) 

    off_values = (cost < iou_thresh) #necessary?
    if off_values.any():
        cost[off_values] = 0 #np.Inf but linear_sum_assignment cant handle inf
    
    cost = cost*(-1) #invert because large iou is better

    row_ind,col_ind=linear_sum_assignment(cost)

    cost = cost*(-1) #invert again to make costs positive (=iou)

    all_rows = np.arange(cost.shape[0])
    all_cols = np.arange(cost.shape[1])
    fp_bbox = len(list(set(all_rows) - set(row_ind)))
    fp_class = 0
    fp_bbox_class = 0

    # Eliminate invalid ones (iou too low): 
    elim = cost[row_ind, col_ind] < iou_thresh
    if elim.any():
        # print('to eliminated2', elim, origin_indices[elim], target_indices[elim])
        row_ind = row_ind[np.logical_not(elim)]
        col_ind = col_ind[np.logical_not(elim)]
        fp_bbox += np.sum(elim)
        for row, col in zip(row_ind, col_ind):
            if source[row]['category_id'] != target[col]['category_id']:
                fp_bbox_class += 1

    if check_class_labels == True:
        for row, col in zip(row_ind, col_ind):
            if source[row]['category_id'] != target[col]['category_id']:
                print('Category id mismatch!!!')
                print(source[row]['category_id'])
                print(target[col]['category_id'])
                row_ind = np.delete(row_ind, np.where(row_ind==row))
                col_ind = np.delete(col_ind, np.where(col_ind==col))
                fp_class += 1

    elif check_class_groups == True:
        class_groups = kwargs.get("class_groups", [])
        for row, col in zip(row_ind, col_ind):
            for group in class_groups:
                if metadata.thing_classes[source[row]['category_id']] in group:
                    if metadata.thing_classes[target[col]['category_id']] not in group:
                        print('Category class mismatch!!!')
                        print(source[row]['category_id'])
                        print(target[col]['category_id'])
                        row_ind = np.delete(row_ind, np.where(row_ind==row))
                        col_ind = np.delete(col_ind, np.where(col_ind==col))
                        break

    # Derive unassigned rows and cols
    row_ind_unassigned = list(set(all_rows) - set(row_ind))
    col_ind_unassigned = list(set(all_cols) - set(col_ind))

    fp_flags = {"bbox_mismatch": int(fp_bbox), "class_mismatch": fp_class, "bbox_class_mismatch": fp_bbox_class}
    iou_list = cost[row_ind,col_ind] #list of chosen iou pairings, in order of source list
    return row_ind, col_ind, row_ind_unassigned, col_ind_unassigned, fp_flags, iou_list



def eval_image(coco_labels_grouped_n, coco_pred_grouped_n, iou_thresh,  metadata, mode="iou+class_labels"):

    dict_template = {'id': 0, 'tp': [], 'fp': [], 'fn': [], 'prec': None, 'rec': None, 'fp_bbox': 0, 'fp_class': 0, 'fp_bbox_class':0, 'iou': iou_thresh, 'iou_list': []}

    # If no gt found in that image -------------------------------------
    if coco_labels_grouped_n == []: 
        print("Warning, no labels found for this image?")
        dict_empty = dict_template
        dict_empty["fp"] = np.array(coco_pred_grouped_n).tolist() #make all preds fps, if exist
        return dict_empty
    else:
        id = coco_labels_grouped_n[0]["image_id"] #image index
        dict_template["id"] = id

    # If no predictions found in that image -------------------------------------
    if coco_pred_grouped_n == []: 
        dict_empty = dict_template
        dict_empty["fn"] = np.array(coco_labels_grouped_n).tolist()
        # dict_empty["prec"] = None
        return dict_empty



    # Match ground truth and predictions: -----------------------------------------
    if mode == 'iou':
        # Match bounding boxes by IoU:
        row, col, row_un, col_un, fp_flags, iou_list = assign_hungarian(source=coco_pred_grouped_n, target=coco_labels_grouped_n, iou_thresh=iou_thresh, metadata=metadata)
        # print(row, col, row_un, col_un)
    elif mode == 'iou+class_labels': 
        # print('Consider Bounding Boxes + Class Labels:')
        row, col, row_un, col_un, fp_flags, iou_list = assign_hungarian_trial(source=coco_pred_grouped_n, target=coco_labels_grouped_n, iou_thresh=iou_thresh, check_class_labels=True, metadata=metadata)
    else:
        print('mode', mode, 'not valid.')
        return

    # elif mode == 'iou+class_groups': 
    #     # print('Consider Bounding Boxes + Class Groups (vehicles, human, misc):')
    #     row, col, row_un, col_un = assign_hungarian(source=coco_pred_grouped_n, target=coco_labels_grouped_n, iou_thresh=iou_thresh, check_class_groups=True, class_groups=class_groups)

    # Calculate tp, fp, fn, prec and recall for that image
    tp = np.array(coco_pred_grouped_n)[row].tolist()
    tp_gt = np.array(coco_labels_grouped_n)[col].tolist()
    fp = np.array(coco_pred_grouped_n)[row_un].tolist()
    assert len(fp) == fp_flags['bbox_class_mismatch']+fp_flags['class_mismatch']+fp_flags['bbox_mismatch']
    fn = np.array(coco_labels_grouped_n)[col_un].tolist()

    if len(tp) + len(fp) > 0:
        prec = len(tp)/(len(tp) + len(fp))
    else:
        prec = None
    if len(tp) + len(fn) > 0:
        rec = len(tp)/(len(tp) + len(fn))
    else:
        rec = None

    dict_real = dict_template
    dict_real["tp"] = tp
    dict_real["fp"] = fp
    dict_real["fn"] = fn
    dict_real["prec"] = prec
    dict_real["rec"] = rec
    dict_real['fp_bbox'] = fp_flags['bbox_mismatch']
    dict_real['fp_class'] = fp_flags['class_mismatch']
    dict_real['fp_bbox_class'] = fp_flags['bbox_class_mismatch']
    dict_real['iou_list'] = iou_list.tolist()

    return dict_real #return dict with tp, fp, fn, reason for fp



def load_all_jsons(folder_path, epoch_nr, typ='ranger'):
    ground_truth_json_file_path = folder_path + '/coco_format.json'
    detection_json_file_path = folder_path + '/orig_model/epochs/0/coco_instances_results_0_epoch.json'
    corr_detection_json_file_path = folder_path + '/corr_model/epochs/' + str(epoch_nr) + '/coco_instances_results_' + str(epoch_nr) + '_epoch.json'

    coco_gt = load_json_indiv(ground_truth_json_file_path)
    coco_orig = load_json_indiv(detection_json_file_path)
    coco_corr = load_json_indiv(corr_detection_json_file_path)

    if os.path.isdir(folder_path + '/{}_model/'.format(typ)): 
        resil_detection_json_file_path = folder_path + '/{}_model/epochs/0/coco_instances_results_0_epoch.json'.format(typ)
        resil_corr_detection_json_file_path = folder_path + '/{}_corr_model/epochs/'.format(typ) + str(epoch_nr) + '/coco_instances_results_' + str(epoch_nr) + '_epoch.json'
        coco_orig_resil = load_json_indiv(resil_detection_json_file_path)
        coco_corr_resil = load_json_indiv(resil_corr_detection_json_file_path)
    else:
        coco_orig_resil = None
        coco_corr_resil = None


    return coco_gt, coco_orig, coco_corr, coco_orig_resil, coco_corr_resil


def add_image_dims(coco_labels_grouped, coco_orig_grouped, coco_corr_grouped, coco_orig_resil_grouped, coco_corr_resil_grouped, img_w, img_h):
    # Add image height, width to coco_labels
    def add_h_w(coco_labels_grouped, img_w, img_h):
        for n in range(len(coco_labels_grouped)):
            for m in range(len(coco_labels_grouped[n])):
                coco_labels_grouped[n][m]["image_width"] = int(img_w[n])
                coco_labels_grouped[n][m]["image_height"] = int(img_h[n])
        return coco_labels_grouped

    coco_labels_grouped = add_h_w(coco_labels_grouped, img_w, img_h)
    coco_orig_grouped = add_h_w(coco_orig_grouped, img_w, img_h)
    coco_corr_grouped = add_h_w(coco_corr_grouped, img_w, img_h)
    if coco_orig_resil_grouped is not None:
        coco_orig_resil_grouped = add_h_w(coco_orig_resil_grouped, img_w, img_h)
    if coco_corr_resil_grouped is not None:
        coco_corr_resil_grouped = add_h_w(coco_corr_resil_grouped, img_w, img_h)

    return coco_labels_grouped, coco_orig_grouped, coco_corr_grouped, coco_orig_resil_grouped, coco_corr_resil_grouped

def filter_out_of_image_boxes(coco_pred_grouped_n, bbox_mode):
    """
    Clips those bounding boxes that are out of the image.
    """
    if len(coco_pred_grouped_n) == 0:
        return coco_pred_grouped_n

    w,h = coco_pred_grouped_n[0]["image_width"], coco_pred_grouped_n[0]["image_height"]
    for n in range(len(coco_pred_grouped_n)):

        bbox = coco_pred_grouped_n[n]['bbox']
        if bbox_mode == 0: #xyxy
            bbox_new = [clamp(bbox[0], 0, w), clamp(bbox[1], 0, h), clamp(bbox[2], 0, w), clamp(bbox[3], 0, h)]
        elif bbox_mode == 1: #xywh
            x_new = clamp(bbox[0], 0, w)
            y_new = clamp(bbox[1], 0, h)
            bbox_new = [x_new, y_new, clamp(bbox[2], 0, w-x_new), clamp(bbox[3], 0, h-y_new)]
        else:
            bbox_new = bbox

        coco_pred_grouped_n[n]['bbox'] = bbox_new

    return coco_pred_grouped_n


def clamp(x, smallest, largest):
    return max(smallest, min(x, largest))

def find_non_annotated_images(coco_gt):
        """
        Returns:
        - Indices of not-annotated images
        - list of IDs of valid images
        - list of indices of valid images
        """

        img_ids_gt = [n["image_id"] for n in coco_gt["annotations"]]
        img_ids_gt2 = [n["id"] for n in coco_gt["images"]]
        print("images without annotations", set(img_ids_gt2) - set(img_ids_gt))
        unannotated_images = list(np.sort(list(set(img_ids_gt2) - set(img_ids_gt))))
        list_nr_images = list(np.sort(list(set(img_ids_gt))))

        index_list = []
        for x in unannotated_images:
            ind = np.where(np.array(img_ids_gt2) == x)[0]
            if ind:
                index_list.append(ind[0])
        
        all_inds = list((range(len(img_ids_gt2))))
        valid_inds = np.sort(list(set(index_list) ^ set(all_inds)))

        return index_list, list_nr_images, valid_inds

def eval_epoch(epoch_nr, folder_path, iou_thresh, metadata, nan_infs, filter_ooI=True, eval_mode="iou+class_labels", typ='ranger', folder_num=0):
    """
    out: dictionary with keys gt, orig, corr, orig_resil, corr_resil.
    Values to keys are lists of dictionaries with entries tp, fp, fn, prec, rec, etc.
    """

    
    coco_gt, coco_orig, coco_corr, coco_orig_resil, coco_corr_resil = load_all_jsons(folder_path, epoch_nr, typ=typ)
    coco_labels = coco_gt["annotations"]

    # Check for unannotated images:
    index_list, list_nr_images, valid_inds = find_non_annotated_images(coco_gt)

    # What is the bbox_mode? Relevant for image filter
    bbox_mode = None
    if len(coco_orig) > 0 and len(coco_corr) > 0:
        assert coco_orig[0]["bbox_mode"] == coco_corr[0]["bbox_mode"]
        bbox_mode = coco_orig[0]["bbox_mode"] #0: #xyxy, 1: xywh

    # Image width and height:
    img_w = [n["width"] for n in coco_gt["images"]]
    img_h = [n["height"] for n in coco_gt["images"]]
    # sort out the image height, width of unannotated imgs:
    img_w = np.array(img_w)[np.array(valid_inds)]
    img_h = np.array(img_h)[np.array(valid_inds)]

    # Info about DUE:
    nan_infs_epoch = nan_infs[0][epoch_nr]
    nan_infs_epoch = np.array(nan_infs_epoch)[np.array(valid_inds)]

    if nan_infs[1] is not None:
        nan_infs_ranger_epoch = nan_infs[1][epoch_nr]
        nan_infs_ranger_epoch = np.array(nan_infs_ranger_epoch)[np.array(valid_inds)]

    # Group per image:
    coco_labels_grouped = group_by_image_id_indiv(coco_labels, list_nr_images)
    coco_orig_grouped = group_by_image_id_indiv(coco_orig, list_nr_images)
    coco_corr_grouped = group_by_image_id_indiv(coco_corr, list_nr_images)
    if coco_orig_resil is not None:
        coco_orig_resil_grouped = group_by_image_id_indiv(coco_orig_resil, list_nr_images)
        coco_corr_resil_grouped = group_by_image_id_indiv(coco_corr_resil, list_nr_images)
    else:
        coco_orig_resil_grouped, coco_corr_resil_grouped = None, None

    # Add image height and width info from gt:
    coco_labels_grouped, coco_orig_grouped, coco_corr_grouped, coco_orig_resil_grouped, coco_corr_resil_grouped = add_image_dims(coco_labels_grouped, coco_orig_grouped, \
        coco_corr_grouped, coco_orig_resil_grouped, coco_corr_resil_grouped, img_w, img_h)

    results = {'gt': [], 'orig': [], 'corr':[], 'orig_resil': [], 'corr_resil': []} 

    for i in tqdm(range(len(coco_labels_grouped)), desc="progress folder {} :".format(folder_num)): #i image counter
        # print('image', i, 'of', len(coco_labels_grouped), 'id (image, object)', [(x["image_id"], x["id"]) for x in coco_labels_grouped[i]])

        if coco_labels_grouped[i] == []:
            print("warning, no labels found for this image. Ground truth missing?")


        # Reduce predictions by those boxes outside of the image, if desired:
        if filter_ooI and bbox_mode is not None:
            coco_labels_grouped[i] = filter_out_of_image_boxes(coco_labels_grouped[i], bbox_mode)
            coco_orig_grouped[i] = filter_out_of_image_boxes(coco_orig_grouped[i], bbox_mode)
            coco_corr_grouped[i] = filter_out_of_image_boxes(coco_corr_grouped[i], bbox_mode)
            if coco_orig_resil_grouped is not None:
                coco_orig_resil_grouped[i] = filter_out_of_image_boxes(coco_orig_resil_grouped[i], bbox_mode)
                coco_corr_resil_grouped[i] = filter_out_of_image_boxes(coco_corr_resil_grouped[i], bbox_mode)



        # Match bounding boxes by IoU:
        result_dict = eval_image(coco_labels_grouped[i], coco_labels_grouped[i], iou_thresh, metadata, mode=eval_mode) #get one dict per image
        result_dict["nan_inf"] = None 
        results["gt"].append(result_dict)

        result_dict = eval_image(coco_labels_grouped[i], coco_orig_grouped[i], iou_thresh, metadata, mode=eval_mode)
        result_dict["nan_inf"] = None 
        results["orig"].append(result_dict)

        result_dict = eval_image(coco_labels_grouped[i], coco_corr_grouped[i], iou_thresh, metadata, mode=eval_mode)
        result_dict["nan_inf"] = int(nan_infs_epoch[i]) #add info about nan or inf at inference, if available. Go to int because of json dump.
        results["corr"].append(result_dict)

        if coco_orig_resil_grouped is not None:
            result_dict = eval_image(coco_labels_grouped[i], coco_orig_resil_grouped[i], iou_thresh, metadata, mode=eval_mode)
            result_dict["nan_inf"] = None 
            results["orig_resil"].append(result_dict)

            result_dict = eval_image(coco_labels_grouped[i], coco_corr_resil_grouped[i], iou_thresh, metadata, mode=eval_mode)
            result_dict["nan_inf"] = int(nan_infs_ranger_epoch[i]) 
            results["corr_resil"].append(result_dict)


    return results, index_list

def eval_experiment(epochs, iou_thresh, folder_path, save_name, metadata, nan_infs, filter_ooI, eval_mode, typ='ranger', folder_num=0):
    """
    output: dictionary (also saved) with keys gt, orig, corr, ... , faults.
    Value is a list of dict corresponding to the epochs.
    """


    results_all_epochs = {'gt': [], 'orig': [], 'corr': [], 'orig_resil': [], 'corr_resil': [], 'unannotated_images': []} 
    for epoch_nr in range(epochs):
        print('epoch', epoch_nr, '...')
        
        results_one_epoch, unannotated_images = eval_epoch(epoch_nr, folder_path, iou_thresh, metadata, nan_infs, filter_ooI, eval_mode=eval_mode, typ=typ, folder_num=folder_num) #corr nan_infs

        results_all_epochs["gt"].append(results_one_epoch["gt"])
        results_all_epochs["orig"].append(results_one_epoch["orig"])
        results_all_epochs["corr"].append(results_one_epoch["corr"])
        if results_one_epoch["orig_resil"] != []:
            results_all_epochs["orig_resil"].append(results_one_epoch["orig_resil"])
            results_all_epochs["corr_resil"].append(results_one_epoch["corr_resil"])

    # Get list of unannotated images (indices extended to all epochs): 
    nr_samples = len(nan_infs[0][0]) #10 #Only relevant if there are not annotated images
    unann_img_ind_ext = []
    for ll in unannotated_images:
        for x in range(epochs):
            unann_img_ind_ext.append(ll + x*nr_samples)
    unann_img_ind_ext = np.sort(unann_img_ind_ext).tolist()
    print("Not annotated images registered (in each epoch: ", unannotated_images, ")")
    results_all_epochs["unannotated_images"] = unann_img_ind_ext

    with open(save_name, "w") as outfile: 
        json.dump(results_all_epochs, outfile) 

    return results_all_epochs, unann_img_ind_ext

def get_sdc_all(res_all):
    """
    Calculates tp, fp, fn and resulting sdc, due.
    Raw tp, fp, fn are saved in dict for transparency.
    Appends everything without epoch average.
    """
    orig_sdc_all = []
    corr_sdc_all = []
    resil_orig_sdc_all = []
    resil_corr_sdc_all = []

    orig_due_all = []
    corr_due_all = []
    resil_orig_due_all = []
    resil_corr_due_all = []

    if res_all["orig_resil"] == [] or res_all["corr_resil"]==[]:
        resil = False
    else:
        resil = True


    dct_save = []
    dct_item = {"id": 0, "orig": {'tp': 0, 'fp': 0, 'fn': 0, 'fp_bbox': 0, 'fp_class': 0, 'fp_bbox_class':0, 'nan_inf': None}, \
        'corr': {'tp': 0, 'fp': 0, 'fn': 0, 'fp_bbox': 0, 'fp_class': 0, 'fp_bbox_class':0, 'nan_inf': None}, \
        'orig_resil': {'tp': 0, 'fp': 0, 'fn': 0, 'fp_bbox': 0, 'fp_class': 0, 'fp_bbox_class':0, 'nan_inf': None},\
        'corr_resil': {'tp': 0, 'fp': 0, 'fn': 0, 'fp_bbox': 0, 'fp_class': 0, 'fp_bbox_class':0, 'nan_inf': None}}

    def fill_dct_item(dct_item, res_all, ep, im, cat):

        dct_item["id"] = res_all["gt"][ep][im]["id"] #save image id

        dct_item[cat]["tp"] = len(res_all[cat][ep][im]["tp"])
        dct_item[cat]["fn"] = len(res_all[cat][ep][im]["fn"])
        dct_item[cat]["fp"] = len(res_all[cat][ep][im]["fp"])
        
        dct_item[cat]["fp_bbox"] = res_all[cat][ep][im]["fp_bbox"]
        dct_item[cat]["fp_class"] = res_all[cat][ep][im]["fp_class"]
        dct_item[cat]["fp_bbox_class"] = res_all[cat][ep][im]["fp_bbox_class"]

        dct_item[cat]["nan_inf"] = res_all[cat][ep][im]["nan_inf"]
        return dct_item

    # res_all["orig"][4][960]
    for ep in range(len(res_all["gt"])):
        # print('epoch', ep)
        # flt = res_all["faults"][ep]
        # --- Meaning for neuron injection: --- #
        # 1. batch index (everywhere)
        # 2. layer (everywhere)

        orig_sdc_ep = []
        corr_sdc_ep = []
        resil_orig_sdc_ep = []
        resil_corr_sdc_ep = []
        orig_due_ep = []
        corr_due_ep = []
        resil_orig_due_ep = []
        resil_corr_due_ep = []

        for im in range(len(res_all["orig"][ep])):
            ddct = deepcopy(dct_item)

            true_obj = res_all["gt"][ep][im]["tp"]

            tp = len(res_all["orig"][ep][im]["tp"])
            fp = len(res_all["orig"][ep][im]["fp"])
            fn = len(res_all["orig"][ep][im]["fn"])

            orig_sdc = (fp + fn)/(2*tp + fp + fn)
            ddct = fill_dct_item(ddct, res_all, ep, im, "orig")

            tp = len(res_all["corr"][ep][im]["tp"])
            fp = len(res_all["corr"][ep][im]["fp"])
            fn = len(res_all["corr"][ep][im]["fn"])
            corr_sdc = (fp + fn)/(2*tp + fp + fn)
            ddct = fill_dct_item(ddct, res_all, ep, im, "corr")

            if resil:
                tp = len(res_all["orig_resil"][ep][im]["tp"])
                fp = len(res_all["orig_resil"][ep][im]["fp"])
                fn = len(res_all["orig_resil"][ep][im]["fn"])
                resil_orig_sdc = (fp + fn)/(2*tp + fp + fn)
                ddct = fill_dct_item(ddct, res_all, ep, im, "orig_resil")

                tp = len(res_all["corr_resil"][ep][im]["tp"])
                fp = len(res_all["corr_resil"][ep][im]["fp"])
                fn = len(res_all["corr_resil"][ep][im]["fn"])
                resil_corr_sdc = (fp + fn)/(2*tp + fp + fn)
                ddct = fill_dct_item(ddct, res_all, ep, im, "corr_resil")
            else:
                resil_orig_sdc = None
                resil_corr_sdc = None

            # if ep ==4 and im ==960:
            #     x=0

            orig_sdc_ep.append(orig_sdc)
            corr_sdc_ep.append(corr_sdc)
            resil_orig_sdc_ep.append(resil_orig_sdc)
            resil_corr_sdc_ep.append(resil_corr_sdc)

            nf = res_all["orig"][ep][im]["nan_inf"]
            if nf is not None:
                orig_due_ep.append(nf)
            nf = res_all["corr"][ep][im]["nan_inf"]
            if nf is not None:
                corr_due_ep.append(nf)
                
            if res_all["orig_resil"] != []:
                nf = res_all["orig_resil"][ep][im]["nan_inf"]
                if nf is not None:
                    resil_orig_due_ep.append(nf)
                nf = res_all["corr_resil"][ep][im]["nan_inf"]
                if nf is not None:
                    resil_corr_due_ep.append(nf)

            dct_save.append(ddct)


        # orig_sdc, corr_sdc, resil_orig_sdc, resil_corr_sdc = np.mean(orig_sdc_ep), np.mean(corr_sdc_ep), np.mean(resil_orig_sdc_ep), np.mean(resil_corr_sdc_ep)
        # print('fault', flt, orig_sdc, corr_sdc)
        orig_sdc_all.append(orig_sdc_ep)
        corr_sdc_all.append(corr_sdc_ep)
        resil_orig_sdc_all.append(resil_orig_sdc_ep)
        resil_corr_sdc_all.append(resil_corr_sdc_ep)

        orig_due_all.append(orig_due_ep)
        corr_due_all.append(corr_due_ep)
        resil_orig_due_all.append(resil_orig_due_ep)
        resil_corr_due_all.append(resil_corr_due_ep)

    return orig_sdc_all, corr_sdc_all, resil_orig_sdc_all, resil_corr_sdc_all, orig_due_all, corr_due_all, resil_orig_due_all, resil_corr_due_all, dct_save