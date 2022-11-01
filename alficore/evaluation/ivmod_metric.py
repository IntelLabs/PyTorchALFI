# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.optimize import linear_sum_assignment

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


    # bb1 = {'x1': bb1[0], 'x2': bb1[0] + bb1[2], 'y1': bb1[1] , 'y2': bb1[1] + bb1[3]}
    # bb2 = {'x1': bb2[0], 'x2': bb2[0] + bb2[2], 'y1': bb2[1] , 'y2': bb2[1] + bb2[3]}
    # bb1 = {'x1': bb1[0], 'x2': bb1[2], 'y1': bb1[1] , 'y2': bb1[3]}
    # bb2 = {'x1': bb2[0], 'x2': bb2[2], 'y1': bb2[1] , 'y2': bb2[3]}
    # print(bb1['x1'], bb1['x2'], bb1['x1'] < bb1['x2'], bb1['x1'] == bb1['x2'], bb1['x1'] > bb1['x2'])

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
        iou = 0
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

    # Do not add that part (14.12.21), leads to strange fps since next best choice is taken from cost matrix:
    # off_values = cost < iou_thresh #low iou should not be penalized when looking for assignment, but will be chosen if no better choice
    # if off_values.any():
    #     cost[off_values] = 0 #np.Inf but linear_sum_assignment cant handle inf. Overwrites -1 to 0 to make class/oobbox a fair balance
    
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
        # for row, col in zip(row_ind, col_ind):
        #     if source[row]['category_id'] != target[col]['category_id']:
        #         fp_bbox_class += 1

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
    check_class_labels= kwargs.get("check_class_labels", False)
    # check_class_groups= kwargs.get("check_class_groups", False)

    cost = np.zeros((len(source), len(target))) #rows: source, cols: targets
    for n in range(len(source)):
        for m in range(len(target)):
            cost[n][m] = get_iou(source[n], target[m]) 

    off_values = (cost < iou_thresh) #necessary?
    if off_values.any():
        cost[off_values] = 0 #np.Inf but linear_sum_assignment cant handle inf
    
    cost = cost*(-1) #invert because large iou is better

    row_ind,col_ind=linear_sum_assignment(cost)
    # print('rows', row_ind)#The row index corresponding to the cost matrix
    # print('cols', col_ind)#The optimally assigned column index corresponding to the row index

    cost = cost*(-1) #invert again to make costs positive (=iou)
    # print('costs', cost[row_ind,col_ind])# Extract the element where the optimal assigned column index of each row index is located to form an array
    # print('costs overall', cost[row_ind,col_ind].sum())#array sum
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

    # Derive unassigned rows and cols
    row_ind_unassigned = list(set(all_rows) - set(row_ind))
    col_ind_unassigned = list(set(all_cols) - set(col_ind))

    fp_flags = {"bbox_mismatch": int(fp_bbox)}
    iou_list = cost[row_ind,col_ind] #list of chosen iou pairings, in order of source list
    return row_ind, col_ind, row_ind_unassigned, col_ind_unassigned, fp_flags, iou_list

def ivmod_metric(coco_labels_grouped_n, coco_pred_grouped_n, iou_thresh, eval_mode="iou+class_labels"):

    dict_template = {'imgId': 0, 'tp': [], 'fp': [], 'fn': [], 'prec': None, 'rec': None, 'fp_bbox': 0, 'fp_class': 0, 'fp_bbox_class':0, 'iou': iou_thresh, 'iou_list': []}

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
    if eval_mode == 'iou':
        # Match bounding boxes by IoU:
        row, col, row_un, col_un, fp_flags, iou_list = assign_hungarian(source=coco_pred_grouped_n, target=coco_labels_grouped_n, iou_thresh=iou_thresh)
        # print(row, col, row_un, col_un)
    elif eval_mode == 'iou+class_labels': 
        # print('Consider Bounding Boxes + Class Labels:')
        row, col, row_un, col_un, fp_flags, iou_list = assign_hungarian_trial(source=coco_pred_grouped_n, target=coco_labels_grouped_n, iou_thresh=iou_thresh, check_class_labels=True)

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
