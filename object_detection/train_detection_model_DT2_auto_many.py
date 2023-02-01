
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from copy import deepcopy
import random
import torch.nn.functional as F
from train_detection_model_LR3 import *
from sklearn import tree
import graphviz
import os
from train_detection_model_DT2 import test_dt, train_dt, get_tp_fp_fn, get_class_acc
from attic.train_detection_model_DT2_auto_ft_red import load_create_result_ft_red_json

# https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
# Feature importance is exp(weights), because of logistic regression sigmoid unit


def where_json(file_name):
    return os.path.exists(file_name)

# def load_create_result_json(result_json, df):
#     if where_json(result_json):
#         pass
#     else:
#         data = {}
#         for i in df:
#             data[i] = {'p': [], 'r': [], 'acc': [], 'acc_cls': [], 'acc_cat':[], 'meta':[]}
#         with open(result_json, 'w') as outfile:  
#             json.dump(data, outfile)

#     result_dict = load_json_indiv(result_json)
#     return result_dict

# def load_create_result_json(result_json, df):
#     if where_json(result_json):
#         pass
#     else:
#         data = {}
#         for i in df:
#             data[i] = {'p': [], 'r': [], 'acc': [], 'acc_cls': [], 'acc_cat':[], 'meta':[], 'due': 0.}
#         with open(result_json, 'w') as outfile:  
#             json.dump(data, outfile)

#     result_dict = load_json_indiv(result_json)
#     for i in df:
#         # if i not in result_dict.keys():
#         result_dict[i] = {'p': [], 'r': [], 'acc': [], 'acc_cls': [], 'acc_cat':[], 'meta':[], 'due': 0.} #overwrite if exists, create if doesnt exist
    
#     return result_dict

def main():
    ##########################################################################################################
    df = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti', 'resnet_imagenet', 'alexnet_imagenet', 'lenet_mnist']
    sv_path = "/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/"
    result_json = sv_path + "dt_train.json"

    
    # Selected data
    target = 'all' #'noise', 'blur', 'blur_noise', 'neurons', 'weights', 'neurons_weights', 'all'
    
    save_trained_model = False
    depth_tree = None #None #10 #None
    ccp_alpha = 1e-5 #pruning factor

    N_epochs = 10
    ##########################################################################################################

    result_dict = load_create_result_ft_red_json(result_json, df)

    for exp in df:
        # Load extracted quantiles: -------------
        data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/' + exp + '_presum/'

        
        fault_class_dict= {'no_sdc': 0, 'neurons': 1, 'weights': 2, 'blur': 3, 'noise': 4, 'contrast':5}
        classes = list(fault_class_dict.keys())
        output_dim = len(classes)
        x_list, y_list, _ = load_data_from_quantile_data(data_folder, target, fault_class_dict)

        feature_labels = create_feature_labels(x_list)


        mask_nan = np.isnan(x_list)
        mask_nan_rows = [n.any() for n in mask_nan]
        x_list = x_list[np.logical_not(mask_nan_rows),:]
        y_list = y_list[np.logical_not(mask_nan_rows)]
        result_dict[exp]['due'] = np.sum(mask_nan_rows)/len(mask_nan_rows) if len(mask_nan_rows) > 0 else 0. 

        assert not (x_list > 1).any() and not (y_list > len(fault_class_dict)-1).any() and not (x_list < 0).any() and not (y_list < 0).any(), 'features not scaled.' #Check that all features are properly normalized between [0,1]
        assert not np.isnan(x_list).any() and not np.isinf(x_list).any(), "x values contain inf or nans!"
        assert not np.isnan(y_list).any() and not np.isinf(y_list).any(), "x values contain inf or nans!"

        # Note
        # x_vector features: 
        # - any layer and any quantile detected sth: 0 or 1 (dim=1)
        # - how many layers detected sth, normalized by nr of layers (per quantile, dim=5)
        # - what was the last layer that detected sth, normalized by nr of layers (per quantile, dim=5)
        # - what was the maximum deviation from bound, normalized to largest value (per quantile, dim = 5)


        # Construct classifier -------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.33, random_state=1) # split 1/3 (test) to 2/3 (train)
        # to_elim = substract_list(feature_labels, fts_to_keep #
        to_elim = elim_by_pattern(['q0_'], feature_labels)
        X_train, _ = eliminate_features(X_train, to_elim, feature_labels)
        X_test, feature_labels = eliminate_features(X_test, to_elim, feature_labels)



        # Training ---------------------------------------------------------------------------------------------
        p_list = []
        r_list = []
        acc_list = []
        acc_cls_list = []
        acc_cat_list = []
        classifier_meta = []
        conf_mat_list = []

        for n in range(N_epochs):
            print('Epoch:', n)
            classifier = train_dt(X_train, y_train, output_dim, target, depth_tree, save_trained_model, classes, feature_labels, ccp_alpha)
            # print('Created classifier of depth:', classifier.get_depth(), 'nr of leaves:', classifier.get_n_leaves())

            # # print('feature importances', classifier.feature_importances_)
            # # feature_imp = classifier.feature_importances_
            # # feature_labels
            # imp_sorted = [x for x,_ in sorted(zip(classifier.feature_importances_,feature_labels))][::-1]
            # labels_sorted = [x for _,x in sorted(zip(classifier.feature_importances_,feature_labels))][::-1]
            # print('importances', imp_sorted[:10])
            # print('labels', labels_sorted[:10])


            # Test -----------------------------------
            conf_matrix = test_dt(X_test, y_test, classifier, output_dim)

            res = get_tp_fp_fn(conf_matrix, fault_class_dict)
            tp_sdc, tp_cat, tp_cls = res['tp_sdc'], res['tp_cat'], res['tp_cls']
            fp, fn, tn = res['fp'], res['fn'], res['tn']
            # cls_conf, cat_conf, sdc_conf = res['cls_conf'], res['cat_conf'], res['sdc_conf']

            # # Accuracies
            # acc_cls = (tp_cls + tn)/np.sum(conf_matrix)
            # acc_cat = (tp_cat + tn)/np.sum(conf_matrix)
            # acc_sdc = (tp_sdc + tn)/np.sum(conf_matrix)
            # print('acc_cls', acc_cls, 'acc_cat', acc_cat, 'acc_sdc', acc_sdc)

            # conf rates (only within sdc "block")
            sdc_cnt = np.sum(conf_matrix) - fp - fn - tn
            print('acc_cls_conf_rate', tp_cls/sdc_cnt, 'acc_cat_conf_rate', tp_cat/sdc_cnt, 'acc_sdc_conf_rate', tp_sdc/sdc_cnt)

            acc_cls, acc_cat = get_class_acc(fault_class_dict, conf_matrix)
            print('acc_cls', acc_cls, 'acc_cat', acc_cat)

            # Prediction precision, recall
            p, r = get_p_r(tp_sdc, fp, fn)
            print('precision', p, 'recall', r, 'acc_cls_conf_rate', tp_cls/sdc_cnt)
            
            p_list.append(p)
            r_list.append(r)
            acc_list.append(tp_cls/sdc_cnt)
            acc_cls_list.append(acc_cls)
            acc_cat_list.append(acc_cat)
            classifier_meta.append([float(classifier.get_depth()), float(classifier.get_n_leaves()), float(classifier.tree_.node_count)])
            conf_mat_list.append(conf_matrix.tolist())

        # Stats:
        result_dict[exp]['p'] = p_list
        result_dict[exp]['r'] = r_list
        result_dict[exp]['acc'] = acc_list
        # result_dict[exp]['acc_cls'] = list(acc_cls_list[0].values())
        # result_dict[exp]['acc_cat'] = list(acc_cat_list[0].values())
        result_dict[exp]['acc_cls'] = acc_cls_list
        result_dict[exp]['acc_cat'] = acc_cat_list
        result_dict[exp]['meta'] = classifier_meta
        result_dict[exp]['conf_mat'] = conf_mat_list


    with open(result_json, 'w') as outfile:  
            json.dump(result_dict, outfile)

if __name__ == "__main__":
    main()
