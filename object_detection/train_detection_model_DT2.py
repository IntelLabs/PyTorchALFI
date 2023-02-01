
# import json
# from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
# import torch
# from tqdm import tqdm
# from copy import deepcopy
# import random
# import torch.nn.functional as F
# from train_detection_model_DT2_auto_ft_red3 import get_tpfpfn_new
from train_detection_model_LR4 import get_balance2, load_data_from_quantile_data, create_feature_labels, get_p_r, elim_by_pattern, eliminate_features, get_tpfpfn_new, substract_list, get_flt_dicts, get_data
from sklearn import tree
import graphviz
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from copy import deepcopy

# https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
# Feature importance is exp(weights), because of logistic regression sigmoid unit

def get_tp_fp_fn_multiclass_dt(x_list, y_list, model, nr_samples, thresh=0.5, output_dim=6):
    #Classes: 0 is no sdc, 1=sdc hw, 2=sdc input
    cnt_tp = [0 for u in range(output_dim)] #here only four classes, since no sdc is no tp, fp, fn
    cnt_fp = [0 for u in range(output_dim)] 
    cnt_fn = [0 for u in range(output_dim)] 
    cnt_tn = 0
    cnt_sdc = 0
    sdc_conf = 0
    conf_matrix = np.zeros((output_dim, output_dim))

    # cnt_sdc = 0
    # sdc_conf = 0
    # sel = range(x_list.shape[0])
    # if nr_samples is not None and nr_samples < len(sel):
    #     sel = random.sample(sel, nr_samples)
        
    prediction_all = model.predict(x_list)

    for n in range(prediction_all.shape[0]):
        # x_n = x_list[n,:]
        prediction=prediction_all[n]
        y_n = y_list[n]

        conf_matrix[y_n, prediction] += 1
        if y_n > 0:
            cnt_sdc += 1


        if prediction == y_n and y_n == 0:
            cnt_tn += 1
        if prediction == y_n and y_n > 0:
            cnt_tp[prediction] += 1
        if prediction == 0 and y_n > 0:
            cnt_fn[y_n] += 1
        if prediction > 0 and y_n == 0:
            cnt_fp[prediction] += 1
        if prediction > 0 and y_n > 0 and prediction != y_n:
            sdc_conf += 1


    cnt_tp = np.array(cnt_tp[1:]).tolist()
    cnt_fp = np.array(cnt_fp[1:]).tolist()
    cnt_fn = np.array(cnt_fn[1:]).tolist()

    return cnt_tp, cnt_fp, cnt_fn, cnt_tn, cnt_sdc, sdc_conf, prediction, conf_matrix

def test_dt(x_list, y_list, model, output_dim=6):

    conf_matrix = np.zeros((output_dim, output_dim))
        
    prediction_all = model.predict(x_list)

    for n in range(prediction_all.shape[0]):
        # x_n = x_list[n,:]
        prediction=prediction_all[n]
        y_n = y_list[n]

        conf_matrix[y_n, prediction] += 1


    return conf_matrix

def train_dt(X_train, y_train, output_dim, target, depth_tree, save_trained_model, classes, feature_labels, ccp_alpha):
    
    class_imbalance = get_balance2(y_train, output_dim)
    weights = dict(zip(list(range(len(classes))), class_imbalance))
    print(weights)

    classifier = tree.DecisionTreeClassifier(class_weight=weights, max_depth=depth_tree, ccp_alpha = ccp_alpha) #some cost-complexity pruning added

    print('Fitting training data...')
    classifier = classifier.fit(X_train, y_train)
    print('Training accuracy is:', classifier.score(X_train, y_train))

    # print(tree.plot_tree(classifier))

    if save_trained_model:
        plot_tree(classifier, feature_labels, classes, "graph_test")
        # dotdata = tree.export_graphviz(classifier, out_file=None) 
        # graphs = graphviz.Source(dotdata) 

        # dotdata = tree.export_graphviz(classifier, out_file=None, 
        #                     feature_names=feature_labels,  
        #                     class_names=np.array(classes)[classifier.classes_].tolist(), 
        #                     filled=True, rounded=True,  
        #                     special_characters=True)  
        # graphs = graphviz.Source(dotdata)

        # save_name = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/decision_tree/graph_test_' + target
        # graphs.render(filename=save_name, format="png") #saves all info
        # print('saved', save_name)
    
    return classifier

def plot_tree(classifier, feature_labels, classes, fig_name):
    # dotdata = tree.export_graphviz(classifier, out_file=None) 
    # graphs = graphviz.Source(dotdata) 

    dotdata = tree.export_graphviz(classifier, out_file=None, 
                        feature_names=feature_labels,  
                        class_names=np.array(classes)[classifier.classes_].tolist(), 
                        filled=True, rounded=True,  
                        special_characters=True)  
    graphs = graphviz.Source(dotdata)

    save_name = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/decision_tree/' + fig_name
    graphs.render(filename=save_name, format="png") #saves all info
    print('saved', save_name)

def train_dt_with_pruning(X_train, y_train, X_test, y_test, output_dim, target, depth_tree, save_trained_model, classes, feature_labels):
    
    class_imbalance = get_balance2(y_train, output_dim)
    weights = dict(zip(list(range(len(classes))), class_imbalance))
    print(weights)

    # ccp_alpha = 0.0 #for pruning increase this number
    clfs = []
    ccp_list = []
    test_accs = []
    train_accs = []
    for x in range(1):
        # ccp_alpha = x*0.5e-5
        # ccp_alpha = x*2.e-5
        ccp_alpha = 1.e-5
        print('Pruning with:', ccp_alpha)
        print('Fitting training data...')
        clf = tree.DecisionTreeClassifier(class_weight=weights, max_depth=depth_tree, ccp_alpha=ccp_alpha)
        clf = clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        print('Train accuracy:', train_acc, 'test acc:', test_acc, 'depth:', clf.get_depth(), clf.get_n_leaves(), clf.tree_.node_count)

        clfs.append(clf)
        ccp_list.append(ccp_alpha)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        

    ind_max = np.where(test_accs == np.max(test_accs))[0][0]
    classifier = clfs[ind_max]
    print('pruning found', ccp_list[ind_max])


    if save_trained_model:
        plot_tree(classifier, feature_labels, classes, "graph_test")
        # dotdata = tree.export_graphviz(classifier, out_file=None) 
        # graphs = graphviz.Source(dotdata) 

        # dotdata = tree.export_graphviz(classifier, out_file=None, 
        #                     feature_names=feature_labels,  
        #                     class_names=np.array(classes)[classifier.classes_].tolist(),  
        #                     filled=True, rounded=True,  
        #                     special_characters=True)  
        # graphs = graphviz.Source(dotdata)

        # save_name = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/decision_tree/graph_test_' + target
        # graphs.render(filename=save_name, format="png") #saves all info
        # print('saved', save_name)
    
    return classifier


# def get_tp_fp_fn(conf_matrix):
#     # tp_sdc, fp_sdc, fn_sdc = np.zeros(len(fault_class_dict)), np.zeros(len(fault_class_dict)), np.zeros(len(fault_class_dict))
#     # tp_cat, fp_cat, fn_cat = np.zeros(len(fault_class_dict)), np.zeros(len(fault_class_dict)), np.zeros(len(fault_class_dict))
#     # tp_cls, fp_cls, fn_cls = np.zeros(len(fault_class_dict)), np.zeros(len(fault_class_dict)), np.zeros(len(fault_class_dict))
#     tp_sdc, tp_cat, tp_cls = 0,0,0
#     fp_all = 0
#     fn_all = 0
#     tn_all = 0
#     cat_conf = 0
#     cls_conf = 0
#     sdc_conf = 0


#     cls_mapping, cats_mapping, sdc_mapping = get_flt_dicts()
#     cls = list(cls_mapping.keys()) #is also order of conf_matrix rows, cols
#     # cats_mapping = {'no_sdc': 0, 'neurons': 1, 'weights': 1, 'blur': 2, 'noise': 2, 'contrast':2}
#     # sdc_mapping = {'no_sdc': 0, 'neurons': 1, 'weights': 1, 'blur': 1, 'noise': 1, 'contrast':1}
#     for n in range(len(cls_mapping)):
#         for m in range(len(cls_mapping)):
#             if n == m: #diagonals
#                 if cls[n] != "no_sdc":
#                     tp_cls += conf_matrix[n,m]
#                     tp_cat += conf_matrix[n,m]
#                     tp_sdc += conf_matrix[n,m]
#                 else:
#                     tn_all += conf_matrix[n,m]
                
#             else: #off-diagonals
#                 cls_conf += conf_matrix[n,m]
#                 if cats_mapping[cls[n]] == cats_mapping[cls[m]]:
#                     tp_cat += conf_matrix[n,m]
#                 else:
#                     cat_conf += conf_matrix[n,m]
#                 if sdc_mapping[cls[n]] == sdc_mapping[cls[m]]:
#                     tp_sdc += conf_matrix[n,m]
#                 else:
#                     sdc_conf += conf_matrix[n,m]
                    
#                 if cats_mapping[cls[n]] == cats_mapping['no_sdc'] and cats_mapping[cls[m]] != cats_mapping['no_sdc'] :
#                     fp_all += conf_matrix[n,m]
#                 if cats_mapping[cls[n]] != cats_mapping['no_sdc'] and cats_mapping[cls[m]] == cats_mapping['no_sdc'] :
#                     fn_all += conf_matrix[n,m]
                    
    
#     return {'tp_cls': tp_cls, 'tp_cat': tp_cat, 'tp_sdc': tp_sdc, 'fn': fn_all, 'fp': fp_all, 'tn': tn_all, 'cat_conf': cat_conf, 'cls_conf': cls_conf, 'sdc_conf': sdc_conf}    

# def get_class_acc(fault_class_dict, conf_matrix):
#     cls = list(fault_class_dict.keys()) #is also order of conf_matrix rows, cols
#     cats_mapping = {'no_sdc': 0, 'neurons': 1, 'weights': 1, 'blur': 2, 'noise': 2, 'contrast':2}    
#     acc_cls ={'no_sdc': 0., 'neurons': 0., 'weights': 0., 'blur': 0., 'noise': 0., 'contrast': 0.}
#     acc_cat ={'no_sdc': 0., 'neurons': 0., 'weights': 0., 'blur': 0., 'noise': 0., 'contrast': 0.}
#     for x in range(1, conf_matrix.shape[0]): #only for non-sdc
#         tp_cls = 0.
#         tp_cat = 0.
#         for y in range(1, conf_matrix.shape[1]): #only for non-sdc
#             if x == y:
#                 tp_cls += conf_matrix[x,y]
#             if cats_mapping[cls[x]] == cats_mapping[cls[y]]:
#                 tp_cat += conf_matrix[x,y]
#         acc_cls[cls[x]] = tp_cls/np.sum(conf_matrix[x, 1:])
#         acc_cat[cls[x]] = tp_cat/np.sum(conf_matrix[x, 1:])
#     return acc_cls, acc_cat


# def get_data(df, target='all'):

#     # Load extracted quantiles: -------------
#     # # # New:
#     # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/yolo_coco_presum/'
#     # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/yolo_kitti_presum/'
#     # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/ssd_coco_presum/'
#     # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/ssd_kitti_presum/'
#     # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/retina_coco_presum/'
#     # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/retina_kitti_presum/'
#     #
#     # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/resnet_imagenet_presum/'
#     # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/alexnet_imagenet_presum/'

#     # NOTE: Use this to find pruning factor and for plausibility check of data completion (diagonals missing?)

#     data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/' + df[0] + '_presum/'

#     fault_class_dict, _, _ = get_flt_dicts()
#     classes = list(fault_class_dict.keys())
#     output_dim = len(classes)
#     x_list, y_list, lay_list = load_data_from_quantile_data(data_folder, target)

#     feature_labels = create_feature_labels(x_list)

#     # TODO: fix later: quantiles are nan because summation becomes too large? replace by 1 in x_list?
#     # # regularize it:
#     mask_nan = np.isnan(x_list)
#     mask_nan_rows = [n.any() for n in mask_nan]
#     x_list = x_list[np.logical_not(mask_nan_rows),:]
#     y_list = y_list[np.logical_not(mask_nan_rows)]
#     if "weights" in target or "neurons" in target:
#         lay_list = lay_list[np.logical_not(mask_nan_rows)] #only for 
#     # mask_inf = np.isinf(x_list)
#     # x_list[mask_inf] = 1
#     # mn = np.min(x_list)
#     # x_list = x_list - mn
#     # mx = np.max(x_list)
#     # x_list = x_list/mx
#     # x_list[mask_inf] = 1 #fix infs
#     # x_list[x_list < 0] = 0 #remove artefacts
    
#     due_rate = np.sum(mask_nan_rows)/len(mask_nan_rows) if len(mask_nan_rows) > 0 else 0. 
#     print('DUE det rate:', due_rate)
    
#     assert not (x_list > 1).any() and not (y_list > len(fault_class_dict)-1).any() and not (x_list < 0).any() and not (y_list < 0).any(), 'features not scaled.' #Check that all features are properly normalized between [0,1]
#     assert not np.isnan(x_list).any() and not np.isinf(x_list).any(), "x values contain inf or nans!"
#     assert not np.isnan(y_list).any() and not np.isinf(y_list).any(), "x values contain inf or nans!"


#     # # Filter for specific layers:
#     # counts, bins = np.histogram(lay_list, bins=74)
#     # print('counts, bins', counts, bins)
#     # y_list = y_list[np.logical_or(lay_list > 42, lay_list < 0)]
#     # x_list = x_list[np.logical_or(lay_list > 42, lay_list < 0)]
    

#     # Note
#     # x_vector features: 
#     # - any layer and any quantile detected sth: 0 or 1 (dim=1)
#     # - how many layers detected sth, normalized by nr of layers (per quantile, dim=5)
#     # - what was the last layer that detected sth, normalized by nr of layers (per quantile, dim=5)
#     # - what was the maximum deviation from bound, normalized to largest value (per quantile, dim = 5)


#     # Construct classifier -------------------------------------------------------------------
#     X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.33, random_state=1) # split 1/3 (test) to 2/3 (train)

#     return X_train, X_test, y_train, y_test, feature_labels, output_dim, classes, fault_class_dict

def get_test_result(conf_matrix):
    from tabulate import tabulate
    print('conf matrix', tabulate(conf_matrix))


    cls_mapping, cats_mapping, sdc_mapping = get_flt_dicts()
    # cls_mapping = fault_class_dict #is also order of conf_matrix rows, cols
    # cats_mapping = {'no_sdc': 0, 'neurons': 1, 'weights': 1, 'blur': 2, 'noise': 2, 'contrast':2}
    # sdc_mapping = {'no_sdc': 0, 'neurons': 1, 'weights': 1, 'blur': 1, 'noise': 1, 'contrast':1}
    cls_mapping = list(cls_mapping.values())
    cats_mapping = list(cats_mapping.values())
    sdc_mapping = list(sdc_mapping.values())


    p_cls, r_cls = get_tpfpfn_new(cls_mapping, conf_matrix)
    p_cats, r_cats = get_tpfpfn_new(cats_mapping, conf_matrix)
    p_sdc, r_sdc = get_tpfpfn_new(sdc_mapping, conf_matrix)

    print('precision (cls, cats, sdc)', p_cls, p_cats, p_sdc, 'recall', r_cls, r_cats, r_sdc)
    


    # res = get_tp_fp_fn(conf_matrix, fault_class_dict)
    # tp_sdc, tp_cat, tp_cls = res['tp_sdc'], res['tp_cat'], res['tp_cls']
    # fp, fn, tn = res['fp'], res['fn'], res['tn']
    # # cls_conf, cat_conf, sdc_conf = res['cls_conf'], res['cat_conf'], res['sdc_conf']

    # # # Accuracies
    # # acc_cls = (tp_cls + tn)/np.sum(conf_matrix)
    # # acc_cat = (tp_cat + tn)/np.sum(conf_matrix)
    # # acc_sdc = (tp_sdc + tn)/np.sum(conf_matrix)
    # # print('acc_cls', acc_cls, 'acc_cat', acc_cat, 'acc_sdc', acc_sdc)

    # # conf rates (only within sdc "block")
    # sdc_cnt = np.sum(conf_matrix) - fp - fn - tn
    # # sdc_cls_conf_rate = (cls_conf - fp - fn)/sdc_cnt #probability to correctly predict sdc but confuse class
    # # sdc_cat_conf_rate = (cat_conf - fp - fn)/sdc_cnt
    # # sdc_sdc_conf_rate = (sdc_conf - fp - fn)/sdc_cnt
    # # print('sdc_cls_conf_rate', sdc_cls_conf_rate, 'sdc_cat_conf_rate', sdc_cat_conf_rate, 'sdc_sdc_conf_rate', sdc_sdc_conf_rate)
    # print('acc_cls_conf_rate', tp_cls/sdc_cnt, 'acc_cat_conf_rate', tp_cat/sdc_cnt, 'acc_sdc_conf_rate', tp_sdc/sdc_cnt)

    
    # acc_cls, acc_cat = get_class_acc(fault_class_dict, conf_matrix)
    # print('acc_cls', acc_cls, 'acc_cat', acc_cat)
    # # Prediction precision, recall
    # p, r = get_p_r(tp_sdc, fp, fn)
    # print('precision', p, 'recall', r)

    return p_cls, r_cls


class Pruner():
    # Helps for visualization but doesnt make a big difference for accuracy etc. Node and leaf count doesnt seem to update.
    def __init__(self):
        self.pruned_ind = 0
        self.dtree = None

    def is_leaf(self, inner_tree, index):
        # Check whether node is leaf node
        return (inner_tree.children_left[index] == TREE_LEAF and 
                inner_tree.children_right[index] == TREE_LEAF)

    def prune_duplicate_leaves(self, mdl):
        
        # Remove leaves if both 
        decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
        self.prune_index(mdl.tree_, decisions)
        self.dtree = mdl
        
    def prune_index(self, inner_tree, decisions, index=0):
        # Start pruning from the bottom - if we start from the top, we might miss
        # nodes that become leaves during pruning.
        # Do not use this directly - use prune_duplicate_leaves instead.
        if not self.is_leaf(inner_tree, inner_tree.children_left[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not self.is_leaf(inner_tree, inner_tree.children_right[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:     
        if (self.is_leaf(inner_tree, inner_tree.children_left[index]) and
            self.is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and 
            (decisions[index] == decisions[inner_tree.children_right[index]])):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            # inner_tree.feature[index] = TREE_UNDEFINED
            print("Pruned {}".format(index))
            self.pruned_ind += 1

    # def get_new_meta(self):
    #     self.dtree.

# def prune_index(inner_tree, index, threshold):
#     if inner_tree.value[index].min() < threshold:
#         # turn node into a leaf by "unlinking" its children
#         inner_tree.children_left[index] = TREE_LEAF
#         inner_tree.children_right[index] = TREE_LEAF
#     # if there are shildren, visit them as well
#     if inner_tree.children_left[index] != TREE_LEAF:
#         prune_index(inner_tree, inner_tree.children_left[index], threshold)
#         prune_index(inner_tree, inner_tree.children_right[index], threshold)


def main():

    df = ['alexnet_imagenet'] #retina_coco, yolo_coco, resnet_imagenet, alexnet_imagenet

    ##########################################################################################################
    # Selected data
    target = 'all' #'noise', 'blur', 'blur_noise', 'neurons', 'weights', 'neurons_weights', 'all'
    
    save_trained_model = False
    depth_tree = None #10 #None
    ##########################################################################################################
    

    X_train, X_test, y_train, y_test, feature_labels, output_dim, classes, _ = get_data(df, target)
    # # Filter out or add new features -------------------------------------------   
    # # in general:
    # to_elim = elim_by_pattern(['q0_'], feature_labels)
    # X_train, _ = eliminate_features(X_train, to_elim, feature_labels)
    # X_test, feature_labels = eliminate_features(X_test, to_elim, feature_labels)

    # # # special check:
    # # # fts_to_keep = ['q10_lay4', 'q10_lay42', 'q10_lay60', 'q10_lay68', 'q20_lay0', 'q30_lay48', 'q50_lay0', 'q50_lay49', 'q60_lay68', 'q80_lay68', 'q100_lay42', 'q100_lay60']
    # # # fts_to_keep = ['q10_lay60', 'q30_lay48', 'q100_lay60']
    # # # fts_to_keep = ['q100_lay74']
    # fts_to_keep = ['q0_lay17', 'q100_lay69']
    # to_elim = substract_list(feature_labels, fts_to_keep) #
    # X_train, _ = eliminate_features(X_train, to_elim, feature_labels)
    # X_test, feature_labels = eliminate_features(X_test, to_elim, feature_labels)
    # print('features used', feature_labels)


    # add features?
    # X_train, _ = add_feature(X_train, X_train[:,-4]**2, feature_labels, 'max_dev_act_q25_2')
    # X_test, feature_labels = add_feature(X_test, X_test[:,-4]**2, feature_labels, 'max_dev_act_q25_2')

    # Training ---------------------------------------------------------------------------------------------
    # classifier = train_dt(X_train, y_train, output_dim, target, depth_tree, save_trained_model, classes, feature_labels, ccp_alpha)
    classifier = train_dt_with_pruning(X_train, y_train, X_test, y_test, output_dim, target, depth_tree, save_trained_model, classes, feature_labels)
    print('Created classifier of depth:', classifier.get_depth(), 'nr of leaves:', classifier.get_n_leaves(), 'node count:', classifier.tree_.node_count)
    cls_before = deepcopy(classifier)

    # # print('feature importances', classifier.feature_importances_)
    # # feature_imp = classifier.feature_importances_
    # # feature_labels
    # imp_sorted = [x for x,_ in sorted(zip(classifier.feature_importances_,feature_labels))][::-1]
    # labels_sorted = [x for _,x in sorted(zip(classifier.feature_importances_,feature_labels))][::-1]
    # print('importances', imp_sorted[:10])
    # print('labels', labels_sorted[:10])

    # Test -----------------------------------
    conf_matrix = test_dt(X_test, y_test, classifier, output_dim)
    p_cls, r_cls = get_test_result(conf_matrix)
    print('before', p_cls, r_cls)
    plot_tree(classifier, feature_labels, classes, "graph_test_before")



    # # # Post-pruning ----------------
    # # # Finding: helps to visually simplify graph but not reflected in tree params? Precision etc much worse
    # # https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier#51407943
    # # https://stackoverflow.com/questions/49428469/pruning-decision-trees

    # pr = Pruner()
    # pr.prune_duplicate_leaves(classifier)
    # pr.pruned_ind

    # # prune_duplicate_leaves(classifier)
    # print('Created classifier of depth:', classifier.tree_.max_depth, 'nr of leaves:', classifier.get_n_leaves(), 'node count:', classifier.tree_.node_count)

    # # print(sum(classifier.tree_.children_left < 0))
    # # # start pruning from the root
    # # prune_index(classifier.tree_, 0, 5)
    # # sum(classifier.tree_.children_left < 0)
    # # print('Created classifier of depth:', classifier.tree_.max_depth, 'nr of leaves:', classifier.tree_.n_leaves, 'node count:', classifier.tree_.node_count)

    # # Test -----------------------------------
    # conf_matrix = test_dt(X_test, y_test, classifier, output_dim)
    # p_cls, r_cls = get_test_result(conf_matrix)
    # print('after', p_cls, r_cls)
    # plot_tree(classifier, feature_labels, classes, "graph_test_after")




if __name__ == "__main__":
    main()
