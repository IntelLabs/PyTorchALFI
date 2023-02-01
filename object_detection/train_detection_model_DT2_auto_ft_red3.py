
import json
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
from train_detection_model_LR3 import eliminate_features, load_json_indiv, load_data_from_quantile_data, create_feature_labels, elim_by_pattern, get_p_r, get_balance2, substract_list, get_tpfpfn_new, get_flt_dicts
from train_detection_model_DT2 import test_dt, train_dt
import os
# from tabulate import tabulate

# https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
# Feature importance is exp(weights), because of logistic regression sigmoid unit

def where_json(file_name):
    return os.path.exists(file_name)

def load_create_result_ft_red_json(result_json, df):
    data = {}
    for i in df:
        data[i] = {'p_sdc': [], 'r_sdc': [], 'p_cats': [], 'r_cats': [], 'p_cls': [], 'r_cls': [],  'acc': [], 'ft': [], 'acc_cls': [], 'acc_cat':[], 'meta':[], 'due': 0., 'conf_mat':[], 'min_alternatives': []}
    with open(result_json, 'w') as outfile:  
        json.dump(data, outfile)

    return data


class DT_trainer:
    def __init__(self, exp, ccp_alpha, sv_path, file_base_name, **kwargs):
        self.exp = exp
        self.ccp_alpha = ccp_alpha
        self.sv_path = sv_path
        self.depth_tree = kwargs.get('depth_tree', None)
        self.target = None
        self.fault_class_dict, _, _ = get_flt_dicts()
        self.classes = list(self.fault_class_dict.keys())
        self.output_dim = len(self.classes)

        self.result_json_path = sv_path + file_base_name + "_" + self.exp + ".json"
        self.result_dict = None
        self.create_result_template()
        self.due_rate = None

        # feature data
        self.feature_labels = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.feature_labels_work = None #working versions
        self.cls_feature_labels_work_sorted = None #working versions
        self.X_train_work = None
        self.X_test_work = None
        
        # Result working lists
        self.p_cls_list_work = []
        self.r_cls_list_work = []
        self.p_cats_list_work = []
        self.r_cats_list_work = []
        self.p_sdc_list_work = []
        self.r_sdc_list_work = []
        self.acc_cls_list_work = []
        self.acc_cat_list_work = [] 
        self.classifier_meta_work = []
        self.ft_list_work = [] 
        self.conf_mat_list_work = []

        # Refs
        self.cls_work = None
        self.cls_eval_work = None
        self.cls_ref = None
        self.cls_eval_ref = None
        self.feature_labels_ref_sorted = None

        
    def create_result_template(self):
        self.result_dict = load_create_result_ft_red_json(self.result_json_path, [self.exp])
        print('Created result dictionary at', self.result_json_path)

    def set_data(self, X_train, y_train, X_test, y_test, feature_labels):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_labels = feature_labels

    def set_data_work(self, X_train, X_test, feature_labels):
        # that is the original setup (no model involved)
        self.X_train_work = X_train
        self.X_test_work = X_test
        self.feature_labels_work = feature_labels

    def reset_data_work(self):
        self.X_train_work = deepcopy(self.X_train)
        self.X_test_work = deepcopy(self.X_test)
        self.feature_labels_work = deepcopy(self.feature_labels)

        self.cls_work = None
        self.cls_eval_work = None
        self.cls_feature_labels_work_sorted = None

    def reset_res_list_work(self):
        # Result working lists
        self.p_cls_list_work = []
        self.r_cls_list_work = []
        self.p_cats_list_work = []
        self.r_cats_list_work = []
        self.p_sdc_list_work = []
        self.r_sdc_list_work = []
        self.acc_cls_list_work = []
        self.acc_cat_list_work = [] 
        self.classifier_meta_work = []
        self.ft_list_work = [] 
        self.conf_mat_list_work = []

    def load_exp_data(self, target='all'):
        # Load extracted quantiles: -------------
        data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/' + self.exp + '_presum/'
        self.target = target

        # classes = list(fault_class_dict.keys())
        # output_dim = len(classes)
        print('loading data from', data_folder)
        x_list, y_list, lay_list = load_data_from_quantile_data(data_folder, self.target)
        feature_labels = create_feature_labels(x_list)

        # Nans that come from summation leading out of range!
        mask_nan = np.isnan(x_list)
        mask_nan_rows = [n.any() for n in mask_nan]
        x_list = x_list[np.logical_not(mask_nan_rows),:]
        y_list = y_list[np.logical_not(mask_nan_rows)]
        # if len(lay_list) > 0:
        if "weights" in target or "neurons" in target:
            lay_list = lay_list[np.logical_not(mask_nan_rows)]

        self.due_rate = np.sum(mask_nan_rows)/len(mask_nan_rows) if len(mask_nan_rows) > 0 else 0. 
        print('DUE det rate:', self.due_rate)

        assert not (x_list > 1).any() and not (y_list > len(self.fault_class_dict)-1).any() and not (x_list < 0).any() and not (y_list < 0).any(), 'features not scaled.' #Check that all features are properly normalized between [0,1]
        assert not np.isnan(x_list).any() and not np.isinf(x_list).any(), "x values contain inf or nans!"
        assert not np.isnan(y_list).any() and not np.isinf(y_list).any(), "x values contain inf or nans!"

        # # Filter for specific layers:
        # y_list = y_list[np.logical_or(lay_list > 50, lay_list < 0)]
        # x_list = x_list[np.logical_or(lay_list > 50, lay_list < 0)]
        # feature_labels.index('q100_lay42')


        # Split data -------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.33, random_state=0) # split 1/3 (test) to 2/3 (train)

        # Elim minimum quantiles ------------------------------
        # to_elim = elim_by_pattern(['q0_'], feature_labels) #todo put back
        # X_train, _ = eliminate_features(X_train, to_elim, feature_labels)
        # X_test, feature_labels = eliminate_features(X_test, to_elim, feature_labels)

        self.set_data(X_train, y_train, X_test, y_test, feature_labels) #original


    def train_cls(self, save_trained_model=False):
        # works on internal classifier and sorts features
        self.cls_work = train_dt(self.X_train_work, self.y_train, self.output_dim, self.target, self.depth_tree, save_trained_model, self.classes, self.feature_labels_work, self.ccp_alpha)
        self.cls_feature_labels_work_sorted = [x for _,x in sorted(zip(self.cls_work.feature_importances_, self.feature_labels_work))][::-1]


    # def eval_cls_old(self):
    #     # works on internal cls and saves internally
    #     classifier = self.cls_work
    #     # Evaluation on test data
    #     conf_matrix = test_dt(self.X_test_work, self.y_test, classifier, self.output_dim)
    #     res = get_tp_fp_fn(conf_matrix, self.fault_class_dict)
    #     tp_sdc, tp_cat, tp_cls = res['tp_sdc'], res['tp_cat'], res['tp_cls']
    #     fp, fn, tn = res['fp'], res['fn'], res['tn']
    #     # # conf rates (only within sdc "block")
    #     # sdc_cnt = np.sum(conf_matrix) - fp - fn - tn

    #     # Prediction precision, recall
    #     p, r = get_p_r(tp_sdc, fp, fn)
        
    #     acc_cls, acc_cat = get_class_acc(self.fault_class_dict, conf_matrix)
    #     acc_cls_m = np.mean(list(acc_cls.values())[1:]) #mean across all classes

    #     coll_res = {'p': p, 'r': r, 'acc_cls_m': acc_cls_m, 'acc_cls': acc_cls, 'acc_cat': acc_cat, 'meta': [float(classifier.get_depth()), float(classifier.get_n_leaves()), float(classifier.tree_.node_count)], 'conf_mat': conf_matrix.tolist()}
    #     print('precision', p, 'recall', r, 'avg acc_cls', acc_cls_m)

    #     self.cls_eval_work = coll_res

    
    def eval_cls(self):
        # works on internal cls and saves internally
        classifier = self.cls_work
        # Evaluation on test data
        conf_matrix = test_dt(self.X_test_work, self.y_test, classifier, self.output_dim)

        cls_mapping, cats_mapping, sdc_mapping = get_flt_dicts()
        cls_mapping = list(cls_mapping.values())
        cats_mapping = list(cats_mapping.values())
        sdc_mapping = list(sdc_mapping.values())


        p_cls, r_cls = get_tpfpfn_new(cls_mapping, conf_matrix)
        p_cats, r_cats = get_tpfpfn_new(cats_mapping, conf_matrix)
        p_sdc, r_sdc = get_tpfpfn_new(sdc_mapping, conf_matrix)

        # res = get_tp_fp_fn(conf_matrix, self.fault_class_dict)
        # tp_sdc, tp_cat, tp_cls = res['tp_sdc'], res['tp_cat'], res['tp_cls']
        # fp, fn, tn = res['fp'], res['fn'], res['tn']
        # # # conf rates (only within sdc "block")
        # # sdc_cnt = np.sum(conf_matrix) - fp - fn - tn

        # # Prediction precision, recall
        # p, r = get_p_r(tp_sdc, fp, fn)
        
        # acc_cls, acc_cat = get_class_acc(self.fault_class_dict, conf_matrix)
        # acc_cls_m = np.mean(list(acc_cls.values())[1:]) #mean across all classes

        
        coll_res = {'p_sdc': p_sdc, 'r_sdc': r_sdc, 'p_cats': p_cats, 'r_cats': r_cats, 'p_cls': p_cls, 'r_cls': r_cls, 'acc_cls_m': [], 'acc_cls': [], 'acc_cat': [], 'meta': [float(classifier.get_depth()), float(classifier.get_n_leaves()), float(classifier.tree_.node_count)],\
             'conf_mat': conf_matrix.tolist()}
        print('precision (cls, cats, sdc)', p_cls, p_cats, p_sdc, 'recall', r_cls, r_cats, r_sdc)
        print('tree attributes:', coll_res['meta'], 'conf_mat', coll_res['conf_mat'])

        self.cls_eval_work = coll_res    


    def save_cls_to_ref(self):
        self.cls_ref = deepcopy(self.cls_work)
        self.cls_eval_ref = deepcopy(self.cls_eval_work)
        self.cls_feature_labels_ref_sorted = deepcopy(self.cls_feature_labels_work_sorted)

    def save_eval_to_list(self):
        # Saves everyting except for ft_lists!
        coll_res = self.cls_eval_work
        if coll_res is None:
            print('evaluation is still None, intended?')
            return
        self.p_cls_list_work.append(coll_res['p_cls'])
        self.r_cls_list_work.append(coll_res['r_cls'])
        self.p_cats_list_work.append(coll_res['p_cats'])
        self.r_cats_list_work.append(coll_res['r_cats'])
        self.p_sdc_list_work.append(coll_res['p_sdc'])
        self.r_sdc_list_work.append(coll_res['r_sdc'])
        # self.acc_cls_list_work.append(coll_res['acc_cls'])
        # self.acc_cat_list_work.append(coll_res['acc_cat'])
        self.classifier_meta_work.append(coll_res['meta'])
        # self.ft_list_work.append(feature_labels)
        self.conf_mat_list_work.append(coll_res['conf_mat'])

    def elim_work_ft(self, to_elim):
        self.X_train_work, _ = eliminate_features(self.X_train_work, to_elim, self.feature_labels_work)
        self.X_test_work, self.feature_labels_work = eliminate_features(self.X_test_work, to_elim, self.feature_labels_work)

    def save_work_lists_to_exp_list(self):
        self.result_dict[self.exp]['p_cls'].append(deepcopy(self.p_cls_list_work))
        self.result_dict[self.exp]['r_cls'].append(deepcopy(self.r_cls_list_work))
        self.result_dict[self.exp]['p_cats'].append(deepcopy(self.p_cats_list_work))
        self.result_dict[self.exp]['r_cats'].append(deepcopy(self.r_cats_list_work))
        self.result_dict[self.exp]['p_sdc'].append(deepcopy(self.p_sdc_list_work))
        self.result_dict[self.exp]['r_sdc'].append(deepcopy(self.r_sdc_list_work))
        # self.result_dict[self.exp]['acc_cls'].append(self.acc_cls_list_work)
        # self.result_dict[self.exp]['acc_cat'].append(self.acc_cat_list_work)
        self.result_dict[self.exp]['meta'].append(deepcopy(self.classifier_meta_work))
        self.result_dict[self.exp]['ft'].append(deepcopy(self.ft_list_work))
        self.result_dict[self.exp]['conf_mat'].append(deepcopy(self.conf_mat_list_work))
        self.result_dict[self.exp]['due'] = deepcopy(self.due_rate) #no need to append as it is a global value
        






def main():
    # Fine
    ##########################################################################################################
    df = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti', 'resnet_imagenet', 'alexnet_imagenet']
    # df = ['ssd_kitti']
    # # # ccp_alphas = [1.5e-5, 1.5e-5, 0.5e-5, 1.e-5, 8.e-5, 2.e-5, 1.e-5, 1.e-5] #old hooks
    # ccp_alphas = [1.e-5, 1.e-5, 0.5e-5, 1.e-5, ?e-5, ?e-5, 1.5e-5, 0.5e-5] #new 
    # ccp_alphas = [1.0e-5, 1.5e-5, 1.0e-5, 1.0e-5, 10.e-5, 1.0e-5, 1.0e-5, 1.0e-5] #new new
    ccp_alphas = [1.5e-5, 1.5e-5, 1.0e-5, 1.5e-5, 2.0e-5, 1.5e-5, 2.0e-5, 1.0e-5] #new new new
    # ccp_alphas = [1.5e-5]

    # df = ['yolo_coco']
    # ccp_alphas = [1.e-5] #new new
    N = 10 #10 # no of test samples for error
    
    ##########################################################################################################

    depth_tree = None #10 #None
    # thres = 1 #maximum nr of layers that are allowed, in percent! Here means only that search granularity is changed.
    sv_path = "/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/"
    file_base_name = "dt_train_ft_red_new_final2" #part of save name
    # Selected data
    target = 'all' #'noise', 'blur', 'blur_noise', 'neurons', 'weights', 'neurons_weights', 'all'
    save_trained_model = False
    fine_tune_thres = 20

    for ex in range(len(df)):
        exp = df[ex] #experiment
        ccp_alpha = ccp_alphas[ex] #pruning value
        # if 'ssd' in exp:
        #     fine_tune_thres = 40
        # else:
        #     fine_tune_thres = 20

        dt_ex = DT_trainer(exp, ccp_alpha, sv_path, file_base_name, depth_tree = depth_tree) #init/overwrite
        dt_ex.load_exp_data(target=target) #load data
        dt_ex.reset_data_work()
        
        # Stat loop ###################################################################
        for rep in range(N):
            print('RUN', exp, 'NUMBER', rep)

            # Reset for new attempt
            dt_ex.reset_data_work()
            dt_ex.reset_res_list_work()

            # Create baseline
            dt_ex.train_cls() #saves to internal self.cls_work
            dt_ex.eval_cls() #saves to internal self.cls_eval_work
            dt_ex.save_cls_to_ref()
            # p_ref,r_ref = dt_ex.cls_eval_ref['p_cls'], dt_ex.cls_eval_ref['r_cls']
            print('Starting with number of features:', len(dt_ex.cls_feature_labels_ref_sorted))
            
            # Run search 
            n = 1
            dn = 1
            # found_thres = False
            while True:
                dt_ex.reset_data_work() #set back work to original

                # Choose subset of most important features in ref
                fts_to_keep = deepcopy(dt_ex.cls_feature_labels_ref_sorted[:n]) #take next one from ref stack
                to_elim = substract_list(dt_ex.cls_feature_labels_ref_sorted, fts_to_keep) #
                print('train with number of features:', len(fts_to_keep))
                print('and keeping:', fts_to_keep)

                dt_ex.elim_work_ft(to_elim) #update work features and X_train, X_test

                # Training ---------------------------------------------------------------------------------------------
                dt_ex.train_cls()
                dt_ex.eval_cls()
                dt_ex.save_eval_to_list() #everything except for features
                dt_ex.ft_list_work.append(deepcopy(dt_ex.cls_feature_labels_work_sorted)) #gets sorted after training

                # p,r = dt_ex.cls_eval_work['p_cls'], dt_ex.cls_eval_work['r_cls']

                
                # if not found_thres and (p > p_ref - thres/100. and r > r_ref - thres/100.):
                #     best_fts = deepcopy(dt_ex.feature_labels_work)
                #     found_thres = True
                #     dn = 10 #increase to steps of 10
                # elif found_thres and (p > p_ref - 0.5*thres/100. and r > r_ref - 0.5*thres/100.):
                #     dn = 100
                if len(dt_ex.feature_labels_work) >= fine_tune_thres: 
                    dn = 10 #increase to steps of #10
                if len(dt_ex.feature_labels_work) >= 100: #100
                    dn = 200 #200

                n += dn

                if n >= len(dt_ex.cls_feature_labels_ref_sorted):
                    # Add ref point (full features) last and exit
                    dt_ex.cls_eval_work = dt_ex.cls_eval_ref
                    dt_ex.save_eval_to_list()
                    dt_ex.ft_list_work.append(deepcopy(dt_ex.cls_feature_labels_ref_sorted))
                    break

            # Save trial:
            dt_ex.save_work_lists_to_exp_list() #Save results



            # # # Find alternatives:  ###################################################################
            # # print('FIND alternatives:')
            # # to_elim_prev = best_fts #dt_ex.feature_labels_work #take from last step above

            # # dt_ex = find_alternative_wrt_previous(dt_ex, to_elim_prev, p_ref, r_ref, acc_cls_m_ref, thres)
            # # p,r,acc_cls_m = dt_ex.cls_eval_work['p'], dt_ex.cls_eval_work['r'], dt_ex.cls_eval_work['acc_cls_m']
            # # print('Found alternative with', p, r, acc_cls_m, "and nr of features", len(dt_ex.cls_feature_labels_work_sorted))
            # # print('Features', dt_ex.cls_feature_labels_work_sorted)

            # # # Save only final features and metrics
            # # dt_ex.result_dict[dt_ex.exp]['min_alternatives'].append({'fts': dt_ex.cls_feature_labels_work_sorted, 'p': p, 'r': r, 'acc_cls_m': acc_cls_m})



        with open(dt_ex.result_json_path, 'w') as outfile:
                json.dump(dt_ex.result_dict, outfile)
        print('Saved results to:', dt_ex.result_json_path)


if __name__ == "__main__":
    main()
