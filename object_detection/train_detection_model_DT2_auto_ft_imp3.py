
import json
import numpy as np
# from sklearn.model_selection import train_test_split
from copy import deepcopy
from train_detection_model_LR3 import eliminate_features, load_json_indiv, load_data_from_quantile_data, create_feature_labels, elim_by_pattern, get_p_r, get_balance2, substract_list
# from train_detection_model_DT2 import test_dt, train_dt, get_tp_fp_fn, get_class_acc
import os
from tabulate import tabulate
from attic.check_dt_train_results_ft_red import get_acc_avgs, get_m_err, get_ind_thres2
from train_detection_model_DT2_auto_ft_red3 import DT_trainer #, find_alternative_wrt_previous
from attic.check_dt_train_results_ft_red import invert_list_orders

# https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
# Feature importance is exp(weights), because of logistic regression sigmoid unit


def acc_avg_single(dct_list):
    sample_list = []
    for dct in dct_list:
        sample_list.append(np.mean(list(dct.values())[1:]))
    return sample_list


def find_alternative_wrt_previous(dt_ex, to_elim_prev, p_ref, r_ref, thres):

    # Build new reference:
    dt_ex.reset_data_work()
    dt_ex.elim_work_ft(to_elim_prev) #use all fts minus the ones that were best before
    print('Alternative: Retraining without previous best features:', len(dt_ex.feature_labels_work))
    dt_ex.train_cls() 
    dt_ex.save_cls_to_ref() #new ref

    # Search for comparable set of features
    n = 1
    while True:
        dt_ex.reset_data_work() #set back work to original
        dt_ex.elim_work_ft(to_elim_prev)

        # Choose subset of most important features in ref
        fts_to_keep = dt_ex.cls_feature_labels_ref_sorted[:n] #take next one from ref stack
        to_elim = substract_list(dt_ex.cls_feature_labels_ref_sorted, fts_to_keep) #
        print('train with number of features:', len(fts_to_keep))
        print('and keeping:', fts_to_keep)

        if len(fts_to_keep) > 20:
            print('More than 20 features, aborting...')
            break


        dt_ex.elim_work_ft(to_elim) #update work features and X_train, X_test

        # Training ---------------------------------------------------------------------------------------------
        dt_ex.train_cls()
        dt_ex.eval_cls()
        # dt_ex.save_eval_to_list() #everything except for features
        # dt_ex.ft_list_work.append(dt_ex.cls_feature_labels_work_sorted) #gets sorted after training

        p,r = dt_ex.cls_eval_work['p_cls'], dt_ex.cls_eval_work['r_cls']

        # Compare
        if (p > p_ref*thres and r > r_ref*thres):
            break
        else:
            n += 1

    return dt_ex


def main():
    ##########################################################################################################
    df = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti', 'resnet_imagenet', 'alexnet_imagenet']
    # # ccp_alphas = [1.0e-5, 1.5e-5, 1.0e-5, 1.0e-5, 10.e-5, 1.0e-5, 1.0e-5, 1.0e-5] #new new
    ccp_alphas = [1.5e-5, 1.5e-5, 1.0e-5, 1.5e-5, 2.0e-5, 1.5e-5, 2.0e-5, 1.0e-5] #new new new

    df = ['retina_coco']
    ccp_alphas = [2.0e-5] #new new new

    thres = 0.95 #maximum nr of layers that are allowed, in percent!
    ##########################################################################################################

    # Save to
    sv_path = "/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/"
    file_base_name = "dt_train_ft_unq2" #to save
    to_load_name = "dt_train_ft_red_new_final2" #to load
    
    # # Selected data
    # target = 'all' #'noise', 'blur', 'blur_noise', 'neurons', 'weights', 'neurons_weights', 'all'
    # save_trained_model = False
    # depth_tree = None

    for ex in range(len(df)):

        exp = df[ex] #experiment
        ccp_alpha = ccp_alphas[ex] #pruning value

        # # Load data
        dt_ex = DT_trainer(exp, ccp_alpha, sv_path, file_base_name) #init/overwrite
        dt_ex.load_exp_data() #load data
        dt_ex.reset_data_work()

        # # Get results from previous exploration:
        data_ft = load_json_indiv(sv_path + to_load_name+ "_" + exp + ".json") #get data from ft reduction run


        # Order list for indix threshold:
        assert len(list(data_ft.keys()))==1, 'More than one model per file, is that intended?'
        if len(data_ft[exp]['ft'])==0:
            print('No features found.')
            continue
        invert_list_orders(data_ft[exp]) #invert order because previously it was from large to small, now the other way round


        N = len(data_ft[exp]['p_cls'])
        # N = 1 #TODO: test only
        # # # Get avgs
        # # p_m, p_err = get_m_err(data[exp]['p'])
        # # r_m, r_err = get_m_err(data[exp]['r'])
        # # acc_cls_m, acc_cls_err = get_acc_avgs(data[exp]['acc_cls'])
        # # # acc_cat_m, acc_cat_err = get_acc_avgs(data[exp]['acc_cat'])
        
        
        # Stat loop over instances ###################################################################
        for rep in range(N):
            print('RUN', exp, 'NUMBER', rep)

            # find best features from previous results
            p_cls, r_cls = data_ft[exp]['p_cls'][rep], data_ft[exp]['r_cls'][rep]
            # p_cat, r_cat= data_ft[exp]['p_cat'][rep], data_ft[exp]['r_cat'][rep]
            # p_sdc, r_sdc= data_ft[exp]['p_sdc'][rep], data_ft[exp]['r_sdc'][rep]
            # acc_cls_m = acc_avg_single(data_ft[exp]['acc_cls'][rep])
            ind = get_ind_thres2(p_cls, r_cls, p_err= None, r_err=None, thres_perc=thres, thres_abs=None) #, acc_cls_m[::-1]) #index for the reversed order!
            best_fts = data_ft[exp]['ft'][rep][ind]

            # skip if that compressed model is the same as before
            if len(dt_ex.result_dict[dt_ex.exp]['min_alternatives']) > 0:
                if set(best_fts) in [set(x['fts']) for x in dt_ex.result_dict[dt_ex.exp]['min_alternatives']]: #test with set
                    print(best_fts, 'exists already in', [x['fts'] for x in dt_ex.result_dict[dt_ex.exp]['min_alternatives']], ', skipping...')
                    continue
            print('previous best', best_fts)

            # Find ref values from previous results
            p_ref, r_ref = p_cls[0], r_cls[0]
            dt_ex.result_dict[dt_ex.exp]['min_alternatives'].append({'fts': best_fts, 'p_cls': p_cls[ind], 'r_cls': r_cls[ind]}) #save again
            

            # Find alternatives:  ###################################################################
            print('FIND alternatives:')
            dpth = 24 #12
            to_elim_prev = deepcopy(best_fts) #dt_ex.feature_labels_work #take from last step above

            for d in range(dpth):
                print('depth trial', d)
                # Elim all of the best features:
                dt_ex = find_alternative_wrt_previous(dt_ex, to_elim_prev, p_ref, r_ref, thres)
                if dt_ex.cls_eval_work is None:
                    continue
                p_cls, r_cls = dt_ex.cls_eval_work['p_cls'], dt_ex.cls_eval_work['r_cls']
                # p_cat, r_cat= data_ft[exp]['p_cats'][rep], data_ft[exp]['r_cats'][rep]
                # p_sdc, r_sdc= data_ft[exp]['p_sdc'][rep], data_ft[exp]['r_sdc'][rep]
                print('Found alternative with', p_cls, r_cls, "and nr of features", len(dt_ex.cls_feature_labels_work_sorted))
                print('Features', dt_ex.cls_feature_labels_work_sorted)

                # Save only final features and metrics
                dt_ex.result_dict[dt_ex.exp]['min_alternatives'].append({'fts': deepcopy(dt_ex.cls_feature_labels_work_sorted), 'p_cls': deepcopy(p_cls), 'r_cls': deepcopy(r_cls)})

                to_elim_prev.extend(dt_ex.cls_feature_labels_work_sorted) #here fix


        with open(dt_ex.result_json_path, 'w') as outfile:  
                json.dump(dt_ex.result_dict, outfile)
        print('Saved results to:', dt_ex.result_json_path)



if __name__ == "__main__":
    main()
