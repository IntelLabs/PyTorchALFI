from train_detection_model_LR3 import load_json_indiv
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter
from train_detection_model_LR3 import get_tpfpfn_new, get_flt_dicts


# def get_mean_err2(x_list):
#     p_m = np.mean(np.array(x_list),0)
#     p_err = np.std(np.array(x_list),0)*1.96/np.sqrt(len(np.array(x_list)))
#     # get_m_err()
#     # return p_m, p_err
#     return [round(x*100,2) for x in p_m], [round(x*100,2) for x in p_err]


def get_m_err(x_list):
    m, err =  np.mean(np.array(x_list), 0), np.std(np.array(x_list), 0)*1.96/np.sqrt(len(x_list)) #95% CI
    # m, err = np.mean(np.array(x_list), 0), np.max(np.abs(np.array(x_list) - np.mean(x_list, 0)), 0) #max error
    try:
        return [round(x*100,1) for x in m], [round(x*100,1) for x in err]
    except: 
        return [round(m*100, 1), round(err*100, 1)]

def save_fig(fig_p, ax_p, ylabel, ft_xrange, ft_yrange, fntsize, pth):
    
    ax_p.set_xlabel('No of features', fontsize = fntsize)
    if ylabel == "Precision":
        ylabel_pic = r'$P_{cls}$'
    elif ylabel == "Recall":
        ylabel_pic = r'$R_{cls}$'
    ax_p.set_ylabel(ylabel_pic, fontsize = fntsize)

    # import math
    # new_list = range(math.floor(min(ft_xrange)), math.ceil(max(ft_xrange))+1)
    # ax_p.xticks(new_list)
    if ft_xrange is not None:
        ax_p.set_xlim(ft_xrange)
    ax_p.invert_xaxis()
    if ft_yrange is not None:
        ax_p.set_ylim(ft_yrange)
    fig_p.legend(bbox_to_anchor=(0.15, 0.45), loc='upper left', borderaxespad=0, fontsize=fntsize-3) #
    
    ax_p.tick_params(axis = 'both', labelsize = fntsize)

    save_name = pth + "features_" + ylabel.lower() + ".png"
    fig_p.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
    print('Saved as', save_name)


def flatten_list(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


def get_ft_ind(p_m, thres_perc=None, thres_abs=None, p_err=None):
    if thres_perc is not None and thres_abs is None:
        p_ref = p_m[0]
        thres = thres_perc
    elif thres_perc is None and thres_abs is not None:
        p_ref = thres_abs
        thres = 0.
    else: 
        print('Both or none of relative and absolute thres set, please decide.')
        return None

    if p_err is None:
        # ind = np.where(np.array(p_m)<p_ref-thres)
        ind = np.where(np.array(p_m)<p_ref*thres) 
    else:
        err_ref = p_err[0]
        # ind = np.where(np.array(p_m)+ np.array(p_err)<p_ref-err_ref-thres)
        ind = np.where(np.array(p_m) + np.array(p_err) < p_ref*thres - err_ref)
    if len(ind[0]) > 0:
        return ind[0][0] #first index
    else:
        return np.Inf #inf because this is no problem


def get_acc_avgs(dict_input):
    acc_cls_ar = np.array(dict_input) #data[x]['acc_cls']
    acc_cls_m = []
    acc_cls_err = []
    for l in range(acc_cls_ar.shape[1]): #loop of steps
        sample_lst = []

        for u in range(acc_cls_ar.shape[0]): #loop of samples
            sample_lst.append(list(acc_cls_ar[u][l].values())[1:]) #mean cls acc samples, steps. First is no_sdc, doesnt count
        avg_ft_mode = np.mean(sample_lst, 1) #mean of 5 fault modes

        m, err = get_m_err(avg_ft_mode)
        # m, err = np.mean(avg_ft_mode), np.std(avg_ft_mode)*1.96/np.sqrt(len(avg_ft_mode))
        # acc_cls_m.append(round(m*100,2)) #mean of 3 (N) samples 
        # acc_cls_err.append(round(err*100,2)) #err of 3 (N) samples 
        acc_cls_m.append(m) #mean of 3 (N) samples 
        acc_cls_err.append(err) #err of 3 (N) samples 

    return acc_cls_m, acc_cls_err


def print_model_info(ind, nr_features, nr_used_layers, nr_used_layers_mean, cap, model_data, p_m, p_err, r_m, r_err, acc_cls_m, acc_cls_err, thres):
    # P, R, Acc
    print('P', p_m[ind], p_err[ind], 'R', r_m[ind], r_err[ind], 'Acc_cls', acc_cls_m[ind], acc_cls_err[ind])

    # Layers
    how_many_features = nr_features[ind]
    how_many_layers = nr_used_layers[ind]
    # which_features = cap[ind]
    print('For a drop of %', thres, 'Resulting min no of features:', how_many_features, 'from no of layers:', how_many_layers, 'on avg:', nr_used_layers_mean[ind])

    # Model meta
    avg_meta = np.mean(model_data['meta'], 0)[ind] #depth, leaves, nodes
    err_meta = np.std(model_data['meta'], 0)[ind]*1.96/(np.array(model_data['meta']).shape[0])
    print('avg meta (depth, leaves, nodes)', avg_meta, err_meta)
    
    # Confusion matrix
    avg_conf_mat = np.mean(model_data['conf_mat'], 0)[ind]
    print('avg conf matrix', tabulate(avg_conf_mat))


def print_model_info2(ind, nr_features, nr_used_layers, nr_used_layers_mean, cap, model_data, thres):
    # P, R, Acc
    p_m, p_err = get_m_err(model_data['p_cls'])
    r_m, r_err = get_m_err(model_data['r_cls'])
    print('P_cls', p_m[ind], p_err[ind], 'R_cls', r_m[ind], r_err[ind])
    p_m, p_err = get_m_err(model_data['p_cats'])
    r_m, r_err = get_m_err(model_data['r_cats'])
    print('P_cats', p_m[ind], p_err[ind], 'R_cats', r_m[ind], r_err[ind])
    p_m, p_err = get_m_err(model_data['p_sdc'])
    r_m, r_err = get_m_err(model_data['r_sdc'])
    print('P_sdc', p_m[ind], p_err[ind], 'R_sdc', r_m[ind], r_err[ind])

    # # Layers
    # how_many_features = nr_features[ind]
    # how_many_layers = nr_used_layers[ind]
    # # which_features = cap[ind]
    # print('For a drop of %', thres, 'Resulting min no of features:', how_many_features, 'from no of layers:', how_many_layers, 'on avg:', nr_used_layers_mean[ind])

    # Model meta
    avg_meta = np.mean(model_data['meta'], 0)[ind] #depth, leaves, nodes
    err_meta = np.std(model_data['meta'], 0)[ind]*1.96/(np.array(model_data['meta']).shape[0])
    print('avg meta (depth, leaves, nodes)', avg_meta, err_meta)
    
    # Confusion matrix
    avg_conf_mat = np.mean(model_data['conf_mat'], 0)[ind]
    print('avg conf matrix', tabulate(avg_conf_mat))

    cls_mapping, cats_mapping, sdc_mapping = get_flt_dicts()
    cls_mapping = list(cls_mapping.values())
    cats_mapping = list(cats_mapping.values())
    sdc_mapping = list(sdc_mapping.values())
    get_tpfpfn_new(cls_mapping, avg_conf_mat)

    # # Best features:
    # if len(cap[ind][0]) < 50: #not for full model
    # #     print('example feature combo:', cap[ind][0])
    #     # combos = np.unique([set(a) for a in cap[ind]]).tolist()
    #     combos = [set(a) for a in cap[ind]]
    #     # print('features common to all options:', set.intersection(*combos))

    #     freqs = Counter(frozenset(sub) for sub in combos)
    #     res = [key for key, val in freqs.items() if val > 1]
    #     combos = [set(x) for x in res]
    #     print('Unique feature combos:', combos)

def get_p_r_tnr(tp, fp, fn, tn):
    if tp+fp>0:
        p = tp/(tp+fp)
    else:
        p = None
    if tp+fn>0:
        r = tp/(tp+fn)
    else:
        r = None
    if tn+fp>0:
        tnr = tn/(tn+fp)
    else:
        tnr = None
    # tpr = recall
    return p,r, tnr

def get_tpfpfntn(mpg, conf_matrix):
    pr_list_fmodes = []
    for n in range(max(mpg)+1):
        if n == 0:
            continue #this is only non-sdc -> no pr wanted
        ind = np.where(np.array(mpg) == n)
        row = conf_matrix[ind,:]
        col = conf_matrix[:, ind]
        diag = conf_matrix[np.min(ind):np.max(ind)+1, np.min(ind):np.max(ind)+1]
        tp = diag.sum()
        fp = col.sum() - tp
        fn = row.sum() - tp
        tn = conf_matrix.sum() - tp - fp - fn

        p,r, tnr = get_p_r_tnr(tp, fp, fn, tn)
        pr_list_fmodes.append([p,r, tnr])

    pr_list_fmodes = [n for n in pr_list_fmodes if n[0] is not None]

    p_m = np.mean([x[0] for x in pr_list_fmodes])
    r_m = np.mean([x[1] for x in pr_list_fmodes])
    tnr_m = np.mean([x[2] for x in pr_list_fmodes])
    # print('p modes, r modes, tnr modes', pr_list_fmodes)
    return p_m, r_m, tnr_m

def print_model_info3(ind, model_data):
    # P, R, Acc
    p_m, p_err = get_m_err(model_data['p_cls'])
    r_m, r_err = get_m_err(model_data['r_cls'])
    print('P_cls', p_m[ind], p_err[ind], 'R_cls', r_m[ind], r_err[ind])
    p_m, p_err = get_m_err(model_data['p_cats'])
    r_m, r_err = get_m_err(model_data['r_cats'])
    print('P_cats', p_m[ind], p_err[ind], 'R_cats', r_m[ind], r_err[ind])
    p_m, p_err = get_m_err(model_data['p_sdc'])
    r_m, r_err = get_m_err(model_data['r_sdc'])
    print('P_sdc', p_m[ind], p_err[ind], 'R_sdc', r_m[ind], r_err[ind])

    # Model meta
    avg_meta = np.mean(model_data['meta'], 0)[ind] #depth, leaves, nodes
    err_meta = np.std(model_data['meta'], 0)[ind]*1.96/(np.array(model_data['meta']).shape[0])
    print('avg meta (depth, leaves, nodes)', avg_meta.round(), err_meta.round())
    
    # Confusion matrix
    avg_conf_mat = np.mean(model_data['conf_mat'], 0)[ind]
    print('avg conf matrix', tabulate(avg_conf_mat))
    mcl_rate = (avg_conf_mat.sum()-np.sum(np.diag(avg_conf_mat)))/(avg_conf_mat.sum())
    print('misclassification rate:', mcl_rate)

    cls_mapping, cats_mapping, sdc_mapping = get_flt_dicts()
    cls_mapping = list(cls_mapping.values())
    cats_mapping = list(cats_mapping.values())
    sdc_mapping = list(sdc_mapping.values())
    p_m, r_m, tnr_m = get_tpfpfntn(cls_mapping, avg_conf_mat)
    print('p_m', round(p_m*100, 1), 'r_m', round(r_m*100,1), 'tnr_m', round(tnr_m*100,1))

    # layers
    cap = [[] for n in range(len(model_data['ft'][0]))]
    for n in range(len(model_data['ft'])):
        for m in range(len(model_data['ft'][n])):
            cap[m].append(model_data['ft'][n][m])

    # nr_features = np.array([len(cap[n][0]) for n in range(len(cap))])

    no_lays = []
    no_fts = []
    for n in range(len(model_data['p_cls'])):
        # p_cls = data[x]['p_cls'][n]
        # r_cls = data[x]['r_cls'][n]

        cap_n = [x[n] for x in cap]
        # cap_n[ind_m]
        # ind_1 = get_ind_thres2(p_cls, r_cls, p_err=None, r_err=None, thres_perc=thres_perc, thres_abs=thres_abs
        # print('Reduced to', 'p', p_cls[ind_1], p_cls[0], 'r', r_cls[ind_1], r_cls[0])
        # assert p_cls[ind_1]>=p_cls[0]*thres_perc
        # assert r_cls[ind_1]>=r_cls[0]*thres_perc
        # cap_n[ind_1]
        no_fts.append(len(cap_n[ind]))
        lays = np.unique([m.split("_")[-1] for m in cap_n[ind]]).tolist()
        no_lays.append(len(lays))

    print('lays', np.mean(no_lays), np.std(no_lays)*1.96/np.sqrt(len(no_lays)))
    print('fts', np.mean(no_fts), np.std(no_fts)*1.96/np.sqrt(len(no_fts)))



def get_ind_thres2(p_m, r_m, p_err=None, r_err=None, thres_perc=None, thres_abs=None):
    ind_p = get_ft_ind(p_m, thres_perc, thres_abs, p_err)
    ind_r = get_ft_ind(r_m, thres_perc, thres_abs, r_err)
    # ind_acc = get_ft_ind(acc_cls_m, thres)
    ind = np.min([ind_p, ind_r])
    if np.isinf(ind):
        return None
    else:
        return int(ind) - 1
    # ind = int(np.min([ind_p, ind_r]))
    # if np.isinf(ind):
    #     return ind
    # else:
    #     return ind - 1 #so not yet dropped by thres

def find_unique_combos(cap_ind):
    uq = []
    for n in cap_ind:
        if n not in uq: #cap is sorted already
            uq.append(n)

    ft_list = []
    cnt_list = []
    for x in range(len(uq)):
        for y in range(len(uq[x])):
            ft_ind = np.where(uq[x][y] == np.array(ft_list))[0]
            if len(ft_ind) > 0:
                cnt_list[ft_ind[0]] += 1
            else:
                ft_list.append(uq[x][y])
                cnt_list.append(1)

    table_1 = [[val for (_, val) in sorted(zip(cnt_list, ft_list), key=lambda x: x[0])][::-1], [val for (val,_) in sorted(zip(cnt_list, ft_list), key=lambda x: x[0])][::-1]]

    return uq, table_1


def invert_list_orders(data_x):
    for n in list(data_x.keys()):
        if isinstance(data_x[n], list) and len(data_x[n]) > 0:
            for m in range(len(data_x[n])):
                data_x[n][m] = data_x[n][m][::-1]


def main():
    ################################################################################
    pth = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/'


    data_list = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti', 'resnet_imagenet', 'alexnet_imagenet']
    # data_list = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'resnet_imagenet', 'alexnet_imagenet']
    data_list = ['alexnet_imagenet']

    fl_names = 'red_new_final2'

    #NOTE: Idea: reduce until 95% instead of 1%?
    ################################################################################


    fig_p = plt.figure()
    ax_p=fig_p.gca()
    fig_r = plt.figure()
    ax_r=fig_r.gca()
    # fig_acc = plt.figure()
    # ax_acc=fig_acc.gca()
    fntsize = 12

    for d in data_list:
        print('Evaluating', d)
        # data = load_json_indiv(pth + "dt_train_ft_red_by_layer_" + d + ".json")
        data = load_json_indiv(pth + "dt_train_ft_" + fl_names + "_" + d + ".json")

        for x in list(data.keys()):
            assert len(list(data.keys()))==1, 'More than one model per file, is that intended?'
            
            invert_list_orders(data[x]) #invert order because previously it was from large to small, now the other way round

            #in case there is an empty ft key:
            if len(data[x]['ft'])==0:
                continue

            # check due:
            print('DUE detection (for classifier) in %', round(data[x]['due']*100,1))


            # Averages overa all attempts
            p_m, p_err = get_m_err(data[x]['p_cls'])
            r_m, r_err = get_m_err(data[x]['r_cls'])

            # Get number of features:
            nr_features = [len(n) for n in data[x]['ft'][0]]


            # Plot averages
            ax_p.errorbar(nr_features, p_m, yerr=p_err, capsize=3, label=x) #color='k', fmt="o", markersize='3'
            ax_r.errorbar(nr_features, r_m, yerr=r_err, capsize=3, label=x) #color='k', fmt="o", markersize='3' 
            # ax_acc.errorbar(nr_features, acc_cls_m, yerr=acc_cls_err, capsize=3, label=x) #yerr=acc_err,
            # ax_acc.errorbar(nr_features, acc_cat_m, yerr=acc_cat_err, capsize=3, label=x)

            print('model', x, 'with max no of layers', int(nr_features[0]/11)) #10 quantiles


            # Find reduced model by step in curves: --------------------------------
            ########################################################################
            thres_perc = 0.95 #3 #None #1. #None #1.
            thres_abs = None #90. #90. #percent drop
            ########################################################################
            ind_m = get_ind_thres2(p_m, r_m, p_err=None, r_err=None, thres_perc=thres_perc, thres_abs=thres_abs) #index where on average p,r have dropped to 95%


            # Print reduced model:
            print('full')
            print_model_info3(0, data[x])

            print('after ' + str(thres_perc) + '%', str(thres_abs) + ' limit')
            print_model_info3(ind_m, data[x])






    ft_xrange = [0,60]
    ft_yrange = None #[50, 100]


    save_fig(fig_p, ax_p, 'Precision', ft_xrange, ft_yrange, fntsize, pth)
    save_fig(fig_r, ax_r, 'Recall', ft_xrange, ft_yrange, fntsize, pth)
    # save_fig(fig_acc, ax_acc, 'Acc_class', ft_xrange, ft_yrange, fntsize, pth)


if __name__ == "__main__":
    main()




# # Compress model --------------
# from train_detection_model_DT2 import get_data, train_dt
# from train_detection_model_LR3 import eliminate_features, substract_list
# X_train, X_test, y_train, y_test, feature_labels, output_dim, classes, fault_class_dict = get_data([d])

# cmbs = np.unique(np.array(cap[ind_1])).tolist()
# if isinstance(cmbs[0], list):
#     fts_list = cmbs
# else:
#     fts_list = [cmbs]

# for fts_to_keep in fts_list:
#     to_elim = substract_list(feature_labels, fts_to_keep) #
#     X_train, _ = eliminate_features(X_train, to_elim, feature_labels)
#     X_test, feature_labels = eliminate_features(X_test, to_elim, feature_labels)
#     print('features used', feature_labels)

# ccp_alpha = 1.e-4
# classifier = train_dt(X_train, y_train, output_dim, 'all', None, True, classes, feature_labels, ccp_alpha)

# p_m[0]*thres_perc
# r_m[0]*thres_perc
# p_m[ind_1]
# r_m[ind_1]
