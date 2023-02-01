from train_detection_model_LR3 import load_json_indiv
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from attic.check_dt_train_results_ft_red import get_acc_avgs, get_m_err, flatten_list
from copy import deepcopy
import numpy as np
# import numpy.random
import matplotlib.pyplot as plt
from matplotlib import cm

# def get_m_err(x_list):
#     m, err =  np.mean(np.array(x_list), 0), np.std(np.array(x_list), 0)*1.96/np.sqrt(len(x_list)) #95% CI
#     # m, err = np.mean(np.array(x_list), 0), np.max(np.abs(np.array(x_list) - np.mean(x_list, 0)), 0) #max error
#     try:
#         return [round(x*100,1) for x in m], [round(x*100,1) for x in err]
#     except: 
#         return [round(m*100, 1), round(err*100, 1)]

# def flatten_list(list_of_lists):
#     return [val for sublist in list_of_lists for val in sublist]

# def get_acc_avgs(dict_input):
    # acc_cls_ar = np.array(dict_input) #data[x]['acc_cls']
    # acc_cls_m = []
    # acc_cls_err = []
    # for l in range(acc_cls_ar.shape[1]): #loop of steps
    #     sample_lst = []
    #     for u in range(acc_cls_ar.shape[0]): #loop of samples
    #         sample_lst.append(list(acc_cls_ar[u][l].values())[1:]) #mean cls acc samples, steps. First is no_sdc, doesnt count
    #     avg_ft_mode = np.mean(sample_lst, 1) #mean of 5 fault modes

    #     m, err = get_m_err(avg_ft_mode)
    #     # m, err = np.mean(avg_ft_mode), np.std(avg_ft_mode)*1.96/np.sqrt(len(avg_ft_mode))
    #     # acc_cls_m.append(round(m*100,2)) #mean of 3 (N) samples 
    #     # acc_cls_err.append(round(err*100,2)) #err of 3 (N) samples 
    #     acc_cls_m.append(m) #mean of 3 (N) samples 
    #     acc_cls_err.append(err) #err of 3 (N) samples 

    # return acc_cls_m, acc_cls_err

def sort_l_q(lays_all, qus_all):
    lays_all_sorted = []
    qus_all_sorted = []
    for n in range(len(lays_all)):
        ls = [int(x) for x,_ in sorted(zip(lays_all[n], qus_all[n]))]
        qs = [int(x) for _, x in sorted(zip(lays_all[n], qus_all[n]))]
        lays_all_sorted.append(ls)
        qus_all_sorted.append(qs)

    lays_all_sorted_0 = deepcopy(lays_all_sorted)
    qus_all_sorted_0 = deepcopy(qus_all_sorted)

    lays_all_sorted.sort(key=len)
    qus_all_sorted.sort(key=len) #order is the same

    # verify that combos were preserved
    list_orig = list(zip(lays_all_sorted_0, qus_all_sorted_0))
    list_new = list(zip(lays_all_sorted, qus_all_sorted))
    for n in range(len(list_orig)):
        assert list_orig[n] in list_new

    return lays_all_sorted, qus_all_sorted


def get_hist3d(ax, x,y):
    alpha = np.linspace(1, 8, 5)
    t = np.linspace(0, 5, 16)
    T, A = np.meshgrid(t, alpha)
    data = np.exp(-T * (1. / A))

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    x1 = T.flatten()
    y1 = A.flatten()
    z1 = np.zeros(data.size)
    dx = .40 * np.ones(data.size)
    dy = .40 * np.ones(data.size)
    dz = data.flatten()

    ax.set_xlabel('T')
    ax.set_ylabel('Alpha')
    ax.bar3d(x1, y1,z1, dx, dy, dz, color='red')
    # plt.show()



################################################################################
pth = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/'

# data_list = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti', 'resnet_imagenet', 'alexnet_imagenet']
# data_list = ['yolo_kitti', 'resnet_imagenet'] #, 'yolo_kitti']
data_list = ['ssd_coco', 'ssd_kitti']
# N_l = [75, 75, 39, 39, 71, 71, 53, 5]
N_l = [39, 39]
fl_names = "unq" #"imp", red_by_layer




# fig_imp, axs = plt.subplots(len(data_list), figsize=(3,15))

fntsize = 14
[]

for xx in range(len(data_list)):
    d = data_list[xx]
    # fig_imp = figs[x]
    # ax_imp = axs[xx]
    fig_imp = plt.figure()
    # ax_imp = fig_imp.gca()
    ax_imp = fig_imp.gca()

    print('Evaluating', d)
    # data = load_json_indiv(pth + "dt_train_ft_red_by_layer_" + d + ".json")
    data = load_json_indiv(pth + "dt_train_ft_" + fl_names + "_" + d + ".json")

    for x in list(data.keys()):
        assert len(list(data.keys()))==1, 'More than one model per file, is that intended?'

        #in case there is an empty ft key:
        if len(data[x]['min_alternatives'])==0:
            print('empty, skipping')
            continue

        # # check due:
        # print('DUE detection (for classifier) in %', round(data[x]['due']*100,1))

        # p_m, p_err = get_m_err(data[x]['p'])
        # r_m, r_err = get_m_err(data[x]['r'])

        # acc_cls_m, acc_cls_err = get_acc_avgs(data[x]['acc_cls'])
        # acc_cat_m, acc_cat_err = get_acc_avgs(data[x]['acc_cat'])
        
        # # key_fts = []
        # # for n in range(len(data[x]['ft'])):
        # #     key_fts.append(flatten_list(data[x]['ft'][n]))
        # # key_fts2 = np.unique(flatten_list(key_fts)).tolist()
        # # print()

        alts = data[x]['min_alternatives']
        alts = alts[:10] #take first ten only
        alts_fts = [n['fts'] for n in alts]
        alts_p = [n['p_cls'] for n in alts]
        alts_r = [n['r_cls'] for n in alts]

        print('features no:', [len(x) for x in alts_fts])

        lays_all = []
        qus_all = []
        for w in alts_fts:
            lays = []
            qus = []
            for v in w:
                lays.append(int(v.split("_")[1][3:]))
                qus.append(int(v.split("_")[0][1:]))
            lays_all.append(lays)
            qus_all.append(qus)

        # sort by layers
        lays_all_sorted, qus_all_sorted = sort_l_q(lays_all, qus_all)

        dist = 0.2
        for n in range(len(lays_all_sorted)):
            get_hist3d(ax_imp, lays_all_sorted, qus_all_sorted)

            # y_artf = np.ones(len(lays_all_sorted[n]))*n*dist
            #
            # ax_imp.scatter(lays_all_sorted[n], y_artf) #, label=d) #plot
            # ax_imp.plot(lays_all_sorted[n], y_artf) #, label=d) #plot
            # for l in range(len(lays_all_sorted[n])):
            #     q_label = [str(x) for x in qus_all_sorted[n]]
            #     ax_imp.text(lays_all_sorted[n][l]-1, y_artf[l]+0.2*dist,  q_label[l], fontsize=6) #str(qus_all_sorted[n])

            # ax_imp.scatter(lays_all_sorted[n], qus_all_sorted[n]) #, label=d) #plot
            # ax_imp.plot(lays_all_sorted[n], qus_all_sorted[n]) #, label=d) #plot


    ax_imp.set_xlim([0, N_l[xx]-1]) #np.max(np.max(lays_all_sorted))+1])
    ax_imp.set_xlabel("Layer")
    ax_imp.set_ylabel(d)
    ax_imp.set_yticklabels("")
    ax_imp.set_yticks([])
    save_name = pth + "ft_combos_" + d + ".png"




    # save_name = pth + "ft_combos"+ ".png"
    fig_imp.legend()
    fig_imp.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
    print('saved as', save_name)