from train_detection_model_LR3 import load_json_indiv
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from attic.check_dt_train_results_ft_red import get_acc_avgs, get_m_err, flatten_list
from copy import deepcopy
from collections import Counter

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


def filter_alts(alts):
    alts_filtered = [] #deepcopy([alts[0]])
    for u in range(len(alts)):
        if u == 0:
            alts_filtered.append(alts[u])
        else:
            if set(alts[u]['fts']) not in [set(n['fts']) for n in alts_filtered]:
                alts_filtered.append(alts[u])
    return alts_filtered


def get_sep_lists_lay_qu(alts_fts):
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
    return lays_all_sorted, qus_all_sorted


def fuse_same_layer_labels(lays_all_sorted, qus_all_sorted):
    
    same_layer_combos = []
    for ux in range(len(lays_all_sorted)):
        # same_layer_combos.append(np.where(lays_all_sorted[ux] in lays_all_sorted[:ux])[0]) #how often in previous
        try:
            same_layer_combos.append(lays_all_sorted[:ux].index(lays_all_sorted[ux])) #how often in previous
        except:
            same_layer_combos.append(None)

    lays_all_sorted = [lays_all_sorted[x] for x in range(len(lays_all_sorted)) if same_layer_combos[x] is None]
    for fu in range(len(qus_all_sorted)):
        if same_layer_combos[fu] is not None:
            qus_all_sorted[same_layer_combos[fu]] += qus_all_sorted[fu]

    qus_all_sorted = [qus_all_sorted[x] for x in range(len(qus_all_sorted)) if same_layer_combos[x] is None]

    for p in range(len(lays_all_sorted)):
        if len(qus_all_sorted[p])>len(lays_all_sorted[p]):
            # qus_all_sorted[p][::2]
            for n in range(len(lays_all_sorted[p])):

                a = qus_all_sorted[p][n::len(lays_all_sorted[p])]
                qus_all_sorted[p][n]  = '/'.join([str(x) for x in a])
            qus_all_sorted[p] = qus_all_sorted[p][:len(lays_all_sorted[p])] #eliminate the rest

    return lays_all_sorted, qus_all_sorted


################################################################################
def main():
    pth = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/'


    data_list = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti', 'resnet_imagenet', 'alexnet_imagenet']
    N_l = [75, 75, 39, 39, 71, 71, 53, 5]


    # data_list = ['yolo_coco', 'yolo_kitti']
    # N_l = [75, 75]
    # data_list = ['ssd_coco', 'ssd_kitti']
    # N_l = [39, 39]
    # data_list = ['retina_coco', 'retina_kitti']
    # N_l = [71, 71]
    # data_list = ['resnet_imagenet']
    # N_l = [53]
    data_list = ['alexnet_imagenet']
    N_l = [5]

    nr_plots = 1



    #visual finetuning
    if 'yolo' in data_list[0] :
        fig_y_size = 3.5*len(data_list) #3.5 for most cases, 1.3
        dist_y = 0.2 #0.2 for most cases, 0.1
        dist_x = 1 #1 for most cases, 0.1
        leg_off = 0.0 #0 for most cases,  0.05 for resnet, 0.2 for alexnet,  for retina
        range_offset = 1.5 #1.5 for most cases, 0.5 for alexnet
        uspecial = 1 #1 for most cases, 0.1
        ylabels_text = "List of minimal layer and quantile marker combinations"
    elif 'ssd' in data_list[0]:
        fig_y_size = 3.5*len(data_list) #3.5 for most cases, 1.3
        dist_y = 0.2 #0.2 for most cases, 0.1
        dist_x = 1 #1 for most cases, 0.1
        leg_off = 0.0 #0 for most cases,  0.05 for resnet, 0.2 for alexnet,  for retina
        range_offset = 1.5 #1.5 for most cases, 0.5 for alexnet
        uspecial = 1 #1 for most cases, 0.1
        ylabels_text = ""
    elif 'retina' in data_list[0]:
        fig_y_size = 3.5*len(data_list) #3.5 for most cases, 1.3
        dist_y = 0.2 #0.2 for most cases, 0.1
        dist_x = 1 #1 for most cases, 0.1
        leg_off = 0. #0 for most cases,  0.05 for resnet, 0.2 for alexnet,  for retina
        range_offset = 1.5 #1.5 for most cases, 0.5 for alexnet
        uspecial = 1 #1 for most cases, 0.1
        ylabels_text = ""
    elif 'resnet' in data_list[0]:
        fig_y_size = 3.5*len(data_list) #3.5 for most cases, 1.3
        dist_y = 0.2 #0.2 for most cases, 0.1
        dist_x = 1 #1 for most cases, 0.1
        leg_off = 0.05 #0 for most cases,  0.05 for resnet, 0.2 for alexnet,  for retina
        range_offset = 1.5 #1.5 for most cases, 0.5 for alexnet
        uspecial = 1 #1 for most cases, 0.1
        ylabels_text = ""
    elif 'alexnet' in data_list[0]:
        fig_y_size = 1.3*len(data_list) #3.5 for most cases, 1.3
        dist_y = 0.1 #0.2 for most cases, 0.1
        dist_x = 0.1 #1 for most cases, 0.1
        leg_off = 0.25 #0 for most cases,  0.05 for resnet, 0.2 for alexnet,  for retina
        range_offset = 0.5 #1.5 for most cases, 0.5 for alexnet
        uspecial = 0.1 #1 for most cases, 0.1
        ylabels_text = ""

    # How many solutions if available < 20 fts
    nr_sel = 10 #None means all
    # yolo: fig: 4, dist_y 0.2, dist_x 1, nr_sel = 10
    # ssd:
    ################################################################################


    fl_names = "unq2" #"imp", red_by_layer

    fig_imp, axs = plt.subplots(nr_plots , figsize=(8,fig_y_size)) #(len(data_list) len(data_list)
    # fig_imp = plt.figure()
    # ax_imp = fig_imp.gca()
    fntsize = 14


    col_blue = [0,0,1]
    col_red = [1,0,0]
    cols = [col_blue, col_red]
    offset = 0

    for xx in range(len(data_list)):

        # fig_imp = figs[x]
        d = data_list[xx]
        print('Evaluating', d)
        # ax_imp = plt.gca()
        if nr_plots==1:
            ax_imp = plt.gca()
        else:
            ax_imp = axs[xx]


        data = load_json_indiv(pth + "dt_train_ft_" + fl_names + "_" + d + ".json")

        for x in list(data.keys()):

            # Basic checks ----------------------------------
            assert len(list(data.keys()))==1, 'More than one model per file, is that intended?'
            #in case there is an empty ft key:
            if len(data[x]['min_alternatives'])==0:
                print('empty, skipping')
                continue

            # Prepare data ---------------------------------
            alts = data[x]['min_alternatives']

            # Eliminate doubles:
            alts = filter_alts(alts)
            # alts = alts[:nr_sel] #eg take first ten only

            alts_fts = [n['fts'] for n in alts]
            alts_p = [n['p_cls'] for n in alts]
            alts_r = [n['r_cls'] for n in alts]
            print('features no:', [len(x) for x in alts_fts])

            lays_all_sorted, qus_all_sorted = get_sep_lists_lay_qu(alts_fts)
            lays_all_sorted, qus_all_sorted = fuse_same_layer_labels(lays_all_sorted, qus_all_sorted)

            if nr_sel is not None:
                lays_all_sorted = lays_all_sorted[:nr_sel] #eg take first ten only
                qus_all_sorted = qus_all_sorted[:nr_sel] #eg take first ten only

            print('minimal no of features for:', d, 'is', len(lays_all_sorted[0]))
            # Plot visuals -------------------------------------
            col = cols[xx]
            
            for n in range(len(lays_all_sorted)): #go through all combos

                no_prev_occ = []
                for ui in range(len(lays_all_sorted[n])):
                    no_prev_occ.append(len(np.where(lays_all_sorted[n][ui] in lays_all_sorted[n][:ui])[0])) #how often in previous
                    
                print('check', lays_all_sorted[n], qus_all_sorted[n], no_prev_occ)
                # dc = Counter(lays_all_sorted[n])
                # cnt = [dc[x] for x in range(len(lays_all_sorted[n]))]
                # # assert np.max(cnt) <= 2, 'too many found'

                y_artf = np.ones(len(lays_all_sorted[n]))*n*10*dist_y + offset
                ax_imp.plot(lays_all_sorted[n], y_artf,zorder=1, color=col) #, label=d) #plot

                for l in range(len(lays_all_sorted[n]))[::-1]: #plot backwards for right foreground/background
                    if n==0 and l==0:
                        label_d = d
                    else:
                        label_d = None

                    q_label = [str(x) for x in qus_all_sorted[n]]
                    x_shift_long_labels = int((len(q_label[0])-2)/1.5)*uspecial
                    ax_imp.text(lays_all_sorted[n][l]-1*uspecial+2*no_prev_occ[l]*dist_x-x_shift_long_labels, y_artf[l]+3.*dist_y+2*no_prev_occ[l]*dist_y,  q_label[l], fontsize=8) #str(qus_all_sorted[n])
                    ax_imp.scatter(lays_all_sorted[n][l]+no_prev_occ[l]*dist_x, y_artf[l]+no_prev_occ[l]*dist_y, color=col, marker = 's', edgecolors='black', s=100, zorder=2, label=label_d) #, label=d) #plot #, (float(q_label[l])/100.)
                
                # ax_imp.scatter(lays_all_sorted[n], qus_all_sorted[n]) #, label=d) #plot
                # ax_imp.plot(lays_all_sorted[n], qus_all_sorted[n]) #, label=d) #plot
                # ax_imp.plot(lays_all_sorted[n], y_artf) #, label=d) #plot

            offset += (len(lays_all_sorted))*dist_y*10

        
        ax_imp.set_xlim([0-range_offset, N_l[xx]-1+range_offset]) #np.max(np.max(lays_all_sorted))+1])
        if len(data_list) <= 1:
            ax_imp.set_ylim([-0.5, (len(lays_all_sorted)+0.1)*dist_y*10])
        ax_imp.set_xlabel("Layer", fontsize=14)
        ax_imp.set_ylabel(ylabels_text, fontsize=14)
        # ax_imp.set_ylabel("List of minimal layer and quantile marker combinations", fontsize=14)
        ax_imp.set_yticklabels("")
        ax_imp.set_yticks([])
        # save_name = pth + "ft_combos_" + d + ".png"



    save_name = pth + "ft_combos"+ ".png"
    # fig_imp.legend()
    fig_imp.legend(bbox_to_anchor=(0.8, 0.95+leg_off), loc='upper right', ncol=2) #, fontsize=9)
    fig_imp.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=250, format='png')
    print('saved as', save_name)


if __name__ == "__main__":
    main()