import os, json
import matplotlib.pyplot as plt
import numpy as np

def load_json_indiv(gt_path):
    with open(gt_path) as f:
        coco_gt = json.load(f)
        f.close()
    return coco_gt

def find_file_by_name(folder_path, name_part):
    return_list_all = []
    for n in folder_path:
        return_list = []
        for _, _, files in os.walk(n):
            for name in files:
                if name_part in name:
                    return_list.append(n + '/' + name)
                    
        return_list_all.append(return_list)
    return return_list_all

def plot_hist(mns, errs, mns_ref, max_mon_range, target, pth):
    fig, axs = plt.subplots(max_mon_range)
    fig.subplots_adjust(left=0.15)
    q_names = ['q0', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90', 'q100']
    for lay in range(max_mon_range): 
        x_pos = list(range(len(mns[lay])))
        axs[lay].hlines(y=0.5, xmin=x_pos[0]-1, xmax=x_pos[-1]+1, linewidth=1, color='k', linestyles='--', label=None)

        if lay == 0:
            lbl = 'SDC'
        else:
            lbl = None
        axs[lay].bar(x_pos, mns[lay], label=lbl)
        axs[lay].errorbar(x_pos, mns[lay], yerr=errs[lay], fmt="o", capsize=3, color='k', markersize='1', label=None) #color='k', fmt="o", markersize='3'
        # axs[lay].bar(x_pos, mns_ref[lay])
        for lns in range(len(mns_ref[lay])):
            if lns==0 and lay==0:
                lbl = 'No SDC (all)'
            else:
                lbl = None
            axs[lay].hlines(y=mns_ref[lay][lns], xmin=x_pos[lns]-0.4, xmax=x_pos[lns]+0.4, linewidth=1, color='orange', label=lbl)
        if lay == max_mon_range - 1:
            axs[lay].set_xticks(ticks=x_pos, labels=q_names)
            axs[lay].set_xlabel("Quantile number")
        else:
            axs[lay].set_xticks(ticks=x_pos, labels="")
        if lay == 0:
            axs[lay].set_ylabel("Fault") # fontsize=9
        else:
            axs[lay].set_ylabel("Fault+" +str(lay))
        axs[lay].set_ylim([0,1])
        axs[lay].set_xlim([x_pos[0]-0.5, x_pos[-1]+0.5])
        # # Put text
        # for n in range(len(mns[lay])):
        #     x = x_pos
        #     y = mns[lay]
        #     txt = [str(round(x,2)) for x in mns[lay]]
        #     axs[lay].text(x[n]-0.1, y[n] + 0.1, txt[n], fontsize=6, rotation=90)

    fig.legend(bbox_to_anchor=(0.9, 1.01))
    fig.suptitle(str(target))
    fig.supylabel("Average quantile shifts per layer")

    

    # Save fig:
    save_name = pth + "qushifts_" + target + ".png"
    fig.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
    print('saved as', save_name)
    # print('next')

def get_mns_errs(data_x, no_layers_shown = None):

    sdc_viols_demo = [np.array(n) for n in data_x["sdc"]]
    nonsdc_viols_demo = [np.array(n) for n in data_x['nonsdc']]

    # Get averages:
    if no_layers_shown is None:
        no_layers_shown = max([n.shape[0] for n in nonsdc_viols_demo])
    # no_layers_shown = 3

    mns, errs = [], []
    for l in range(no_layers_shown):
        l_shift = [n[l,:] for n in sdc_viols_demo if n.shape[0] > l]
        # l_shifts.append([n[l,:] for n in sdc_viols_demo if n.shape[0] >= l])
        if len(l_shift) > 0:
            m, err = np.mean(l_shift, 0), np.std(l_shift, 0)*1.96/np.sqrt(len(l_shift))
        else:
            m, err = None, None
        mns.append(m)
        errs.append(err)

    # Non sdc demo:
    mns_nosdc, errs_nosdc = [], []
    for l in range(no_layers_shown):
        l_shift = [n[l,:] for n in nonsdc_viols_demo if n.shape[0] > l]
        if len(l_shift) > 0:
            m, err = np.mean(l_shift, 0), np.std(l_shift, 0)*1.96/np.sqrt(len(l_shift))
        else:
            m, err = None, None
        mns_nosdc.append(m)
        errs_nosdc.append(err)

    return mns, err, mns_nosdc, errs_nosdc, no_layers_shown

def main():
    ##############################################################################
    folder_path = ['/home/fgeissle/pytorchalfi/object_detection/quantile_detection_data/']
    file_list = find_file_by_name(folder_path, 'quantile_shift_stats')[0]
    
    no_layers_shown = 3 #None #None means we extract from data, otherwise give explicitly
    ##############################################################################

    # Extract and consolidate fault modes: 
    dct_all = []
    non_sdc_all = []
    sdc_mem = []
    non_sdc_mem = []
    for f in file_list:
        data = load_json_indiv(f)
        non_sdc_all.extend(data['nonsdc'])
        if 'hwfault' in data['target']:
            sdc_mem.extend(data["sdc"])
            non_sdc_mem.extend(data['nonsdc'])
        else:
            dct_all.append(data)
    dct_all.append({'target': 'Memory_fault', 'sdc': sdc_mem, 'nonsdc': non_sdc_mem})

    
    dct_ref = {'target': 'No_sdc', 'sdc': [], 'nonsdc': non_sdc_all}
    _, _, mns_nosdc_ref, errs_nosdc_ref, no_layers_shown = get_mns_errs(dct_ref, no_layers_shown)

    # Plot it:
    # TODO: subtract non-sdc?
    for x in dct_all:
        
        target = x['target']
        mns, errs, _, _, _ = get_mns_errs(x, no_layers_shown)

        # Plot:
        plot_hist(mns, errs, mns_nosdc_ref, no_layers_shown, target, folder_path[0])
        


if __name__ == "__main__":
    main()