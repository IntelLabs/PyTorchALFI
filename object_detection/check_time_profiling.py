# from train_detection_model_LR3 import load_json_indiv
import numpy as np
import matplotlib.pyplot as plt
# from tabulate import tabulate
# from attic.check_dt_train_results_ft_red import get_acc_avgs, get_m_err, flatten_list
# from copy import deepcopy
# from collections import Counter


def load_txt_data(txt_file):

    avgs_cpu = []
    errs_cpu = []
    avgs_gpu = []
    errs_gpu = []

    cpu_nohooks = txt_file[0]
    avgs_cpu.append(float(cpu_nohooks[0]))
    errs_cpu.append(float(cpu_nohooks[-1].split("\n")[0]))
    gpu_nohooks = txt_file[1]
    avgs_gpu.append(float(gpu_nohooks[0]))
    errs_gpu.append(float(gpu_nohooks[-1].split("\n")[0]))

    cpu_qu = txt_file[2]
    avgs_cpu.append(float(cpu_qu[0]))
    errs_cpu.append(float(cpu_qu[-1].split("\n")[0]))
    gpu_qu = txt_file[3]
    avgs_gpu.append(float(gpu_qu[0]))
    errs_gpu.append(float(gpu_qu[-1].split("\n")[0]))

    cpu_qu2 = txt_file[4]
    avgs_cpu.append(float(cpu_qu2[0]))
    errs_cpu.append(float(cpu_qu2[-1].split("\n")[0]))
    gpu_qu2 = txt_file[5]
    avgs_gpu.append(float(gpu_qu2[0]))
    errs_gpu.append(float(gpu_qu2[-1].split("\n")[0]))

    cpu_fmaps = txt_file[6]
    avgs_cpu.append(float(cpu_fmaps[0]))
    errs_cpu.append(float(cpu_fmaps[-1].split("\n")[0]))
    gpu_fmaps = txt_file[7]
    avgs_gpu.append(float(gpu_fmaps[0]))
    errs_gpu.append(float(gpu_fmaps[-1].split("\n")[0]))

    return avgs_cpu, errs_cpu, avgs_gpu, errs_gpu



################################################################################
pth = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/'

data_list = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti', 'resnet_imagenet', 'alexnet_imagenet']
# data_list = ['yolo_coco', 'yolo_kitti', 'ssd_coco', 'ssd_kitti', 'retina_coco', 'retina_kitti']
# data_list = ['resnet_imagenet', 'alexnet_imagenet']

################################################################################

# col_blue = [0.3,0.3, 1]
# col_red = [1,0.1,0.1]
# col_green = [0.,1,0.5]
# cols = [col_blue, col_red, col_green]
# ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
#  '#7f7f7f', '#bcbd22', '#17becf']
cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


fig = plt.figure(figsize=(8,4))
ax = fig.gca()
x_labels = ["original (CPU)", "quantiles red (CPU)", "quantiles full (CPU)", 'fmap trace (CPU)']
x_labels2 = ["original (GPU)", "quantiles red (GPU)", "quantiles full (GPU)", 'fmap trace (GPU)']
stretch = 5
w = 1.

for xx in range(len(data_list)):

    d = data_list[xx]
    print('Evaluating', d)


    txt_file = []
    with open(pth + 'time_profiling_' + d + '.txt', "r") as file:
        for line in file:
            txt_file.append(line.split(","))

    Nr_imgs = int(txt_file[8][0].split("\n")[0])
    avgs_cpu, errs_cpu, avgs_gpu, errs_gpu = load_txt_data(txt_file) #order: no hooks, quantiles, fmaps
    avgs_cpu = np.array(avgs_cpu)/Nr_imgs*1e-3 #scale from us to ms
    errs_cpu = np.array(errs_cpu)/Nr_imgs*1e-3
    avgs_gpu = np.array(avgs_gpu)/Nr_imgs*1e-3
    errs_gpu = np.array(errs_gpu)/Nr_imgs*1e-3

    if xx == 0:
        label_d =x_labels
        label_d2 = x_labels2
    else:
        label_d = None
        label_d2 = None
    distr = (np.array(list(range(len(avgs_cpu))))-np.mean(range(len(avgs_cpu))) ) *w #bar distribution
    ax.bar(np.array([xx*stretch]*len(avgs_cpu)) + distr, avgs_cpu, width = w, label = label_d, color=cols, edgecolor='k', alpha = 1) #CPU
    ax.bar(np.array([xx*stretch]*len(avgs_gpu)) + distr, avgs_gpu, bottom = avgs_cpu, width = w, label = label_d2, color=cols, edgecolor='k', alpha =0.5) #GPU
    ax.errorbar(np.array([xx*stretch]*len(avgs_cpu)) + distr, avgs_cpu, yerr=errs_cpu, fmt="o", capsize=3, color='k', markersize='1', label=None) #CPU
    ax.errorbar(np.array([xx*stretch]*len(avgs_gpu)) + distr, avgs_gpu + avgs_cpu, yerr=errs_gpu, fmt="o", capsize=3, color='k', markersize='1', label=None) #GPU
    # Put text
    for n in range(len(avgs_gpu)):
        x = np.array([xx*stretch]*len(avgs_gpu)) + distr*1.
        y = avgs_gpu + avgs_cpu
        txt = [str(round(x,1)) for x in  avgs_gpu + avgs_cpu]
        ax.text(x[n]-0.4, y[n] + 1.5, txt[n], fontsize=7, rotation=90)

    tt = avgs_cpu + avgs_gpu
    # print('slowdown', d, 'qu vs orig', tt[1]/tt[0], 'fmap vs qu', tt[2]/tt[1])
    rel = 100
    print('slowdown', d, 'qu vs orig (%)', round(100*tt[1]/tt[0]-rel,1), 'qu2 vs orig (%)', round(100*tt[2]/tt[0]-rel,1), 'fmap vs orig (%)', round(100*tt[3]/tt[0]-rel,1), 'fmap vs qu (%)', round(100*tt[3]/tt[1]-rel,1), 'fmap vs qu2 (%)', round(100*tt[3]/tt[2]-rel,1))

ax.set_xticks(ticks=np.array(range(len(data_list)))*stretch, labels=data_list, rotation=90, fontsize=14)
ax.set_ylabel('Inference time per image (ms)', fontsize=14)
save_name = pth + "timings"+ ".png"
ax.set_ylim(np.array(ax.get_ylim()) + [0,5])
fig.legend(bbox_to_anchor=(1.2, 0.7), loc='upper right', ncol=1, fontsize=12)
fig.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=250, format='png')
print('saved as', save_name)
