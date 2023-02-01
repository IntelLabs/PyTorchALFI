# import csv
import numpy as np
import matplotlib.pyplot as plt



def plot_data(mdls, ydata1, yerror1, ydata2, yerror2, xtitle, ytitle, leg, save_name):
    fig, ax = plt.subplots()
    x_pos = range(len(mdls))
    # ax.bar(x_pos, ydata, yerr=yerror, align='center', alpha=0.5, ecolor='black', capsize=10)
    # plt.scatter(x_pos, ydata1)
    ax.errorbar(x_pos, ydata1, yerr=yerror1, alpha=0.5, ecolor='black', capsize=10)
    ax.errorbar(x_pos, ydata2, yerr=yerror2, alpha=0.5, ecolor='black', capsize=10)

    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mdls)
    # ax.set_title('Accuracy top 1 (in %)')
    ax.yaxis.grid(True)
    #
    # Save the figure and show
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4, left=0.15)
    # mn, mx = min(ydata), max(ydata)
    # plt.ylim([0.9*mn,1.1*mx])
    # plt.xticks(rotation=90, fontsize=8)
    
    plt.legend(leg)

    # # get handles
    # handles, labels = ax.get_legend_handles_labels()
    # # remove the errorbars
    # handles = [h[0] for h in handles]
    # # use them in the legend
    # ax.legend(handles, labels, loc='upper left')


    plt.savefig(save_name, dpi=300)
    plt.show()




# x_faults = [0,1,10]
# acc_1 = [74.6, 71.43,  43.1856] #unprot
# acc_2 = [74.6, 74.0904, 69.4356] #prot
# err_acc_1 = [0,1.297438696,3.186570041]
# err_acc_2 = [0,0.436061792,1.198830896]

x_faults = [0,1,10]
acc_1 = [65.85, 61.893200000000036,  28.147299999999994] #unprot
acc_2 = [65.85, 65.11600000000003, 53.3076] #prot
err_acc_1 = [0,3.0647218199941344,5.663647661718669]
err_acc_2 = [0,0.8016352557316824,3.6097504390863144]

# Prec
x_faults = [0,1]
acc_1 = [0.8390906068985641,0.7258857234290244]
err_acc_1 = list(np.array([1.1102230246251565e-16,0.2830262593827349])*1.96/np.sqrt(100))
acc_2 = [0.8390906068985641,0.8404483559461916]
err_acc_2 = list(np.array([1.1102230246251565e-16,0.005560764548974382])*1.96/np.sqrt(100))
print(err_acc_1, err_acc_2)

# Rec
acc_1 = [0.7561333627867528,0.651887047174466]
err_acc_1 = list(np.array([1.1102230246251565e-16,0.25883279162625805])*1.96/np.sqrt(100))
acc_2 = [0.7561333627867528,0.7498805896619303]
err_acc_2 = list(np.array([1.1102230246251565e-16,0.047366625212472764])*1.96/np.sqrt(100))

# unprot,61.893200000000036,3.0647218199941344,72.3063,3.580419866353799,501.9918576836586,9.521843407021315
# prot,65.11600000000003,0.8016352557316824,76.12369999999999,0.9021225999280584,561.2187590408325,90.895167705327
# unprot,28.147299999999994,5.663647661718669,29.4907,7.00189848745668,354.4315155863762,30.521915099069425
# prot,53.3076,3.6097504390863144,62.58509999999999,4.258723707923753,533.4346143722535,39.117129361781835

# Accuracy plot
# ydata = acc_1
# yerror = err_acc_1
xtitle = 'Nr of faults'
# ytitle = 'Accuracy top 1 (in %)'
# ytitle = 'mAP (in %)'
# ytitle = 'Precision'
ytitle = 'Recall'
save_name = 'plots/evaluation/yolo_ppp_fi_rec.png'
leg = ['No protection', 'Ranger']
plot_data(x_faults, acc_1, err_acc_1, acc_2, err_acc_2, xtitle, ytitle, leg, save_name)
