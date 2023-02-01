import numpy as np
import matplotlib.pyplot as plt
import random 
import matplotlib.pyplot as plt 
from train_detection_model_LR3 import get_tp_fp_fn, load_json_indiv
from copy import deepcopy

def plot_v_line(q, qname, **kwargs):
    vshift = kwargs.get('vup', 0.)
    plt.axvline(x=q, color='b')
    plt.text(q, 1.03 +vshift, qname + '=' + str(round(q,2)), horizontalalignment='center', color='black', fontsize=14) #fontweight='bold'

def plot_h_line(q0, q10, perc_diff):
    # plt.axvline(x=q0, color='b')
    # print(perc_diff/(q10-q0), q0, q10)
    # plt.axhline(y=perc_diff/(q10-q0), xmin=q0, xmax=q10, color='g')
    plt.plot([q0, q10], [perc_diff/(q10-q0), perc_diff/(q10-q0)], color='g')



# Load data- ------------------------------------
print('Loading data...')
file_act_example = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_1_trials/neurons_injs/per_image/objDet_20220524-155333_1_faults_[0_32]_bits/coco2017/val/yolov3_ultra_test_random_sbf_neurons_inj_1_1_1bs_quantiles.json'
act_example = load_json_indiv(file_act_example)

nums_all = act_example['corr'][0]
suppl = ''


# # # Create data- ------------------------------------
# nums = [] 
# samples = 1000
# mu = 10
# sigma = 1
# random.seed(0)
    
# for i in range(samples): 
#     temp = random.gauss(mu, sigma)
#     nums.append(temp) 

# nums = np.sort(nums).tolist()
# nums_all = [nums]
# suppl = '_theo'



# Calculate quantiles ------------------------------------------
mod = False #True, False
if suppl == '_theo':
    output_folder = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/theory_generic'
else:
    output_folder = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/theory'

list_lays = range(len(nums_all))
# list_lays = [5,15,25,35,45,55,65]

lay_size = []
for x in list_lays:
    print('layer', x)
    nums = nums_all[x]
    lay_size.append(len(nums))

    # Modify data
    if mod:
        # # Move top
        # print(nums[1], 'to', 20)
        # nums[1] = 20

        # # Move bulk 
        # print(nums[1], 'to', 20)
        nums = np.array(nums)
        nums[1:100] = nums[1:100] + 1

    # Get quantiles    
    q0 = np.quantile(nums, 0.0)
    q10 = np.quantile(nums, 0.10)
    q25 = np.quantile(nums, 0.25)
    q50 = np.quantile(nums, 0.50)
    q75 = np.quantile(nums, 0.75)
    q100 = np.quantile(nums, 1.0)
    print(q0, q10, q25, q50, q75, q100)


    # plotting a graph -------------------------------------------
    # # nums = nums/np.sum(np.abs(nums)) #to scale bars to be <=1
    # df = deepcopy(q0)
    # # if q0 < 0:
    # #     df = q0
    # # else:
    # #     df = q0
    # nums = nums - df
    # q0 = q0 - df #do last 
    # q10 = q10 - df
    # q25 = q25 - df
    # q50 = q50 - df
    # q75 = q75 - df
    # q100 = q100 - df
   

    plt.figure(figsize=(13, 10))
    plt.hist(nums, bins=100, density=True) #density=True)
    plt.ylim([0,1])
    plt.xlabel('activation magnitude', fontsize=14)
    plt.ylabel('Density (N=' + str(len(nums)) + ')', fontsize=14)

    plot_v_line(q0, 'q0', vup=-0.03)
    plot_v_line(q10, 'q10', vup=0.0)
    plot_v_line(q25, 'q25', vup=0.03)
    plot_v_line(q50, 'q50', vup=0.06)
    plot_v_line(q75, 'q75',  vup=0.09)
    plot_v_line(q100, 'q100', vup=0.)

    # # Probability estimation with quantiles:
    # plot_h_line(q0, q10, 0.1)
    # plot_h_line(q10, q25, 0.15)
    # plot_h_line(q25, q50, 0.25)
    # plot_h_line(q50, q75, 0.25)
    # plot_h_line(q75, q100, 0.25)

    plt.show()
    if mod:
        nm = output_folder + "/quants_mod_" + str(x) + suppl + ".png"
        plt.savefig(nm)
        print('saved', nm)
    else:
        nm = output_folder + "/quants_" + str(x)  +  suppl + ".png"
        plt.savefig(nm)
        print('saved', nm)

print('lay size', lay_size, 'max', np.max(lay_size))