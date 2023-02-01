from train_detection_model_LR3 import load_json_indiv
import numpy as np
from attic.check_dt_train_results_ft_red import get_acc_avgs

def get_mean_err(x_list):
        # return np.sum( np.abs(np.array(x_list) - np.mean(x_list)) )/len(x_list)
        m, err = np.mean(x_list), np.max( np.abs(np.array(x_list) - np.mean(x_list)) )
        return round(m*100,1), round(err*100,1)

data = load_json_indiv("/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/dt_train.json")

for x in list(data.keys()):
    p_list = data[x]['p']
    r_list = data[x]['r']
    acc_list = data[x]['acc']
    acc_cls_m, acc_cls_err = get_acc_avgs(data[x]['acc_cls'])
    acc_cat_m, acc_cat_err = get_acc_avgs(data[x]['acc_cat'])

    m, err = get_mean_err(p_list)
    print(x, 'precision', m, err)
    m, err = get_mean_err(r_list)
    print(x, 'recall', m, err)
    m, err = get_mean_err(acc_list)
    print(x, 'acc_cls', m, err)
