
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from copy import deepcopy
# from test_detection_model_LR import get_tp_fp_fn
import random
# import torch.nn.functional as F
from quantiles_extract_features_plot4 import flatten_list

# https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
# Feature importance is exp(weights), because of logistic regression sigmoid unit

# Prepare data: --------------------------------------------------------------------------------------

def get_flt_dicts():
    # cls_mapping = {'no_sdc': 0, 'neurons': 1, 'weights': 2, 'blur': 3, 'noise': 4, 'contrast':5}
    # cats_mapping = {'no_sdc': 0, 'neurons': 1, 'weights': 1, 'blur': 2, 'noise': 2, 'contrast':2}
    # sdc_mapping = {'no_sdc': 0, 'neurons': 1, 'weights': 1, 'blur': 1, 'noise': 1, 'contrast':1}

    cls_mapping = {'no_sdc': 0, 'memory': 1, 'blur': 2, 'noise': 3, 'contrast':4}
    cats_mapping = {'no_sdc': 0, 'memory': 1, 'blur': 2, 'noise': 2, 'contrast':2}
    sdc_mapping = {'no_sdc': 0, 'memory': 1, 'blur': 1, 'noise': 1, 'contrast':1}
    return cls_mapping, cats_mapping, sdc_mapping


def load_json_indiv(gt_path):
    with open(gt_path) as f:
        coco_gt = json.load(f)
        f.close()
    return coco_gt

def substract_list(test_list1, test_list2):
    # does list1-list2
    res = [ ele for ele in test_list1 ]
    for a in test_list2:
        if a in test_list1:
            res.remove(a)
    return res

def elim_by_pattern(pattern, feature_labels):
    to_elim = []
    for p in pattern:
        for x in feature_labels:
            if p[-1] != "_":
                p2 = p + "_"
            else:
                p2 = p
            x2 = x + "_"
            if p2 in x2: #'lay7' but not 'lay70'
                to_elim.append(x)
    return to_elim


# https://dair.ai/notebooks/machine%20learning/beginner/logistic%20regression/2020/03/18/pytorch_logistic_regression.html
class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs


class FccDetector_L4(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FccDetector_L4, self).__init__()
        self.out_features = output_dim
        n_w = 64
        self.fc1 = torch.nn.Linear(input_dim, n_w)
        self.fc2 = torch.nn.Linear(n_w, n_w)
        self.fc3 = torch.nn.Linear(n_w, n_w)
        self.fc4 = torch.nn.Linear(n_w, output_dim)

        self.bn = torch.nn.BatchNorm1d(num_features=n_w)
        self.relu = torch.nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.fc1(x)

        x = self.fc2(x)
        # if self.training:
        #     x = self.bn(x)
        # x = self.relu(x)

        x = self.fc3(x)
        # if self.training:
        #     x = self.bn(x)
        # x = self.relu(x)

        x = self.fc4(x)

        outputs = torch.sigmoid(x)
        return outputs

class FccDetector_L2(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FccDetector_L2, self).__init__()
        self.out_features = output_dim
        n_w = 64
        self.fc1 = torch.nn.Linear(input_dim, n_w)
        self.fc2 = torch.nn.Linear(n_w, output_dim)

        self.bn = torch.nn.BatchNorm1d(num_features=n_w)
        self.relu = torch.nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.fc1(x)
        # if self.training:
        #     x = self.bn(x)
        # x = self.relu(x)
        x = self.fc2(x)
        outputs = torch.sigmoid(x)
        return outputs


def eliminate_features(X, fts_to_elim, feature_labels):
    if len(fts_to_elim) == 0:
        return X, feature_labels
        
    fts_to_elim_num = []
    for u in fts_to_elim:
        fts_to_elim_num.append(np.where([m == u for m in feature_labels])[0][0])

    feature_labels_red = []
    dummy = np.zeros(X.shape[0])
    for n in range(X.shape[1]):
        if n not in fts_to_elim_num:
            dummy = np.vstack((dummy, X[:,n]))
            feature_labels_red.append(feature_labels[n])
        # else:
        #     print('eliminated:', feature_labels[n])
        
    return dummy[1:,:].T, feature_labels_red

def add_feature(x_orig, x_extra, feature_labels, label_extra):
    f_new = deepcopy(feature_labels)
    x_new = np.vstack((x_orig.T, x_extra)).T
    f_new.append(label_extra)
    return x_new, f_new


def get_tpfpfn_new(mpg, conf_matrix):
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
        p,r = get_p_r(tp, fp, fn)

        pr_list_fmodes.append([p,r])

    pr_list_fmodes = [n for n in pr_list_fmodes if n[0] is not None]

    p_m = np.mean([x[0] for x in pr_list_fmodes])
    r_m = np.mean([x[1] for x in pr_list_fmodes])
    print('p modes, r modes', pr_list_fmodes)
    return p_m, r_m


def get_tp_fp_fn(x_list, y_list, model, nr_samples, thresh=0.5):
    cnt_tp = 0
    cnt_fp = 0
    cnt_fn = 0
    cnt_tn = 0
    cnt_sdc = 0

    sel = range(x_list.shape[0])
    if nr_samples is not None and nr_samples < len(sel):
        sel = random.sample(sel, nr_samples)
        
    for n in sel:
        x_n = torch.tensor(x_list[n,:]).type(torch.FloatTensor)
        y_n = y_list[n]
        if y_n == 1:
            cnt_sdc += 1

        with torch.no_grad():
            prediction = model(x_n)

            if int(prediction > thresh) == y_n and y_n == 1:
                cnt_tp += 1
            elif int(prediction > thresh) == y_n and y_n == 0:
                cnt_tn += 1
            elif y_n == 1 and int(prediction > thresh) == 0:
                cnt_fn += 1
            elif y_n == 0 and int(prediction > thresh) == 1:
                cnt_fp += 1
            

    return cnt_tp, cnt_fp, cnt_fn, cnt_tn, cnt_sdc


def get_tp_fp_fn_multiclass_all_classes(x_list, y_list, model, nr_samples, thresh=0.5):
    # #Classes: 0 is no sdc, 1=sdc hw, 2=sdc input
    # cnt_tp = [0 for u in range(model.linear.out_features)] #here only four classes, since no sdc is no tp, fp, fn
    # cnt_fp = [0 for u in range(model.linear.out_features)]
    # cnt_fn = [0 for u in range(model.linear.out_features)]

    # cnt_tn = 0
    cnt_sdc = 0
    # sdc_conf = 0
    try:
        conf_matrix = np.zeros((model.linear.out_features, model.linear.out_features))
    except:
        conf_matrix = np.zeros((model.out_features, model.out_features))

    sel = range(x_list.shape[0])
    if nr_samples is not None and nr_samples < len(sel):
        sel = random.sample(sel, nr_samples)
        
    for n in sel:
        x_n = x_list[n,:].clone().detach().type(torch.float)
        # x_n = torch.tensor(x_list[n,:]).float
        y_n = y_list[n]

        
        if y_n > 0:
            cnt_sdc += 1

        with torch.no_grad():
            prediction = model(x_n)
            prediction = int(torch.argmax(prediction))

            conf_matrix[y_n, prediction] += 1

            # if prediction == y_n and y_n == 0:
            #     cnt_tn += 1
            # if prediction == y_n and y_n > 0:
            #     cnt_tp[prediction] += 1
            # if prediction == 0 and y_n > 0:
            #     cnt_fn[y_n] += 1
            # if prediction > 0 and y_n == 0:
            #     cnt_fp[prediction] += 1
            # if prediction > 0 and y_n > 0 and prediction != y_n:
            #     sdc_conf += 1


    # cnt_tp = np.array(cnt_tp[1:]).tolist()
    # cnt_fp = np.array(cnt_fp[1:]).tolist()
    # cnt_fn = np.array(cnt_fn[1:]).tolist()

    return conf_matrix
    


def get_p_r(tp, fp, fn):
    if tp+fp>0:
        p = tp/(tp+fp)
    else:
        p = None
    if tp+fn>0:
        r = tp/(tp+fn)
    else:
        r = None
    return p,r

class loss_fscore(torch.nn.Module):
    
    def __init__(self):
        super(loss_fscore,self).__init__()
    
    def forward(self, outputs, labels):
        # reshape labels to give a flat vector of length batch_size*seq_len
        # f_out = outputs[0]

        mask_pred_one = outputs.round() == 1
        mask_pred_zero = outputs.round() == 0
        mask_label_one = labels == 1
        mask_label_zero = labels == 0

        tp = torch.sum(torch.logical_and(mask_pred_one, mask_label_one))
        fp = torch.sum(torch.logical_and(mask_pred_one, mask_label_zero))
        fn = torch.sum(torch.logical_and(mask_pred_zero, mask_label_one))

        f = tp/(tp+1/2*(fp+fn))
        # f_out.data = 1-f
        # cross entropy loss for all non 'PAD' tokens
        f.requires_grad = True

        return 1-f

def get_balance(y_train):

    class_dist = np.array([np.sum(np.array(y_train)==0)/len(y_train), np.sum(np.array(y_train)==1)/len(y_train), np.sum(np.array(y_train)==2)/len(y_train)])
    class_imbalance = np.ones(len(class_dist))
    for n in range(3):
        if class_dist[n] == 0:
            class_imbalance[n] = 0 #no weight if doesnt exist
        else:
            class_imbalance[n] = 1/class_dist[n]
    # Note: class 0 should never have highest weight of existing classes, otherwise gets stuck
    class_imbalance[0] = np.max([np.min(class_imbalance[1:])-0.3, 0.01]) #penalty for no sdc
    
    return [float(n) for n in class_imbalance]

def get_balance2(y_train, output_dim):

    class_dist = np.array([np.sum(np.array(y_train)==n) for n in range(output_dim)])
    print('Class distribution:', class_dist)
    # class_dist = np.array([np.sum(np.array(y_train)==0), np.sum(np.array(y_train)==1), np.sum(np.array(y_train)==2)])
    class_imbalance = np.zeros(len(class_dist))
    for n in range(len(class_dist)):
        if class_dist[n] > 0:
            class_imbalance[n] = 1/(class_dist[n])
    # Note: class 0 should never have highest weight of existing classes, otherwise gets stuck
    # class_imbalance[0] = np.max([np.min(class_imbalance[1:])-0.3, 0.01]) #penalty for no sdc
    class_imbalance = class_imbalance/max(class_imbalance)

    # class_imbalance[1:] = class_imbalance[1:]*(output_dim-1) #extra weight on recall 

    return [float(n) for n in class_imbalance]

def load_data_from_quantile_data(data_folder, target):

    fault_class_dict, _, _ = get_flt_dicts()
    files = list(Path(data_folder).glob('**/*.json'))
    file_list =[]
    for u in range(len(files)):
        if "ftraces" in str(files[u]) or 'feature' in str(files[u]): #Here NO ftraces used
            continue

        if target == 'all':
            file_list.append(files[u])
            print('added', files[u])
            continue #(no double add)

        if target in str(files[u]):
            file_list.append(files[u])
            print('added', files[u])


    # Transform to x, y vectors ---------------------------------------------------------------------
    x_list = []
    y_list = []
    lay_list = []
    # feature_labels = ['nr_act_lay_q0', 'nr_act_lay_q10', 'nr_act_lay_q25', 'nr_act_lay_q50', 'nr_act_lay_q75', 'nr_act_lay_q100', 'last_act_lay_q0', 'last_act_lay_q10', 'last_act_lay_q25', 'last_act_lay_q50', 'last_act_lay_q75', \
    #     'last_act_lay_q100', 'max_dev_act_q0', 'max_dev_act_q10', 'max_dev_act_q25', 'max_dev_act_q50', 'max_dev_act_q75', 'max_dev_act_q100'] #, 'fmap_disp_max', 'fmap_disp_sum']
    
    
    for u in range(len(file_list)):

        dt = load_json_indiv(file_list[u]) #first file
        flt_type = str(file_list[u])
        if "memory" in list(fault_class_dict.keys()):
            flt_type = flt_type.replace('neurons', 'memory')
            flt_type = flt_type.replace('weights', 'memory')

        # #rescale fmap_disp features
        # x = np.array(dt['features']['disp_max_sum'])[:,0]
        # x = x - np.min(x)
        # x = x/np.max(x)
        # v = np.array(dt['features']['disp_max_sum'])[:,1]
        # v = v - np.min(v)
        # v = v/np.max(v)
        # dt['features']['disp_max_sum'] = np.vstack((x,v)).T.tolist()

        
        # if 'neurons' not in flt_type and 'weights' not in flt_type: 
        # if we have non-platform fault
        samples = np.array(dt['glob_features']['is_sdc'])
        sdc_samples =  (samples == 1)

        # create layer list as well
        if 'sdc_lays' in dt['glob_features'].keys() and len(dt['glob_features']['sdc_lays']) > 0:
            sdc_lays = dt['glob_features']['sdc_lays']
            all_lays = -1*np.ones(len(samples))
            all_lays[sdc_samples] = sdc_lays
            lay_list.extend([int(x) for x in all_lays])

        for c in range(len(fault_class_dict)):
            if list(fault_class_dict.keys())[c] in flt_type:
                samples[sdc_samples] = list(fault_class_dict.values())[c]
                y_list.extend(samples.tolist())

        # elif 'weights' in flt_type:
        #     samples[sdc_samples] = 2
        #     y_list.extend(samples.tolist())
        # elif 'blur' in flt_type:
        #     samples[sdc_samples] = 3
        #     y_list.extend(samples.tolist())
        # elif 'noise' in flt_type:
        #     samples[sdc_samples] = 4
        #     y_list.extend(samples.tolist())
        # elif 'contrast' in flt_type:
        #     samples[sdc_samples] = 5
        #     y_list.extend(samples.tolist())
        # else:
        #     y_list.extend(samples.tolist())

        nr_samples = len(samples)
        
        print('check', u,np.array(dt['features']['q_df_by_layer']).shape)

        for v in range(nr_samples):
            x_s = []
            # x_s.extend(dt['features']['nr_act_lay'][v]) #list of 5 for 5 quantiles
            # x_s.extend(dt['features']['last_act_lay'][v]) #list of 5 for 5 quantiles
            # x_s.extend(dt['features']['max_dev_act'][v]) #list of 5 for 5 quantiles
            # # x_s.extend(dt['features']['disp_max_sum'][v]) #list of disp max, sum

            x_s.extend(dt['features']['q_df_by_layer'][v])
            

            x_list.append(x_s)
        # x_list.append(dt['features']['ftraces'])

    x_list, y_list = np.array(x_list), np.array(y_list) #features: 26K?
    lay_list = np.array(lay_list)

    return x_list, y_list, lay_list


def create_feature_labels(x_list):
    q_list = [0,10,20,30,40,50,60,70,80,90,100] #NOTE: take from hook_functions
    q_names = ['q'+str(n) for n in q_list]
    nr_layers = int(x_list.shape[1]/len(q_names))
    # feature_labels = []
    # for u in range(len(q_names)):
    #     feature_labels.extend([q_names[u] + '_lay' + str(n) for n in range(nr_layers)])
    #     [u + '_lay' +str(n) for u in q_names]
    feature_labels = flatten_list([[u + '_lay' +str(n) for u in q_names] for n in range(nr_layers)])
    return feature_labels


def condense_labels(y_in):
    y = deepcopy(y_in)
    y[y == 2] = 1 # 1: neurons ->1, 2: weighs -> 1, 3: blur ->2, 4: noise -> 2
    y[y == 3] = 2
    y[y == 4] = 2
    return y

def update_correct_list(correct_list, predicted, y, y_full, fault_class_dict):
    for n in range(len(y)):
        if predicted[n] == y[n]:
            if y[n] > 0:
                flt_key = list(fault_class_dict.keys())[list(fault_class_dict.values()).index(y_full[n])]
                correct_list[flt_key] += 1
            # else:
            #     print('no sdc')
            # fault_class_dict
            # if y_full[n] == 1:
            #     correct_list['neurons'] += 1
            # if y_full[n] == 2:
            #     correct_list['weights'] += 1
            # if y_full[n] == 3:
            #     correct_list['blur'] += 1
            # if y_full[n] == 4:
            #     correct_list['noise'] += 1
    return correct_list

def get_data(df, target='all'):

    # Load extracted quantiles: -------------
    # # # New:
    # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/yolo_coco_presum/'
    # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/yolo_kitti_presum/'
    # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/ssd_coco_presum/'
    # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/ssd_kitti_presum/'
    # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/retina_coco_presum/'
    # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/retina_kitti_presum/'
    #
    # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/resnet_imagenet_presum/'
    # data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/alexnet_imagenet_presum/'

    # NOTE: Use this to find pruning factor and for plausibility check of data completion (diagonals missing?)

    data_folder = '/nwstore/florian/LR_detector_data_auto/quantile_detection_data/' + df[0] + '_presum/'

    fault_class_dict, _, _ = get_flt_dicts()
    classes = list(fault_class_dict.keys())
    output_dim = len(classes)
    x_list, y_list, lay_list = load_data_from_quantile_data(data_folder, target)

    feature_labels = create_feature_labels(x_list)

    # TODO: fix later: quantiles are nan because summation becomes too large? replace by 1 in x_list?
    # # regularize it:
    mask_nan = np.isnan(x_list)
    mask_nan_rows = [n.any() for n in mask_nan]
    x_list = x_list[np.logical_not(mask_nan_rows),:]
    y_list = y_list[np.logical_not(mask_nan_rows)]
    if "weights" in target or "neurons" in target:
        lay_list = lay_list[np.logical_not(mask_nan_rows)] #only for 
    # mask_inf = np.isinf(x_list)
    # x_list[mask_inf] = 1
    # mn = np.min(x_list)
    # x_list = x_list - mn
    # mx = np.max(x_list)
    # x_list = x_list/mx
    # x_list[mask_inf] = 1 #fix infs
    # x_list[x_list < 0] = 0 #remove artefacts
    
    due_rate = np.sum(mask_nan_rows)/len(mask_nan_rows) if len(mask_nan_rows) > 0 else 0. 
    print('DUE det rate:', due_rate)
    
    assert not (x_list > 1).any() and not (y_list > len(fault_class_dict)-1).any() and not (x_list < 0).any() and not (y_list < 0).any(), 'features not scaled.' #Check that all features are properly normalized between [0,1]
    assert not np.isnan(x_list).any() and not np.isinf(x_list).any(), "x values contain inf or nans!"
    assert not np.isnan(y_list).any() and not np.isinf(y_list).any(), "x values contain inf or nans!"


    # # Filter for specific layers:
    # counts, bins = np.histogram(lay_list, bins=74)
    # print('counts, bins', counts, bins)
    # y_list = y_list[np.logical_or(lay_list > 42, lay_list < 0)]
    # x_list = x_list[np.logical_or(lay_list > 42, lay_list < 0)]
    

    # Note
    # x_vector features: 
    # - any layer and any quantile detected sth: 0 or 1 (dim=1)
    # - how many layers detected sth, normalized by nr of layers (per quantile, dim=5)
    # - what was the last layer that detected sth, normalized by nr of layers (per quantile, dim=5)
    # - what was the maximum deviation from bound, normalized to largest value (per quantile, dim = 5)


    # Construct classifier -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.33, random_state=1) # split 1/3 (test) to 2/3 (train)

    return X_train, X_test, y_train, y_test, feature_labels, output_dim, classes, fault_class_dict


def train_net(model, X_train, y_train, bs, class_imbalance, learning_rate):
    
    number_of_bs = int(X_train.shape[0]/bs)
    correct = 0
    # correct_list = dict(zip(list(fault_class_dict.keys())[1:], list(np.zeros(output_dim-1, dtype=int))))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for nr_b in range(number_of_bs):
        # print('batch nr', nr_b, '/', int(X_train.shape[0]/bs))

        x = X_train[nr_b*bs:(nr_b+1)*bs, :]
        y = y_train[nr_b*bs:(nr_b+1)*bs]
        # y_full = y_train[nr_b*bs:(nr_b+1)*bs]
        # weight_train = torch.ones(bs)
        # weight_train[y!=1] = class_imbalance
        weight_train = torch.tensor(class_imbalance)
        criterion_train = torch.nn.CrossEntropyLoss(weight=weight_train) #weight = weight_train
        # criterion_train = torch.nn.CrossEntropyLoss() #weight = weight_train

        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(x)

        # assert not (torch.squeeze(outputs) > 1).any() and not (y > 1).any() and not (torch.squeeze(outputs) < 0).any() and not (y < 0).any(), 'features not scaled.' #Check that all features are properly normalized between [0,1]
        loss = criterion_train(torch.squeeze(outputs), y)
        # loss = F.cross_entropy(torch.squeeze(outputs), y)

        # print(list(model.parameters())[0], list(model.parameters())[1])
        loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
        optimizer.step() # Updates weights and biases with the optimizer (SGD)

        predicted = torch.argmax(outputs, 1)
        # # print(predicted)
        correct += np.sum(predicted.detach().numpy() == y.detach().numpy())
        # correct_list = update_correct_list(correct_list, predicted, y, y_full, fault_class_dict)
        # # print(correct_list)

    total = number_of_bs*bs
    accuracy = 100 * correct/total

    return model, accuracy, loss


def test_net(model, X_test, y_test):

    # # Calculating the loss and accuracy for the train dataset -----------------
    # total = number_of_bs*bs
    # correct = np.sum(predicted.detach().numpy() == y.detach().numpy())
    # accuracy = 100 * correct/total
    # print('Train accuracy:', accuracy)
    # losses.append(loss.item())
    # Iterations.append(iter)
    criterion_test = torch.nn.CrossEntropyLoss()

    # Calculating the loss and accuracy for the test dataset -------------------
    model_copy = deepcopy(model)
    model_copy.eval()

    correct_test = 0
    total_test = 0
    outputs_test = torch.squeeze(model_copy(X_test))
    loss_test = criterion_test(outputs_test, y_test)
    
    # predicted_test = outputs_test.round().detach().numpy()
    predicted_test = torch.argmax(outputs_test, 1)

    total_test += y_test.size(0)
    correct_test += np.sum(predicted_test.detach().numpy() == y_test.detach().numpy())
    accuracy_test = 100 * correct_test/total_test
    # losses_test.append(loss_test.item())
    
    
    # Test , R, P:
    conf_matrix = get_tp_fp_fn_multiclass_all_classes(X_test, y_test, model_copy, None) #TODO TODO why not working for multiclass?
                    
    return conf_matrix, accuracy_test, loss_test

def main():

    ##########################################################################################################
    # Selected data
    df = ['yolo_kitti']

    target = 'all' #'noise', 'blur', 'blur_noise', 'neurons', 'weights', 'neurons_weights', 'all'

    epochs = 1000
    # input_dim = X_train.shape[1] # Two inputs x1 and x2
    learning_rate = 0.0001 #0.1, 0.05 #L2: 0.003

    bs = 100
    save_trained_model = False
    N_attempts = 1
    ##########################################################################################################
    
    X_train, X_test, y_train, y_test, feature_labels, output_dim, classes, fault_class_dict = get_data(df, target)


    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train).to(torch.long),torch.Tensor(y_test).to(torch.long)
    


    # Do training: ---------------------------------------------------
    # Preparation:
    input_dim = X_train.shape[1] # Two inputs x1 and x2 
    class_imbalance = get_balance2(y_train, output_dim)
    print('balances', class_imbalance)

    cls_mapping, cats_mapping, sdc_mapping = get_flt_dicts()
    cls_mapping = list(cls_mapping.values())
    cats_mapping = list(cats_mapping.values())
    sdc_mapping = list(sdc_mapping.values())

    # Loops:
    for n in range(N_attempts): #different initializations? TODO

        # # Define model architecture ------------------------------
        # model = LogisticRegression(input_dim,output_dim)
        # model = FccDetector_L2(input_dim,output_dim)
        model = FccDetector_L4(input_dim,output_dim)
        


        losses = []
        losses_test = []
        model_best = None
        iter = 0
        model.train()

        for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
            
            model, train_accuracy, loss = train_net(model, X_train, y_train, bs, class_imbalance, learning_rate)
            print('training loss', loss, 'Train accuracy:', train_accuracy)
            if losses == [] or loss < min(losses):
                model_best = deepcopy(model) #save best model
            losses.append(loss.item())
            

            iter+=1
            # Evaluate intermediate result: ----------------------------------------
            if iter%(int(max(1,epochs/20)))==0:
                with torch.no_grad():

                    # Show train accuracy
                    # print('Train accuracy:', train_accuracy)
                    # losses.append(loss.item())

                    # Test:
                    conf_matrix, accuracy_test, loss_test = test_net(model, X_test, y_test)
                    print('Test accuracy:', accuracy_test)

                    losses_test.append(loss_test.item())
                    print('confusion matrix (real, pred):', conf_matrix)

                    p_cls, r_cls = get_tpfpfn_new(cls_mapping, conf_matrix)
                    p_cats, r_cats = get_tpfpfn_new(cats_mapping, conf_matrix)
                    p_sdc, r_sdc = get_tpfpfn_new(sdc_mapping, conf_matrix)
                    print('precision (cls, cats, sdc)', p_cls, p_cats, p_sdc, 'recall', r_cls, r_cats, r_sdc)

        # Best model:
        conf_matrix, accuracy_test, loss_test = test_net(model_best, X_test, y_test)
        print('best: confusion matrix (real, pred):', conf_matrix)

        p_cls, r_cls = get_tpfpfn_new(cls_mapping, conf_matrix)
        p_cats, r_cats = get_tpfpfn_new(cats_mapping, conf_matrix)
        p_sdc, r_sdc = get_tpfpfn_new(sdc_mapping, conf_matrix)
        print('best: precision (cls, cats, sdc)', p_cls, p_cats, p_sdc, 'recall', r_cls, r_cats, r_sdc)


    # # Save results ---------------------------------------------------------------
    if save_trained_model:
        save_path = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/detector_model_' + target + '.pt'
        torch.save(model.state_dict(), save_path)
        print('Saved', save_path)

    # # # check: load model to see if saved correctly
    # print(model.state_dict()['linear.weight'])
    # model2 = LogisticRegression(input_dim,output_dim)
    # model2.load_state_dict(torch.load(save_path))
    # print(model2.state_dict()['linear.weight'])


    # # # Analyse resulting weights --------------------------------------------
    # import math
    # b = model.state_dict()["linear.bias"].numpy()
    # w = model.state_dict()["linear.weight"].numpy()[0]

    # # Sort
    # w_sorted = [x for x,_ in sorted(zip((math.e**w).tolist(),feature_labels))][::-1]
    # feature_labels_sorted = [x for _,x in sorted(zip((math.e**w).tolist(),feature_labels))][::-1]
    # print(target)
    # # print('feature importance', w_sorted, 'bias:', (math.e**b).astype(float))
    # # print('feature labels', feature_labels_sorted)
    # assert len(w_sorted) == len(feature_labels_sorted)



    # if save_trained_model:
    #     feature_info = {'feature_importance': w_sorted, 'bias importance': [float(n) for n in (math.e**b)], 'labels_sorted': feature_labels_sorted, 'input_size': input_dim, 'output_size': output_dim, 'test_precision': p, 'test_recall': r, 'test_acc': accuracy_test, 'train_acc': accuracy}
    #     json_name = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/detector_model_' + target + '_meta.json'

    #     with open(json_name, "w") as outfile:
    #         json.dump(feature_info, outfile)
    #     print('saved:', json_name)



    # # # # Consistency check:
    # # # m = y_test == 1
    # # # x_sdc = X_test[m]
    # # # pos = np.where(np.array(feature_labels) == feature_labels_sorted[0])[0]
    # # # pos2 = np.where(np.array(feature_labels) == feature_labels_sorted[1])[0]
    # # # pred = model(x_sdc[0,:])
    # # # pred0 = torch.sigmoid(torch.tensor(np.sum(np.array(w)*np.array(x_sdc[0,:])) + b))
    # # # torch.sigmoid(torch.tensor(np.sum(np.array(w)[pos]*np.array(x_sdc[0,pos])) + b))
    # # # w[pos]*np.array(x_sdc[0,pos]) #is the main feature responsible for prediction?



# NOTE: some numbers
# Yolo coco (full features) with LR for 1000 epochs and lr=0.01: 
# precision (cls, cats, sdc) 0.7986123181115794 0.8647582603988638 0.8401073317023123 recall 0.8940391752133217 0.9455738543624377 0.9550082985690576

if __name__ == "__main__":
    main()
