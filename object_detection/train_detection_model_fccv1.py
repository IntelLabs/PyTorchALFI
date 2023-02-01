
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from copy import deepcopy
# from test_detection_model_LR import get_tp_fp_fn
import random
import torch.nn.functional as F
from train_detection_model_LR3 import get_balance2

# https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
# Feature importance is exp(weights), because of logistic regression sigmoid unit

# Prepare data: --------------------------------------------------------------------------------------
def load_json_indiv(gt_path):
    with open(gt_path) as f:
        coco_gt = json.load(f)
        f.close()
    return coco_gt

# https://dair.ai/notebooks/machine%20learning/beginner/logistic%20regression/2020/03/18/pytorch_logistic_regression.html
class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

class FccDetector(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(FccDetector, self).__init__()
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
        if self.training:
            x = self.bn(x)
        x = self.relu(x)

        x = self.fc3(x)
        if self.training:
            x = self.bn(x)
        x = self.relu(x)

        x = self.fc4(x)

        outputs = torch.sigmoid(x)
        return outputs

class FccDetector_v2(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(FccDetector_v2, self).__init__()
         n_w = 64
         self.fc1 = torch.nn.Linear(input_dim, n_w)
         self.fc2 = torch.nn.Linear(n_w, output_dim)

         self.bn = torch.nn.BatchNorm1d(num_features=n_w)
         self.relu = torch.nn.ReLU(inplace = True)

     def forward(self, x):
        x = self.fc1(x)
        if self.training:
            x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        outputs = torch.sigmoid(x)
        return outputs


def eliminate_features(X, fts_to_elim, feature_labels):
    fts_to_elim_num = []
    for u in fts_to_elim:
        fts_to_elim_num.append(np.where([m == u for m in feature_labels])[0][0])

    feature_labels_red = []
    dummy = X[:,0]
    for n in range(X.shape[1]):
        if n not in fts_to_elim_num:
            dummy = np.vstack((dummy, X[:,n]))
            feature_labels_red.append(feature_labels[n])
        else:
            print('eliminated:', feature_labels[n])
        
    return dummy[1:,:].T, feature_labels_red

def add_feature(x_orig, x_extra, feature_labels, label_extra):
    f_new = deepcopy(feature_labels)
    x_new = np.vstack((x_orig.T, x_extra)).T
    f_new.append(label_extra)
    return x_new, f_new


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


def main():

    # All data
    data_folder =  '/nwstore/florian/LR_detector_data/quantile_detection_data/maxs/'
    files = list(Path(data_folder).glob('**/*.json'))

    ##########################################################################################################
    # Selected data
    target = 'neurons_weights' #'noise', 'blur', 'blur_noise', 'neurons', 'weights', 'neurons_weights', 'all'
    ##########################################################################################################


    file_list =[]
    for u in range(len(files)):
        if "ftraces" not in str(files[u]):
            continue

        if target == 'all':
            file_list.append(files[u])
            print('added', files[u])
            continue #(no double add)

        if ('noise' in target and 'noise' in str(files[u])):
            file_list.append(files[u])
            print('added', files[u])
        if ('blur' in target and 'blur' in str(files[u])):
            file_list.append(files[u])
            print('added', files[u])
        if ('neurons' in target and 'neurons' in str(files[u])):
            file_list.append(files[u])
            print('added', files[u])
        if ('weights' in target and 'weights' in str(files[u])):
            file_list.append(files[u])
            print('added', files[u])



    # Transform to x, y vectors ---------------------------------------------------------------------
    x_list = []
    y_list = []
    # feature_labels = ['nr_act_lay_q0', 'nr_act_lay_q10', 'nr_act_lay_q25', 'nr_act_lay_q50', 'nr_act_lay_q75', 'nr_act_lay_q100', 'last_act_lay_q0', 'last_act_lay_q10', 'last_act_lay_q25', 'last_act_lay_q50', 'last_act_lay_q75', \
    #     'last_act_lay_q100', 'max_dev_act_q0', 'max_dev_act_q10', 'max_dev_act_q25', 'max_dev_act_q50', 'max_dev_act_q75', 'max_dev_act_q100']


    for u in range(len(file_list)):    # len(file_list)
        print('loading...', file_list[u])
        dt = load_json_indiv(file_list[u]) #first file
        
        y_list.extend(dt['glob_features']['is_sdc'])

        # nr_samples = len(dt['glob_features']['is_sdc'])
        
        # for v in range(nr_samples):
        #     x_s = []
        #     x_s.extend(dt['features']['ftraces'][v]) #list of 5 for 5 quantiles
        #     # x_s.extend(dt['qu_features']['last_act_lay'][v]) #list of 5 for 5 quantiles
        #     # x_s.extend(dt['qu_features']['max_dev_act'][v]) #list of 5 for 5 quantiles
        #     # mean violation together with max?

        #     x_list.append(x_s)
        x_list.extend(dt['features']['ftraces'])

    x_list, y_list = np.array(x_list), np.array(y_list) #features: 26K?


    # # regularize it:
    # mask_inf = np.isinf(x_list)
    # x_list[mask_inf] = 1000
    # mask_nan = np.isnan(x_list)
    # x_list[mask_nan] = 0
    # # mn = np.min(x_list)
    # # x_list = x_list - mn
    # # mx = np.max(x_list)
    # # x_list = x_list/mx
    # # x_list[mask_inf] = 1 #fix infs
    # # x_list[x_list < 0] = 0 #remove artefacts
    
    # assert not (x_list > 1).any() and not (y_list > 1).any() and not (x_list < 0).any() and not (y_list < 0).any(), 'features not scaled.' #Check that all features are properly normalized between [0,1]
    assert not np.isnan(x_list).any() and not np.isinf(x_list).any(), "x values contain inf or nans!"
    assert not np.isnan(y_list).any() and not np.isinf(y_list).any(), "x values contain inf or nans!"

    # Note
    # x_vector features: 
    # - any layer and any quantile detected sth: 0 or 1 (dim=1)
    # - how many layers detected sth, normalized by nr of layers (per quantile, dim=5)
    # - what was the last layer that detected sth, normalized by nr of layers (per quantile, dim=5)
    # - what was the maximum deviation from bound, normalized to largest value (per quantile, dim = 5)


    # Construct classifier -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.33, random_state=42) # split 1/3 (test) to 2/3 (train)

    # Filter out or add new features -------------------------------------------
    # to_elim = ['last_act_lay_q0', 'last_act_lay_q25', 'last_act_lay_q50', 'last_act_lay_q75', 'last_act_lay_q100']
    # to_elim = ['nr_act_lay_q0', 'nr_act_lay_q25', 'nr_act_lay_q50', 'nr_act_lay_q75', 'nr_act_lay_q100', 'last_act_lay_q0', 'last_act_lay_q25', 'last_act_lay_q50', 'last_act_lay_q75', \
    #     'last_act_lay_q100', 'max_dev_act_q0', 'max_dev_act_q50', 'max_dev_act_q75', 'max_dev_act_q100']
    # to_elim = ['nr_act_lay_q0', 'nr_act_lay_q25', 'nr_act_lay_q50', 'nr_act_lay_q75', 'nr_act_lay_q100', 'last_act_lay_q0', 'last_act_lay_q25', 'last_act_lay_q50', 'last_act_lay_q75', \
    #     'last_act_lay_q100', 'max_dev_act_q0', 'max_dev_act_q25', 'max_dev_act_q50', 'max_dev_act_q75']
    # to_elim = ['last_act_lay_q10', 'nr_act_lay_q10', 'max_dev_act_q10']
    # to_elim = ['nr_act_lay_q0', 'nr_act_lay_q10', 'nr_act_lay_q25', 'nr_act_lay_q50', 'nr_act_lay_q75', 'nr_act_lay_q100', 'last_act_lay_q0', 'last_act_lay_q10', 'last_act_lay_q25', 'last_act_lay_q50', 'last_act_lay_q75', \
    #     'last_act_lay_q100', 'max_dev_act_q0', 'max_dev_act_q10', 'max_dev_act_q50', 'max_dev_act_q75', 'max_dev_act_q100']
    # to_elim = ['nr_act_lay_q0', 'nr_act_lay_q10', 'nr_act_lay_q25', 'nr_act_lay_q50', 'nr_act_lay_q75', 'nr_act_lay_q100', 'last_act_lay_q0', 'last_act_lay_q10', 'last_act_lay_q25', 'last_act_lay_q50', 'last_act_lay_q75', \
    #     'last_act_lay_q100', 'max_dev_act_q0', 'max_dev_act_q10', 'max_dev_act_q25', 'max_dev_act_q50', 'max_dev_act_q100']
    # X_train, _ = eliminate_features(X_train, to_elim, feature_labels)
    # X_test, feature_labels = eliminate_features(X_test, to_elim, feature_labels)

    # add features?
    # X_train, _ = add_feature(X_train, X_train[:,-4]**2, feature_labels, 'max_dev_act_q25_2')
    # X_test, feature_labels = add_feature(X_test, X_test[:,-4]**2, feature_labels, 'max_dev_act_q25_2')

    #---------------------------------------------------------------------------------------------

    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train),torch.Tensor(y_test)
    # print(outputs.grad)


    ##############################################################
    epochs = 10_000
    input_dim = X_train.shape[1] # Two inputs x1 and x2 
    output_dim = 1 # Single binary output 
    learning_rate = 0.1 #0.1, 0.05
    class_imbalance = 1.0 #0.5 #get_balance2(y_train) #1.0 #0.9 #0<x<1 for weight on non-sdc class (1.0 is equal weights), 0.9 (what worked well: 20K epochs, lr 0.05, imb 0.9)

    bs = 100
    save_trained_model = True
    ################################################################

    model = FccDetector(input_dim,output_dim)


    # weight_train = torch.ones(X_train.shape[0])
    # weight_train[y_train!=1] = class_imbalance
    criterion_train = torch.nn.BCELoss() #will be replaced later
    criterion_test = torch.nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1-class_balance_sdc, class_balance_sdc]))

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(gamma = 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




    losses = []
    losses_test = []
    Iterations = []
    iter = 0
    model.train()
    number_of_bs = int(X_train.shape[0]/bs)

    for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
        
        correct = 0
        for nr_b in range(number_of_bs):
            # print('batch nr', nr_b, '/', int(X_train.shape[0]/bs))

            x = X_train[nr_b*bs:(nr_b+1)*bs, :]
            y = y_train[nr_b*bs:(nr_b+1)*bs]
            weight_train = torch.ones(bs)
            weight_train[y!=1] = class_imbalance
            criterion_train = torch.nn.BCELoss(weight = weight_train)


            optimizer.zero_grad() # Setting our stored gradients equal to zero
            outputs = model(x) #batch size = 3350, all samples?

            loss = criterion_train(torch.squeeze(outputs), y)

            # print(list(model.parameters())[0], list(model.parameters())[1])
            loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
            optimizer.step() # Updates weights and biases with the optimizer (SGD)

            correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y.detach().numpy())

        iter+=1
        # Evaluate intermediate result: ----------------------------------------
        if iter%(int(epochs/20))==0:
            with torch.no_grad():

                # Calculating the loss and accuracy for the train dataset -----------------
                total = number_of_bs*bs
                accuracy = 100 * correct/total
                losses.append(loss.item())
                Iterations.append(iter)


                # Calculating the loss and accuracy for the test dataset -------------------
                model_copy = deepcopy(model)
                model_copy.eval()

                # correct_test = 0
                # total_test = 0
                outputs_test = torch.squeeze(model_copy(X_test))
                loss_test = criterion_test(outputs_test, y_test)
                
                predicted_test = outputs_test.round().detach().numpy()
                total_test = y_test.size(0)
                correct_test = np.sum(predicted_test == y_test.detach().numpy())
                accuracy_test = 100 * correct_test/total_test
                losses_test.append(loss_test.item())
                

                # Test , R, P:
                tp, fp, fn, tn, sdc = get_tp_fp_fn(X_test, y_test, model_copy, X_test.shape[0])
                if tp+fp>0:
                    p = tp/(tp+fp)
                else:
                    p = None
                if tp+fn>0:
                    r = tp/(tp+fn)
                else:
                    r = None

                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}, P: {p}, R: {r}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")
                print('tp:', tp, 'fp:', fp, 'fn:', fn)



    # Need more data for so many features!
    # # Save results ---------------------------------------------------------------
    if save_trained_model:
        save_path = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/exp_1_fccv1_' + target + '.pt'
        torch.save(model.state_dict(), save_path)
        print('Saved', save_path)




    # # Analyse resulting weights --------------------------------------------
    # import math
    # b = model.state_dict()["linear.bias"].numpy()
    # w = model.state_dict()["linear.weight"].numpy()[0]

    # # Sort
    # w_sorted = [x for x,_ in sorted(zip((math.e**w).tolist(),feature_labels))][::-1]
    # feature_labels_sorted = [x for _,x in sorted(zip((math.e**w).tolist(),feature_labels))][::-1]
    # print(target)
    # print('feature importance', w_sorted, 'bias:', math.e**b)
    # print('feature labels', feature_labels_sorted)


if __name__ == "__main__":
    main()
