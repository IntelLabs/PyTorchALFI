# import torch
# import numpy as np
# # from copy import deepcopy
# import psutil
# import sys
# # from tqdm import tqdm
# # import collections
# import gc
# from util.hook_functions import set_ranger_hooks, set_ranger_hooks_ReLU, get_max_min_lists
# from util.helper_functions import save_Bounds_minmax
# from torch.autograd import Variable
# import pickle

import torch
import numpy as np
# from copy import deepcopy
import psutil
import sys
from tqdm import tqdm
# import collections
import gc
from util.hook_functions import set_ranger_hooks, set_ranger_hooks_ReLU, get_max_min_lists
from alficore.dataloader.objdet_baseClasses.common import pad_to_square, resize
from util.helper_functions import save_Bounds_minmax
from torch.autograd import Variable
import pickle
import torch
from typing import Dict, List

cuda_device = 0
if torch.cuda.is_available():
    device = torch.device(
        "cuda:{}".format(cuda_device))    
else:
    device = torch.device("cpu")



cuda_device = 0
if torch.cuda.is_available():
    device = torch.device(
        "cuda:{}".format(cuda_device))    
else:
    device = torch.device("cpu")
# Activation/bound extraction -----------------

def update_act_minmaxlists(act_in_global_min, act_in_global_max, act_in):
    """
    Update the global separate lists for min and max with the data of the current data in act_in (min, max in a single list)
    :param act_in_global_min: array of all mins (Nx0 shape)
    :param act_in_global_max: array of all maxs (Nx0 shape)
    :param act_in: array of shape BxNx2 with mins and max, B batch number, N numer of ranger layers
    :return: act_in_global_min: array of all mins (Nx0 shape)
    :return: act_in_global_max: array of all mins (Nx0 shape)
    """

    for btch in range(len(act_in)):
        act_in_btch = act_in[btch]

        if act_in_global_min is None: #first time
            act_in_global_min = np.array(act_in_btch[:, 0])
        else:
            act_in_global_min = np.vstack((act_in_global_min, act_in_btch[:, 0]))  # makes stack of rows of format Nx0
            act_in_global_min = np.min(act_in_global_min, axis=0)

        if act_in_global_max is None:
            act_in_global_max = np.array(act_in_btch)[:, 1]
        else:
            act_in_global_max = np.vstack((act_in_global_max, act_in_btch[:, 1]))  # rows are samples here
            act_in_global_max = np.max(act_in_global_max, axis=0)


    return act_in_global_min, act_in_global_max


def combine_act_minmaxlists(act_in_global_min, act_in_global_max):
    """
    Transforms the separate global lists for min max in one single dict.
    :param act_in_global_min: array of mins shape Nx0
    :param act_in_global_max: array of max shape Nx0
    :return: one list with [min, max] as rows
    """
    val = np.vstack((act_in_global_min, act_in_global_max))
    val = (val.T).tolist()

    return val


def extract_ranger_bounds(data_loader, net, file_name, dataset='imagenet'):
    """
    Extracts the ranger bounds from all the images in test_loader, for network net and saves them in bounds directory.
    Applies to all activation layers (ReLU) not the Ranger. Only saves the input bounds.
    :param data_loader: dataset loader, pytorch dataloader
    :param net: network (pretrained), pytorch model
    :param file_name: Name for bounds file, string
    :param dataset_name: Name of the dataset using which Ranger bounds are generated
    :return: list of min, max ranger inputs of the form list of [[min, max], [min, max], ...]
    :return: list of min, max ranger output of the form list of [[min, max], [min, max], ...]
    """

    act_in_global_min = None  # Lists that collect all mins and max across the image runs
    act_in_global_max = None
    act_out_global_min = None
    act_out_global_max = None
    net = net.to(device)
    net.eval()

    # memory debugging ---
    # pid = os.getpid()
    # py = psutil.Process(pid)
    # init_mem = py.memory_info()[0] / 2. ** 30
    # print(f"Torch Memory: Init memory = {init_mem:.3e}Gb")
    # ----------------------

    # save_input, save_output, hook_handles_in, hook_handles_out = set_ranger_hooks(net) #classes that save activations via hooks
    save_input, save_output, hook_handles_in, hook_handles_out = set_ranger_hooks_ReLU(net)  # classes that save activations via hooks

    with torch.no_grad():
        print('Extracting bounds from data set' + '(' + str(data_loader.__len__()) + ' batches)' + '...')
        # for j, (images, labels) in enumerate(data_loader):
        #     if j % 100 == 0:
        #         print(' Batch nr:', j)

        #     net(images) #have to call forward pass to activate the hooks, output doesnt matter here
        # data_loader = iter(data_loader)
        for i, x in enumerate(data_loader):
            if dataset == 'miovision':
                val_images = Variable(x['image']).to(device)
            elif dataset == 'imagenet':    
                val_images = x[0].to(device)
            elif dataset == 'MS CoCo' or dataset == 'coco':
                val_images = x[1].to(device)
            else:
                print('Please choose a supported dataset.')

            net(val_images.float())
            print('Batch number:', i)
            act_in = save_input.inputs
            act_out = save_output.outputs

            save_input.clear() #clear the hook lists, otherwise memory leakage
            save_output.clear()

            # # Check activations (extract only min, max from the full list of activations)
            act_in, act_out = get_max_min_lists(act_in, act_out)

            act_in_global_min, act_in_global_max = update_act_minmaxlists(act_in_global_min, act_in_global_max, act_in) #flat lists
            act_out_global_min, act_out_global_max = update_act_minmaxlists(act_out_global_min, act_out_global_max, act_out)

            # Debug memory ---------------------------------
            # memoryUse = py.memory_info()[0] / 2. ** 30
            # print(f"Torch Memory: Memory = {memoryUse:.3e}Gb")
            # print('object sizes net:', sys.getsizeof(net), net.__sizeof__())
            # print('object sizes output:', sys.getsizeof(output))
            # --------------------------------------------------
            gc.collect() # collect garbage

            # if j == 3: #break if smaller dataset should be used
            #     break

    # below loop required to solve memory leakage
    for i in range(len(hook_handles_in)):
        hook_handles_in[i].remove()
        hook_handles_out[i].remove()
        
    act_in_global = combine_act_minmaxlists(act_in_global_min, act_in_global_max)
    act_out_global = combine_act_minmaxlists(act_out_global_min, act_out_global_max)

    save_Bounds_minmax(act_out_global, file_name) #save outgoing (incoming?) activations, already in Bounds folder

    return act_in_global, act_out_global





def extract_ranger_bounds_objdet(data_loader, net, file_name):
    """
    Extracts the ranger bounds from all the images in test_loader, for network net and saves them in bounds directory.
    Applies to all activation layers (ReLU) not the Ranger. Only saves the input bounds.
    :param data_loader: dataset loader, pytorch dataloader
    :param net: network (pretrained), pytorch model
    :param file_name: Name for bounds file, string
    :param dataset_name: Name of the dataset using which Ranger bounds are generated
    :return: list of min, max ranger inputs of the form list of [[min, max], [min, max], ...]
    :return: list of min, max ranger output of the form list of [[min, max], [min, max], ...]
    """

    act_in_global_min = None  # Lists that collect all mins and max across the image runs
    act_in_global_max = None
    act_out_global_min = None
    act_out_global_max = None
    net = net.to(device)
    net.eval()

    # memory debugging ---
    # pid = os.getpid()
    # py = psutil.Process(pid)
    # init_mem = py.memory_info()[0] / 2. ** 30
    # print(f"Torch Memory: Init memory = {init_mem:.3e}Gb")
    # ----------------------

    # save_input, save_output, hook_handles_in, hook_handles_out = set_ranger_hooks(net) #classes that save activations via hooks
    save_input, save_output, hook_handles_in, hook_handles_out = set_ranger_hooks_ReLU(net)  # classes that save activations via hooks

    with torch.no_grad():
        print('Extracting bounds from data set' + '(' + str(data_loader.dataset_length) + ' batches)' + '...')
        # for j, (images, labels) in enumerate(data_loader):
        #     if j % 100 == 0:
        #         print(' Batch nr:', j)

        #     net(images) #have to call forward pass to activate the hooks, output doesnt matter here
        # data_loader = iter(data_loader)
        i = 0
        pbar = tqdm(total = data_loader.dataset_length)
        while data_loader.data_incoming:
            data_loader.datagen_itr()
            images = data_loader.data
            # images = preprocess_image_yolo_model(data_loader.data)
            net(images)
            act_in = save_input.inputs
            act_out = save_output.outputs

            save_input.clear() #clear the hook lists, otherwise memory leakage
            save_output.clear()

            # # Check activations (extract only min, max from the full list of activations)
            act_in, act_out = get_max_min_lists(act_in, act_out)

            act_in_global_min, act_in_global_max = update_act_minmaxlists(act_in_global_min, act_in_global_max, act_in) #flat lists
            act_out_global_min, act_out_global_max = update_act_minmaxlists(act_out_global_min, act_out_global_max, act_out)

            # Debug memory ---------------------------------
            # memoryUse = py.memory_info()[0] / 2. ** 30
            # print(f"Torch Memory: Memory = {memoryUse:.3e}Gb")
            # print('object sizes net:', sys.getsizeof(net), net.__sizeof__())
            # print('object sizes output:', sys.getsizeof(output))
            # --------------------------------------------------
            gc.collect() # collect garbage
            pbar.update(data_loader.curr_batch_size)
            i+=1
        pbar.close()
            # if j == 3: #break if smaller dataset should be used
            #     break

    # below loop required to solve memory leakage
    for i in range(len(hook_handles_in)):
        hook_handles_in[i].remove()
        hook_handles_out[i].remove()
        
    act_in_global = combine_act_minmaxlists(act_in_global_min, act_in_global_max)
    act_out_global = combine_act_minmaxlists(act_out_global_min, act_out_global_max)

    save_Bounds_minmax(act_out_global, file_name) #save outgoing (incoming?) activations, already in Bounds folder

    return act_in_global, act_out_global





# Accuracy-related evaluation ------------------------------------


def evaluateAccuracy(net, test_loader, classes_gt, classes_pred, top_nr, ranger_activity):
    """
    Automatic accuracy evaluation and getting active ranger layers.
    :param net: pytorch model (with bounds)
    :param test_loader: dataset loader
    :param classes_gt: classes for ground truth
    :param classes_pred: classes for predictions (can be different from above)
    :param top_nr: top n accuracy is evaluated (e.g. 1,3,5)
    :param ranger_activity: True (active ranger layers are evaluated), False (or not).
    :return: correct (scalar, nr of correct samples)
    :return: total (scalar, nr of total images checked)
    :return: Ranger activity: Gives a list of length N (number of images), each entry in the list is number of activated ranger layers for that image.
        List is empty if flag was false.
    """

    print('Evaluate accuracy')
    # Initialize
    correct = 0
    total = 0
    batch_size = test_loader.batch_size
    ranger_act_list = []

    if ranger_activity:
        save_input, save_output = set_ranger_hooks(net) #classes that save activations via hooks

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i % 100 == 0:
                print(' Batch nr:', i)

            gt_labels = [classes_gt[n] for n in labels.tolist()]  # original labels

            output = net(images)  # have to call forward pass to activate the hooks

            # Check activations
            if ranger_activity:
                act_in, act_out = get_max_min_lists(save_input.inputs, save_output.outputs)
                save_input.clear() #clear the hook lists, otherwise memory leakage
                save_output.clear()
                activated = 0
                for n in range(len(act_in)):
                    if (act_in[n,0] < act_out[n,0]) or (act_in[n,1] > act_out[n,1]):
                        activated += 1
                        # print('in', act_in, 'out', act_out, 'active ranger')
                ranger_act_list.append(activated) #nr of active ranger layers per image


            predictions = torch.nn.functional.softmax(output, dim=1) #normalized tensors

            top_val, top_ind = torch.topk(predictions, top_nr, dim=1) #it is by default sorted
            predicted = []
            for n in range(batch_size):
                predicted.append([classes_pred[top_ind[n, m]] for m in range(top_nr)])
            # print('top val', top_val, 'top ind', top_ind)
            # print('check', predicted)
            # print('gt', gt_labels)

            total += len(gt_labels)
            correct += np.sum([gt_labels[z] in predicted[z] for z in range(batch_size)])

            # if i == 10:
            #     break


    macc = (100. * float(correct) / float(total))
    print('Accuracy of the network on the ' + str(i) + ' test images: ' + str(macc))
    return correct, total, ranger_act_list



def save_accuracy_values(list, file_name):
    """
    Saves Ranger bounds
    :param: list of format [a, b, c] to be saved
    :param file_name: name of file as string
    :return: saves to a txt file in /bounds
    """
    if list == [] or list is None:
        bnd_path = str('./result_files/' + file_name + '.txt')
        f = open(bnd_path, "w+")
        f.close()
    else:
        bnd_path = str('./result_files/' + file_name + '.txt')
        f = open(bnd_path, "w+")
        for u in range(len(list)-1):
            f.write(str(list[u]) + ", ")
        f.write(str(list[len(list)-1])) #last without comma
        f.close()

    print('Results saved as ' + file_name)


def save_metadata(a, b, c, d, file_name):
    """
    Saves 3 metadata parameters.
    :param: parameters (scalar)
    :param file_name: name of file as string
    :return: saves to a txt file in /bounds
    """

    bnd_path = str('./result_files/' + file_name + '.txt')
    f = open(bnd_path, "w+")
    f.write(str(a) + ", " + str(b) + ", " + str(c) + ", " + str(d))
    f.close()

    print('Metadata saved as ' + file_name)


def save_to_dict(dicti, file_name):
    """
    Saves a dictionary to pickle.
    :param: parameters (scalar)
    :param file_name: name of file as string
    :return: saves to a txt file in /bounds
    """

    f = open(str('./result_files/' + file_name + ".pkl"),"wb")
    pickle.dump(dicti,f)
    f.close()
    print('Dictionary saved as ' + file_name)








# def save_ranger_values(list, file_name):
#     """
#     Save ranger activations
#     """
#     bnd_path = str('./result_files/' + file_name + '.txt')
#     f = open(bnd_path, "w+")
#     for v in range(len(list)):
#         for u in range(len(list[v])-1):
#             f.write(str(list[v][u]) + ", ")
#         f.write(str(list[v][len(list[v])-1]) + "\n") #last without comma
#     f.close()

#     print('Activations saved as ' + file_name)





# (LeNet)

# def evaluateAccuracy_LeNet(net, fi_net, bnds, test_loader, inj_layer):
#     """
#     :param net: healty net (weights already loaded)
#     :param fi_net: faulty net
#     :param bnds: ranger bounds (or None)
#     :param test_loader: dataset
#     :param layer in which fault was injected
#     :return: mean accuracy, fi mean accuracy, ranger activations, fi ranger activations, precision, recall
#     """
#
#
#     # About dataset: ------------------------
#     nr_samples = len(test_loader.dataset.data)
#     batch_size = test_loader.batch_size
#
#     # Get nets ready ---------------------------
#     net.eval()
#     fi_net.eval()
#
#     if bnds is not None:
#         bounds_max = (np.array(bnds)[:, 1]).tolist()
#     else:
#         bounds_max = None
#
#
#     # Initialize
#     correct = 0
#     fi_correct = 0
#     total = 0
#     ranger_count = np.zeros(6) #per ranger layer. 0 is ranger layer 1 etc.
#     fi_ranger_count = np.zeros(6)
#
#
#     print('Checking test data set...')
#     with torch.no_grad():
#         i = 0
#         for data in test_loader:
#             images, labels = data
#
#             # normal net without fault --------------
#             outputs, list_act = net(images, bnds, True)
#
#             if bounds_max is not None:
#                 ranger_active_layers = get_active_ranger(list_act, bounds_max) #note: counts batches not images
#                 ranger_count = ranger_count + ranger_active_layers
#
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             # net WITH fault ---------------------
#             fi_outputs, fi_list_act = fi_net(images, bnds, True)  # 4x10 tensor
#
#             if bounds_max is not None:
#                 fi_ranger_active_layers = get_active_ranger(fi_list_act, bounds_max) #note: counts batches not images
#                 fi_ranger_count = fi_ranger_count + fi_ranger_active_layers
#
#             _, fi_predicted = torch.max(fi_outputs, 1)  # select highest value
#             fi_correct += (fi_predicted == labels).sum().item()
#
#             i = i + 1
#
#     macc = (100. * float(correct) / float(total))
#     print('Accuracy of the network on the 10000 test images: %d %%' % macc)
#     print('Correct classifications (absolute)', correct)
#     print('Ranger per layer activated in 2500 test batches:', ranger_count)
#
#     fi_macc = (100. * float(fi_correct) / float(total))
#     print('Accuracy of the fi network on the 10000 test images: %d %%' % fi_macc)
#     print('Correct classifications (absolute)', fi_correct)
#     print('Ranger per layer activated in 2500 test batches:', fi_ranger_count) #refers to batch samples
#
#
#     # Precision, recall
#     nr_btch = nr_samples/batch_size
#
#     if bounds_max is not None:
#         inj_faults = [0., 0., 0., 0., 0., 0.] #per layer
#         inj_faults[inj_layer] = nr_btch
#
#         tp_layer = [fi_ranger_count[n] if (inj_faults[n] >= fi_ranger_count[n]) else inj_faults[n] for n in range(len(fi_ranger_count))]
#         fp_layer = [fi_ranger_count[n] - inj_faults[n] if fi_ranger_count[n] - inj_faults[n] > 0. else 0. for n in range(len(fi_ranger_count))]
#         fn_layer = [inj_faults[n] - fi_ranger_count[n] if inj_faults[n] - fi_ranger_count[n] > 0. else 0. for n in range(len(fi_ranger_count))]
#
#         prec_layer = [np.array(tp_layer[x]) / (np.array(tp_layer[x]) + np.array(fp_layer[x])) if (np.array(tp_layer[x]) + np.array(fp_layer[x])) else np.nan for x in range(len(tp_layer))] #prec per layer
#         rec_layer = [np.array(tp_layer[x]) / (np.array(tp_layer[x]) + np.array(fn_layer[x])) if (np.array(tp_layer[x]) + np.array(fn_layer[x])) else np.nan for x in range(len(tp_layer))] #rec per layer
#         print('Precision per layer', prec_layer, 'Recall per layer', rec_layer)
#
#
#         tp = sum(tp_layer)
#         fp = sum(fp_layer)
#         fn = sum(fn_layer)
#         prec = tp/(tp+fp) if tp+fp>0 else np.nan
#         rec = tp/(tp+fn) if tp+fn>0 else np.nan
#         print('Precision', prec, 'Recall', rec)
#     else:
#         print('Ranger not active')
#         prec = np.nan
#         rec = 0.
#
#
#     return macc, fi_macc, ranger_count, fi_ranger_count, prec, rec



# def get_active_ranger(list_act, bounds_max):
#     """
#     Takes the activations list for one *batch sample*, finds the max and compares it with the bounds_max.
#     Gives back an array of dimension 1xranger layers indicating with 0 or 1 if for that batch sample the bounds were
#     exceeded before ranger cut them. For fault detection.
#     If listact is empty gives back an array of zeros. Gives +1 count for a whole batch.
#     :param list_act:
#     :param bounds_max:
#     :return: array of dimension 1xranger with 0 or 1 if for that batch sample ranger was active (1) or not (0) in that layer.
#     """
#
#     max_act = []
#     if list_act:
#         max_act = [max(sublist) for sublist in list_act]
#
#     max_act = max_act[:len(bounds_max)] #if output last layer is also saved this is ignored for the ranger_active
#     # print('max act', max_act)  # before ranger
#     # print('bounds', bounds_max)  # bounds
#
#     ranger_active_layers = np.zeros(len(bounds_max))
#     if max_act:
#         ranger_active_layers = (np.array(max_act) > np.array(bounds_max)).astype(float)
#     # print('ranger active', ranger_active_layers)
#
#     return ranger_active_layers