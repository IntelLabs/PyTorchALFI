import torch
import numpy as np
import json
import os
import glob
from collections import Iterable
import subprocess

def show_gpu(msg):

    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',

                '--format=csv,nounits,noheader'],

            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[0])


    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')

# ImageNet -----------------------------------------------


def prepare_imageNet_data():
    """
    Puts images in separate folders depending on classes.
    Renames images, i.e. shifts them.
    :return:
    """

    # label extraction and data preprocessing
    # imagenet_groundtruth = "./ILSVRC2012_validation_ground_truth.txt"
    imagenet_mapping = "./ILSVRC2012_mapping.txt"
    # imagenet_labels = "./labels.txt"
    json_read = "./imagenet_class_index.json"

    class_idx = json.load(open(json_read))

    label_dict = {}
    for i in range(len(list(class_idx.keys()))):
        class_mapping = class_idx[list(class_idx.keys())[i]][0]
        class_label = class_idx[list(class_idx.keys())[i]][1]
        label_dict[class_mapping] = class_label

    # json_read = "/home/qutub/PhD/git_repos/intel_gitlab_repos/example_images/imagenet/imagenet_class_index.json"
    # class_idx = json.load(open(json_read))

    with open(imagenet_mapping) as f:
        mapping = [line.strip() for line in f.readlines()]

    # prepare dataset
    VAL_CLASS_PATH = "./ILSVRC2012_validation_ground_truth.txt"
    VAL_DATA_PATH = "./data/imagenet_val/"
    VAL_ORI_DATA_PATH = "./data/imagenet_val_1000/*.JPEG"

    val_class = []
    with open(VAL_CLASS_PATH) as val_file:
        rows = val_file.readlines()
        for row in rows:
            row = int(row.strip())
            val_class.append(row)
    val_files = glob.glob(VAL_ORI_DATA_PATH, recursive=True)
    for file in val_files:
        file = file.replace("\\","/")
        seq_num = int(file.split("/")[-1].split("_")[-1].split(".")[0])
        print(seq_num)
        class_id = val_class[seq_num - 1]
        class_mapping = mapping[class_id - 1].split()[1]
        class_name = label_dict[class_mapping]

        if not os.path.isdir(VAL_DATA_PATH + class_name):
            os.mkdir(VAL_DATA_PATH + class_name)

        os.rename(file, VAL_DATA_PATH + class_name + "/" + file.split("/")[-1])





#onnx --------------------------------------------------

def save_torch_to_onnx(model, image_shape, onnx_file):
    dummy_input = torch.ones(image_shape, dtype=torch.float32)
    dummy_input = dummy_input[None]
    torch.onnx.export(model, dummy_input, onnx_file, export_params=True, verbose=True)



# Load and save bounds ------------------------------------


def save_Bounds_minmax(activations_in, bnds_name):
    """
    Saves Ranger bounds
    :param activations_in: list of format [[min, max], [min, max], ... ]
    :param bnds_name: 'Vgg16_bounds_dog' for example
    :return: saves to a txt file in /bounds
    """

    bnd_path = str('./bounds/' + bnds_name + '.txt')
    f = open(bnd_path, "w+")
    for u in range(len(activations_in)):
        f.write(str(activations_in[u][0]) + " , " + str(activations_in[u][1]))
        f.write("\n")
    f.close()

    print('Bounds saved as ' + bnds_name)




def get_savedBounds_minmax(filename):
    f = open(filename, "r")
    bounds = []
    if f.mode == 'r':
        contents = f.read().splitlines()
        bounds = [u.split(',') for u in contents]
    f.close()

    def str_to_fl(n):
        x = []
        for u in range(len(n)):
            x.append(float(n[u]))
        return x

    bounds = [str_to_fl(n) for n in bounds] #make numeric

    return bounds



# Load weights in presence of ranger layers -----------------------------------------------

def getdict_ranger(PATH, net):
    """
    Identifies which are the correct weight and bias layers of the modified network net, and changes the dictionary in path accordingly.
    :param PATH: path to saved dict with weights
    :param net: net with ranger
    :return: modified dict of the same form as the one in path
    """

    dict_vgg = torch.load(PATH)
    list_pa = list(net.named_parameters())
    list_weights_ranger = [list_pa[i][0] for i in range(len(list_pa))]
    # list_weights = list(dict_vgg.keys()) #for comparison
    dict_vgg_ranger = dict(zip(list_weights_ranger, list(dict_vgg.values())))

    return dict_vgg_ranger




z = 0


# Other
def flatten(x):
    """
    Flatten any list to a single level.
    """
    global z
    z += 1
    # print(z)
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


# def get_max_min_dicts(activations_in, activations_out):
#     """
#     Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
#     :param activations_in: dict
#     :param activations_out: dict
#     :return: act_in, act_out dicts of form name: layer name, value: [min, max]
#     """
#
#     activations_in2 = {}
#     for name, outputs in activations_in.items():
#         max_lists = [torch.max(torch.cat(outputs[n])).tolist() for n in range(len(outputs))]
#         min_lists = [torch.min(torch.cat(outputs[n])).tolist() for n in range(len(outputs))]
#         activations_in2[name] = [np.min(min_lists), np.max(max_lists)]
#
#     activations_out2 = {}
#     for name, outputs in activations_out.items():
#         max_lists = [torch.max(outputs[n]).tolist() for n in range(len(outputs))]
#         min_lists = [torch.min(outputs[n]).tolist() for n in range(len(outputs))]
#         activations_out2[name] = [np.min(min_lists), np.max(max_lists)]
#
#     return activations_in2, activations_out2



# LeNet -------------------------------------------------
#
# def get_savedBounds(filename):
#     f = open(filename, "r")
#     bounds = []
#     if f.mode == 'r':
#         contents = f.read().splitlines()
#         bounds = [float(u) for u in contents]
#     f.close()
#
#     return bounds
#
#
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


# VGG16 -------------------------------------------
