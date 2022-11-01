# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import torch
import numpy as np
import json
import os
import glob
from collections import Iterable
import subprocess
import numpy as np
# ImageNet -----------------------------------------------

class TEM_Dataloader_attr:
    """
    Test_Error_Models_Dataloader_attr:
    One class for all attributes related to dataloaders of pytorchFI-Wrapper.
    """
    def __init__(self, dl_dataset_name="CoCo2017", dl_shuffle=True, dl_random_sample=False, dl_sampleN=1000, dl_scenes=[-1], dl_sensor_channels=["None"], dl_mode="image", \
        dl_batch_size=1, dl_num_workers=1, dl_dataset_type="val", dl_device=None, dl_dirname=None, dl_transform=None, dl_img_root=None, dl_gt_json=None) -> None:
        self.dl_dataset_name    = dl_dataset_name
        self.dl_shuffle         = dl_shuffle
        self.dl_random_sample   = dl_random_sample
        self.dl_sampleN         = dl_sampleN
        self.dl_scenes          = dl_scenes
        self.dl_sensor_channels = dl_sensor_channels
        self.dl_mode            = dl_mode
        self.dl_batch_size      = dl_batch_size
        self.dl_num_workers     = dl_num_workers
        self.dl_dataset_type    = dl_dataset_type
        self.dl_device          = dl_device
        self.dl_dirname         = dl_dirname
        self.dl_transform       = dl_transform
        self.dl_img_root        = dl_img_root
        self.dl_gt_json         = dl_gt_json

def show_gpu(cuda_device, msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[cuda_device])
    
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')

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


# def save_Bounds_minmax(activations_in, bnds_name):
#     """
#     Saves Ranger bounds
#     :param activations_in: list of format [[min, max], [min, max], ... ]
#     :param bnds_name: 'Vgg16_bounds_dog' for example
#     :return: saves to a txt file in /bounds
#     """

#     bnd_path = str('./bounds/' + bnds_name + '.txt')
#     f = open(bnd_path, "w+")
#     for u in range(len(activations_in)):
#         f.write(str(activations_in[u][0]) + " , " + str(activations_in[u][1]))
#         f.write("\n")
#     f.close()

#     print('Bounds saved as ' + bnds_name)



def save_Bounds_minmax(activations_in, bnds_name):
    """
    Saves Ranger bounds
    :param activations_in: list of format [[min, max], [min, max], ... ]
    :param bnds_name: 'Vgg16_bounds_dog' for example
    :return: saves to a txt file in /bounds
    """

    bnd_path = './bounds/' + str(bnds_name) + '.txt'
    f = open(bnd_path, "w+")
    for u in range(len(activations_in)):
        sv = ""
        for v in range(len(activations_in[u])-1):
            sv += str(activations_in[u][v]) + " , "
        sv += str(activations_in[u][len(activations_in[u])-1])
        # f.write(str(activations_in[u][0]) + " , " + str(activations_in[u][1]))
        f.write(sv)
        f.write("\n")
    f.close()

    print('Bounds saved to ' + bnds_name)


def get_savedBounds_minmax(filename):
    f = open(filename, "r")
    bounds = []
    if f.mode == 'r':
        contents = f.read().splitlines()
        bounds = [u.split(',') for u in contents]
    f.close()

    # bounds = [[float(n[0]), float(n[1])] for n in bounds] #make numeric
    bounds = [[float(n) for n in m] for m in bounds] #make numeric

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

# Other
def flatten(x):
    """
    Flatten any list to a single level.
    """
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def get_max_min_lists_in(activations_in):
    """
    Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
    :param activations_in: list of tuple of tensors
    :param activations_out: list of tensors
    :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
    :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
    """
    #Note:
    #activations_in structure: nr ranger layers, silly bracket, batch size, channel, height, width
    #activations_out structure: nr ranger layers, batch size, channel, height, width


    # a = max([max([torch.max(activations_in[i][0][n]).tolist() for n in range(len(activations_in[i][0]))]) for i in range(len(activations_in))])
    # print('incoming max', a) #debugging

    batch_nr = activations_in[0][0].size()[0]
    nr_rangers = len(activations_in)
    activations_in2 = []

    for b in range(batch_nr): #walk through a batch, i.e. through images
        ranger_list_in = []

        for r in range(nr_rangers):

            rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
            rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
            ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])

            # if rmax_perIm_in > 100:
            #     print('in getfct', rmax_perIm_in, rmax_perIm_out) #todo

        activations_in2.append(ranger_list_in)

    return np.array(activations_in2)


def get_max_min_lists(activations_in, activations_out, get_perc=False):
    """
    Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
    :param activations_in: list of tuple of tensors
    :param activations_out: list of tensors
    :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
    :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
    """
    #Note:
    #activations_in structure: nr ranger layers, silly bracket, batch size, channel, height, width
    #activations_out structure: nr ranger layers, batch size, channel, height, width

    
    batch_nr = activations_out[0].size()[0]
    nr_rangers = len(activations_out)
    activations_in2 = []
    activations_out2 = []

    for b in range(batch_nr): #walk through a batch (here usually just 1)
        ranger_list_in = []
        ranger_list_out = []

        for r in range(nr_rangers): #walk through a layer
            if activations_in is not None:
                rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
                rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
                ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])



            rmax_perIm_out = torch.max(activations_out[r][b]).tolist()
            rmin_perIm_out = torch.min(activations_out[r][b]).tolist()
            if get_perc:
                act_num = activations_out[r][b].cpu().numpy()
                p25 = np.percentile(act_num, 25)
                p50 = np.percentile(act_num, 50)
                p75 = np.percentile(act_num, 75)
                # p0 = min, p100 = max

                ranger_list_out.append([rmin_perIm_out, p25, p50, p75, rmax_perIm_out])
            else:
                ranger_list_out.append([rmin_perIm_out, rmax_perIm_out])


        if activations_in is not None:
            activations_in2.append(ranger_list_in)
        activations_out2.append(ranger_list_out)

    return np.array(activations_in2), np.array(activations_out2)
