# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import random
import logging
import torch
from pytorchfi.pytorchfi import core
import torch.nn as nn
import numpy as np

def random_batch_element(pfi_model):
    return random.randint(0, pfi_model.get_total_batches() - 1)

def random_layer_selected(pfi_model):
    return pfi_model.get_random_layer_selected()

def random_layer_element(pfi_model):
    return random.randint(0, pfi_model.get_total_conv() - 1)

def random_layer_weighted(pfi_model, rnd_type):
    if rnd_type == "weights":
        weights = pfi_model.get_layer_weight_weights() 
        population = list(range(len(np.unique(pfi_model.OUTPUT_LOOKUP))))
    else:
        weights = pfi_model.get_layer_neuron_weights()
        population = list(range(pfi_model.get_total_conv()))
    ret = random.choices(population=population, weights=weights,k=1)
    return ret[0]

def random_neuron_location(pfi_model, conv=-1):
    if conv == -1:
        conv = random.randint(0, pfi_model.get_total_conv() - 1)

    c = pfi_model.get_fmaps_num(conv)
    if not c == -1:
        c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
    d = pfi_model.get_fmaps_D(conv)
    if not d == -1:
        d = random.randint(0, d - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

    return (c, d, h, w)

def random_weight_location(pfi_model, conv=-1):
    loc = list()

    if conv == -1:
        corrupt_layer = random.randint(0, pfi_model.get_total_conv() - 1)
    else:
        corrupt_layer = conv
    loc.append(corrupt_layer)

    curr_layer = 0
    layer_dim = 0
    looptest = 0
    for module_name, module in pfi_model.get_original_model().named_modules():
    #for name, param in pfi_model.get_original_model().named_parameters():
        #print(type(module))
        #if "conv" in name and "weight" in name:
        # todo verify position of k
        #print("Type {}".format(type(module)))
        if __layer_check(pfi_model, module, module_name):
            for name, param in module.named_parameters():
                if curr_layer == corrupt_layer and "weight" in name:
                    #logging.info("Layer for injection: {}".format(name))
                    layer_dim = len(param.size()) + 1
                    for dim in param.size():
                        loc.append(random.randint(0, dim - 1))
            #print (curr_layer)
            curr_layer += 1
        # print ("loop {}".format(looptest))
        # looptest += 1

    assert curr_layer == len(np.unique(pfi_model.OUTPUT_LOOKUP))
    assert len(loc) == layer_dim

    return tuple(loc)

def get_number_of_weights(pfi_model):
    # size = 0
    # for module in pfi_model.get_original_model().modules():
    #     if __layer_check(pfi_model, module):
    #         size += sum(param.numel() for name, param in module.named_parameters()
    #                     if "weight" in name)
    # return size
    return pfi_model.get_weight_num()

def get_number_of_neurons(pfi_model):
    return pfi_model.get_neuron_num()

def get_input_number_of_neurons(pfi_model):
    return pfi_model.get_input_neuron_num()
    
def __layer_check(pfi_model, module, module_name=None):
    ret = False
    if pfi_model.LAYER_TYPES_NAMED:
        check_layer_types_named = any([layer_types_named in module_name for layer_types_named in pfi_model.LAYER_TYPES_NAMED])
        ## filtering NonDynamicallyQuantizableLinear layers:
        check_layer_types_named = check_layer_types_named and not('out_proj' in module_name)
        if isinstance(module, nn.Linear) and pfi_model.LAYER_TYPE_ATT and check_layer_types_named:
            ret = True
        if isinstance(module, nn.modules.sparse.Embedding) and pfi_model.LAYER_TYPE_ATT_EMBED and check_layer_types_named:
            ret = True
        if isinstance(module, nn.Linear) and pfi_model.LAYER_TYPE_FCC and check_layer_types_named:
            ret = True
    else:
        if isinstance(module, nn.Conv2d) and pfi_model.LAYER_TYPE_CONV2D:
            ret = True
        elif isinstance(module, nn.Conv3d) and pfi_model.LAYER_TYPE_CONV3D:
            ret = True
        elif isinstance(module, nn.Linear) and pfi_model.LAYER_TYPE_FCC:
            ret = True
        elif isinstance(module, nn.modules.activation.LeakyReLU) and pfi_model.LAYER_TYPE_LeakyRelu:
            ret = True
        elif isinstance(module, nn.modules.batchnorm.BatchNorm2d) and pfi_model.LAYER_TYPE_BatchNorm:
            ret = True
    if pfi_model.LAYER_TYPES_EXCLUDE:
        check_layer_types_exclude = any([layer_types_exclude in module_name for layer_types_exclude in pfi_model.LAYER_TYPES_EXCLUDE])
        if check_layer_types_exclude:
            ret = False
    return ret

def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)

def compare_weights(orig, corr):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    orig_weights = [np for np in orig.named_parameters()]
    corr_weights = [np for np in corr.named_parameters()]
    different = False
    diffs = []
    count = 0

    for (name, p0), (name2, p1) in zip(orig_weights, corr_weights):
        if not torch.allclose(p0.to(device).data, p1.to(device).data):
            different = True
            d = torch.eq(p0.to(device).data, p1.to(device).data).int()
            d = d.cpu()
            
            diff = (d == 0).nonzero(as_tuple=False)
            count = count + len(diff)
            val0 = [p0.to(device).data[
                tuple(diff[n])].data.cpu().item()
                      for n in range(len(diff))]
            val1 = [p1.to(device).data[
                tuple(diff[n])].data.cpu().item()
                      for n in range(len(diff))]
            diffs.append({
                "conv_name": name,
                "diff_count": len(diff),
                "diff_pos": diff.data.cpu().numpy(),
                "diff_val0": val0,
                "diff_val1": val1})
    return different, count, diffs
