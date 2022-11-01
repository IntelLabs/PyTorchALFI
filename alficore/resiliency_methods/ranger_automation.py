import torch
from torch import nn as nn
import torchvision
from alficore.resiliency_methods.ranger import Ranger, Ranger_trivial, Ranger_BackFlip, Ranger_Clip, Ranger_FmapAvg, Ranger_FmapRescale
from alficore.ptfiwrap_utils.helper_functions import *
import numpy as np
# from yolov3_utils.dataset_loaders import *
# from alficore.utils.helper_functions import getdict_ranger, get_savedBounds_minmax
# from alficore.ptfiwrap_utils.visualization import *
# from modelsummary import summary
from collections.abc import Iterable
from copy import deepcopy

# "https://download.pytorch.org/models/vgg16-397923af.pth"
# https://discuss.pytorch.org/t/module-children-vs-module-modules/4551
# https://discuss.pytorch.org/t/dynamically-insert-new-layer-at-middle-of-a-pre-trained-model/66531
# https://github.com/pytorch/vision/tree/master/torchvision/models
# wget https://download.pytorch.org/models/resnet50-19c8e357.pth

# todo: replace activations? (what if leaky relu?)
# todo: make model a class?


class My_Reshape(nn.Module):
    """ Usually this is done in the forward fct implicitly, but we make it explicit for automated Ranger insertion. """
    def __init__(self):
        super(My_Reshape, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        return x


def bn_get_flatlist(bn_module):
    """
    flat module list of bottleneck without repetitions, i.e. not counting sequentials or bottlenecks.
    Checked for resnet skips.
    :param bn_module: bottleneck module
    :return: flat list of modules without any repetitions.
    """
    allmod = list(bn_module.modules())
    ind_lay = [l for l in allmod if (type(l) is not nn.Sequential) and (type(l) is not torchvision.models.resnet.Bottleneck) \
               and (type(l) is not My_Bottleneck)]
    return ind_lay

def bn_get_len(bn_module):
    """
    Find the length of the bottleneck as it appears in module list without repetitions.
    :param bn_module: bottleneck module
    :return: length (scalar)
    """
    ind_lay = bn_get_flatlist(bn_module)
    return len(ind_lay)

def find_bn_pos(mdl_list_indv):
    """
    Finds the list of Bottleneck containers from a flat module list.
    :param mdl_list_indv: flat list of modules
    :return: list of positions where bottlenecks are
    """
    types = [type(mdl) for mdl in mdl_list_indv]
    bn_ind = []
    if torchvision.models.resnet.Bottleneck in types:
        for x in range(len(types)):
            if types[x] == torchvision.models.resnet.Bottleneck:
                bn_ind.append(x)

    return bn_ind


def add_reshape(mdl_list_indv):
    """
    Adds reshape layer before first linear layer.
    :param mdl_list_indv:
    :return: mdl_list_indv
    """
    for ind, mdl in enumerate(mdl_list_indv):
        if type(mdl) == nn.Linear:
            mdl_list_indv.insert(ind, My_Reshape())
            # print('added reshape before first linear')
            break

    return mdl_list_indv


class My_Bottleneck(nn.Module):
    """
    Replaces the bottleneck container in ResNet.
    Here, Relu are explicit layers. Skip connections show up in .modules() or .children() list, but will be effective only during the forward pass.
    Does not include Ranger layers.
    :param: bottleneck module to be replaced
    :return: new bottleneck module with relu explicit, skip connection hidden in forward part.
    """

    def __init__(self, bn):
        super(My_Bottleneck, self).__init__()

        if type(bn) is not torchvision.models.resnet.Bottleneck:
            print('Can only convert Bottleneck modules into My_Bottleneck_Ranger.')
            return

        # add relus explicitly
        mdl_bn = list(bn.children())
        mdl_bn.insert(2, nn.ReLU())
        mdl_bn.insert(4+1, nn.ReLU()) #Note: inplace=True removed!

        # remove downsample sequential for main path
        dwnsmple_save = None
        if type(mdl_bn[-1]) == nn.Sequential:
            dwnsmple_save = mdl_bn[-1]  # save as list, so it will not show up in modules or children
            mdl_bn.pop(-1)

        mdl_bn.pop(-1) #pop last relu also
        self.bn_model1 = nn.Sequential(*mdl_bn)
        self.downsample = dwnsmple_save
        self.bn_model2 = nn.ReLU() #has to be split of for skip connection


    def forward(self, x):
        identity = x
        out = self.bn_model1(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.bn_model2(out)

        return out



class My_Bottleneck_Ranger(nn.Module):
    """
    Takes a MyBottleneck module and adds Ranger layers to it.
    :param: bottleneck module to be replaced
    :param: Ranger bound section, two lines for two Rangers in this bottleneck.
    :return: new bottleneck module with Ranger.
    """

    def __init__(self, bn, bnds, resil=Ranger):
        super(My_Bottleneck_Ranger, self).__init__()

        if type(bn) is not My_Bottleneck:
            print('Can only convert My_Bottleneck into My_Bottleneck_Ranger.')
            return

        self.Bounds = bnds

        mdl_bn = list(bn.bn_model1.modules())[1:]
        mdl_bn.insert(3, resil(self.Bounds[0]))
        mdl_bn.insert(7, resil(self.Bounds[1]))
        self.bn_model1 = nn.Sequential(*mdl_bn)
        self.downsample = bn.downsample
        self.bn_model2 = bn.bn_model2


    def forward(self, x):
        identity = x
        out = self.bn_model1(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.bn_model2(out)

        return out


def flatten_model(model):
    """
    Flattens the module to a list -> sequ as follows:
    - No explicit sequentials in the highest level unless they have skip connections
    - Bottlenecks remain a container because of skip connections, type: Bottleneck
    - internal structure flattened where possible
    :param model: a model to be flattened (sequential module)
    :return: a sequential module with a flattened structure (see above).
    """

    mdl_list = list(model.modules())[1:] # Create fully flat module list (contains repetitions for seqs and bottlenecks).


    mdl_list_indv = [mdl for mdl in mdl_list if type(mdl) is not nn.Sequential] # remove all sequentials
    bn_ind = find_bn_pos(mdl_list_indv) #check if network has modules with skip connections (here bottlenecks)

    # If NO skip forward (bottlenecks here) exist, we are good to go with a flat list.
    if bn_ind == []:
        mdl_list_indv = add_reshape(mdl_list_indv)
        flat_model = nn.Sequential(*mdl_list_indv)
        return flat_model
        # return mdl_list_indv?

    # Otherwise: Transform into list that is flat except for sequentials with skip connections (My_Bottleneck)
    # Remove duplicated indv layers from mdl_list_indv
    cnt_del = 0
    for pos in bn_ind:
        ind_pos = pos - cnt_del
        bn = mdl_list_indv[ind_pos]
        bn_len = bn_get_len(bn)
        del (mdl_list_indv[(ind_pos + 1):(ind_pos + 1 + bn_len)])
        cnt_del += bn_len  # so many elements less

    # Replace bottlenecks with modified bottlenecks (keep skip connection)
    bn_ind2 = find_bn_pos(mdl_list_indv)
    for x in bn_ind2:
        bln_new = My_Bottleneck(mdl_list_indv[x])
        mdl_list_indv.pop(x)
        mdl_list_indv.insert(x, bln_new)

    # Add reshape (before hidden in forward pass). Assumption is reshape is before first linear layer once.
    mdl_list_indv = add_reshape(mdl_list_indv)
    flat_model = nn.Sequential(*mdl_list_indv)
    return flat_model


def find_Ranger_targets(model):
    """
    Finds the target positions to insert Ranger layers.
    :param model:
    :return: target_list (list) # indices of layers after which ranger should be inserted
    :return: target_type_list (list) # list of the respective layers for info
    """

    # Note:
    # - Model_test can not have any direct sequentials since it was flattened, but can have My_Bottlenecks
    # - Within bottlenecks there is no propagation of ranger (relu after bnorm and before conv2d). Always 3 ReLus per bottleneck, i.e. 2xranger, last ranger is set outside
    chld_list = list(model.children())  # flat list if no bottlenecks were there, flat + bottlenecks otherwise.

    # find all ranger positions ---------
    protection_list = [My_Reshape, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d]
    target_list = []  # indices of layers after which ranger should be inserted
    target_type_list = []  # type of that layers

    for ind, mdl in enumerate(chld_list):

        # My_Bottlenecks always end with a ReLU so ranger after that.
        if type(mdl) == nn.ReLU or type(mdl) == My_Bottleneck:
            target_list.append(ind)
            target_type_list.append(mdl)
            # print(ind, type(mdl))

            # check next:
            if type(chld_list[ind + 1]) in protection_list:
                target_list.append(ind + 1)
                target_type_list.append(chld_list[ind + 1])
                # print('also', ind + 1, chld_list[ind + 1])

                go_deeper = 1
                while type(chld_list[ind + 1 + go_deeper]) in protection_list:
                    # if type(mdl_list_indv[ind + 1 + go_deeper]) in protection_list:
                    target_list.append(ind + 1 + go_deeper)
                    target_type_list.append(chld_list[ind + 1 + go_deeper])
                    # print('also', ind + 1 + go_deeper, chld_list[ind + 1 + go_deeper])
                    go_deeper += 1

    return target_list, target_type_list


def insert_Ranger(model, target_list, target_type_list, bnds_orig, resil=Ranger_trivial):
    """
    Adds Ranger layers
    :param model: flat_model where Ranger should be inserted
    :param bnds_orig: Ranger bound list
    :return: flat model with Ranger layers
    :return: bnds, updated with skip connections
    """

    if bnds_orig is None:
        print('no bounds can be assigned.')
        return model, bnds_orig

    chld_list = list(model.children())  # could be reused also, debugging
    bnds = deepcopy(bnds_orig)
    bnds_ranger = []

    # Insert layers at correct positions
    cnt = 0  # inserted layer number
    nr_bnd = -1  # nr of bound used (start with -1 so the first is 0)
    sel_bnd = [None, None]

    for nr_ind in range(len(target_list)):
        ind = target_list[nr_ind]  # index of layer in target_list
        lay = target_type_list[nr_ind]  # layer at above index
        tp = type(lay)  # type of above index layer

        # print(ind, tp)

        if tp == nn.ReLU:
            # print('relu found')
            nr_bnd += 1  # go to next ranger layer
            sel_bnd = bnds[nr_bnd]  # the bound used for that layer
            bnds_ranger.append(sel_bnd)
            chld_list.insert(ind + 1 + cnt, resil(bnds=sel_bnd))  # add ranger layer AFTER ind, with resp. bound
            cnt += 1  # count + 1 for next layer index since one Ranger layer was added.
            # print('used bound', sel_bnd)

        elif tp == My_Bottleneck:

            # Replace bn by bn_with_Ranger:
            # print('replace bn')
            nr_bnd += 1
            sel_bnd = [bnds[nr_bnd], bnds[nr_bnd + 1]]  # take next two bounds
            bnds_ranger.append(bnds[nr_bnd])
            bnds_ranger.append(bnds[nr_bnd + 1])
            nr_bnd += 1  # to match the next one
            # print('found', tp)
            bn_ranger = My_Bottleneck_Ranger(lay, sel_bnd, resil=resil)
            chld_list.pop(ind + cnt)
            chld_list.insert(ind + cnt, bn_ranger)
            # print('used bound', sel_bnd)

            # Add Ranger after bn:
            # print('last relu in bn')
            nr_bnd += 1  # go to next ranger layer
            sel_bnd1 = bnds[nr_bnd]  # the bound used for the last relu in the bn that ends here
            if nr_bnd - 3 >= 0:
                sel_bnd2 = bnds[nr_bnd - 3]  # bound of the last act before that bn
                cat_min = min(sel_bnd1[0], sel_bnd2[0])
                cat_max = max(sel_bnd1[1], sel_bnd2[1])
                sel_bnd = [cat_min, cat_max]
                bnds_ranger.append(sel_bnd)
            else:
                sel_bnd = sel_bnd1
                bnds_ranger.append(sel_bnd)

            chld_list.insert(ind + 1 + cnt, resil(bnds=sel_bnd))  # add ranger layer AFTER ind, with resp. bound
            cnt += 1  # count + 1 for next layer index since one Ranger layer was added.
            bnds[nr_bnd] = sel_bnd  # update the bound if it was changed due to skips
            # print('used bound', sel_bnd)

        else:
            # print(tp, 'found. reuse last bound.')
            chld_list.insert(ind + 1 + cnt, resil(bnds=sel_bnd))  # reuse latest bounds, dont have to change nothing
            bnds_ranger.append(sel_bnd)
            cnt += 1
            # print('used bound', sel_bnd)

    model_ranger = nn.Sequential(*chld_list)

    return model_ranger, np.array(bnds_ranger)


def get_Ranger_protection(net, bnds, resil=Ranger_trivial):
    """
    Tested for alexnet, vgg16, resnet50.
    :param net: original net, pretrained
    :param bnds: bounds for ranger, extracted from all activation fcts
    :return: net with Ranger layers.
    :return: bounds list that is modified such as not only to measure activations but also skip connections etc
    """

    model = deepcopy(net)  # weights are already loaded here, deepcoy also copies other functions like forward!
    model.eval()

    model = flatten_model(model)
    # Metadata: ---
    # 16 Bottleneck layers, + 1 Relu at the beginning = 49 relu layers
    # 16*3+1 conv layers, 1 linear layer = 50

    target_list, target_type_list = find_Ranger_targets(model)

    model, bnds_mod = insert_Ranger(model, target_list, target_type_list, bnds, resil=resil) #could give modified bounds as second parameter

    return model, bnds_mod


## Mixed ranger protection


def get_Ranger_protection_mixed(net, bnds, resil_1=Ranger, resil_2=Ranger, spec=[]):
    """
    Like get_Ranger_protection but adds protection layers of type resil_1 in all protection layers specified in spec.
    Tested for alexnet, vgg16, resnet50.
    :param net: original net, pretrained
    :param bnds: bounds for ranger, extracted from all activation fcts
    :param: Ranger method, e.g. Ranger, Ranger_Clip, ... for the special layers.
    :param: Ranger method for the remaining layers
    :param: spec is plain list of layer indices, e.g. [0,1,2] referring to the index of Ranger layers.
    :return: net with Ranger layers.
    :return: bounds list that is modified such as not only to measure activations but also skip connections etc
    """

    model = deepcopy(net)  # weights are already loaded here, deepcoy also copies other functions like forward!
    model.eval()

    model = flatten_model(model)
    # Metadata: ---
    # 16 Bottleneck layers, + 1 Relu at the beginning = 49 relu layers
    # 16*3+1 conv layers, 1 linear layer = 50

    target_list, target_type_list = find_Ranger_targets(model)

    model, bnds_mod = insert_Ranger_mixed(model, target_list, target_type_list, bnds, resil_a=resil_1, resil_b=resil_2, spec_layers=spec) #could give modified bounds as second parameter

    return model, bnds_mod



def insert_Ranger_mixed(model, target_list, target_type_list, bnds_orig, resil_a=Ranger, resil_b=Ranger, spec_layers=[]):
    """
    If layer in spec_layers is a bottleneck it will currently replace only the first protection layer in the bottleneck with new method.
    Adds Ranger layers
    :param model: flat_model where Ranger should be inserted
    :param bnds_orig: Ranger bound list
    :return: flat model with Ranger layers
    :return: bnds, updated with skip connections
    """

    if bnds_orig is None:
        print('no bounds can be assigned.')
        return model, bnds_orig

    chld_list = list(model.children())  # could be reused also, debugging
    bnds = deepcopy(bnds_orig)
    bnds_ranger = []

    # Insert layers at correct positions
    cnt = 0  # inserted layer number
    nr_bnd = -1  # nr of bound used (start with -1 so the first is 0)
    sel_bnd = [None, None]

    for nr_ind in range(len(target_list)):
        ind = target_list[nr_ind]  # index of layer in target_list
        lay = target_type_list[nr_ind]  # layer at above index
        tp = type(lay)  # type of above index layer

        if nr_ind in spec_layers: #special layer

            if tp == nn.ReLU:
                # print('relu found')
                nr_bnd += 1  # go to next ranger layer
                sel_bnd = bnds[nr_bnd]  # the bound used for that layer
                bnds_ranger.append(sel_bnd)
                chld_list.insert(ind + 1 + cnt, resil_a(bnds=sel_bnd))  # add ranger layer AFTER ind, with resp. bound
                cnt += 1  # count + 1 for next layer index since one Ranger layer was added.
                # print('used bound', sel_bnd)

            elif tp == My_Bottleneck:

                # Replace bn by bn_with_Ranger:
                # print('replace bn')
                nr_bnd += 1
                sel_bnd = [bnds[nr_bnd], bnds[nr_bnd + 1]]  # take next two bounds
                bnds_ranger.append(bnds[nr_bnd])
                bnds_ranger.append(bnds[nr_bnd + 1])
                nr_bnd += 1  # to match the next one
                # print('found', tp)
                bn_ranger = My_Bottleneck_Ranger_mixed(lay, sel_bnd, resil1=resil_a, resil2=resil_b)
                chld_list.pop(ind + cnt)
                chld_list.insert(ind + cnt, bn_ranger)
                # print('used bound', sel_bnd)

                # Add Ranger after bn:
                # print('last relu in bn')
                nr_bnd += 1  # go to next ranger layer
                sel_bnd1 = bnds[nr_bnd]  # the bound used for the last relu in the bn that ends here
                if nr_bnd - 3 >= 0:
                    sel_bnd2 = bnds[nr_bnd - 3]  # bound of the last act before that bn
                    cat_min = min(sel_bnd1[0], sel_bnd2[0])
                    cat_max = max(sel_bnd1[1], sel_bnd2[1])
                    sel_bnd = [cat_min, cat_max]
                    bnds_ranger.append(sel_bnd)
                else:
                    sel_bnd = sel_bnd1
                    bnds_ranger.append(sel_bnd)

                chld_list.insert(ind + 1 + cnt, resil_b(bnds=sel_bnd))  # add ranger layer AFTER ind, with resp. bound. Here b after bottleneck.
                cnt += 1  # count + 1 for next layer index since one Ranger layer was added.
                bnds[nr_bnd] = sel_bnd  # update the bound if it was changed due to skips
                # print('used bound', sel_bnd)

            else:
                # print(tp, 'found. reuse last bound.')
                chld_list.insert(ind + 1 + cnt, resil_a(bnds=sel_bnd))  # reuse latest bounds, dont have to change nothing
                bnds_ranger.append(sel_bnd)
                cnt += 1
                # print('used bound', sel_bnd)

        else: #other layer
            if tp == nn.ReLU:
                # print('relu found')
                nr_bnd += 1  # go to next ranger layer
                sel_bnd = bnds[nr_bnd]  # the bound used for that layer
                bnds_ranger.append(sel_bnd)
                chld_list.insert(ind + 1 + cnt, resil_b(bnds=sel_bnd))  # add ranger layer AFTER ind, with resp. bound
                cnt += 1  # count + 1 for next layer index since one Ranger layer was added.
                # print('used bound', sel_bnd)

            elif tp == My_Bottleneck:

                # Replace bn by bn_with_Ranger:
                # print('replace bn')
                nr_bnd += 1
                sel_bnd = [bnds[nr_bnd], bnds[nr_bnd + 1]]  # take next two bounds
                bnds_ranger.append(bnds[nr_bnd])
                bnds_ranger.append(bnds[nr_bnd + 1])
                nr_bnd += 1  # to match the next one
                # print('found', tp)
                bn_ranger = My_Bottleneck_Ranger(lay, sel_bnd, resil=resil_b)
                chld_list.pop(ind + cnt)
                chld_list.insert(ind + cnt, bn_ranger)
                # print('used bound', sel_bnd)

                # Add Ranger after bn:
                # print('last relu in bn')
                nr_bnd += 1  # go to next ranger layer
                sel_bnd1 = bnds[nr_bnd]  # the bound used for the last relu in the bn that ends here
                if nr_bnd - 3 >= 0:
                    sel_bnd2 = bnds[nr_bnd - 3]  # bound of the last act before that bn
                    cat_min = min(sel_bnd1[0], sel_bnd2[0])
                    cat_max = max(sel_bnd1[1], sel_bnd2[1])
                    sel_bnd = [cat_min, cat_max]
                    bnds_ranger.append(sel_bnd)
                else:
                    sel_bnd = sel_bnd1
                    bnds_ranger.append(sel_bnd)

                chld_list.insert(ind + 1 + cnt, resil_b(bnds=sel_bnd))  # add ranger layer AFTER ind, with resp. bound
                cnt += 1  # count + 1 for next layer index since one Ranger layer was added.
                bnds[nr_bnd] = sel_bnd  # update the bound if it was changed due to skips
                # print('used bound', sel_bnd)

            else:
                # print(tp, 'found. reuse last bound.')
                chld_list.insert(ind + 1 + cnt, resil_b(bnds=sel_bnd))  # reuse latest bounds, dont have to change nothing
                bnds_ranger.append(sel_bnd)
                cnt += 1
                # print('used bound', sel_bnd)

    model_ranger = nn.Sequential(*chld_list)

    return model_ranger, np.array(bnds_ranger)



class My_Bottleneck_Ranger_mixed(nn.Module):
    """
    Takes a MyBottleneck module and adds Ranger layers to it.
    :param: bottleneck module to be replaced
    :param: Ranger bound section, two lines for two Rangers in this bottleneck.
    :return: new bottleneck module with Ranger.
    """

    def __init__(self, bn, bnds, resil1=Ranger, resil2=Ranger):
        super(My_Bottleneck_Ranger_mixed, self).__init__()

        if type(bn) is not My_Bottleneck:
            print('Can only convert My_Bottleneck into My_Bottleneck_Ranger_mixed.')
            return

        self.Bounds = bnds

        mdl_bn = list(bn.bn_model1.modules())[1:]
        mdl_bn.insert(3, resil1(self.Bounds[0]))
        mdl_bn.insert(7, resil2(self.Bounds[1]))
        self.bn_model1 = nn.Sequential(*mdl_bn)
        self.downsample = bn.downsample
        self.bn_model2 = bn.bn_model2


    def forward(self, x):
        identity = x
        out = self.bn_model1(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.bn_model2(out)

        return out



# ## added for convenience only

# def get_Ranger_protection_trivial(net, bnds):
#     """
#     Tested for alexnet, vgg16, resnet50.
#     :param net: original net, pretrained
#     :param bnds: bounds for ranger, extracted from all activation fcts
#     :return: net with Ranger layers.
#     """

#     model = deepcopy(net)  # weights are already loaded here, deepcoy also copies other functions like forward!
#     model.eval()

#     model = flatten_model(model)
#     # Metadata: ---
#     # 16 Bottleneck layers, + 1 Relu at the beginning = 49 relu layers
#     # 16*3+1 conv layers, 1 linear layer = 50

#     target_list, target_type_list = find_Ranger_targets(model)

#     model, bnds_mod = insert_Ranger_trivial(model, target_list, target_type_list, bnds) #could give modified bounds as second parameter

#     return model, bnds_mod



# def insert_Ranger_trivial(model, target_list, target_type_list, bnds_orig):
#     """
#     Adds Ranger layers
#     :param model: flat_model where Ranger should be inserted
#     :param bnds_orig: Ranger bound list
#     :return: flat model with Ranger layers
#     :return: bnds, updated with skip connections
#     """

#     if bnds_orig is None:
#         print('no bounds can be assigned.')
#         return model, bnds_orig

#     chld_list = list(model.children())  # could be reused also, debugging
#     bnds = deepcopy(bnds_orig)
#     bnds_ext = []

#     # Insert layers at correct positions
#     cnt = 0  # inserted layer number
#     nr_bnd = -1  # nr of bound used (start with -1 so the first is 0)
#     sel_bnd = [None, None]

#     for nr_ind in range(len(target_list)):
#         ind = target_list[nr_ind]  # index of layer in target_list
#         lay = target_type_list[nr_ind]  # layer at above index
#         tp = type(lay)  # type of above index layer

#         # print(ind, tp)

#         if tp == nn.ReLU:
#             # print('relu found')
#             nr_bnd += 1  # go to next ranger layer
#             sel_bnd = bnds[nr_bnd]  # the bound used for that layer
#             chld_list.insert(ind + 1 + cnt, Ranger_trivial(bnds=sel_bnd))  # add ranger layer AFTER ind, with resp. bound
#             cnt += 1  # count + 1 for next layer index since one Ranger layer was added.
#             bnds_ext.append(sel_bnd)
#             # print('used bound', sel_bnd)

#         elif tp == My_Bottleneck:

#             # Replace bn by bn_with_Ranger:
#             # print('replace bn')
#             nr_bnd += 1
#             sel_bnd = [bnds[nr_bnd], bnds[nr_bnd + 1]]  # take next two bounds
#             nr_bnd += 1  # to match the next one
#             # print('found', tp)
#             bn_ranger = My_Bottleneck_Ranger(lay, sel_bnd)
#             chld_list.pop(ind + cnt)
#             chld_list.insert(ind + cnt, bn_ranger)
#             bnds_ext.append(sel_bnd[0])
#             bnds_ext.append(sel_bnd[1])
#             # print('used bound', sel_bnd)

#             # Add Ranger after bn:
#             # print('last relu in bn')
#             nr_bnd += 1  # go to next ranger layer
#             sel_bnd1 = bnds[nr_bnd]  # the bound used for the last relu in the bn that ends here
#             if nr_bnd - 3 >= 0:
#                 sel_bnd2 = bnds[nr_bnd - 3]  # bound of the last act before that bn
#                 cat_min = min(sel_bnd1[0], sel_bnd2[0])
#                 cat_max = max(sel_bnd1[1], sel_bnd2[1])
#                 sel_bnd = [cat_min, cat_max]
#             else:
#                 sel_bnd = sel_bnd1

#             chld_list.insert(ind + 1 + cnt, Ranger_trivial(bnds=sel_bnd))  # add ranger layer AFTER ind, with resp. bound
#             cnt += 1  # count + 1 for next layer index since one Ranger layer was added.
#             bnds[nr_bnd] = sel_bnd  # update the bound if it was changed due to skips
#             bnds_ext.append(sel_bnd)
#             # print('used bound', sel_bnd)

#         else:
#             # print(tp, 'found. reuse last bound.')
#             chld_list.insert(ind + 1 + cnt, Ranger_trivial(bnds=sel_bnd))  # reuse latest bounds, dont have to change nothing
#             cnt += 1
#             bnds_ext.append(sel_bnd)
#             # print('used bound', sel_bnd)

#     model_ranger = nn.Sequential(*chld_list)

#     return model_ranger, bnds_ext

