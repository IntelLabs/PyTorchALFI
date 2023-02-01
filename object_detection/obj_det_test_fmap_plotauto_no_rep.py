import argparse
import json
import logging.config
import os
import sys
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from alficore.models.yolov3.darknet import Darknet
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize
from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_trivial, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg
from alficore.dataloader.objdet_baseClasses.boxes import Boxes, BoxMode
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.models.yolov3.utils import non_max_suppression, rescale_boxes
from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
import json
from obj_det_evaluate_jsons import eval_image, assign_hungarian, filter_out_of_image_boxes
from alficore.evaluation.visualization.visualization import Visualizer, simple_xx_coverage, simple_visualization
from copy import deepcopy
from pathlib import Path

# sys.path.append('/home/fgeissle/ranger_repo/ranger/yolov3/')
# from yolov3.yolov3_lyft import yolov3
# from yolov3.utils.general import (non_max_suppression, print_args)
# from yolov3.utils.general import (non_max_suppression, scale_coords, print_args)
# from yolov3.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
# FILE = Path(__file__).resolve()
sys.path.append("..")
sys.path.append('/home/fgeissle/ranger_repo/ranger/yolov3_ultra')
from yolov3_ultra.yolov3 import yolov3
from yolov3_ultra.yolov3_utils.general import (non_max_suppression, scale_coords, print_args)
FILE = Path(__file__).resolve()
from yolov3_ultra.yolov3_utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox



from pytorchfi.pytorchfi.errormodels import CircularBuffer
from pytorchfi.pytorchfi.errormodels import single_bit_flip_func
from random import randrange, randint
from os.path import exists

import warnings
warnings.filterwarnings("ignore")
cuda_device = 0


def annos2instance(annos, imgsize=(1200, 1920), change_bbox_mode=False):

    out_instance = Instances(imgsize)
    boxes = [n["bbox"] for n in annos]
    if change_bbox_mode and len(boxes)>0:
        boxes = np.array(boxes)
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]
        boxes = boxes.tolist()

    classes = [n["category_id"] for n in annos]
    scores = [n["score"] if 'score' in n.keys() else 1 for n in annos]
    # if 'score' in annos[0].keys():
    #     if isinstance(annos['orig'][0][0]['tp'][0]['score'], list):
    #         scores = annos['orig'][0][0]['tp'][0]['score']
    #     else:
    #         scores = [annos['orig'][0][0]['tp'][0]['score']]
    # else:
    #     scores = [1 for n in annos]
    out_instance.set("pred_boxes", Boxes(boxes))
    out_instance.set("pred_classes", torch.tensor(classes).type(torch.ByteTensor))
    out_instance.set("scores", scores)

    return [{"instances": out_instance}]


class Map_Dict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


def run_FI_stochastic(model, images, check_mode, channel_nr=3, conv_i=0, k=0, c_i=0, h_i=0, w_i=0, bit_pos=0, injection_type='weights', cuda_device=0, clip=0, test_batch_size=1, **kwargs):
    top_nr = 1

    # pcore = single_bit_flip_func(model, parsed, c=channel_nr, h=32, w=32,
    #                              batch_size=test_batch_size, cuda_device=cuda_device)  # automate the dimensions here.
    # pcore = single_bit_flip_func.from_kwargs(model, c=channel_nr, h=416, w=416,
    #                              batch_size=test_batch_size, cuda_device=cuda_device)
    model_attr = Map_Dict({"ptf_C": channel_nr, "ptf_H":h_i, "ptf_W": w_i, "ptf_D":clip, "ptf_batch_size":test_batch_size, "rnd_value_type":'bitflip'})
    # [42, 389, 60, -1, 1, 2, 5] #weights
    # # [5, -1, 36, -1, 92, 43, 4] #neurons

    pcore = single_bit_flip_func(model, model_attr) #, cuda_device=cuda_device
                                 
    # print('bits here', bit_pos, conv_i, k, c_i, h_i, w_i)
    pcore.set_bit_loc(value=bit_pos)
    # if weight function=pytorchfi.single_bit_flip_weights
    # if neurons function=pytorchfi.single_bit_flip_signed_across_batch
    if injection_type == 'weights':
        fi_model = pcore.declare_weight_fi(conv_num=conv_i, k=k, c=c_i, clip=clip, h=h_i, w=w_i, function=pcore.single_bit_flip_weights)
    elif injection_type == 'neurons': 
        fi_model  = pcore.declare_neuron_fi(conv_num=conv_i, batch=0, c=c_i, h=h_i, w=w_i, function=pcore.single_bit_flip_signed_across_batch)
        # (orig_value, corrupt_value)
        x=0
    # fi_net = pcore.declare_weight_fi(conv_num=(conv_i, conv_i), k=(k, k+1), c=(c_i,c_i), h=(h_i, h_i), w=(w_i, w_i), value=(inj_value_i, 20))
    fi_model.eval()
    # print('check weights', fi_model.__getattr__('0').weight[k][c_i][h_i][w_i])

    # Predict ----------------------------
    with torch.no_grad():  # turn off gradient calculation
        # outputs = model(images)
        # outputs_corr = fi_model(images)
        # outputs, input_tensors, output_tensors = run_with_hooks_stochastics(model, images)  # input and output of Ranger layers
        # outputs_corr, fi_input_tensors, fi_output_tensors = run_with_hooks_stochastics(fi_model, images)
        outputs_corr, fi_input_tensors, fi_output_tensors = run_with_hooks_stochastics(fi_model, images, check_mode)


        # outputs_corr = fi_model( images)
    # if injection_type == 'neurons':
    #     orig_value  = pcore.prev_value
    #     corrupt_value = pcore.new_value
    # _, predicted = torch.max(outputs, 1)  # select highest value (~instead of softmax)
    # _, fi_predicted = torch.max(outputs_corr, 1)  # select highest value
    # torch.set_printoptions(precision=4, sci_mode=False)
    # predicted_logits = get_logit_prob(outputs, top_nr)
    # fi_predicted_logits = get_logit_prob(outputs_corr, top_nr)
    del fi_model
    return outputs_corr, fi_input_tensors, fi_output_tensors


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        # print('check min max here (out)', module, torch.min(module_out), torch.max(module_out))# 
    def clear(self):
        self.outputs = []


class SaveInput:
    def __init__(self):
        self.inputs = []

    def __call__(self, module, module_in, module_out):
        self.inputs.append(module_in)
        # print('check min max here (in)', module, torch.min(module_in), torch.max(module_in)) 
        # print('check hook', module_in[0].device.index) #check on which device the hook is set

    def clear(self):
        self.inputs = []


def run_with_hooks_stochastics(net, images, check_mode):
    """
    Creates hooks for net, executes an input image, and eliminates hooks again. Avoids GPU memory leakage issues.
    :param net: pytorch model
    :param inputs: image batch
    :return outputs_corr: (tensor) inference results
    :return activated: (list of size batchsize), nr of ranger layers that got activated in one image, collected by batch
    """
    
    # Hooks to measure Ranger activation:
    save_input = SaveInput()
    save_output = SaveOutput()
 
    hook_handles_in = [] #handles list could be used to remove hooks later, here not used
    hook_handles_out = []
 
    ranger_count_check = 0
    for name, m in net.named_modules():
        if type(m) == Ranger or type(m) == Ranger_Clip or type(m) == Ranger_trivial:
        # if type(m) == Stochastic_Monitors:
            # print('Ranger hook set')
            ranger_count_check += 1
            handle_in = m.register_forward_hook(save_input)
            handle_out = m.register_forward_hook(save_output)
 
            hook_handles_in.append(handle_in)
            hook_handles_out.append(handle_out)
 

    # Hooks to check nan/inf during inference
    from util.hook_functions import set_nan_inf_hooks, run_nan_inf_hooks
    save_nan_inf, hook_handles, hook_layer_names = set_nan_inf_hooks(net)

    outputs_corr = net(images) 
 
    print("nan/inf caught (corr):", np.array(save_nan_inf.outputs).any())
    if check_mode == "due" and not np.array(save_nan_inf.outputs).any():
        print('Corr sample should have nan/inf but none found!')
        sys.exit()
    # Clean up, check nan/infs ----------------------------------------------
    nan_dict_corr, inf_dict_corr = run_nan_inf_hooks(save_nan_inf, hook_handles, hook_layer_names) #TODO: needed?

    # Check activations ---------------------------------------
    # print('act_out vs outputs', act_out, save_output.outputs)
    input_tensors = save_input.inputs
    output_tensors = save_output.outputs
    save_input.clear()  # clear the hook lists, otherwise memory leakage
    save_output.clear()
 
    for i in range(len(hook_handles_in)):
        hook_handles_in[i].remove()
        hook_handles_out[i].remove()
    return outputs_corr, input_tensors, output_tensors


def find_fault_examples(yolov3_ErrorModel, img, annos, bit_nr, attempts, injection_type):

    yolov3 = yolov3_ErrorModel.ORIG_MODEL
    # Search interesting faults:
    # Dimensions for weights:
    layer_names = [list(yolov3.named_parameters())[x][0] for x in range(len(list(yolov3.named_parameters())))]
    layer_params = [list(yolov3.named_parameters())[x][1] for x in range(len(list(yolov3.named_parameters())))]
    conv_mask = [True if "conv" in x and "bias" not in x else False for x in layer_names] #75 of them are true (vs 72 ranger layers)
    conv_dims = [np.array(layer_params)[conv_mask][x].shape for x in range(np.sum(conv_mask))]

    # Dimensions for neurons:
    conv_neurons_dims = yolov3_ErrorModel.model_wrapper.pytorchfi.OUTPUT_SIZE

    # # which layers?
    # yolov3_ErrorModel.model_wrapper.pytorchfi.OUTPUT_SIZE
    # yolov3_ErrorModel.model_wrapper.pytorchfi.OUTPUT_LOOKUP
    # # list(yolov3.named_modules())[3]
    # convs = [list(yolov3.named_modules())[x] for x in range(len(list(yolov3.named_modules()))) if type(list(yolov3.named_modules())[x][1]) == torch.nn.Conv2d]
    # rels = [list(yolov3.named_modules())[x] for x in range(len(list(yolov3.named_modules()))) if type(list(yolov3.named_modules())[x][1]) == torch.nn.LeakyReLU]
    # list(yolov3.named_parameters())[3][1].shape

    list_crit_vs_orig = []
    list_crit_vs_gt = []
    cnt = 0
    while cnt < attempts:
        print('COUNTER', cnt)
        bit_queue = CircularBuffer()

        if injection_type == "weights":
            lay = randrange(len(conv_dims))

            conv_i=[lay]
            k=[randrange(conv_dims[lay][0])]
            c_i=[randrange(conv_dims[lay][1])]
            clip=[-1]
            h_i=[randrange(conv_dims[lay][2])]
            w_i=[randrange(conv_dims[lay][3])]

        elif injection_type == "neurons":
            lay = randrange(len(conv_neurons_dims))

            conv_i=[lay]
            # k=[randrange(conv_neurons_dims[lay][0])]
            k = [-1]
            c_i=[randrange(conv_neurons_dims[lay][1])]
            clip=[-1]
            h_i=[randrange(conv_neurons_dims[lay][2])]
            w_i=[randrange(conv_neurons_dims[lay][3])]

        
        bpos = [randint(bit_nr[0], bit_nr[1])]

        print('fault', [conv_i, k, c_i, clip, h_i, w_i, bpos])
        bit_queue.enqueue(bpos[0]) #for retina net count 3 faults per injection due to reuse?!
        bit_queue.enqueue(bpos[0])
        bit_queue.enqueue(bpos[0])
        bit_queue.enqueue(bpos[0])
        bit_queue.enqueue(bpos[0])
        bit_queue.enqueue(bpos[0])

        # print('queue', bit_queue)
        outputs_orig = yolov3(img) #no fault
        outputs_corr, fi_input_tensors, fi_output_tensors = run_FI_stochastic(yolov3, img, conv_i=conv_i[0], k=k[0], c_i=c_i[0], clip=clip[0], h_i=h_i[0], w_i=w_i[0], bit_pos=bit_queue, injection_type=injection_type, cuda_device=cuda_device)
        # outputs_corr_resil, fi_input_tensors2, fi_output_tensors2 = run_FI_stochastic(yolov3, img, conv_i=conv_i[0], k=k[0], c_i=c_i[0], clip=clip[0], h_i=h_i[0], w_i=w_i[0], bit_pos=bit_queue, injection_type=injection_type, cuda_device=cuda_device)

        

        # Simple check how bad the fault was:
        dict_real = eval_simple(annos, outputs_orig, img)
        tp = len(dict_real["tp"])
        fp = len(dict_real["fp"])
        fn = len(dict_real["fn"])
        sdc_1 = (fp + fn)/(2*tp + fp + fn)
        print('orig vs gt', 'tp', tp, 'fp', fp, 'fn', fn, 'sdc', sdc_1, 'fp_flags (bbox, class, both)', [dict_real["fp_bbox"], dict_real["fp_bbox_class"], dict_real["fp_bbox_class"]], 'iou_list', dict_real["iou_list"])

        dict_real = eval_simple(annos, outputs_corr, img)
        tp = len(dict_real["tp"])
        fp = len(dict_real["fp"])
        fn = len(dict_real["fn"])
        sdc_2 = (fp + fn)/(2*tp + fp + fn)
        print('corr vs gt', 'tp', tp, 'fp', fp, 'fn', fn, 'sdc', sdc_2, 'fp_flags (bbox, class, both)', [dict_real["fp_bbox"], dict_real["fp_bbox_class"], dict_real["fp_bbox_class"]], 'iou_list', dict_real["iou_list"])

        dict_real = eval_simple(outputs_orig, outputs_corr, img)
        tp = len(dict_real["tp"])
        fp = len(dict_real["fp"])
        fn = len(dict_real["fn"])
        sdc_3 = (fp + fn)/(2*tp + fp + fn)
        print('corr vs orig', 'tp', tp, 'fp', fp, 'fn', fn, 'sdc', sdc_3, 'fp_flags (bbox, class, both)', [dict_real["fp_bbox"], dict_real["fp_bbox_class"], dict_real["fp_bbox_class"]], 'iou_list', dict_real["iou_list"])

        if sdc_2 >= 0.1 and sdc_1 <= 0.1:
            print('fault has impact')
            list_crit_vs_gt.append([lay, k[0], c_i[0], clip[0], h_i[0], w_i[0], bpos[0]])


        if sdc_3 >= 0.1:
            print('fault has impact')
            list_crit_vs_orig.append([lay, k[0], c_i[0], clip[0], h_i[0], w_i[0], bpos[0]])

        cnt += 1 #add up

    return list_crit_vs_gt, list_crit_vs_orig


def eval_simple(outputs, outputs_corr, img, filter_ooI, mode="iou+class_labels"):

    w, h = img[0]["width"], img[0]["height"]

    bboxes_output = outputs[0]["instances"].get("pred_boxes") #format will be (x1, y1, x2, y2)?
    bboxes_output = bboxes_output.tensor.cpu().tolist()
    cls_output = outputs[0]["instances"].get("pred_classes")
    cls_output = cls_output.cpu().tolist()
    # bring to format like in json files:
    output_format = [{"image_id": 0, 'category_id': cls_output[x], 'bbox': bboxes_output[x], "score": 1, "bbox_mode": 0, "image_width": w, "image_height": h} for x in range(len(bboxes_output))] 

    bboxes_fi_output = outputs_corr[0]["instances"].get("pred_boxes")
    bboxes_fi_output = bboxes_fi_output.tensor.cpu().tolist()
    cls_fi_output = outputs_corr[0]["instances"].get("pred_classes")
    cls_fi_output = cls_fi_output.cpu().tolist()
    # bring to format like in json files:
    fi_output_format = [{"image_id": 0, 'category_id': cls_fi_output[x], 'bbox': bboxes_fi_output[x], "score": 1, "bbox_mode": 0, "image_width": w, "image_height": h} for x in range(len(bboxes_fi_output))] 


    if filter_ooI:
        bbox_mode = 0 #means xyxy when coming from network prediction directly 
        output_format = filter_out_of_image_boxes(output_format, bbox_mode)
        fi_output_format = filter_out_of_image_boxes(fi_output_format, bbox_mode)

    dict_real = eval_image(output_format, fi_output_format, 0.5,  None, mode=mode)
    return dict_real


def analyse_activations(fi_output_tensors, bnds, save_file_path):

    bnds_min = [i[0] for i in bnds]
    bnds_max = [i[1] for i in bnds]

    act_plot = fi_output_tensors
    if act_plot == []:
        print('No activations available because the model does not have any Ranger layers that detect activations. Skipping this part.')
        return

    act = [np.array(act_plot[lay].detach().cpu()) for lay in range(len(act_plot))]
    for lay in range(len(act)):
        act[lay] = act[lay][~np.isnan(act[lay])]
    # act = np.array([x[0] for x in act])

    max_vis = 1000 #replace inf, nan to make it plottable
    act_min = np.array([np.min(act[u]) if act[u] != np.array([]) else np.nan for u in range(len(act))])
    act_min[act_min < -max_vis] = -max_vis
    act_min[act_min > max_vis] = max_vis

    act_max = np.array([np.max(act[u]) if act[u] != np.array([]) else np.nan for u in range(len(act))])
    act_max[act_max < -max_vis] = -max_vis
    act_max[act_max > max_vis] = max_vis

    nr_ranger = np.sum(np.logical_or(act_min < bnds_min, act_max > bnds_max))
    print('Ranger activations:', nr_ranger)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)

    x = np.arange(len(bnds))

    ax.plot(x, bnds_min, 'r', x, bnds_max, 'r')
    ax.plot(x, act_min, 'b', x, act_max, 'b')
    plt.scatter(x, act_min, s=4, c='b', marker='o') #also plot points in case they are nan
    plt.scatter(x, act_max, s=4, c='b', marker='o') #also plot points in case they are nan

    # Plot last layers before detections -----------
    # xcoords = [57, 64, 71]
    # colors = ['k','k','k']
    # for xc,c in zip(xcoords,colors):
    #     plt.axvline(x=xc, label='line at x = {}'.format(xc), c=c, linestyle='dashed')

    plt.ylim([-30,100])
    plt.xlabel('Ranger layers')
    plt.ylabel('activation magnitude')
    
    fig.savefig(save_file_path)


def filter_by_height(annos, h, factor):
    bboxes = annos[0]["instances"].pred_boxes
    h_mask = torch.max(bboxes.tensor[:, 1], bboxes.tensor[:,3]) > h*factor

    out_instance = Instances((1200, 1920))
    out_instance.set("pred_boxes",  annos[0]["instances"].pred_boxes[h_mask])
    out_instance.set("pred_classes",  annos[0]["instances"].pred_classes[h_mask])
    out_instance.set("scores",  np.array(annos[0]["instances"].scores)[h_mask.numpy()].tolist())
    return [{"instances": out_instance}]


def set_new_ids(yolov3_ErrorModel, img_counter, ids_old, factor):
    """
    Takes the model and gets next image. Always refers to the first image in the batch!
    Counter needs to track image id to get respective annotations from gt.
    Assigns an id to next annos and plots picture with ids.
    Annos of gt are in xyxy format.
    """

    img = yolov3_ErrorModel.dataloader.datagen_iter.next() #Image

    assert yolov3_ErrorModel.dataloader.dataset_dict[img_counter+len(img)]["file_name"] == img[0]["file_name"], "img and annotation do not match!"

    annos = yolov3_ErrorModel.dataloader.dataset_dict[img_counter+len(img)]["annotations"] #Corresponding ground truth
    annos = annos2instance(annos) #transform to instances

    # Filter for foreground
    h = img[0]["height"]
    annos = filter_by_height(annos, h, factor)

    # ids_new = list(np.arange(annos[0]["instances"].__len__())) #initial ids
    ids_new = np.arange(max(ids_old)+1,max(ids_old)+1+annos[0]["instances"].__len__())
    annos[0]["ids"] = ids_new #add ids to dict

    # save_path = "plots/fmaps/test_gt_" + str(img_counter) + ".png"
    # simple_visualization(img[0], annos[0]["instances"], save_path, ids=ids_new)

    return img, annos


def pseudo_set_new_ids(yolov3_ErrorModel, img_counter, ids_old, factor):
    """
    Takes the model and gets next image. Always refers to the first image in the batch!
    Counter needs to track image id to get respective annotations from gt.
    Assigns an id to next annos and plots picture with ids.
    Annos of gt are in xyxy format.
    """

    img = yolov3_ErrorModel.dataloader.datagen_iter.next() #Image

    assert yolov3_ErrorModel.dataloader.dataset_dict[img_counter+len(img)]["file_name"] == img[0]["file_name"], "img and annotation do not match!"

    annos = yolov3_ErrorModel.dataloader.dataset_dict[img_counter+len(img)]["annotations"] #Corresponding ground truth
    annos = annos2instance(annos) #transform to instances  imgsize=(img[0]["height"], img[0]["width"])

    # # Filter for foreground
    # h = img[0]["height"]
    # annos = filter_by_height(annos, h, factor)

    # ids_new = list(np.arange(annos[0]["instances"].__len__())) #initial ids
    # ids_new = np.arange(max(ids_old)+1,max(ids_old)+1+annos[0]["instances"].__len__())
    annos[0]["ids"] = np.arange(0, annos[0]["instances"].__len__()) #add ids to dict
    

    # save_path = "plots/fmaps/test_gt_" + str(img_counter) + ".png"
    # simple_visualization(img[0], annos[0]["instances"], save_path, ids=ids_new)

    return img, annos


def annos2dictlist(annos, img_counter):
    # annos to dict for hungarian
    bboxes = annos[0]["instances"].pred_boxes.tensor.detach().numpy()
    classes = annos[0]["instances"].pred_classes.detach().numpy()
    scores = annos[0]["instances"].scores

    annos_dictlist = [{'image_id': img_counter, 'category_id': classes[x], 'bbox': bboxes[x], 'score': scores[x], 'bbox_mode': 0} for x in range(len(bboxes))]
    return annos_dictlist


def match_ids(annos_old, annos, img_counter, iou_thres):
    """
    Matches and replaces the ids in annos with counterparts from annos_old, as determind by bbox iou.
    """

    annos_old_dictlist = annos2dictlist(annos_old, img_counter-1)
    annos_dictlist = annos2dictlist(annos, img_counter)
    row_ind, col_ind, row_ind_unassigned, col_ind_unassigned, fp_flags, iou_list = assign_hungarian(annos_dictlist, annos_old_dictlist, iou_thres)
    print('before matching: ', annos[0]["ids"], 'old', annos_old[0]["ids"])
    print('ioulist', iou_list, 'assigned:', len(row_ind), len(col_ind), 'not assigned rows', row_ind_unassigned, "not assigned cols", col_ind_unassigned)

    # row_ind = row_ind[:-2]
    # col_ind = col_ind[:-2]
    print('new (assigned): ', annos[0]["ids"][row_ind], 'old (assigned)', annos_old[0]["ids"][col_ind])
    annos[0]["ids"][row_ind] = annos_old[0]["ids"][col_ind] #replace matched ids, keep unmatched ones
    print('new (all): ', annos[0]["ids"], 'old (all)', annos_old[0]["ids"])
    return annos


def prepare_annos(annos, dataset_name, imgsize):

    # Change bbox format from xywh to xyxy
    if dataset_name == 'coco2017' or dataset_name == 'coco2014':
        from alficore.dataloader.objdet_baseClasses.boxes import BoxMode
        boxes = np.array([i["bbox"] for i in annos]) #xywh
        for i in range(len(annos)):
            annos[i]["bbox"] = BoxMode.convert(boxes[i].tolist(), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    # Make annos of form instance
    annos = annos2instance(annos, imgsize=imgsize) #transform to instances

    # Transform annos class labels from 0-80 to 1-91 for comparison. This is automatically done for the ground truth already during coco_api evaluation (coco_loader id_map)
    if dataset_name == 'coco2017':
        metadata = MetadataCatalog.get('coco2017/val')

        dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        annos[0]["instances"].pred_classes = torch.tensor([reverse_id_mapping[x] for x in annos[0]["instances"].pred_classes.tolist()], dtype=torch.uint8)
    elif  dataset_name == 'coco2014':
        metadata = MetadataCatalog.get('coco2014/val')

        dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        annos[0]["instances"].pred_classes = torch.tensor([reverse_id_mapping[x] for x in annos[0]["instances"].pred_classes.tolist()], dtype=torch.uint8)

    return annos


def prepare_origs(outputs_orig, dataset_name):
    # Transform annos class labels from 0-80 to 1-91 for comparison. This is automatically done for the ground truth already during coco_api evaluation (coco_loader id_map)
    if dataset_name == 'coco2017':
        metadata = MetadataCatalog.get('coco2017/val')

        dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        outputs_orig[0]["instances"].pred_classes = torch.tensor([reverse_id_mapping[x] for x in outputs_orig[0]["instances"].pred_classes.tolist()], dtype=torch.uint8)
    elif dataset_name == 'coco2014':
        metadata = MetadataCatalog.get('coco2014/val')

        dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        outputs_orig[0]["instances"].pred_classes = torch.tensor([reverse_id_mapping[x] for x in outputs_orig[0]["instances"].pred_classes.tolist()], dtype=torch.uint8)

    return outputs_orig


def load_json_indiv(gt_path):
    with open(gt_path) as f:
        coco_gt = json.load(f)
        f.close()
    return coco_gt


def clip_boxes(outputs_orig, img):
    from obj_det_evaluate_jsons import clamp
    bbox_mode = 0 #means xyxy when coming from network prediction directly
    w, h = img[0]['width'], img[0]['height']
    bboxes = outputs_orig[0]["instances"].pred_boxes.tensor.tolist()
    bboxes_new = []

    for bbox in bboxes:
        if bbox_mode == 0: #xyxy
            bbox_new = [clamp(bbox[0], 0, w), clamp(bbox[1], 0, h), clamp(bbox[2], 0, w), clamp(bbox[3], 0, h)]
        elif bbox_mode == 1: #xywh
            x_new = clamp(bbox[0], 0, w)
            y_new = clamp(bbox[1], 0, h)
            bbox_new = [x_new, y_new, clamp(bbox[2], 0, w-x_new), clamp(bbox[3], 0, h-y_new)]
        else:
            bbox_new = bbox
        bboxes_new.append(bbox_new)

    outputs_orig[0]["instances"].pred_boxes = torch.tensor(bboxes_new)

    return outputs_orig


def search_image_ids(loader, nr_samples, ids, unann_imgs):
    ind = 0
    cnt_list = []
    img_list = []

    print('Searching image ids...')
    for n in range(nr_samples):
        # img = model_ErrorModel.dataloader.datagen_iter.next() #Image
        img = loader.datagen_iter.next() #Image
        
        if img[0]["image_id"] in ids: # 13659:
            cnt_list.append(ind)
            img_list.append(img)
            # print('image id found.')
            # break
        # TODO: dont count up if img is not annotated
        if n not in unann_imgs:
            ind += 1
        if len(cnt_list) == len(ids):
            break

    # ids and faults are in specific order, cnt_list and img_list are not:
    ids_compare = [n[0]["image_id"] for n in img_list]
    img_list_sorted = []
    cnt_list_sorted = []
    for x in ids:
        img_list_sorted.append(img_list[ids_compare.index(x)])
        cnt_list_sorted.append(cnt_list[ids_compare.index(x)])
    img_list = img_list_sorted
    cnt_list = cnt_list_sorted

    return img_list, cnt_list


def parse_opt(weights_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights_path, help='model path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='/home/qutub/PhD/git_repos/github_repos/yolov3/yolov3.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=None, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default=cuda_device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half',  default=False, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def flatten(t):
    return [item for sublist in t for item in sublist]

def get_epoch_list_sdc(results_comp2, mode):
    def flatten(t):
        return [item for sublist in t for item in sublist]

    if mode == 'sdc':
    # ids_flat = results_comp2["img_ids_sdc"]
        ids_count = results_comp2['nr_img_sdc']
    elif mode == "due":
        ids_count = results_comp2['nr_img_due']
    else:
        print('Please choose either sdc or due mode.')

    # list with epcoh nrs it belongs to
    all_ep = []
    for ep_nr in range(len(ids_count)):
        # ep_nr = 0
        if ids_count[ep_nr] > 0:
            all_ep.append([ep_nr for n in range(ids_count[ep_nr])])

    all_ep = flatten(all_ep)
    return all_ep


def add_outputs(a, b):
    # c = deepcopy(a)
    a = deepcopy(a[0]['instances'])
    b = deepcopy(b[0]['instances'])

    # pred_boxes
    a_list = a.pred_boxes.tolist()
    b_list = b.pred_boxes.tolist()
    for n in b_list:
        a_list.append(n)
    boxes = torch.tensor(a_list)
    # c[0]['instances'].num_instances = len(a_list)
    # c[0]['instances'].pred_boxes = torch.tensor(a_list)

    # classes
    a_list = a.pred_classes.tolist()
    b_list = b.pred_classes.tolist()
    for n in b_list:
        a_list.append(n)
    classes = torch.tensor(a_list)
    # c[0]['instances'].pred_classes = torch.tensor(a_list)

    # scores
    a_list = a.scores
    b_list = b.scores
    for n in b_list:
        a_list.append(n)
    scores = a_list
    # c[0]['instances'].scores = a_list

    c = Instances(image_size = a.image_size, pred_boxes=boxes, pred_classes = classes, scores=scores)

    return [{'instances': c}]


def main(argv):

    # Extract faults: -----------------
    # import pickle
    # def read_fault_file(file):
    #     _file = open(file, 'rb')
    #     return pickle.load(_file)

    # faults_path = '/home/fgeissle/ranger_repo/ranger/yolov3_test_random_sbf_weights_inj_corr_1_100_25bs_kitti_fault_locs.bin'

    # faults = read_fault_file(faults_path)
    # ## epoch = fault id
    # epoch = 45 #17, 45, 62, 64, 65, 86
    # img_id = 2  #[50, 100]
    # print(faults[:,epoch*(1000)+img_id])


    torch.cuda.empty_cache()
    logging.config.fileConfig('fi.conf')
    log = logging.getLogger()

    cuda_device = 0
    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

    ####################################################################################
    model_name = 'yolov3' #retina005, yolov3, yolov3u_silu, #detectron
    dataset_name = 'coco2017' #lyft #coco2017 'kitti' #'ppp' ## CHANGE DATASET HERE #kitti done
    check_mode = "sdc" # "sdc", "due"

    injection_type='neurons' #'neurons' #'weights'
    
    nr_faults_studied = 'all' #'all' #or 'all'
    plot_or_not = True
    get_bbox_coverage = False
    plot_example_pics = False
    # Avg boxes extracted: retina+coco (n+w), yolo+lyft (n,w), yolo+coco (n, w), yolo+kitti (n,w), frcnn+coco (n,w), frcnn+kitti (n, w)
    # retina005+coco, yolo+coco, yolo + kitti, yolo + lyft, det+ kitti, det + coco

    # results file in folder dataset/val:
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220331-125720_1_faults_[1,4]_bits/coco2017/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220331-143056_1_faults_[1,4]_bits/robo/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_2_trials/neurons_injs/per_image/objDet_20220331-151035_1_faults_[1,4]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_100_trials/neurons_injs/per_image/objDet_20220407-110545_1_faults_[0,8]_bits/robo/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_1000_trials/weights_injs/per_epoch/objDet_20220411-191405_1_faults_[0_32]_bits/ppp/val'
    # folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_1000_trials/neurons_injs/per_epoch/objDet_20220412-160634_1_faults_[0_32]_bits/ppp/val'
    # folder = "/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_100_trials/neurons_injs/per_image/objDet_20220407-110545_1_faults_[0,8]_bits/robo/val"
    folder = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220502-190646_1_faults_[0,8]_bits/coco2017/val'
    



    # Below choose true if you want to plot Ranger, if not ranger leave at false
    # Note: need to run twice to get both (with and without ranger)
    plot_ranger = False #True, False if true then ranger is used, otherwise normal corr
    plot_ranger_with_orig_imgs = False #if this is True then Ranger images are plotted for sdc due based on the orig vs corr (i.e. even if ranger orig vs ranger corr is the same)
    

    ####################################################################################
    # eval_mode = 'iou+class_labels' #'iou+class_labels' # iou+class_labels, iou
    # nr_samples = 999 #of experiment: 1000 coco, 999 kitti, lyft 1008


    try:
        link_all = folder + '/results.json'
    except:
        print('results.json file does not exist, please run evaluation first to create that file.')
    link_compact = link_all[:-5] + '_sdc_due.json'
    link_compact_ranger = link_all[:-5] + '_sdc_due_ranger.json'
    link_save = folder + '/vis/images_' + check_mode + '/'
    if not os.path.exists(link_save):
        os.makedirs(os.path.dirname(link_save))
        
    if plot_ranger:
        sfx = '_ranger' #'_ranger' # '', 'ranger'
    else:
        sfx=''

    # Sample size:
    if dataset_name == 'coco2017' or dataset_name == 'coco2014':
        nr_samples = 1000
        dl_sampleN= nr_samples # 1000 #0.0247 #for 1K images in coco
    elif dataset_name == "kitti":
        nr_samples = 999
        dl_sampleN= nr_samples # 1000 #0.1337 #for 1K images in coco
    elif dataset_name == 'lyft':
        nr_samples = 1008
        dl_sampleN= nr_samples
    elif dataset_name == "robo":
        nr_samples = 21
        dl_sampleN= nr_samples
    elif dataset_name == 'ppp':
        nr_samples = 51
        dl_sampleN= nr_samples
    else:
        nr_samples = 1000
        dl_sampleN = nr_samples


    ## set dataloader attributes
    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = False
    dl_attr.dl_batch_size  = 1
    dl_attr.dl_shuffle     = False
    dl_attr.dl_sampleN     = dl_sampleN
    # dl_attr.dl_mode        = "sequence" # "image/sequence"
    # dl_attr.dl_scenes      = [0]
    dl_attr.dl_num_workers = 4
    dl_attr.dl_device      = device
    dl_attr.dl_sensor_channels  = ["CAM_FRONT"]
    dl_attr.dl_dataset_type = 'val'
    dl_attr.dl_dataset_name     = dataset_name



    # Load data: -------------------------------------------------------------------------------------------------
    #model_none
    # model_ErrorModel = TestErrorModels_ObjDet(model=None, resil_model=None, model_name=model_name, dataset=dataset_name,\
    #     config_location=yml_file, dl_sampleN=dl_sampleN, dl_shuffle=False, device=device, ranger_detector=False) 


    if dataset_name == 'ppp':
        from alficore.dataloader.ppp_loader import PPP_obj_det_dataloader
        dataloader = PPP_obj_det_dataloader(dl_attr=dl_attr, dnn_model_name = model_name)
    if dataset_name == 'kitti':
        from alficore.dataloader.kitti_loader import Kitti2D_dataloader
        dataloader = Kitti2D_dataloader(dl_attr=dl_attr, dnn_model_name = model_name)
    if dataset_name == 'coco2017' or dataset_name == 'coco2014':
        from alficore.dataloader.coco_loader import CoCo_obj_det_dataloader
        dataloader = CoCo_obj_det_dataloader(dl_attr=dl_attr, dnn_model_name = model_name)
    if dataset_name == 'lyft':
        from alficore.dataloader.lyft_loader import Lyft_dataloader
        dataloader = Lyft_dataloader(dl_attr=dl_attr, dnn_model_name = model_name)
    if dataset_name == 'robo':
        from alficore.dataloader.robo_loader import Robo_obj_det_dataloader
        dataloader = Robo_obj_det_dataloader(dl_attr=dl_attr, dnn_model_name = model_name)



    # Automatically check faults -----------------------------

    if model_name == 'yolov3' and dataset_name == 'lyft':
        model_name = 'yolov3u_silu'

    # Load from file saved in obj_det_analysis5.py:
    # Inference results
    # json_path = model_name + "_" + dataset_name + "_" + "results_1_" + injection_type + "_backup" + suffix + ".json"
    json_path = link_all
    results_comp1 = load_json_indiv(json_path)
    unann_imgs = results_comp1['unannotated_images']
    print('loaded:', json_path)

    # Evaluated compact form
    # json_path = model_name + "_" + dataset_name + "_" + "results_1_" + injection_type + "_images" + suffix + ".json"

    # Checks with ids are used:
    if plot_ranger and not plot_ranger_with_orig_imgs:
        json_path = link_compact_ranger
    else:
        json_path = link_compact
    results_comp2 = load_json_indiv(json_path)
    print('loaded:', json_path)
    # results_orig = load_json_indiv('/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220117-020417_1_faults_[0, 31]_bits/kitti/val/orig_model/epochs/0/coco_instances_results_0_epoch.json')
    # print()

    if model_name == 'yolov3' and dataset_name == 'lyft':
        model_name = 'yolov3'

    


    # Choose sdc or due ----------------------------------------------------------------------------
    if check_mode == "sdc":
        print('SDC data:')
        fltst = results_comp2["flts_sdc"]
        ids = results_comp2["img_ids_sdc"]
        ep_list = get_epoch_list_sdc(results_comp2, 'sdc')
        ratio = np.sum(results_comp2["nr_img_sdc"])/np.sum(results_comp2["nr_img_all"])
        print('ratio of affected images', ratio, results_comp2["nr_img_sdc"], results_comp2["nr_img_all"])

        tp_target_orig = results_comp2["metrics"]["sdc"]["tpfpfn_orig"]
        tp_target_corr = results_comp2["metrics"]["sdc"]["tpfpfn_corr"]

    if check_mode == "due":
        print('DUE data:')
        fltst = results_comp2["flts_due"]
        ids = results_comp2["img_ids_due"]
        ep_list = get_epoch_list_sdc(results_comp2, 'due')
        ratio = np.sum(results_comp2["nr_img_due"])/np.sum(results_comp2["nr_img_all"])
        print('ratio of affected images', ratio, results_comp2["nr_img_due"], results_comp2["nr_img_all"])

        tp_target_orig = results_comp2["metrics"]["due"]["tpfpfn_orig"]
        tp_target_corr = results_comp2["metrics"]["due"]["tpfpfn_corr"]



    # -----------------------------------------------------------------------------------

    # Only take some faults:
    print('overall nr of candidates:', len(ids))
    if nr_faults_studied == 'all':
        nr_faults_studied = len(ids)
    tp_target_orig = tp_target_orig[:nr_faults_studied]
    tp_target_corr = tp_target_corr[:nr_faults_studied]

    fltst = np.array(fltst)[:, :nr_faults_studied]
    
    ids = ids[:nr_faults_studied]
    ep_list = ep_list[:nr_faults_studied]
    print('Epochs with sdc:', np.unique(ep_list))
    from collections import Counter
    print(Counter(ep_list).most_common())

    # manual test:

    # nan/inf caught: True 776 {'batch': 0, 'conv_num': 23, 'vtype': 'neurons', 'c': 207, 'd': -1, 'h': 18, 'w': 10, 'value': 1}
    # nan/inf caught: True 3934 {'batch': 0, 'conv_num': 14, 'vtype': 'neurons', 'c': 52, 'd': -1, 'h': 30, 'w': 21, 'value': 1}
    # nan/inf caught: True 5060 {'batch': 0, 'conv_num': 36, 'vtype': 'neurons', 'c': 392, 'd': -1, 'h': 4, 'w': 14, 'value': 1}

    # fltst = np.array([
    #         [  0,   0,   0],
    #        [  23,  14,  36],
    #        [ 207, 52,   392],
    #        [ -1,  -1,  -1],
    #        [ 18,  30,  4],
    #        [10,  21,   14],
    #        [  1,  1,   1]])
    # ids = [776, 3934, 5060]
    # fltst = np.array([
    #     [  0],
    #     [  0],
    #     [ 0],
    #     [ -1],
    #     [ 0],
    #     [0],
    #     [0]])
    # ids = [191]



    # Get image and annotations, plot images:  -------------------------------------------------------------------------
    # img_list, cnt_list = search_image_ids(model_ErrorModel, nr_samples, ids, unann_imgs) #get the images, sample, e.g. 10 samples
    img_list, cnt_list = search_image_ids(dataloader, nr_samples, ids, unann_imgs) #get the images, sample, e.g. 10 samples
    print('...done searching')
    fp_collection = {'sizes_orig': [], 'sizes_corr': [], 'scores_orig': [], 'scores_corr': [], 'classes_orig': [], 'classes_corr': [], 'fp_occ': [], 'fn_occ': []}

    # 'unannotated_images': [250, 370, 384, 431, 516, 566, 838, 862, 1250, ...]


    if plot_ranger:
        orig_str = 'orig_resil'
        corr_str = 'corr_resil'
    else:
        orig_str = 'orig'
        corr_str = 'corr'

    for AUTOMATED_ISSUE_INDEX in range(len(ids)):

        print(AUTOMATED_ISSUE_INDEX)

        
        cnt = cnt_list[AUTOMATED_ISSUE_INDEX]
        ep = ep_list[AUTOMATED_ISSUE_INDEX]
        id_current = ids[AUTOMATED_ISSUE_INDEX]
        img = img_list[AUTOMATED_ISSUE_INDEX]
        err = fltst[:, AUTOMATED_ISSUE_INDEX].tolist()
        print('info:', ep, cnt, err)





        # Replace by loaded data ---------------------------------------
        outputs_orig_tp = annos2instance(results_comp1[orig_str][ep][cnt]['tp'], change_bbox_mode=True) #for tp
        outputs_corr_tp = annos2instance(results_comp1[corr_str][ep][cnt]['tp'], change_bbox_mode=True) # for tp

        outputs_orig_fp = annos2instance(results_comp1[orig_str][ep][cnt]['fp'], change_bbox_mode=True) #for fp
        outputs_corr_fp = annos2instance(results_comp1[corr_str][ep][cnt]['fp'], change_bbox_mode=True) # for fp

        outputs_orig_fn = annos2instance(results_comp1[orig_str][ep][cnt]['fn'], change_bbox_mode=True) #for tp
        outputs_corr_fn = annos2instance(results_comp1[corr_str][ep][cnt]['fn'], change_bbox_mode=True) # for tp

        filter_ooI = True
        if filter_ooI:
            outputs_orig_tp = clip_boxes(outputs_orig_tp, img)
            outputs_corr_tp = clip_boxes(outputs_corr_tp, img)

            outputs_orig_fp = clip_boxes(outputs_orig_fp, img)
            outputs_corr_fp = clip_boxes(outputs_corr_fp, img)

            outputs_orig_fn = clip_boxes(outputs_orig_fn, img)
            outputs_corr_fn = clip_boxes(outputs_corr_fn, img)

        fp_sizes_orig = [n['bbox'][2]*n['bbox'][3] for n in results_comp1[orig_str][ep][cnt]['fp']]
        fp_sizes_corr = [n['bbox'][2]*n['bbox'][3] for n in results_comp1[corr_str][ep][cnt]['fp']]
        fp_scores_orig = [n['score'] for n in results_comp1[orig_str][ep][cnt]['fp']]
        fp_scores_corr = [n['score'] for n in results_comp1[corr_str][ep][cnt]['fp']]
        fp_classes_orig = [n['category_id'] for n in results_comp1[orig_str][ep][cnt]['fp']]
        fp_classes_corr = [n['category_id'] for n in results_comp1[corr_str][ep][cnt]['fp']]
        fp_collection['sizes_orig'].append(fp_sizes_orig)
        fp_collection['sizes_corr'].append(fp_sizes_corr)
        fp_collection['scores_orig'].append(fp_scores_orig)
        fp_collection['scores_corr'].append(fp_scores_corr)
        fp_collection['classes_orig'].append(fp_classes_orig)
        fp_collection['classes_corr'].append(fp_classes_corr)


        tp = len(results_comp1[orig_str][ep][cnt]['tp'])
        fp = len(results_comp1[orig_str][ep][cnt]['fp'])
        fn = len(results_comp1[orig_str][ep][cnt]['fn'])
        if dataset_name == 'robo':
            orig_label = "FP: " + str(fp)
        else:
            orig_label = "TP: " + str(tp) + ", " + "FP: " + str(fp) + ", " + "FN: " + str(fn)
        tp = len(results_comp1[corr_str][ep][cnt]['tp'])
        fp = len(results_comp1[corr_str][ep][cnt]['fp'])
        fn = len(results_comp1[corr_str][ep][cnt]['fn'])
        if injection_type == 'weights':
            if dataset_name == 'robo': #special as there is no ground truth
                corr_label = "FP: " + str(fp) + ", " + "b: " + str([int(err[6])][0]) + ", " + "l: " + str([int(err[0])][0])
            else:
                corr_label = "TP: " + str(tp) + ", " + "FP: " + str(fp) + ", " + "FN: " + str(fn) + ", " + "b: " + str([int(err[6])][0]) + ", " + "l: " + str([int(err[0])][0])
        elif injection_type == "neurons":
            if dataset_name == 'robo': #special as there is no ground truth
                corr_label = "FP: " + str(fp) + ", " + "b: " + str([int(err[6])][0]) + ", " + "l: " + str([int(err[1])][0])
            else:
                corr_label = "TP: " + str(tp) + ", " + "FP: " + str(fp) + ", " + "FN: " + str(fn) + ", " + "b: " + str([int(err[6])][0]) + ", " + "l: " + str([int(err[1])][0])

        
        
        if get_bbox_coverage:
            # plot_example_pics = True #False #plot blob examples

            outputs_orig_tp_fp = add_outputs(outputs_orig_tp, outputs_orig_fp)
            outputs_corr_tp_fp = add_outputs(outputs_corr_tp, outputs_corr_fp)

            # Plots for testing
            if plot_example_pics:
                output_path_orig = 'plots/blobs/' + model_name + "_" + dataset_name+ '/' + injection_type + '/' + model_name + "_" + dataset_name + '_' + str(id_current) + '_' + str(ep) + '_' + orig_str + '.png'
                simple_visualization(img[0], outputs_orig_tp[0]["instances"], output_path_orig, dataset_name, inset_str=orig_label, extra_boxes=outputs_orig_fp[0]["instances"], extra_boxes2=outputs_orig_fn[0]["instances"])
                output_path_corr = 'plots/blobs/' + model_name + "_" + dataset_name+ '/' + injection_type + '/' + model_name + "_" + dataset_name + '_' + str(id_current) + '_' + str(ep) + '_' + corr_str + '.png'
                simple_visualization(img[0], outputs_corr_tp[0]["instances"], output_path_corr, dataset_name, inset_str=corr_label, extra_boxes=outputs_corr_fp[0]["instances"], extra_boxes2=outputs_corr_fn[0]["instances"])

            # Create binary blobs
            tp_fp_frame_orig = simple_xx_coverage(img[0], outputs_orig_tp_fp[0]["instances"])
            tp_fp_frame_corr = simple_xx_coverage(img[0], outputs_corr_tp_fp[0]["instances"])
            total_area, tp_fp_area_orig = np.prod(tp_fp_frame_orig.shape[:2]), np.sum(tp_fp_frame_orig[:,:,0] > 0)

            fp_blob = (np.array(tp_fp_frame_corr).astype('int') - np.array(tp_fp_frame_orig).astype('int'))
            fp_blob[fp_blob < 0] = 0
            fp_blob[fp_blob <127] = 0
            fp_blob[fp_blob > 127] = 255

            fn_blob = (np.array(tp_fp_frame_orig).astype('int') - np.array(tp_fp_frame_corr).astype('int'))
            fn_blob[fn_blob < 0] = 0
            fn_blob[fn_blob <127] = 0
            fn_blob[fn_blob > 127] = 255

            # Plot the blobs
            if plot_example_pics:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                fig.gca().imshow(fp_blob)
                fig.gca().axis("off")
                fig.tight_layout()
                fig.savefig('plots/blobs/test_fp_blob.png', dpi=300, bbox_inches='tight',pad_inches = 0)
                fig = plt.figure()
                fig.gca().imshow(fn_blob)
                fig.gca().axis("off")
                fig.tight_layout()
                fig.savefig('plots/blobs/test_fn_blob.png', dpi=300, bbox_inches='tight',pad_inches = 0)


            # Get occupancy numbers in pixel ratios
            fp_blob_area, fn_blob_area = np.sum(fp_blob[:,:,0]>0), np.sum(fn_blob[:,:,0]>0)
            occ_fp = fp_blob_area/total_area
            if tp_fp_area_orig == 0:
                occ_fn = 0
            else:
                occ_fn = fn_blob_area/tp_fp_area_orig
            # if occ_fp <0 or occ_fp>1 or occ_fn <0 or occ_fn > 1:
            #     print()
            assert occ_fp <= 1 and occ_fp>=0 and occ_fn <= 1 and occ_fn>=0
            # if not np.isnan(occ_fn):
            #     assert occ_fn <= 1 and occ_fn>=0

            fp_collection['fp_occ'].append(occ_fp)
            fp_collection['fn_occ'].append(occ_fn)
            print(occ_fp, occ_fn, total_area, tp_fp_area_orig)
            if occ_fp > 0.05 and occ_fn > 0.05:
                print() #breakpoint here for example pics
            else:
                if plot_example_pics:
                    os.remove(output_path_orig)
                    os.remove(output_path_corr)


        # Plot ------------------------------------
        if plot_or_not:
            # link_all
            # Only orig
            # output_path = 'plots/fmaps/' + model_name + "_" + dataset_name+ '/' + injection_type + '/' + model_name + "_" + dataset_name + '_' + str(id_current) + '_' + str(ep) + sfx + '_' + orig_str + '.png'
            output_path = link_save  + model_name + "_" + dataset_name + '_' + str(ep) + '_' + str(cnt) + '_' + str(id_current)  + sfx + '_' + orig_str + '.png'
            
            if not exists(output_path):
                if dataset_name == 'robo': #special case because here gt is missing
                    simple_visualization(img[0], outputs_orig_tp[0]["instances"], output_path, dataset_name, inset_str=orig_label, extra_boxes=outputs_orig_fp[0]["instances"])
                else:
                    simple_visualization(img[0], outputs_orig_tp[0]["instances"], output_path, dataset_name, inset_str=orig_label, extra_boxes=outputs_orig_fp[0]["instances"], extra_boxes2=outputs_orig_fn[0]["instances"])
            else:
                print('file already exists, skipping...')

            # Corr
            # output_path = 'plots/fmaps/' + model_name + "_" + dataset_name + '/' + injection_type + '/' + model_name + "_" + dataset_name + '_' + str(id_current) + '_' + str(ep) + sfx + '_' + corr_str + '.png'
            output_path = link_save + model_name + "_" + dataset_name + '_' + str(ep) + '_' + str(cnt) + '_' + str(id_current)  + sfx + '_' + corr_str + '.png'
            
            
            if not exists(output_path):
                if dataset_name == 'robo': #special case because here gt is missing
                    simple_visualization(img[0], outputs_corr_tp[0]["instances"], output_path, dataset_name, inset_str=corr_label, extra_boxes=outputs_corr_fp[0]["instances"])
                else:
                    simple_visualization(img[0], outputs_corr_tp[0]["instances"], output_path, dataset_name, inset_str=corr_label, extra_boxes=outputs_corr_fp[0]["instances"], extra_boxes2=outputs_corr_fn[0]["instances"])
            else:
                print('file already exists, skipping...')

        # Analyse activations -----------------------------------
        # save_file_path = 'plots/fmaps/' + model_name + "_" + dataset_name + '/' + injection_type + '/' + model_name + "_" + dataset_name + '_' + str(id_current) + '_activations.png'
        # analyse_activations(fi_output_tensors, bnds, save_file_path)


    # # FP analysis: ----------------------------------------------------------------------------------------
    # print('analyse FPs further...')
    # sz_orig = np.mean(flatten(fp_collection['sizes_orig']))
    # sz_corr = np.mean(flatten(fp_collection['sizes_corr']))
    # sc_orig = np.mean(flatten(fp_collection['scores_orig']))
    # sc_corr = np.mean(flatten(fp_collection['scores_corr']))
    # fp_flat = fp_collection['fp_occ']
    # fp_area, fp_err = np.mean(fp_flat), np.std(fp_flat)*1.96/len(fp_flat)
    # fn_flat = fp_collection['fn_occ']
    # fn_area, fn_err = np.mean(fn_flat), np.std(fn_flat)*1.96/len(fn_flat)

    # cls_orig = flatten(fp_collection['classes_orig'])
    # cls_orig_cnt = {x:cls_orig.count(x) for x in cls_orig}

    # cls_corr = flatten(fp_collection['classes_corr'])
    # cls_corr_cnt = {x:cls_corr.count(x) for x in cls_corr}
    # print()

    # json_file = model_name + "_" + dataset_name + "_" + "results_1_" + injection_type + "_fpcheck" + suffix + ".json"
    # fp_collection_save = {'sizes': {'orig': sz_orig, 'corr': sz_corr}, 'scores': {'orig': sc_orig, 'corr': sc_corr}, 'classes': {'orig': cls_orig_cnt, 'corr': cls_corr_cnt}, 'occ': {'fp': [fp_area, fp_err], 'fn': [fn_area, fn_err]} }

    # with open(json_file, "w") as outfile: 
    #     json.dump(fp_collection_save, outfile)
    # print('saved:', json_file)



if __name__ == "__main__":
    # ctx = mp.get_context("spawn")
    # ctx.set_start_method('spawn')onc
    main(sys.argv)










