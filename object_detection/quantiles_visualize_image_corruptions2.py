import pickle
import argparse
import os, sys
from pathlib import Path
import json
import numpy as np
from copy import deepcopy
sys.path.append("/home/fgeissle/fgeissle_ranger")
# from alficore.ptfiwrap_utils.evaluate import extract_ranger_bounds, extract_ranger_bounds_auto2, get_ranger_bounds_quantiles
from alficore.ptfiwrap_utils.helper_functions import get_savedBounds_minmax
import matplotlib.pyplot as plt
import torchvision
# import tqdm
# sys.path.append("..")
# sys.path.append('/home/fgeissle/ranger_repo/ranger/yolov3_ultra')
# from yolov3_ultra.yolov3 import yolov3
# from yolov3_ultra.yolov3_utils.general import (non_max_suppression, scale_coords, print_args)
FILE = Path(__file__).resolve()
# from yolov3_ultra.yolov3_utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox

import sys
import os
from os.path import dirname as up
from collections import namedtuple
# sys.path.append(sys.argv[0])
# sys.path.insert(0, up(up(up(os.getcwd()))))
## comment this for debug
sys.path.append(up(up(up(os.getcwd()))))

## uncomment this for debug
# sys.path.append(os.getcwd())

from alficore.evaluation.visualization.visualization import Visualizer, ColorMode
from alficore.ptfiwrap_utils.utils import  read_json
from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
from PIL import Image, TiffImagePlugin
TiffImagePlugin.DEBUG = False

# from min_test import build_objdet_native_model, parse_opt
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
import torch
from os.path import exists
from obj_det_evaluate_jsons import eval_image, assign_hungarian, filter_out_of_image_boxes
from alficore.evaluation.visualization.visualization import Visualizer, simple_xx_coverage, simple_visualization, simple_visualization_direct_img
from alficore.evaluation.ivmod_metric import ivmod_metric
from alficore.dataloader.objdet_baseClasses.boxes import BoxMode, Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances

from det_quant_test_auto2 import set_up_mmdet_model, assign_val_train

def plot_img(img, file_name):
    fig, ax = plt.subplots()
    ax.imshow(img.permute(1,2,0))
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(file_name)
    plt.close()
    print('saved', file_name)


def create_instance(img_size=416, annotations=None):
    """
    pred_boxes should be passed in raw XYWH format and internally this function converts them back to native
    format preferred by pytorchfiWrapper functions viz XYXY  format.
    """
    pred_boxes_xywh = [annotations[i]['bbox'] for i in range(len(annotations))]
    pred_boxes_xyxy = [BoxMode.convert(box, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for box in pred_boxes_xywh]
    pred_boxes = [torch.Tensor(pred_boxes) for pred_boxes in pred_boxes_xyxy]
    try:
        pred_scores = [annotations[i]['score'] for i in range(len(annotations))]
    except:
        pred_scores = [0]*len(pred_boxes)
    # if "coco" in self.metadata.name.lower() or "robo" in self.metadata.name.lower() or "ppp" in self.metadata.name.lower():
    #     """
    #     for CoCo datasets
    #     """
    #     try:
    #         pred_classes = [self.metadata.thing_dataset_id_to_contiguous_id[annotations[i]['category_id']] for i in range(len(annotations))]
    #     except:
    #         print("Input has the wrong class labels (0-80) instead of (1-91)?)")
    #         sys.exit()
    # else:
    pred_classes = [annotations[i]['category_id'] for i in range(len(annotations))]
    
    if not len(pred_classes):
        pred_classes = [-1]*len(pred_boxes)
    if not len(pred_scores):
        pred_scores = [0]*len(pred_boxes)
    
    instance = Instances(image_size=img_size)
    # instance.set("image_size", img_size)
    instance.set("pred_boxes", Boxes(torch.stack(pred_boxes, 0)) if len(pred_boxes)>0 else  Boxes([]))
    instance.set("pred_classes", torch.Tensor(pred_classes).type(torch.ByteTensor))
    instance.set("scores", torch.Tensor(pred_scores))
    return instance


def annos2instance(annos, imgsize=(1200, 1920)):

    out_instance = Instances(imgsize)
    boxes = [n["bbox"] for n in annos]
    classes = [n["category_id"] for n in annos]
    scores = [1 for n in annos]
    out_instance.set("pred_boxes", Boxes(boxes))
    out_instance.set("pred_classes", torch.tensor(classes).type(torch.ByteTensor))
    out_instance.set("scores", scores)

    return [{"instances": out_instance}]


def get_tp_fp_fn_list(output_orig):
    # Convert output_orig to dict list
    boxes = output_orig[0]['instances'].pred_boxes.tensor.cpu().tolist()
    boxes = np.array(boxes)
    if len(boxes) > 0:
        boxes[:,2] = boxes[:,2] - boxes[:, 0] #convert xyxy to xywh to compare to gt. TODO: only for coco?
        boxes[:,3] = boxes[:,3] - boxes[:, 1]
    boxes = boxes.tolist()

    classes = output_orig[0]['instances'].pred_classes.tolist()
    scores = output_orig[0]['instances'].scores.tolist()
    output_anno = []
    for u in range(len(classes)):
        output_anno.append({'bbox': boxes[u], 'category_id': classes[u], 'image_id': 0, 'score': scores[u]})


    # Ground truth
    gt = img_data[0]['annotations']
    # imgsize= (img_data[0]["height"], img_data[0]["width"])
    # Change bbox format from xywh to xyxy
    # if dataset_name == 'coco2017' or dataset_name == 'coco2014':
    #     from alficore.dataloader.objdet_baseClasses.boxes import BoxMode
        # boxes = np.array([i["bbox"] for i in gt]) #xywh
        # for i in range(len(gt)):
        #     gt[i]["bbox"] = BoxMode.convert(boxes[i].tolist(), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    for v in range(len(gt)):
        gt[v]['image_id'] = 0

    # # Make annos of form instance
    # gt2 = annos2instance(gt, imgsize=imgsize) #transform to instances
    # gt_boxes = np.array([n['bbox'] for n in gt])
    # gt_classes = [n['category_id'] for n in gt]


    tpfpfn = ivmod_metric(gt, output_anno, 0.5, eval_mode="iou+class_labels")
    output_orig_tp = [{'instances': create_instance(annotations=tpfpfn['tp'])}]
    output_orig_fp = [{'instances': create_instance(annotations=tpfpfn['fp'])}]
    output_orig_fn = [{'instances': create_instance(annotations=tpfpfn['fn'])}]

    orig_label = "TP: " + str(len(tpfpfn['tp'])) + ", " + "FP: " + str(len(tpfpfn['fp'])) + ", " + "FN: " + str(len(tpfpfn['fn']))

    return output_orig_tp, output_orig_fp, output_orig_fn, orig_label



# Check blur impact etc. -----------------------

####################################################################
dataset_name = 'coco'
model_name = "yolov3"
#####################################################################

cuda_device = 0
device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu") 

## set dataloader attributes ----------------------------------------
dl_attr = TEM_Dataloader_attr()
dl_attr.dl_batch_size  = 1
dl_attr.dl_sampleN     = 100 #NOTE: <= actual dataset, e.g. for pp <=51
dl_attr.dl_random_sample  = False
dl_attr.dl_shuffle     = False
dl_attr.dl_mode        = "sequence" # "image/sequence"
dl_attr.dl_scenes      = [0,1]
dl_attr.dl_num_workers = 4
dl_attr.dl_device      = device
dl_attr.dl_sensor_channels  = ["CAM_FRONT"]
dl_attr.dl_dataset_type = 'val' #train, val
dl_attr.dl_dataset_name = dataset_name

dl_attr = assign_val_train(dl_attr)


# if dataset_name=='kitti':
#     from alficore.dataloader.kitti_loader import Kitti2D_dataloader
#     dataloader = Kitti2D_dataloader(dl_attr=dl_attr, dnn_model_name = None)
if 'coco' in dataset_name:
    from alficore.dataloader.coco_loader import CoCo_obj_det_native_dataloader
    dataloader = CoCo_obj_det_native_dataloader(dl_attr=dl_attr)
# if dataset_name=='ppp':
#     from alficore.dataloader.ppp_loader import PPP_obj_det_dataloader
#     dataloader = PPP_obj_det_dataloader(dl_attr=dl_attr)
# if dataset_name=='lyft':
#     from alficore.dataloader.lyft_loader import Lyft_dataloader
#     dataloader = Lyft_dataloader(dl_attr=dl_attr)
# if dataset_name=='robo':
#     from alficore.dataloader.robo_loader import Robo_obj_det_dataloader
#     dataloader = Robo_obj_det_dataloader(dl_attr=dl_attr)



# # Model : ---------------------------------------------------------

model = set_up_mmdet_model(model_name, dataset_name, device)



# Visualize -----------------------------------------------------------------------
output_folder = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/vis_input_corruption'


# # Check corruption input -----------------------------------------
n = 0 #image number #3, 17, 24

for n in range(n, n+1):

    #Orig  ---------------------------------------------------------
    corr_type = 'orig'
    # img_data = dataloader.datagen_iter.next()
    img_data = [dataloader.datagen_iter._dataset[n]]
    img = img_data[0]['image']
    img_test_data = img_data
    # plot_img(img, output_folder + "/test_orig.png")

    # import captum
    # from captum.attr import LayerConductance
    # layer_cond = LayerConductance(model, 'model.2.cv1.conv') #list(model.named_parameters())[0][0]
    # input = torch.randn(1, 3, 416, 416, requires_grad=True)
    # # Computes layer conductance for class 3.
    # # attribution size matches layer output, Nx12x32x32
    # attribution = layer_cond.attribute(input, target=3)


    output = model(img_test_data)
    output_tp, output_fp, output_fn, label = get_tp_fp_fn_list(output)
    output_path = output_folder + '/test_bbox_' + corr_type + '_' + str(n) + '.png'

    simple_visualization_direct_img(img_test_data[0], output_tp[0]["instances"], output_path, dataset_name, inset_str=label, extra_boxes=output_fp[0]["instances"], extra_boxes2= output_fn[0]["instances"], classes=model.model.CLASSES)


    # Blur --------------------------------------------------------
    corr_type = 'blur'
    img_test_data = deepcopy(img_data)
    sig = 3 #sigma = 1,2,3
    img_blur = torchvision.transforms.GaussianBlur((5,9), sigma=(sig,sig))(img)


    img_test_data[0]['image'] = img_blur
    del img_test_data[0]['file_name'] #delete file_name so that image tensor is used
    # plot_img(img_blur, output_folder + "/test_blur.png")

    output = model(img_test_data)
    output_tp, output_fp, output_fn, label = get_tp_fp_fn_list(output)
    output_path = output_folder + '/test_bbox_' + corr_type + str(sig) + '_' + str(n) + '.png'
    simple_visualization_direct_img(img_test_data[0], output_tp[0]["instances"], output_path, dataset_name, inset_str=label, extra_boxes=output_fp[0]["instances"], extra_boxes2= output_fn[0]["instances"], classes=model.model.CLASSES)


    # Noise --------------------------------------------------------
    corr_type = 'noise'
    mean = 0
    std = 1 #10 #1,5,10
    # noise = mean + std*torch.randn(img.shape) #10 leads to 40% SDC, 1 to 17% SDC

    img_test_data = deepcopy(img_data)
    del img_test_data[0]['file_name'] #delete file_name so that image tensor is used
    noise = np.random.normal(mean, std, img.shape)
    # noise = (10**0.5)*torch.randn(img.shape) #10 leads to 40% SDC, 1 to 17% SDC
    img_noise = (img + noise).to(torch.uint8)
    img_test_data[0]['image'] = img_noise
    # plot_img(img_noise, output_folder + "/test_noise.png")


    output = model(img_test_data)
    output_tp, output_fp, output_fn, label = get_tp_fp_fn_list(output)
    output_path = output_folder + '/test_bbox_' + corr_type + str(std) + '_' + str(n) + '.png'
    simple_visualization_direct_img(img_test_data[0], output_tp[0]["instances"], output_path, dataset_name, inset_str=label, extra_boxes=output_fp[0]["instances"], extra_boxes2= output_fn[0]["instances"], classes=model.model.CLASSES)


    # Contrast --------------------------------------------------------
    corr_type = 'contrast'
    contrast_factor = 0.1 # 0=gray, 1=original
    img_contrast = torchvision.transforms.functional.adjust_contrast(img, contrast_factor) #new fault type #TODO:

    img_test_data = deepcopy(img_data)
    del img_test_data[0]['file_name'] #delete file_name so that image tensor is used
    img_test_data[0]['image'] = img_contrast



    output = model(img_test_data)
    output_tp, output_fp, output_fn, label = get_tp_fp_fn_list(output)
    output_path = output_folder + '/test_bbox_' + corr_type + str(std) + '_' + str(n) + '.png'
    simple_visualization_direct_img(img_test_data[0], output_tp[0]["instances"], output_path, dataset_name, inset_str=label, extra_boxes=output_fp[0]["instances"], extra_boxes2= output_fn[0]["instances"], classes=model.model.CLASSES)
