import argparse
import json
import logging.config
import os
from pickle import FALSE
import sys
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms, models
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import pandas as pd

# sys.path.append("..")
# sys.path.append('/home/fgeissle/ranger_repo/ranger/yolov3/')
# from yolov3.yolov3_lyft import yolov3
# from yolov3.utils.general import (non_max_suppression, scale_coords, print_args)
# FILE = Path(__file__).resolve()
# from yolov3.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
sys.path.append("..")
sys.path.append('/home/fgeissle/ranger_repo/ranger/yolov3_ultra')
from yolov3_ultra.yolov3 import yolov3
from yolov3_ultra.yolov3_utils.general import (non_max_suppression, scale_coords, print_args)
FILE = Path(__file__).resolve()
from yolov3_ultra.yolov3_utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox


from alficore.dataloader.coco_loader import CoCo_obj_det_dataloader
from alficore.dataloader.imagenet_loader import imagenet_Dataloader
from alficore.dataloader.coco_loader import coco_loader
from alficore.wrapper.ptfiwrap import ptfiwrap
from alficore.ptfiwrap_utils.helper_functions import getdict_ranger, get_savedBounds_minmax, show_gpu, save_Bounds_minmax, get_max_min_lists
from alficore.ptfiwrap_utils.hook_functions import set_ranger_hooks_ReLU
from alficore.ptfiwrap_utils.evaluate import extract_ranger_bounds, extract_ranger_bounds_auto2, get_ranger_bounds_quantiles
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
# from alficore.models.yolov3.darknet import Darknet
# from darknet_Ranger import Darknet_Ranger
# from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg, Ranger_trivial
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize
# from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg



from alficore.dataloader.objdet_baseClasses.boxes import Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.models.yolov3.utils import rescale_boxes
from alficore.ptfiwrap_utils.build_native_model import build_native_model

# from alficore.ptfiwrap_utils.evaluate import get_Ranger_bounds2

from typing import Dict, List
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
# from ranger_as_detector.functions import get_max_min_lists, set_ranger_hooks_ReLU,

# logging.config.fileConfig('fi.conf')
# log = logging.getLogger()
cuda_device = 0
model_name = 'yolov3_ultra'
transform = transforms.Compose([            #[1]
            transforms.Resize(256),                    #[2]
            transforms.CenterCrop(224),                #[3]
            transforms.ToTensor(),                     #[4]
            transforms.Normalize(                      #[5]
            mean=[0.485, 0.456, 0.406],                #[6]
            std=[0.229, 0.224, 0.225]                  #[7]
            )])





class build_objdet_native_model(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, opt, dataset_name):
        super().__init__(model=model, device=opt.device)
        self.dataset_name = dataset_name
        ### img_size, preprocess and postprocess can also be inialised using kwargs which will be set in base class

        if 'coco' in self.dataset_name or self.dataset_name == 'robo' or self.dataset_name=='ppp':
            self.img_size = 416
        # elif self.dataset_name == 'robo':
        #     self.img_size = 480
        # elif self.dataset_name == 'ppp':
        #     self.img_size = 400
        elif 'kitti' in self.dataset_name:
            self.img_size = 640
        elif 'lyft' in self.dataset_name:
            self.img_size = 640 
        else:
            self.img_size = 416

        self.preprocess = True
        self.postprocess = True
        self.opt = opt

    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """


        if 'coco' in self.dataset_name:
            images = [resize(x['image'], self.img_size) for x in batched_inputs] # coco
            images = [x/255. for x in images]
        elif 'kitti' in self.dataset_name or self.dataset_name == 'robo' or self.dataset_name == 'ppp' or self.dataset_name == 'lyft':
            images = [letterbox(x["image"], self.img_size, HWC=True, use_torch=True)[0] for x in batched_inputs]  # kitti
            images = [x/255. for x in images]
        # elif 'lyft' in self.dataset_name:
        #     # images = [x["image"]/255. for x in batched_inputs]
        #     # # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        #     # # Pad to square resolution
        #     # padded_imgs = [pad_to_square(img, 0)[0] for img in images]
        #     # # Resize
        #     # images = [resize(img, self.img_size) for img in padded_imgs]

        #     images = [letterbox(x["image"], self.img_size, HWC=True, use_torch=True)[0] for x in batched_inputs]
        #     images = [x/255. for x in images]
        #     # Convert to tensor
        else:
            print('dataset_name not known, aborting.')
            sys.exit()
  
        

        # # images = [letterbox(x["image"], self.img_size, HWC=True, use_torch=True)[0] for x in batched_inputs]
        # images = [resize(x['image'], self.img_size) for x in batched_inputs]
        # images = [x/255. for x in images]

        # Convert to tensor
        images = torch.stack(images).to(self.device)
        ## normalisde the input if neccesary
        return images

    def rescale_boxes(self, boxes, current_dim, original_shape):
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
        boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
        return boxes

    def postprocess_output(self, output, original_shapes, current_dim):
        """
        the returning output should be stored as dictionary
        Output['instances] = fields containing pred_boxes, scores, classes
        viz. it should align to attributes as used in function instances_to_coco_json() in coco evaluation file.
        Output['instances].pred_boxes = [[2d-bb_0], [2d-bb_1], [2d-bb_2]...]
        Output['instances].scores     = [score_0, score_1, .....]
        Output['instances].classes     = [car, pedetrian, .....]

        ex: for batch size 1
        Output = [{}]
        Output['instances'] = output
        return Output        
        """
        Output, _ = non_max_suppression(output, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=self.opt.max_det)
        out_list = []
        for idx, output in enumerate(Output): # for each image in batch                
            out_instance = Instances(self.img_size)
            if len(original_shapes):
                # boxes = Boxes(output[:,:4])
                # boxes.rescale_boxes(current_dim=[self.img_size]*2, original_shape=original_shapes[idx])
                boxes = Boxes(scale_coords(current_dim, output[:, :4], original_shapes[idx]).round())
            else:
                boxes = Boxes(output[:,:4])
            out_instance.set("pred_boxes", boxes)
            out_instance.set("pred_classes", output[:,-1].type(torch.ByteTensor))
            out_instance.set("scores", output[:,4])

            if 'robo' in self.dataset_name: #filter for only classes with label 0 (persons)
                red_mask = out_instance.pred_classes == 0
                cls = out_instance.pred_classes[red_mask]
                bxs = out_instance.pred_boxes[red_mask]
                scrs = out_instance.scores[red_mask]
                out_instance = Instances(self.img_size)
                out_instance.set("pred_classes", cls)
                out_instance.set("pred_boxes", bxs)
                out_instance.set("scores", scrs)

            out_list.append({'instances': out_instance})
        return out_list

    def __getattr__(self, method):
        if method.startswith('__'):
            raise AttributeError(method)
        try:
        # if hasattr(self.model, method):
            
            try:
                func = getattr(self.model.model, method)
            except:
                func = getattr(self.model, method)
            ## running pytorch model (self.model) inbuilt functions like eval, to(device)..etc
            ## assuming the executed method is not changing the model but rather
            ## operates on the execution level of pytorch model.
            def wrapper(*args, **kwargs):            
                if (method=='to'):
                    return self
                else:
                    return  func(*args, **kwargs)
            return wrapper
        except KeyError:
            raise AttributeError(method)

    def __call__(self, input, dummy=False):
        input = pytorchFI_objDet_inputcheck(input)
        try:
            original_shapes = [(input[i]['height'], input[i]['width']) for i in range(len(input))]
        except:
            original_shapes = []
        _input = input
        if self.preprocess:
            _input = self.preprocess_input(input)
        ## pytorchFI core checks model with dummy tensor with batch size sample
        # show_gpu(cuda_device, 'GPU memory usage before inference:')
        output = self.model(_input)
        # show_gpu(cuda_device, 'GPU memory usage after clearing cache:')
        if self.postprocess:
            output = self.postprocess_output(output, original_shapes, _input.shape[2:])

        output = pytorchFI_objDet_outputcheck(output)
        return output

def parse_opt(dataset_name):
    parser = argparse.ArgumentParser()

    if dataset_name == 'kitti':
        parser.add_argument('--weights', nargs='+', type=str, default='/nwstore/ralf/train_log/ultralytics/results_new/kitti_siLu/exp25/weights/best.pt', help='model path(s)') #kitti
    elif dataset_name == 'lyft':
        parser.add_argument('--weights', nargs='+', type=str, default='/nwstore/ralf/train_log/ultralytics/results_new/lyft_siLu_latest/exp32/weights/best.pt', help='model path(s)') #lyft
    elif 'coco' in dataset_name or dataset_name == 'robo' or dataset_name == 'ppp':
        parser.add_argument('--weights', nargs='+', type=str, default='/home/fgeissle/ranger_repo/ranger/yolov3/yolov3/yolov3.pt', help='model path(s)') #coco 

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


def main(argv):
    CUDA_LAUNCH_BLOCKING=1
    ####################################################################
    dataset_name = 'coco2017' #'coco2017' lyft, kitti, 'robo', ppp
    ####################################################################

    opt = parse_opt(dataset_name)
    
    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")    
    yolov3_model = yolov3(**vars(opt))
    # device = torch.device("cpu")
    opt.device = device
    if not any('--' in s for s in argv):
        argv = []
    else:
        for i, args in enumerate(argv):
            if '--' in args:
                argv = argv[i:]
                break
    parser = argparse.ArgumentParser("resnet_pytorch")
    parser.add_argument( '-r',   '--gen_ranger_bounds', default=False, help="generate ranger bounds out of the models")

    args = parser.parse_args(argv)
    gen_ranger_bounds = args.gen_ranger_bounds
    # print(torch.__version__, torchvision.__version__)
    # print(torchvision.__path__, torch.__path__)


    # Model : ---------------------------------------------------------
    

    yolov3_model = yolov3_model.to(device)
    model = build_objdet_native_model(model=yolov3_model, opt=opt, dataset_name=dataset_name)

    ####################################################################
    ## set dataloader attributes
    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = False
    dl_attr.dl_batch_size  = 10
    dl_attr.dl_shuffle     = False
    dl_attr.dl_sampleN     = 100 #NOTE: <= actual dataset, e.g. for pp <=51
    dl_attr.dl_mode        = "sequence" # "image/sequence"
    dl_attr.dl_scenes      = [0,1]
    dl_attr.dl_num_workers = 4
    dl_attr.dl_device      = device
    dl_attr.dl_sensor_channels  = ["CAM_FRONT"]
    dl_attr.dl_dataset_type = 'val' #train, val
    dl_attr.dl_dataset_name     = dataset_name
    ####################################################################

    # Ranger bounds: ---------------------------------------------------------
    gen_ranger_bounds = False #if true new bounds are generated (overwritten)
    get_percentiles = False # True, False
    get_ftraces = False
    # nr_samples = 21
    # batch_size = 1
    # ranger_file_name = 'yolov3_bounds_CoCo_train20p_act_v2'
    # ranger_file_name = 'yolov3_bounds_CoCo_custom'
    # ranger_file_name = 'yolov3_coco_bounds_test_perc'
    # ranger_file_name = 'yolov3_bounds_kitti_train20p_act_v2'
    # ranger_file_name = 'yolov3_ultra_lyft_bounds_train20p'
    # ranger_file_name = 'yolov3_bounds_lyft_train20p_act_v3'
    # ranger_file_name = 'yolov3_ultra_bounds_robo'
    # ranger_file_name = 'yolov3_ultra_bounds_ppp'
    ranger_file_name = 'yolov3_bounds_CoCo_train20p_act_v9'
    # ranger_file_name = 'yolov3_bounds_kitti_train20p_act_v3'


    bnds, bnds_min_max, bnds_qu = get_ranger_bounds_quantiles(model, dl_attr, ranger_file_name, gen_ranger_bounds, get_percentiles, get_ftraces)
    print()




    # # Inference runs: ---------------------------------------------------------



    # Inference with fault injection -------------------------
    num_faults = 1
    # yml_file = 'default_min_test_stuckat.yml'
    # yml_file = 'yolov3_ultra_test_random_sbf_neurons_inj_1_200_1bs_lyft_model_scenario.yml'
    yml_file = 'default_min_test.yml'

    
    # resil_model ------------------
    # ff = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220502-190646_1_faults_[0,8]_bits/coco2017/val/yolov3_ultra_test_random_sbf_neurons_inj_1_50_1bs_coco2017_fault_locs.bin'
    yolov3_ErrorModel = TestErrorModels_ObjDet(model=model, model_name=model_name, resil_name='ranger_trivial', dl_attr=dl_attr, num_faults=num_faults,\
        config_location=yml_file, inf_nan_monitoring=True, ranger_bounds=bnds, ranger_detector=True, disable_FI=False, quant_monitoring=True, ftrace_monitoring = False, exp_type="Gaussian_blur") #, fault_file=ff, copy_yml_scenario=True)

    yolov3_ErrorModel.test_rand_ObjDet_SBFs_inj()
    print()




if __name__ == "__main__":
    main(sys.argv)
