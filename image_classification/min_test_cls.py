import sys
import numpy as np
import torch
from torch._C import device
from torch.multiprocessing import Process
import torchvision
from torchvision import transforms, models
from torch import nn

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch

sys.path.append("..")
sys.path.append('/home/fgeissle/ranger_repo/ranger/yolov3_ultra')
from yolov3_ultra.yolov3 import yolov3
from yolov3_ultra.yolov3_utils.general import (non_max_suppression, scale_coords, print_args)
FILE = Path(__file__).resolve()
from yolov3_ultra.yolov3_utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox

from alficore.ptfiwrap_utils.evaluate import extract_ranger_bounds, extract_ranger_bounds_auto2, get_ranger_bounds_quantiles
from alficore.wrapper.test_error_models_imgclass import TestErrorModels_ImgClass
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize
from alficore.ptfiwrap_utils.build_native_model import build_native_model

from typing import Dict, List
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
from image_classification.MMAL.auto_load_resume import auto_load_resume
# from ranger_as_detector.functions import get_max_min_lists, set_ranger_hooks_ReLU,

# logging.config.fileConfig('fi.conf')
# log = logging.getLogger()
cuda_device = 0

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])




class build_objdet_native_model_mmal(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, device):
        super().__init__(model=model)
        ### img_size, preprocess and postprocess can also be inialised using kwargs which will be set in base class
        self.postprocess = True
        self.preprocess = False
        self.device = device
        

        auto_load_resume(model,  "/home/fgeissle/ranger_repo/ranger/image_classification/MMAL/air_epoch146.pth", status='test') #load actual pretrained parameters!
        self.model = model

    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """

        if 'coco' in self.dataset_name or self.dataset_name == 'robo':
            images = [resize(x['image'], self.img_size) for x in batched_inputs] # coco
            images = [x/255. for x in images]
        elif 'kitti' in self.dataset_name:
            images = [letterbox(x["image"], self.img_size, HWC=True, use_torch=True)[0] for x in batched_inputs]  # kitti
            images = [x/255. for x in images]
        elif 'lyft' in self.dataset_name:
            images = [x["image"]/255. for x in batched_inputs]
            # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            # Pad to square resolution
            padded_imgs = [pad_to_square(img, 0)[0] for img in images]
            # Resize
            images = [resize(img, self.img_size) for img in padded_imgs]
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

    def postprocess_output(self, output):
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
        output = output[-2]
        # output = output[-2].max(1, keepdim=True)[1].T

        return output

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
        # input = pytorchFI_objDet_inputcheck(input)

        self.model = self.model.to(self.device)
        input = input.to(self.device)

        _input = input
        # if self.preprocess:
        #     _input = self.preprocess_input(input)

        output = self.model(_input, DEVICE=self.device)

        if self.postprocess:
            output = self.postprocess_output(output)

        # output = pytorchFI_objDet_outputcheck(output)
        return output



def main(argv):
    model_name = 'alexnet' #'mmal' alexnet
    dataset_name = 'imagenet' # imagenet 'fgvc' #'fgvc' #'coco2017' lyft, kitti, robo


    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu") 
    
    # Model : ---------------------------------------------------------
    if model_name == 'mmal':
        from MMAL.config import input_size, root, proposalN, channels
        from MMAL.mainnet import MainNet
        model = MainNet(proposalN=proposalN, num_classes=100, channels=channels)
        model = build_objdet_native_model_mmal(model, device)

    if model_name == 'resnet_fgvc':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.load_state_dict(torch.load('/home/fgeissle/ranger_repo/ranger/image_classification/MMAL/resnet50-19c8e357.pth'))
        model.fc = nn.Linear(num_ftrs, 100)
        model = model.to(device)

    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True, progress=True)
        model = model.to(device)


    ## set dataloader attributes
    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = False
    dl_attr.dl_batch_size  = 1
    dl_attr.dl_shuffle     = False
    dl_attr.dl_sampleN     = 100 #in percent for imagenet, absolute nr for fgvc
    dl_attr.dl_mode        = "sequence" # "image/sequence"
    dl_attr.dl_scenes      = [0]
    dl_attr.dl_num_workers = 1
    dl_attr.dl_device      = device
    dl_attr.dl_sensor_channels  = ["CAM_FRONT"]
    dl_attr.dl_dataset_type = 'train' #val, train
    dl_attr.dl_dataset_name     = dataset_name
    if dataset_name == 'imagenet':
        dl_attr.dl_transform     = transform
    if model_name == 'mmal' and dl_attr.dl_batch_size > 1:
        print('MMal model does not support bs>1.')
        sys.exit()


    # Ranger bounds: ---------------------------------------------------------
    gen_ranger_bounds = False #if true new bounds are generated (overwritten), True, False
    get_percentiles = False
    get_ftraces = False

    ranger_file_name = 'yolov3_bounds_alexnet_Imagenet_act_v1'
    # ranger_file_name = 'yolov3_ultra_bounds_robo'
    # ranger_file_name = 'resnet_bounds_fgvc_qu_v2'
    # ranger_file_name = 'alexnet_bounds_imagenet_qu_v3'

    bnds, bnds_min_max, bnds_qu = get_ranger_bounds_quantiles(model, dl_attr, ranger_file_name, gen_ranger_bounds, get_percentiles, get_ftraces)
    print()



    # # Inference runs: ---------------------------------------------------------


    # Inference with fault injection -------------------------
    num_faults = 1
    yml_file = 'default_min_test_cls.yml' #'default_neurons.yml' #'default_min_test_cls.yml'

    
    # resil_model ------------------
    # ff = '/home/fgeissle/ranger_repo/ranger/result_files/alexnet_2_trials/neurons_injs/per_image/objDet_20220405-175606_1_faults_[0,8]_bits/imagenet/val/alexnet_test_random_sbf_neurons_inj_1_2_1bs_imagenet_fault_locs.bin'
    ErrorModel = TestErrorModels_ImgClass(model=model, model_name=model_name, resil_name='ranger_trivial', dl_attr=dl_attr, num_faults=num_faults,\
        config_location=yml_file, inf_nan_monitoring=True, ranger_bounds=bnds, ranger_detector=True, disable_FI=False, quant_monitoring=True, ftrace_monitoring = True, exp_type="hwfault")

    ErrorModel.test_rand_ImgClass_SBFs_inj()
    print()




if __name__ == "__main__":
    main(sys.argv)
