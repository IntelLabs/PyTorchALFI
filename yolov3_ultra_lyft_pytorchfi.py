import argparse
import json
import logging.config
import os
import sys
import time

import numpy as np
import torch
from torch._C import device
from torch.multiprocessing import Process
from torchvision import transforms, models


import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from yolov3_ultra.yolov3 import yolov3
from yolov3_ultra.yolov3_utils.general import (non_max_suppression, scale_coords, print_args)
from yolov3_ultra.yolov3_utils.augmentations import letterbox
FILE = Path(__file__).resolve()


from alficore.dataloader.imagenet_loader import imagenet_Dataloader
from alficore.ptfiwrap_utils.evaluate import get_ranger_bounds_quantiles
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
import pandas as pd
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck
from alficore.resiliency_methods.ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg

from alficore.dataloader.objdet_baseClasses.boxes import Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.ptfiwrap_utils.build_native_model import build_native_model
from typing import Dict, List
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr

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

def runInParallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


def test_imagenet_images(model, device):
    total_accuracy = 0
    __TOP_RES = 1
    batch_size = 20
    incorrect_classified_file_paths = []
    model.eval()
    _, imagenet_dataloader, _ = imagenet_Dataloader(root='val', shuffle=False, batch_size=batch_size, transform=transform,
                                                        sampling=False)
    for i, x in enumerate(imagenet_dataloader):
        images = x[0].to(device)
        labels = x[1].to(device)
        image_paths = x[2]
        outputs = model(images)
        print('Batch no.', i)
        num_of_images = len(outputs) 
        correctly_classified_images = 0 
        for i in range(num_of_images):
            _output = outputs[i]
            _output = torch.unsqueeze(_output, 0)
            # percentage = torch.nn.functional.softmax(_output, dim=1)[0] * 100
            _, output_index = torch.sort(_output, descending=True)
            # output_perct = np.round(percentage[output_index[0][:__TOP_RES]].cpu().detach().numpy(), decimals=2)
            output_index = output_index[0][:__TOP_RES].cpu().detach().numpy()
            if output_index[0] == labels[i]:
                correctly_classified_images += 1
            else:
                incorrect_classified_file_paths.append(image_paths[i])

        accuracy = (correctly_classified_images * 100) / num_of_images
        total_accuracy += accuracy

    print('Total accuracy:', total_accuracy/len(imagenet_dataloader))  
    df = pd.DataFrame(incorrect_classified_file_paths, columns=['incorrect_image_fps'])
    df.to_csv('incorrect_fps.csv', index=False) 

class build_objdet_native_model(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, opt):
        super().__init__(model=model, device=opt.device)
        ### img_size, preprocess and postprocess can also be inialised using kwargs which will be set in base class
        self.img_size = 640
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
        images = [letterbox(x["image"], self.img_size, HWC=True, use_torch=True)[0] for x in batched_inputs]
        images = [x/255. for x in images]
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

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='/nwstore/ralf/train_log/ultralytics/lyft_leakyRelu-2-2/exp2/weights/best.pt', help='model path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='/nwstore/ralf/train_log/ultralytics/results_new/lyft_siLu_latest/exp32/weights/best.pt', help='model path(s)')
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

def main(argv):
    opt = parse_opt()
    device = torch.device(
        "cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")
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
    # Model   ----------------------------------------------------------
    yolov3_model = yolov3_model.to(device)
    model = build_objdet_native_model(model=yolov3_model, opt=opt)
    get_percentiles = False

    ## set dataloader attributes
    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = False
    dl_attr.dl_batch_size  = 10
    dl_attr.dl_shuffle     = False
    dl_attr.dl_sampleN     = 126
    dl_attr.dl_mode        = "sequence" # "image/sequence"
    dl_attr.dl_scenes      = [0]
    dl_attr.dl_num_workers = 1
    dl_attr.dl_device      = device
    dl_attr.dl_sensor_channels  = ["CAM_FRONT"]
    dl_attr.dl_dataset_name     = "lyft"

    ranger_file_name = 'yolov3_ultra_lyft_bounds_train20p'
    bnds, bnds_min_max, bnds_qu = get_ranger_bounds_quantiles(model, dl_attr, ranger_file_name, gen_ranger_bounds, get_percentiles)
    # Inference runs: ---------------------------------------------------------
    apply_ranger = True #TODO:

    # replace with pytorchfi wrapper code
    # batchsize is set in scenarios/default.yml -> ptf_batch_size
    num_faults  = [1]
    fault_files = ["/home/qutub/PhD/git_repos/intel_git_repos/pfd_uet/result_files/result_files_paper/yolov3_ultra_200_trials/weights_injs/per_epoch/objDet_20220211-172719_1_faults_[0, 8]_bits/lyft/val/yolov3_ultra_test_random_sbf_weights_inj_1_200_1bs_lyft_fault_locs.bin"]
    # fault_files = [None]s
    resume_dir  = None
    yml_file = 'default_yolo.yml'
    # choose fault injection policy None (as before), None, 'per_batch' or "per_epoch"
    inj_policy = None
    
    for id, _num_faults in enumerate(num_faults):
            if apply_ranger:
                # Change network architecture to add Ranger
                # net_for_protection = model
                # protected_model, _ = get_Ranger_protection(net_for_protection, bnds, resil=_resil)
                # protected_model = protected_model.to(device)
                yolov3_ErrorModel = TestErrorModels_ObjDet(model=None, resil_model=model, resil_name='ranger', model_name=model_name, config_location='default_neurons.yml', \
                    ranger_bounds=np.array(bnds), device=device,  inf_nan_monitoring=True, disable_FI=False, dl_attr=dl_attr, num_faults=_num_faults, fault_file=fault_files[id], \
                        resume_dir=resume_dir, copy_yml_scenario = True)

            else:
                yolov3_ErrorModel = TestErrorModels_ObjDet(model=model, model_name=model_name, dataset='coco2017', dl_attr=dl_attr,\
        config_location='default_neurons.yml', inf_nan_monitoring=False)

            yolov3_ErrorModel.test_rand_ObjDet_SBFs_inj()

if __name__ == "__main__":
    # ctx = mp.get_context("spawn")
    # ctx.set_start_method('spawn')onc
    main(sys.argv)
