import argparse
import json
import logging.config
import os
import sys
import time

import numpy as np
import torch
from torch.multiprocessing import Process
from torchvision import transforms, models

from alficore.dataloader.imagenet_loader import imagenet_Dataloader
from alficore.dataloader.coco_loader import coco_loader
from alficore.wrapper.ptfiwrap import ptfiwrap
from util.helper_functions import getdict_ranger, get_savedBounds_minmax
from util.evaluate import extract_ranger_bounds, extract_ranger_bounds_objdet
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from util.ranger_automation import flatten_model
import pandas as pd

from alficore.models.yolov3.darknet import Darknet
from darknet_Ranger import Darknet_Ranger

from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg, Ranger_trivial
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize
from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg

from alficore.dataloader.objdet_baseClasses.boxes import Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.models.yolov3.utils import non_max_suppression, rescale_boxes
from alficore.ps_utils.build_native_model import build_native_model
from typing import Dict, List

# logging.config.fileConfig('fi.conf')
# log = logging.getLogger()
cuda_device = 0
model_name = 'yolov3'
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


def get_Ranger_bounds(resnet50, batch_size, ranger_file_name, dataset_name, sample_Percentage=20):
    print('Loading ImageNet...')
    _, imagenet_dataloader, _ = imagenet_Dataloader(root='train', shuffle=False, batch_size=batch_size, transform=transform,
                                                    sampling=True, sampleN=sample_Percentage)
    net_for_bounds = flatten_model(resnet50) #to make Relus explicit
    act_input, act_output = extract_ranger_bounds(imagenet_dataloader, net_for_bounds, ranger_file_name, dataset_name) # gets also saved automatically
    print('check Ranger input', act_input)
    print('check Ranger output', act_output)
    sys.exit()

def get_Ranger_bounds_yolov3(model, batch_size, ranger_file_name, dataset_name='coco', sample_Percentage=20):
    # Confirmed: 72 leaky relu layers, no relu layers
    if dataset_name == 'coco':
        print('Loading coco...')
        dataloader, _ = coco_loader(root='train', shuffle=False, batch_size=batch_size, sampling=True, sampleN=sample_Percentage) #train TODO:
    if dataset_name == 'coco':
        print('Loading kitti...')
        dataloader = Kitti2D_dataloader(dataset_type='val', batch_size=batch_size, sampleN=0.2) #device=self.device)
    # net_for_bounds = flatten_model(resnet50) #to make Relus explicit
    net_for_bounds = model
    act_input, act_output = extract_ranger_bounds_objdet(dataloader, net_for_bounds, ranger_file_name) # gets also saved automatically
    print('check Ranger input', act_input)
    print('check Ranger output', act_output)
    sys.exit()

class build_objdet_native_model(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        ### img_size, preprocess and postprocess can also be inialised using kwargs which will be set in base class
        self.img_size = 416
        self.preprocess = True
        self.postprocess = True

    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """
        images = [x["image"]/255. for x in batched_inputs]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
         # Pad to square resolution
        padded_imgs = [pad_to_square(img, 0)[0] for img in images]
        # Resize
        images = [resize(img, self.img_size) for img in padded_imgs]
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

    def postprocess_output(self, output, original_shapes):
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
        Output = non_max_suppression(output)
        out_list = []
        for idx, output in enumerate(Output): # for each image in batch                
            out_instance = Instances(self.img_size)
            if len(original_shapes):
                boxes = rescale_boxes(output[:,:4], current_dim=416, original_shape=original_shapes[idx])
            else:
                boxes = output[:,:4]
            out_instance.set("pred_boxes", Boxes(boxes))
            out_instance.set("pred_classes", output[:,-1].type(torch.ByteTensor))
            out_instance.set("scores", output[:,4])
            out_list.append({'instances': out_instance})
        return out_list

def main(argv):
    device = torch.device(
        "cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

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

    # dataset: ---------------------------------------------------------
    dataset_name = 'coco'  

   # Model: ---------------------------------------------------------
    model = Darknet("./alficore/models/yolov3/config/yolov3.cfg").to(device)
    model.load_darknet_weights("./alficore/models/yolov3/weights/yolov3.weights")
    model = build_objdet_native_model(model=model, device=device)

    # Ranger bounds: ---------------------------------------------------------
    gen_ranger_bounds = False #if true new bounds are generated (overwritten)
    # ranger_file_name = 'Resnet50_bounds_ImageNet_train20p_act_new'
    ranger_file_name = 'yolov3_bounds_CoCo_train20p_act_v2'
    if gen_ranger_bounds:
        print('New bounds to be saved as:', ranger_file_name)
        # get_Ranger_bounds(model, 10, ranger_file_name, dataset_name, sample_Percentage=20)
        get_Ranger_bounds_yolov3(model, 10, ranger_file_name, dataset_name, sample_Percentage=100)
    else:
        print('Bounds loaded:', ranger_file_name)
    
    # Inference runs: ---------------------------------------------------------
    apply_ranger = True #TODO:
    bnds = get_savedBounds_minmax('./bounds/' + ranger_file_name + '.txt')
    # resil = [Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg]
    resil = [Ranger]

    # Model with trivial ranger: ---------------------------------------------------------
    model = Darknet_Ranger("./alficore/models/yolov3/config/yolov3.cfg", resil=Ranger_trivial, bnds=bnds).to(device)
    model.load_darknet_weights("./alficore/models/yolov3/weights/yolov3.weights")
    model = build_objdet_native_model(model=model, device=device)

    # replace with pytorchfi wrapper code
    # batchsize is set in scenarios/default.yml -> ptf_batch_size
    num_faults = [1]
    fault_files = ['/home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_0p/yolov3_100_trials/neurons_injs/per_image/objDet_2ef57520-aca2-47ec-983b-b3b3a0371d38_1_faults_[0, 9]_bits/kitti/val/yolov3_test_random_sbf_neurons_inj_1_100_25bs_kitti_fault_locs.bin']
    # fault_files = ['/home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files/yolov3_100_trials/weights_injs/per_batch/objDet_0b481f2f-8a12-4646-a80e-aa8df4f1cc27_1_faults_[0, 9]_bits/kitti/val/yolov3_test_random_sbf_weights_inj_1_100_25bs_kitti_fault_locs.bin']

    fault_files = ['/home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_10p/yolov3_200_trials/neurons_injs/per_image/objDet_83788bf6-3421-462d-a476-51491db3564e_1_faults_[0, 31]_bits/kitti/val/yolov3_test_random_sbf_neurons_inj_1_200_25bs_kitti_fault_locs.bin']
    # fault_files = ['/home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_0p/yolov3_200_trials/neurons_injs/per_image/objDet_cd330c8e-9e3c-47b5-a725-e49991ad5514_1_faults_[0, 31]_bits/coco2017/val/yolov3_test_random_sbf_neurons_inj_1_200_25bs_coco2017_fault_locs.bin']
    # fault_files = ['/home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_0p/yolov3_100_trials/weights_injs/per_batch/objDet_0b481f2f-8a12-4646-a80e-aa8df4f1cc27_1_faults_[0, 9]_bits/kitti/val/yolov3_test_random_sbf_weights_inj_1_100_25bs_kitti_fault_locs.bin']
    
    ## test fault file weights 5 runs
    # fault_file = ['/home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_0p/yolov3_100_trials/neurons_injs/per_image/objDet_2ef57520-aca2-47ec-983b-b3b3a0371d38_1_faults_[0, 9]_bits/kitti/val/yolov3_test_random_sbf_neurons_inj_1_100_25bs_kitti_fault_locs.bin']
    fault_files = [None]
    # resume_dir = '/home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_M_5p/yolov3_200_trials/weights_injs/objDet_edfa2c22-593d-4524-88c5-4bd70a2eeee6_1_faults_[0, 31]_bits'
    resume_dir = None
    yml_file = 'default_yolo.yml'
    # choose fault injection policy None (as before), None, 'per_batch' or "per_epoch"
    inj_policy = None
    
    for id, _num_faults in enumerate(num_faults):
        for _resil in resil:
            if apply_ranger:
                # Change network architecture to add Ranger
                bnds = get_savedBounds_minmax('./bounds/' + ranger_file_name + '.txt')
                
                # net_for_protection = model
                # protected_model, _ = get_Ranger_protection(net_for_protection, bnds, resil=_resil)
                # protected_model = protected_model.to(device)
                protected_model = Darknet_Ranger("./alficore/models/yolov3/config/yolov3.cfg", _resil, bnds).to(device)
                protected_model.load_darknet_weights("./alficore/models/yolov3/weights/yolov3.weights")

                protected_model = build_objdet_native_model(model=protected_model, device=device)
                yolov3_ErrorModel = TestErrorModels_ObjDet(model=model, resil_model=protected_model, model_name=model_name, dataset='coco2017', dl_sampleN=10,\
        config_location='default_neurons.yml', ranger_bounds=np.array(bnds), device=device)

            else:
                yolov3_ErrorModel = TestErrorModels_ObjDet(model=model, model_name=model_name, dataset='coco2017', dl_sampleN=0.024690674053554944,\
        config_location='default_neurons.yml')

            yolov3_ErrorModel.test_rand_ObjDet_SBFs_inj(num_faults=_num_faults, fault_file=fault_files[id], resume_dir=resume_dir, resume_inj=False, inj_policy=inj_policy)

if __name__ == "__main__":
    # ctx = mp.get_context("spawn")
    # ctx.set_start_method('spawn')onc
    main(sys.argv)
