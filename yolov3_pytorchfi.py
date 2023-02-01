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
from tqdm import tqdm

from alficore.dataloader.coco_loader import coco_loader
from alficore.wrapper.ptfiwrap import ptfiwrap
# from alficore.wrapper.ptfiwrap import TestErrorModels
from util.helper_functions import getdict_ranger, get_savedBounds_minmax
from util.evaluate import extract_ranger_bounds
from util.ranger_automation import get_Ranger_protection
import pandas as pd
from typing import Dict, List, Optional, Tuple

from alficore.models.yolov3.darknet import Darknet
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize

from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg

from alficore.dataloader.objdet_baseClasses.boxes import Boxes, BoxMode
from alficore.dataloader.objdet_baseClasses.instances import Instances
# from alficore.utils.utils import non_max_suppression
from alficore.models.yolov3.utils import non_max_suppression, rescale_boxes
from visualization import *



torch.cuda.empty_cache()
logging.config.fileConfig('fi.conf')
log = logging.getLogger()
cuda_device = 0
model_name = 'yolov3'
# transform = transforms.Compose([            #[1]
#             transforms.Resize(256),                    #[2]
#             transforms.CenterCrop(224),                #[3]
#             transforms.ToTensor(),                     #[4]
#             transforms.Normalize(                      #[5]
#             mean=[0.485, 0.456, 0.406],                #[6]
#             std=[0.229, 0.224, 0.225]                  #[7]
#             )])



# def runInParallel(*fns):
#     proc = []
#     for fn in fns:
#         p = Process(target=fn)
#         p.start()
#         proc.append(p)
#     for p in proc:
#         p.join()


# def test_imagenet_images(model, device):
#     total_accuracy = 0
#     __TOP_RES = 1
#     batch_size = 20
#     incorrect_classified_file_paths = []
#     model.eval()
#     _, imagenet_dataloader, _ = coco_loader(root='val', shuffle=True, batch_size=batch_size, transform=transform,
#                                                         sampling=False)
#     for i, x in enumerate(imagenet_dataloader):
#         images = x[0].to(device)
#         labels = x[1].to(device)
#         image_paths = x[2]
#         outputs = model(images)
#         print('Batch no.', i)
#         num_of_images = len(outputs) 
#         correctly_classified_images = 0 
#         for i in range(num_of_images):
#             _output = outputs[i]
#             _output = torch.unsqueeze(_output, 0)
#             # percentage = torch.nn.functional.softmax(_output, dim=1)[0] * 100
#             _, output_index = torch.sort(_output, descending=True)
#             # output_perct = np.round(percentage[output_index[0][:__TOP_RES]].cpu().detach().numpy(), decimals=2)
#             output_index = output_index[0][:__TOP_RES].cpu().detach().numpy()
#             if output_index[0] == labels[i]:
#                 correctly_classified_images += 1
#             else:
#                 incorrect_classified_file_paths.append(image_paths[i])

#         accuracy = (correctly_classified_images * 100) / num_of_images
#         total_accuracy += accuracy

    # print('Total accuracy:', total_accuracy/len(imagenet_dataloader))
    # df = pd.DataFrame(incorrect_classified_file_paths, columns=['incorrect_image_fps'])
    # df.to_csv('incorrect_fps.csv', index=False) 


class build_d2_model(object):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.img_size = 416
        self.preprocess = True ## detectron2 has inuilt preprocess
        self.postprocess = True
        self.device = torch.device(
        "cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

    def __getattr__(self, method):
        if method.startswith('__'):
            raise AttributeError(method)
        try:
        # if hasattr(self.model, method):
            func = getattr(self.model, method)
            ## running pytorch model (self.model) inbuilt functions like eval, to(device)..etc
            ## assuming the executed method is not changing the model but rather
            ## operates on the execution level of pytorch model.
            def wrapper(*args, **kwargs):            
                if (method=='to'):
                    return self
                else:
                    return  func(*args, **kwargs)
            return  wrapper 
        except KeyError:
            raise AttributeError(method)

    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """
        # [{'image': tensor([[[-0.7156,  ...='cuda:0'), 'image_id': 0}]
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

    def __call__(self, input, dummy=False):
        input = pytorchFI_objDet_inputcheck(input)
        try:
            original_shapes = [(input[i]['height'], input[i]['width']) for i in range(len(input))]
        except:
            original_shapes = []
        _input = input
        if self.preprocess:
            _input = self.preprocess_image(input)
        ## pytorchFI core checks model with dummy tensor with batch size sample

        output = self.model(_input)
        if self.postprocess:
            output = self.postprocess_output(output, original_shapes)

        output = pytorchFI_objDet_outputcheck(output)
        return output



def main(argv):
    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

    # if "--" not in argv:
    #     argv = []  # as if no args are passed
    # else:
    #     argv = argv[argv.index("--") + 1:]  # get all args after "--"
    if not any('--' in s for s in argv):
        argv = []
    else:
        for i, args in enumerate(argv):
            if '--' in args:
                argv = argv[i:]
                break
    parser = argparse.ArgumentParser("yolov3_pytorch")
    parser.add_argument( '-r',   '--gen_ranger_bounds', default=False, help="generate ranger bounds out of the models")

    args = parser.parse_args(argv)
    gen_ranger_bounds = args.gen_ranger_bounds
    apply_ranger = False
    # gen_ranger_bounds = True
    dataset_name = 'MS CoCo'
    ranger_file_name = 'yolov3_bounds_CoCo_train20p_act'
    batch_size = 100


    # Run Yolo fault injection
    ## Parameterize the input with config path as - model = Darknet(opt.model_def).to(device)
    yolov3 = Darknet("alficore/models/yolov3/config/yolov3.cfg").to(device)
    ## Parameterize the weight input as - model = Darknet(opt.model_def).to(device)
    yolov3.load_darknet_weights("alficore/models/yolov3/weights/yolov3.weights")
    # yolov3.load_darknet_weights("/home/sreetama/code/fault-injection-yolov3/weights/yolov3-kitti.weights")

    yolov3 = build_d2_model(yolov3)
    yolov3_ErrorModel = TestErrorModels_ObjDet(model=yolov3, model_name=model_name, dataset='coco',\
        config_location='default_yolo.yml', dl_sampleN=0.001, dl_shuffle=False, device=device)
    # det2_fasterrcnn_ErrorModel.postprocess_model_output
    yolov3_ErrorModel.test_rand_ObjDet_SBFs_inj(fault_file='')
    
    



    # Visualize from saved files:
    from alficore.dataloader.coco_loader import CoCo_obj_det_dataloader
    from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog

    # orig
    ground_truth_json_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/coco/val/coco_format.json'
    detection_json_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/coco/val/orig_model/epochs/0/coco_instances_results_0_epoch.json'
    corr_detection_json_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/coco/val/corr_model/epochs/0/coco_instances_results_0_epoch.json'

    # coco_dataloader = CoCo_obj_det_dataloader(dataset_type='val', batch_size=2) #do not need to register again if already done in TestErrorModels_ObjDet
    metadata = MetadataCatalog.get('coco/val')
    image = np.random.randn(416,416,3)
    img_ids = [42, 73]

    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE, vis_mode='offline', gt_json_file=ground_truth_json_file_path)

    annos_gt = visualizer.visualize_output(image_ids=img_ids, save_name='plots/test_viz_gt_', json_file=ground_truth_json_file_path)
    annos_orig = visualizer.visualize_output(image_ids=img_ids, save_name='plots/test_viz_orig_', json_file=detection_json_file_path)
    annos_corr = visualizer.visualize_output(image_ids=img_ids, save_name='plots/test_viz_corr_', json_file=corr_detection_json_file_path)
    print('plotted')



    # if gen_ranger_bounds:
    #     coco_dataloader, class_names = coco_loader(root='val', shuffle=False, batch_size=batch_size, sampling=True, sampleN=20)
    #     net_for_bounds = yolov3
    #     # dataloader = iter(coco_dataloader)
    #     act_input, act_output = extract_ranger_bounds(coco_dataloader, net_for_bounds, ranger_file_name, dataset_name) # gets also saved automatically
    #     print('check Ranger input', act_input)
    #     print('check Ranger output', act_output)
    #     sys.exit()

    # apply_ranger = False
    # # resil = [Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg]
    # resil = [Ranger]
    
    # # replace with pytorchfi wrapper code
    # # batchsize is set in scenarios/default.yml -> ptf_batch_size
    # num_faults = [1]
    # # fault_files = ["./logs/fault_rates_alexnet_1_500.bin", "./logs/fault_rates_alexnet_10_500.bin" ]

    # save_fault_file_dir = 'result_files'

    # for id, _num_faults in enumerate(num_faults):
    #     for _resil in resil:
    #         if apply_ranger:
    #             bnds = get_savedBounds_minmax('./bounds/' + ranger_file_name + '.txt')
    #             # Change network architecture to add Ranger
    #             net_for_protection = yolov3
    #             protected_yolov3, _ = get_Ranger_protection(net_for_protection, bnds, resil=_resil)
    #             protected_yolov3 = protected_yolov3.to(device)

    #         yolov3_ErrorModel = TestErrorModels(model=yolov3, resil_model=None, model_name=model_name, cuda_device=cuda_device,
    #                                 dataset='coco', store_json=True, resil_method=_resil.__name__, config_location='default.yml')
    #         yolov3_ErrorModel.test_random_single_bit_flip_inj(num_faults=_num_faults, fault_file=None, save_fault_file_dir=save_fault_file_dir)

if __name__ == "__main__":
    # ctx = mp.get_context("spawn")
    # ctx.set_start_method('spawn')
    main(sys.argv)
