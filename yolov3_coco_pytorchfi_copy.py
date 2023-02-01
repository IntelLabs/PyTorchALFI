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
from util.helper_functions import getdict_ranger, get_savedBounds_minmax
from util.evaluate import extract_ranger_bounds, extract_ranger_bounds_objdet
from util.ranger_automation import get_Ranger_protection
import pandas as pd
from typing import Dict, List, Optional, Tuple
from alficore.models.yolov3.darknet import Darknet
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize

from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg

from alficore.dataloader.objdet_baseClasses.boxes import Boxes, BoxMode
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.models.yolov3.utils import non_max_suppression, rescale_boxes
from visualization_florian import *

from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
from alficore.dataloader.ppp_loader import PPP_obj_det_dataloader
import pickle
from darknet_Ranger import Darknet_Ranger

from obj_det_evaluate_jsons_florian import group_by_image_id_indiv, read_fault_file, load_json_indiv, eval_image, eval_epoch, eval_experiment, get_sdc_mean_per_epoch
import csv
import json 


torch.cuda.empty_cache()
logging.config.fileConfig('fi.conf')
log = logging.getLogger()
cuda_device = 1
model_name = 'yolov3'


class build_d2_model(object):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, conf_thres=0.25, iou_thres=0.45):
        self.model = model
        self.model.eval()
        self.img_size = 416
        self.preprocess = True ## detectron2 has inuilt preprocess
        self.postprocess = True
        self.device = torch.device(
        "cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres

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
        # images = [x["image"]/255. for x in batched_inputs]
        images = [x["image"]/255. for x in batched_inputs]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
         # Pad to square resolution
        padded_imgs = [pad_to_square(img, 0)[0] for img in images]
        # Resize
        images = [resize(img, self.img_size) for img in padded_imgs]

        # transform = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #TODO: right transform?
        # # images = transform(images)

        # transform = torch.nn.Sequential(
        #     # transforms.CenterCrop(10),
        #     # transforms.Resize((256, 256)),
        #     # transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     )
        # scripted_transforms = torch.jit.script(transform)
        # images = [scripted_transforms(x) for x in images]

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
        Output = non_max_suppression(output, self.conf_thres, self.iou_thres)
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


# def save_to_csv(result_line, csv_file):
#     """
#     Saves result_line by appending it to an existing csv file.
#     """
#     to_create = not os.path.exists(csv_file) #to create header only once

#     with open(csv_file, mode='a+') as csv_f:
#         fieldnames = ['gt', 'orig', 'corr', 'orig_resil', 'corr_resil']
#         writer = csv.writer(csv_f) #, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

#         if to_create:
#             writer.writerow(fieldnames)
#         writer.writerow(result_line)


def main(argv):
    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

    dataset_name = 'coco'
    ranger_file_name = 'yolov3_bounds_CoCo_train20p_act'
    bnds =  get_savedBounds_minmax('./bounds/' + ranger_file_name + '.txt')

    


    # # # Run Yolo fault injection model_output -------------------------------------------------------------------------------------------------
    yolov3 = Darknet("alficore/models/yolov3/config/yolov3.cfg").to(device)
    yolov3.load_darknet_weights("alficore/models/yolov3/weights/yolov3.weights")

    yolov3_resil = Darknet_Ranger("alficore/models/yolov3/config/yolov3.cfg", Ranger, bnds).to(device)
    yolov3_resil.load_darknet_weights("alficore/models/yolov3/weights/yolov3.weights")

    conf_thres=0.25
    iou_thres=0.45
    yolov3 = build_d2_model(yolov3, conf_thres=conf_thres, iou_thres=iou_thres)
    yolov3_resil = build_d2_model(yolov3_resil, conf_thres=conf_thres, iou_thres=iou_thres)

    yolov3_ErrorModel = TestErrorModels_ObjDet(model=yolov3, resil_model=yolov3_resil, model_name=model_name, dataset='coco',\
        config_location='default_yolo.yml', dl_sampleN=1, dl_shuffle=False, device=device) #0.1 not working for some reason, 0.01 works

    nr_faults = 1
    yolov3_ErrorModel.test_rand_ObjDet_SBFs_inj(fault_file='', num_faults=nr_faults, inj_policy=None) #only writes fault file if doesnt exist already!



    # # Visualization --------------------------------------------------------------------------------------------------------------------------------
    # ground_truth_json_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/ppp/val/coco_format.json'
    # detection_json_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/ppp/val/orig_model/epochs/0/coco_instances_results_0_epoch.json'
    # corr_detection_json_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/ppp/val/corr_model/epochs/0/coco_instances_results_0_epoch.json'
    # #
    # resil_detection_json_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/ppp/val/resil_model/epochs/0/coco_instances_results_0_epoch.json'
    # resil_corr_detection_json_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/ppp/val/resil_corr_model/epochs/0/coco_instances_results_0_epoch.json'

    # faults = read_fault_file('/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1_trials/weights_injs/1_faults/yolov3_test_rand_objDet_sbf_weights_inj_1_1_1bs_ppp_fault_locs.bin')
    # print(faults)

    # ppp_dataloader = PPP_obj_det_dataloader(dataset_type='val', sampleN = 0.01) #do not need to register again if already done in TestErrorModels_ObjDet
    # metadata = MetadataCatalog.get('ppp/val')
    # image = np.random.randn(416,416,3)
    # visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE, vis_mode='offline', gt_json_file=ground_truth_json_file_path)
    # # possible_img_ids = visualizer.get_image_ids()
    # img_ids = 'all' #[1, 2] #'all' #[1] #list [1,2,3] or 'all'

    # # Check predictions vs gt
    # annos_gt = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp/gt/test_viz_gt_', json_file=ground_truth_json_file_path)
    # annos_orig = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp/orig/test_viz_orig_', json_file=detection_json_file_path)
    # annos_corr = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp/corr/test_viz_corr_', json_file=corr_detection_json_file_path)

    # annos_orig_resil = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp/orig_resil/test_viz_orig_resil_', json_file=resil_detection_json_file_path)
    # annos_corr_resil = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp/corr_resil/test_viz_corr_resil_', json_file=resil_corr_detection_json_file_path)
    # print('Images saved')



    # # Evaluation --------------------------------------------------------------------------------------------------------------------------------
    # Part I: extraction -----

    # # faults_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_100_trials/weights_injs/per_image/1_faults/ppp/val/yolov3_test_random_sbf_weights_inj_1_100_10bs_ppp_fault_locs.bin'
    # # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_100_trials/weights_injs/per_image/1_faults/ppp/val'
    # faults_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1000_trials/weights_injs/per_image/1_faults/ppp/val/yolov3_test_random_sbf_weights_inj_1_1000_10bs_ppp_fault_locs.bin'
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1000_trials/weights_injs/per_image/1_faults/ppp/val'

    # iou_thresh = 0.5
    # epochs = range(1000) #can be list [0,1,2] or e.g. range(3) so 0 to 2
    # save_name = "yolo_ppp_results_2.json"

    # eval_experiment(epochs, iou_thresh, folder_path, save_name, faults_path)

    # Part II: analysis -----
    # res_all = load_json_indiv("yolo_ppp_results_2_backup.json")

    # orig_sdc, corr_sdc, resil_orig_sdc, resil_corr_sdc = get_sdc_mean_per_epoch(res_all)

    # # orig_sdc_all
    # print('orig_sdc', np.mean(orig_sdc), 'corr_sdc', np.mean(corr_sdc), 'resil_orig_sdc', np.mean(resil_orig_sdc), 'resil_corr_sdc', np.mean(resil_corr_sdc))


    # # compare to: res_all["faults"]
    # flts = res_all["faults"]
    # bpos = [n[6] for n in flts]
    # lays = [n[0] for n in flts]

    # import matplotlib.pyplot as plt

    # # average by bit position
    # fig, ax = plt.subplots(1,2)

    # orig_av = []
    # corr_av = []
    # for b in np.arange(0,8+1):
    #     orig_av.append(np.mean(np.array(orig_sdc)[np.array(bpos)==b]))
    #     corr_av.append(np.mean(np.array(corr_sdc)[np.array(bpos)==b]))


    # ax[0].scatter(np.arange(0,8+1), orig_av)
    # ax[0].set_xlabel('bit position')
    # ax[0].set_ylabel('SDC rate - orig')
    # ax[1].scatter(np.arange(0,8+1), corr_av)
    # ax[1].set_xlabel('bit position')
    # ax[1].set_ylabel('SDC rate - corr')

    # fig.tight_layout()
    # plt.savefig("plots/evaluation/bpos_vs_sdc_2.png", dpi=300)
    # plt.show()

    # # average by layer index
    # fig, ax = plt.subplots(1,2)

    # orig_av = []
    # corr_av = []
    # for b in np.arange(0,72+1):
    #     orig_av.append(np.mean(np.array(orig_sdc)[np.array(lays)==b]))
    #     corr_av.append(np.mean(np.array(corr_sdc)[np.array(lays)==b]))


    # ax[0].scatter(np.arange(0,72+1), orig_av)
    # ax[0].set_xlabel('layer')
    # ax[0].set_ylabel('SDC rate - orig')
    # ax[1].scatter(np.arange(0,72+1), corr_av)
    # ax[1].set_xlabel('layer')
    # ax[1].set_ylabel('SDC rate - corr')
    # fig.tight_layout()

    # plt.savefig("plots/evaluation/lay_vs_sdc_2.png", dpi=300)
    # plt.show()


    # what do the faulty pictures look like? ---------------------------
    # res_all = load_json_indiv("yolo_ppp_results_2_backup.json")
    # flts = res_all["faults"]
    # bpos = [n[6] for n in flts]
    # lays = [n[0] for n in flts]

    # # find images that are problematic
    # faulty_examples = []
    # for y in range(len(res_all["corr"])):
    #     for z in range(len(res_all["corr"][y])):
    #         tp = res_all["corr"][y][z]["tp"]
    #         fp = res_all["corr"][y][z]["fp"]
    #         fn = res_all["corr"][y][z]["fn"]
    #         sdc = (fp + fn)/(fp + fn + 2*tp)
    #         if sdc > 0.9:
    #             faulty_examples.append([y,z])
    #             break

    # faulty_examples

    # #visualize them
    # eps = np.array([i[0] for i in faulty_examples])
    # # example_faults = np.array(flts)[eps]
    # bpos_example = np.array(bpos)[eps]
    # lays_example = np.array(lays)[eps]


    # # faults_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1000_trials/weights_injs/per_image/1_faults/ppp/val/yolov3_test_random_sbf_weights_inj_1_1000_10bs_ppp_fault_locs.bin'
    # folder_path = '/home/fgeissle/ranger_repo/ranger/result_files/yolov3_1000_trials/weights_injs/per_image/1_faults/ppp/val'
    # ground_truth_json_file_path = folder_path + '/coco_format.json'

    # ppp_dataloader = PPP_obj_det_dataloader(dataset_type='val', sampleN = 0.1) #do not need to register again if already done in TestErrorModels_ObjDet
    # metadata = MetadataCatalog.get('ppp/val')
    # image = np.random.randn(416,416,3)
    # visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE, vis_mode='offline', gt_json_file=ground_truth_json_file_path)

    # for a in faulty_examples:
    #     epoch_nr = a[0]
    #     img_nr = a[1] + 1
    #     print('epoch', epoch_nr, 'image', img_nr, 'fault:',  flts[epoch_nr])

    #     detection_json_file_path = folder_path + '/orig_model/epochs/0/coco_instances_results_0_epoch.json'
    #     corr_detection_json_file_path = folder_path + '/corr_model/epochs/' + str(epoch_nr) + '/coco_instances_results_' + str(epoch_nr) + '_epoch.json'
    #     resil_detection_json_file_path = folder_path + '/ranger_model/epochs/0/coco_instances_results_0_epoch.json'
    #     resil_corr_detection_json_file_path = folder_path + '/ranger_corr_model/epochs/' + str(epoch_nr) + '/coco_instances_results_'+ str(epoch_nr) + '_epoch.json'

    #     # possible_img_ids = visualizer.get_image_ids()
    #     img_ids = [img_nr] #[1, 2] #'all' #[1] #list [1,2,3] or 'all'

    #     # Check predictions vs gt
    #     annos_gt = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp_fault_test/gt/test_viz_gt_' + str(epoch_nr) + '_', json_file=ground_truth_json_file_path)
    #     annos_orig = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp_fault_test/orig/test_viz_orig_' + str(epoch_nr) + '_', json_file=detection_json_file_path)
    #     annos_corr = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp_fault_test/corr/test_viz_corr_' + str(epoch_nr) + '_', json_file=corr_detection_json_file_path)

    #     annos_orig_resil = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp_fault_test/orig_resil/test_viz_orig_resil_' + str(epoch_nr) + '_', json_file=resil_detection_json_file_path)
    #     annos_corr_resil = visualizer.visualize_output(image_ids=img_ids, save_name='plots/ppp_fault_test/corr_resil/test_viz_corr_resil_' + str(epoch_nr) + '_', json_file=resil_corr_detection_json_file_path)
    #     print('Images saved')




    


if __name__ == "__main__":
    # ctx = mp.get_context("spawn")
    # ctx.set_start_method('spawn')
    main(sys.argv)










    # # qutubs version
    # # from alficore.evaluation.visualization.visualization import Visualizer, ColorMode
    # visualizer = Visualizer(metadata=metadata, instance_mode=ColorMode.IMAGE, vis_mode='offline')
    # visualizer.draw_instance_gt_pred_offline(img_ids=[1, 2, 3], viz_cdt=True, viz_rdt=False, viz_rcdt=False, gt_json_file=ground_truth_json_file_path,
    #     dt_json_file=detection_json_file_path, resil_name='ranger', epoch = 0)

    
    # # This is ok:
    # yolov3_ErrorModel._ptfi_dataloader()
    # yolov3_ErrorModel.dataloader.datagen_itr()
    # img = yolov3_ErrorModel.dataloader.data
    # # img = img[0]["image"].to(device)
    # output = yolov3(img) #why not working?
    # print(output[0]["instances"].get("pred_boxes"))