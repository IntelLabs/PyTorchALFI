import os
import sys
import torch
import argparse
import yaml
from copy import deepcopy
import numpy as np
from typing import Dict, List
import datetime


# Add wrapper:
# sys.path.append("..")
sys.path.append("/home/fgeissle/fgeissle_ranger")
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize
from alficore.dataloader.objdet_baseClasses.boxes import Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.ptfiwrap_utils.build_native_model import build_native_model
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr, assign_val_train
from alficore.dataloader.objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog


# Add mmdet:
sys.path.append("/home/fgeissle/mmdetection")
from mmdet.apis import init_detector
from mmdet.apis.inference import inference_detector, show_result_pyplot




class build_objdet_native_model_mmdet(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, device, dataset_name, conf_thres=0.25):
        super().__init__(model=model, device=device)
        self.dataset_name = dataset_name
        self.img_size = [(0,0)]
        self.conf_thres = conf_thres
        self.preprocess = False #not needed but function must exist due to abstract template
        self.postprocess = True

    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """
        # Preprocessing done by mmnet pipeline
        return

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

        out_list = []
        for _, im in enumerate(output):
            mask = [len(n) > 0 for n in im]
            objects = np.array(im,dtype=object)[mask]
            if len(objects) > 0:
                obj_class_counts = [u.shape[0] for u in objects]
                objects = np.vstack(objects) #make one single list

                classes_found = np.array(self.model.CLASSES)[mask]
                classes_sorted = []
                for a,b in list(zip(obj_class_counts, classes_found)):
                    for x in range(a):
                        ind = np.where(np.array(self.model.CLASSES)== b)[0][0] #go from class label to index
                        classes_sorted.append(ind)
                
                scores = objects[:,4]
            else:
                classes_sorted = []
                scores = []

            out_instance = Instances(self.img_size)

            # Filter by confidence:
            mask_high_conf = np.array(scores)>=self.conf_thres
            scores = np.array(scores)[mask_high_conf]
            objects = np.array(objects)[mask_high_conf]
            classes_sorted = np.array(classes_sorted)[mask_high_conf].tolist()
            try:
                boxes = Boxes(objects[:,:4])
            except:
                boxes = Boxes(torch.tensor([]))
            out_instance.set("pred_boxes", boxes)
            out_instance.set("pred_classes", torch.tensor(classes_sorted).type(torch.ByteTensor))
            out_instance.set("scores", scores)

            # if 'robo' in self.dataset_name: #filter for only classes with label 0 (persons)
            #     red_mask = out_instance.pred_classes == 0
            #     cls = out_instance.pred_classes[red_mask]
            #     bxs = out_instance.pred_boxes[red_mask]
            #     scrs = out_instance.scores[red_mask]
            #     out_instance = Instances(self.img_size)
            #     out_instance.set("pred_classes", cls)
            #     out_instance.set("pred_boxes", bxs)
            #     out_instance.set("scores", scrs)

            out_list.append({'instances': out_instance})
        return out_list

    def get_class_dict(self):
        # Extract class info
        classes = self.model.CLASSES
        indices = list(range(len(classes)))
        class_dict = dict(zip(classes, indices))
        return class_dict

    def get_img_scale(self):
        return self.model.cfg.test_pipeline[1]['img_scale']

    def __getattr__(self, method):
        if method.startswith('__'):
            raise AttributeError(method)
        try:
        # if hasattr(self.model, method):
            # print(self.model, 'method:', method)
            # print(self.model.model)

            try:
                func = getattr(self.model.model, method)
            except:
                func = getattr(self.model, method)
            ## running pytorch model (self.model) inbuilt functions like eval, to(device)..etc
            ## assuming the executed method is not changing the model but rather
            ## operates on the execution level of pytorch model.
            # print('check', func)
            def wrapper(*args, **kwargs):            
                if (method=='to'):
                    return self
                else:
                    return  func(*args, **kwargs)
            # print('check', wrapper, method, self)
            return wrapper
        except KeyError:
            raise AttributeError(method)

    def __call__(self, input, dummy=False):
        input = pytorchFI_objDet_inputcheck(input, dummy=dummy) #wraps image with metadata


        if "file_name" in input[0].keys(): #load from original image
            img_path = [n['file_name'] for n in input]
            self.img_size = [(n['height'], n['width']) for n in input] #original image size, gets resclaed during inference_detector data transformation
            result = inference_detector(self.model, img_path)
            # # test plot of first image in batch:
            # show_result_pyplot(self.model, img_path[0], result[0], out_file = '/home/fgeissle/ranger_repo/ranger/object_detection/demo_boxes.jpg')
        
        else: #load from given tensor (for example for dummy tensor)
            img_path = []
            self.img_size = [tuple(n['image'].shape[1:]) for n in input] #original image size
            result = inference_detector(self.model, [np.transpose(n['image'].cpu().numpy(), (1,2,0)) for n in input])

            # # test plot of first image in batch:
            # show_result_pyplot(self.model, np.transpose(input[0]['image'].cpu().numpy(), (1,2,0)), result[0], out_file = '/home/fgeissle/ranger_repo/ranger/object_detection/demo_boxes.jpg')
        

        if self.postprocess:
            output = self.postprocess_output(result)

        output = pytorchFI_objDet_outputcheck(output)
        return output

def set_state(yml_file_path, field_name, field_state):
    
    with open(yml_file_path) as f:
        doc = yaml.safe_load(f)

    doc[field_name] = field_state

    with open(yml_file_path, 'w') as f:
        yaml.dump(doc, f)

def save_to_nwstore(yml_file_path):
    e = datetime.datetime.now()
    folder_name = str(e.year) + "-" + str(e.month).zfill(2) + "-" + str(e.day).zfill(2) + "-" + str(e.hour).zfill(2) + "-" + str(e.minute).zfill(2)
    save_path = '/nwstore/florian/LR_detector_data_auto/' + folder_name + '/'
    # save_path = '/nwstore/florian/LR_detector_data_auto/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    set_state(yml_file_path, 'save_fault_file_dir', save_path) #path
    return save_path

def set_up_mmdet_model(model_name, dataset_name, device):
    """
    NOTE: If fault injection has out of bound error: Need to go to config file and manually change ALL parameters in transforms -> keep_ratio to False.
    Needs to be done in multiple spots (test_pipeline, train_pipeline, data, etc).
    """

    # Retina net
    if 'retina' in model_name.lower():

        if 'coco' in dataset_name.lower():
            config_file = '/home/fgeissle/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/Retina-Net/pretrained_weights/retinanet-r50_fpn_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
            
        elif 'kitti' in dataset_name.lower():
            # config_file = '/home/fgeissle/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_kitti.py'
            config_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/Retina-Net/pretrained_weights/retinanet-r50_fpn_kitti/retinanet_r50_fpn_1x_kitti.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/Retina-Net/pretrained_weights/retinanet-r50_fpn_kitti/latest.pth'
    
    # Yolov3
    elif "yolo" in model_name.lower():

        if 'coco' in dataset_name.lower():
            config_file = '/home/fgeissle/mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/YoloV3/pretrained_weights/yolo_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'

        if 'kitti' in dataset_name.lower():
            # config_file = '/home/fgeissle/mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_kitti.py'
            config_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/YoloV3/pretrained_weights/yolo_kitti/yolov3_d53_mstrain-416_273e_kitti.py' #original file
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/YoloV3/pretrained_weights/yolo_kitti/latest.pth'


    elif "ssd" in model_name.lower():

        if 'coco' in dataset_name.lower():
            config_file = '/home/fgeissle/mmdetection/configs/ssd/ssd512_coco.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/SSD/pretrained_weights/ssd_512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth'

        if 'kitti' in dataset_name.lower():
            # config_file = '/home/fgeissle/mmdetection/configs/ssd/ssd512_kitti.py'
            config_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/SSD/pretrained_weights/ssd_512_kitti/ssd512_kitti.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/SSD/pretrained_weights/ssd_512_kitti/latest.pth'



    else:
        print('Model and dataset combination not found.')
        sys.exit()
        
    model = init_detector(config_file, checkpoint_file, device=device) #takes care of eval, to(device)
    
    model = build_objdet_native_model_mmdet(model=model, device=device, dataset_name=dataset_name)

    return model

class exp_class():

    def __init__(self, dataset_name, model_name, yml_file_path, dl_attr, bs, model, yml_file, num_faults, quant_monitoring, ftrace_monitoring):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.yml_file_path = yml_file_path
        self.yml_file = yml_file
        self.dl_attr = dl_attr
        self.bs = bs
        self.model = model
        self.num_faults = num_faults
        self.quant_monitoring = quant_monitoring
        self.ftrace_monitoring = ftrace_monitoring

    def run(self, flt_type, noise_magn, exp_name, target_list, inj_policy, rnd_mode, rnd_bit_range):

        target_list.append(exp_name)
        print('Running experiment:', target_list[-1])
        set_state(self.yml_file_path, 'inj_policy', inj_policy) #
        # dl_attr.dl_batch_size = deepcopy(bs)
        set_state(self.yml_file_path, 'ptf_batch_size', deepcopy(self.bs)) 
        set_state(self.yml_file_path, 'rnd_mode', rnd_mode)
        set_state(self.yml_file_path, 'rnd_bit_range', rnd_bit_range)
        ####

        yolov3_ErrorModel = TestErrorModels_ObjDet(model=self.model, model_name=self.model_name, resil_name='ranger_trivial', dl_attr=self.dl_attr, num_faults=self.num_faults,\
            config_location=self.yml_file, inf_nan_monitoring=True, ranger_bounds=[], ranger_detector=False, disable_FI=False, quant_monitoring=self.quant_monitoring, ftrace_monitoring = self.ftrace_monitoring, exp_type=flt_type, corr_magn=noise_magn) #, fault_file=ff, copy_yml_scenario=True)
        yolov3_ErrorModel.test_rand_ObjDet_SBFs_inj()

        DatasetCatalog.remove(self.dataset_name)
        MetadataCatalog.remove(self.dataset_name)


def main(argv):
    """
    Workflow: 
    1. Run this file for automated experiments, additionally for range extraction. Results are in nwstore or local.
    2. Run quantiles_extract_features_plot3.py for quantile extraction and save it to nwstore.
    3. Run train_detection_model_LR3.py or ..._DT2.py or ..._fcc1.py to train anomaly detector.
    """

    ms = ['yolo'] #['yolov3', 'ssd', 'retina', 'yolov3', 'ssd', 'retina']
    ds = ['kitti']
    # ms = ['ssd', 'retina', 'retina'] #, 'yolov3', 'ssd', 'retina']
    # ds = ['kitti', 'coco', 'kitti']

    for a in range(len(ms)):
        # Define dataset and model specifications
        ####################################################################
        dataset_name = ds[a] #'coco', kitti, 'robo', ppp, lyft
        model_name = ms[a] #'yolov3', 'retina_net', 'ssd'

        batch_size = 20 #batchsize for neurons
        num_faults = 1 #faults
        num_runs = 1 #number of runs #500
        sample_size = 100 #nr of images (sequential)
        dataset_type = 'val' #'train', 'val'

        save_to_nw = False #Save directly to nw store?

        quant_monitoring = False
        ftrace_monitoring = True

        # TODO: Issues with retina_net:
        # - run with smaller bs (10), but >1 in general ok
        # - ftrace_monitoring not done yet
        # - only layer empty for monitoring, will be skipped
        # - use quantiles without presum fails (too large layers?)
        ####################################################################

        # Set device ---------------------------------------------------------
        cuda_device = 0
        device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu") 
        print('running on', device)

        # Storing location ---------------------------------------------------------
        yml_file = 'default_min_quant_test_auto.yml'
        yml_file_path = "scenarios/" + yml_file
        if save_to_nw:
            save_path = save_to_nwstore(yml_file_path)
        else:
            set_state(yml_file_path, 'save_fault_file_dir', 'result_files/')


        ## set dataloader attributes ----------------------------------------
        dl_attr = TEM_Dataloader_attr()
        dl_attr.dl_batch_size  = deepcopy(batch_size)
        dl_attr.dl_sampleN     = sample_size #NOTE: <= actual dataset, e.g. for pp <=51
        dl_attr.dl_random_sample  = False
        dl_attr.dl_shuffle     = False
        dl_attr.dl_mode        = "sequence" # "image/sequence"
        dl_attr.dl_scenes      = [0,1]
        dl_attr.dl_num_workers = 4
        dl_attr.dl_device      = device
        dl_attr.dl_sensor_channels  = ["CAM_FRONT"]
        dl_attr.dl_dataset_type = dataset_type #train, val
        dl_attr.dl_dataset_name = dataset_name
        dl_attr = assign_val_train(dl_attr)
        

        set_state(yml_file_path, 'num_runs', num_runs) #nr of runs
        set_state(yml_file_path, 'dataset_size', sample_size) #nr of runs
        set_state(yml_file_path, 'ptf_C', 3) #color channels

        # # Model : ---------------------------------------------------------
        model = set_up_mmdet_model(model_name, dataset_name, device)


        # Adjust image scale in scenario ---------------------------------------------
        # model.model.cfg.test_pipeline[1]['transforms'][0]['keep_ratio'] = False #test added
        class_dict = model.get_class_dict()
        if 'kitti' in dataset_name.lower():
            #kitti has homogeneous image sizes, need to adjust FI to size detected in build_model -> call -> input
            (ptfw, ptfh) = (1224, 370) 
        elif 'coco' in dataset_name.lower():
            (ptfw, ptfh) = model.get_img_scale()
        set_state(yml_file_path, 'ptf_H', ptfh)
        set_state(yml_file_path, 'ptf_W', ptfw)
        # Note: for kitti: put break point in call_size to find new size and overwrite in scenario.yaml, keep keep_ratio = True in config file 


        
        # # Ranger bounds: ---------------------------------------------------------
        # gen_ranger_bounds = True #if true new bounds are generated (overwritten), used 2000 images
        # get_percentiles = True # True, False
        # get_ftraces = False
        

        # # non-presum too large for quantiles!?
        # ranger_file_name = model_name + '_' + dataset_name + '_v3_presum'

        # from alficore.ptfiwrap_utils.evaluate import get_ranger_bounds_quantiles
        # bnds = get_ranger_bounds_quantiles(model, dl_attr, ranger_file_name, gen_ranger_bounds, get_percentiles, get_ftraces)



        

        # # Inference runs: ---------------------------------------------------------
        # #################################################################################
        target_list = []
        experiment_template = exp_class(dataset_name, model_name, yml_file_path, dl_attr, batch_size, model, yml_file, num_faults, quant_monitoring, ftrace_monitoring)

        # # Gaussian noise
        dl_attr.dl_batch_size = deepcopy(batch_size)
        experiment_template.run("Gaussian_noise", 0.1, "noise_lvl01", target_list, 'per_image', 'neurons', [0,31])
        experiment_template.run("Gaussian_noise", 1, "noise_lvl1", target_list, 'per_image', 'neurons', [0,31])
        experiment_template.run("Gaussian_noise", 10, "noise_lvl10", target_list, 'per_image', 'neurons', [0,31])


        # # # Blur 
        dl_attr.dl_batch_size = deepcopy(batch_size)
        experiment_template.run("Gaussian_blur", 0.3, "blur_lvl03", target_list, 'per_image', 'neurons', [0,31])
        experiment_template.run("Gaussian_blur", 1, "blur_lvl1", target_list, 'per_image', 'neurons', [0,31])
        experiment_template.run("Gaussian_blur", 3, "blur_lvl3", target_list, 'per_image', 'neurons', [0,31])


        # # Low contrast
        dl_attr.dl_batch_size = deepcopy(batch_size)
        experiment_template.run("Adjust_contrast", 0.9, "contrast_lvl09", target_list, 'per_image', 'neurons', [0,31])
        experiment_template.run("Adjust_contrast", 0.5, "contrast_lvl05", target_list, 'per_image', 'neurons', [0,31])
        experiment_template.run("Adjust_contrast", 0.1, "contrast_lvl01", target_list, 'per_image', 'neurons', [0,31])


        # # # # Neuron faults
        dl_attr.dl_batch_size = deepcopy(batch_size)
        experiment_template.run("hwfault", 1, "neurons_32", target_list, 'per_image', 'neurons', [0,31])
        # experiment_template.run("hwfault", 2, "neurons_03", target_list, 'per_image', 'neurons', [1,3])

        # # # # Weight faults
        dl_attr.dl_batch_size = 1
        experiment_template.run("hwfault", 1, "weights_32", target_list, 'per_batch', 'weights', [0,31])
        # experiment_template.run("hwfault", 2, "weights_03", target_list, 'per_batch', 'weights', [1,3])




if __name__ == "__main__":
    main(sys.argv)

