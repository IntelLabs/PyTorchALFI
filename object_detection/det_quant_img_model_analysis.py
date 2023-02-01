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
from alficore.wrapper.test_error_models_imgclass import TestErrorModels_ImgClass
# from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize
from alficore.dataloader.objdet_baseClasses.boxes import Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.ptfiwrap_utils.build_native_model import build_native_model
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr, assign_val_train
from alficore.dataloader.objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog


# # Add mmdet:
# sys.path.append("/home/fgeissle/mmdetection")
# from mmdet.apis import init_detector
# from mmdet.apis.inference import inference_detector, show_result_pyplot
import torchvision
import warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
from LeNet5.LeNet5 import LeNet_orig
from det_quant_model_analysis import set_hooks_info

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


class  build_objdet_native_model_img_cls(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, device, dataset_name):
        super().__init__(model=model, device=device)
        ### img_size, preprocess and postprocess can also be inialised using kwargs which will be set in base class
        self.preprocess = True
        self.postprocess = False
        self.model_name = model._get_name().lower()
        if "lenet" in self.model_name:
            self.img_size = 32
        elif "alex" in self.model_name:
            self.img_size = 256
        elif "res" in self.model_name:
            self.img_size = 232
        else:
            self.img_size = 416

        self.transform = None
        if "imagenet" in dataset_name.lower():
            self.transform = transforms.Compose([            #[1]
                transforms.Lambda(lambda x: x/255.),
                transforms.Normalize(                      #[5]
                mean=[0.485, 0.456, 0.406],                #[6]
                std=[0.229, 0.224, 0.225]                  #[7]
                )])
        elif "mnist" in dataset_name.lower():
            self.transform = transforms.Compose([           
                transforms.Lambda(lambda x: x/255.),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
                
        self.corr_mode = None
        self.corr_magn = None


    def set_image_corruption(self, corr_mode, corr_magn):
        if corr_mode in ['Gaussian_blur', 'Gaussian_noise', 'Adjust_contrast']:
            self.corr_mode = corr_mode
            self.corr_magn = corr_magn
        elif corr_mode is None:
            self.corr_mode = None
            self.corr_magn = None
        else:
            print('Fault mode not known, please choose from either of: Gaussian_blur, Gaussian_noise, Adjust_contrast')


    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """


        if "lenet" in self.model_name:
            images = [resize(x['image'], self.img_size) for x in batched_inputs]
        elif "alex" in self.model_name:
            images = [x['image'] for x in batched_inputs]
        elif "res" in self.model_name:
            images = [x['image'] for x in batched_inputs]


        # Convert to tensor
        images = torch.stack(images).to(self.device)
        return images

    def postprocess_output(self):
        return

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
        #TODO: input modification for corr and resil_corr is different (random)!
        
        if (self.corr_mode is not None) and (self.corr_magn is not None):
            input = add_img_corr(deepcopy(input), self.corr_mode, self.corr_magn) 
            # print('Added input corr:', self.corr_mode, 'strength', self.corr_magn)
        input = self.transform(input) #retransform ImageNet images from rbg 255 to 0-1

        input = pytorchFI_objDet_inputcheck(input, dummy=dummy) #transforms input to dict list

        _input = input
        if self.preprocess:
            _input = self.preprocess_input(input)
        output = self.model(_input)
        # if self.postprocess:
        #     output = self.postprocess_output(output, input, original_shapes, _input.shape[2:])

        # output = pytorchFI_objDet_outputcheck(output)
        return output


def add_img_corr(input, corr_mode, corr_magn):

    for n in range(len(input)):
        orig_img = input[n]

        # Blur
        if corr_mode == "Gaussian_blur":
            sig = corr_magn
            # sig = 3 #sigma = 1,2,3
            orig_img = torchvision.transforms.GaussianBlur((5,9), sigma=(sig,sig))(orig_img) #(21,21), sigma=(3,3) 


        # Noise
        elif corr_mode == "Gaussian_noise":
            mean = 0
            # std = 10 #1,5,10
            std = corr_magn
            # noise = mean + std*torch.randn(orig_img.shape) #10 leads to 40% SDC, 1 to 17% SDC
            noise = np.random.normal(mean, std, orig_img.shape)
            noise = torch.tensor(noise).to(orig_img.device)
            # cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            orig_img = (orig_img + noise).to(torch.uint8)

        elif corr_mode == "Adjust_contrast":
            contrast_factor = corr_magn # 0=gray, 1=original;  #use 0.5, 0.3
            orig_img = torchvision.transforms.functional.adjust_contrast(orig_img, contrast_factor) #new fault type 
    
        input[n] = orig_img

    return input


def set_up_img_model(model_name, dataset_name, device):
    """
    NOTE: If fault injection has out of bound error: Need to go to config file and manually change ALL parameters in transforms -> keep_ratio to False.
    Needs to be done in multiple spots (test_pipeline, train_pipeline, data, etc).
    """

    if 'alex' in model_name.lower() and "imagenet" in dataset_name.lower():
        # # AlexNet
        alex_net = torchvision.models.alexnet(pretrained=True, progress=True)
        alex_net = alex_net.to(device)
        alex_net.eval()
        model = alex_net
    elif "resnet" in model_name.lower() and "imagenet" in dataset_name.lower():
        #ResNet
        from torchvision.models import resnet50
        res_net = resnet50(pretrained=True)
        res_net = res_net.to(device)
        res_net.eval()
        model = res_net
    elif "lenet" in model_name.lower() and "mnist" in dataset_name.lower():
        leNet = LeNet_orig(color_channels=1)
        leNet.load_state_dict(torch.load('LeNet5/lenet5-mnist.pth')) #load the pretrained weights
        leNet = leNet.to(device)
        leNet.eval()
        model = leNet
    else:
        print('Model not supported.')
        return None

    wrapped_model = build_objdet_native_model_img_cls(model, device, dataset_name)

    return wrapped_model


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

class exp_class_img():

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

        img_ErrorModel = TestErrorModels_ImgClass(model=self.model, model_name=self.model_name, resil_name='ranger_trivial', dl_attr=self.dl_attr, num_faults=self.num_faults,\
            config_location=self.yml_file, inf_nan_monitoring=True, ranger_bounds=[], ranger_detector=False, disable_FI=False, quant_monitoring=self.quant_monitoring, ftrace_monitoring = self.ftrace_monitoring, exp_type=flt_type, corr_magn=noise_magn) #, fault_file=ff, copy_yml_scenario=True)
        img_ErrorModel.test_rand_ImgClass_SBFs_inj()

        # net_Errormodel = TestErrorModels_ImgClass(model=wrapped_model, resil_model=None, resil_name=None, model_name=model._get_name(), config_location=opt.config_file, \
        #             ranger_bounds=None, device=device, ranger_detector=False, inf_nan_monitoring=True, disable_FI=False, dl_attr=dl_attr, num_faults=0, fault_file=fault_files, \
        #                 resume_dir=None, copy_yml_scenario = False)
        # net_Errormodel.test_rand_ImgClass_SBFs_inj()

        # DatasetCatalog.remove(self.dataset_name)
        # MetadataCatalog.remove(self.dataset_name)


def main(argv):
    """
    Workflow: 
    1. Run this file for automated experiments, additionally for range extraction. Results are in nwstore or local.
    2. Run quantiles_extract_features_plot3.py for quantile extraction and save it to nwstore.
    3. Run train_detection_model_LR3.py or ..._DT2.py or ..._fcc1.py to train anomaly detector.
    """

    # Define dataset and model specifications
    ####################################################################
    dataset_name = 'imagenet' #'imagenet, mnist
    model_name = 'alexnet' #'alexnet', 'resnet', 'lenet'

    batch_size = 1 #batchsize for neurons
    num_faults = 1 #faults
    num_runs = 1 #number of runs #500
    sample_size = 10 #nr of images (sequential)
    dataset_type = 'val' #'train', 'val'

    save_to_nw = False #Save directly to nw store?

    quant_monitoring = True
    ftrace_monitoring = False

    ####################################################################

    # Set device ---------------------------------------------------------
    cuda_device = 1
    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu") 

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

    if "lenet" in "lenet" in model_name.lower():
        set_state(yml_file_path, 'ptf_C', 1) #color channels
    else:
        set_state(yml_file_path, 'ptf_C', 3) #color channels

    

    # # Model : ---------------------------------------------------------
    model = set_up_img_model(model_name, dataset_name, device)


    # # Test
    dummy_input = torch.rand(1,3,100,100).to(device)
    
    hook_list, hook_handles_list, setting_list = set_hooks_info(model)

    output = model(dummy_input)

    info_names = [n[0] for n in setting_list]
    info_types = [n.outputs[0][0] for n in hook_list]
    info_kn = [n.outputs[0][1] for n in hook_list]
    info_sz = [n.outputs[0][2] for n in hook_list]
    for i in range(len(hook_handles_list)):
        hook_handles_list[i].remove()
        hook_list[i].clear()

    # Print no of conv activations
    no_kn = np.sum([np.prod(x) for x in info_kn])
    no_activations = np.sum([np.prod(x) for x in info_sz])
    no_fmaps = np.sum([np.prod(x[:2]) for x in info_sz])
    print(f"no of kernels: {no_kn:e}, model: {model_name}, {dataset_name}")
    print(f"no of activations: {no_activations:e}, model: {model_name}, {dataset_name}")
    print(f"no of fmaps: {no_fmaps:e}, model: {model_name}, {dataset_name}")

    with open('layer_names_info.txt', 'w') as f:
        for n in info_names:
            f.write(str(n)+"\n")
    print('saved layer_names_info.txt')


    # # # # Specific layers:
    # # mlist = list(model.model.named_modules())
    # cnt_act = 0
    # cnt_conv = 0
    # for _, m in model.model.named_modules(): 
    #     print(type(m))
    #     if type(m) in [torch.nn.Conv2d, torch.nn.Linear]: 
    #         print('yes')
    #         cnt_conv += 1
    #     if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]: 
    #         print('yes act')
    #         cnt_act += 1

    # print('compare', cnt_act, cnt_conv)
    # print('done')


if __name__ == "__main__":
    main(sys.argv)

