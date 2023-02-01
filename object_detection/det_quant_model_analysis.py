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
from alficore.ptfiwrap_utils.hook_functions import set_quantiles_hooks, set_features_all_hooks
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr, assign_val_train
from alficore.dataloader.objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog
from det_quant_test_auto2 import set_up_mmdet_model, save_to_nwstore, set_state


# # Add mmdet:
# sys.path.append("/home/fgeissle/mmdetection")
# from mmdet.apis import init_detector
# from mmdet.apis.inference import inference_detector, show_result_pyplot

class SaveInfo:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append([module._get_name(), list(module.named_parameters())[0][1].shape, module_out.shape])

    def clear(self):
        self.outputs = []



def set_hooks_info(net):

    hook_handles_out = []
    hook_list = []
    setting_list = []
    try:
        cnt = 0
        for name, m in net.named_modules():
            if type(m) in [torch.nn.Conv2d]:
                act_hook = SaveInfo()
                handle_out = m.register_forward_hook(act_hook)
                hook_handles_out.append(handle_out)
                hook_list.append(act_hook)
                # print('info hook set', cnt, name, list(m.named_parameters())[0][1].shape)
                cnt += 1
                setting_list.append([name, list(m.named_parameters())[0][1].shape])
    except:
        print('Not able to integrate hooks, format of bound file correct?')

    return hook_list, hook_handles_out, setting_list



def main(argv):
    """
    Workflow: 
    1. Run this file for automated experiments, additionally for range extraction. Results are in nwstore or local.
    2. Run quantiles_extract_features_plot3.py for quantile extraction and save it to nwstore.
    3. Run train_detection_model_LR3.py or ..._DT2.py or ..._fcc1.py to train anomaly detector.
    """

    # Define dataset and model specifications
    ####################################################################
    dataset_name = 'coco' #'coco', kitti, 'robo', ppp, lyft
    model_name = 'ssd' #'yolov3', 'retina_net', 'ssd'

    batch_size = 1 #batchsize for neurons
    num_faults = 1 #faults
    num_runs = 1 #number of runs #500
    sample_size = 10 #nr of images (sequential)
    dataset_type = 'val' #'train', 'val'

    save_to_nw = False #Save directly to nw store?

    quant_monitoring = True
    ftrace_monitoring = False

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


    # # Profiling: ---------------------------------------------------------
    # # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    # from torch.profiler import profile, record_function, ProfilerActivity

    # dummy_input = torch.rand(1,3,100,100).to(device)
    # # a = torch.cuda.max_memory_allocated(device=device) #in bytes
    # # hook_list, hook_handles_list, setting_list = set_hooks_info(model)
    # hook_handles_quant, hook_list_quant = set_quantiles_hooks(model)
    # # hook_handles_quant, hook_list_quant = set_features_all_hooks(model)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         model(dummy_input)
    # # b= torch.cuda.max_memory_allocated(device=device) #since beginning of program
    # # print(a, b)
    # # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1))
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=1))
    # # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
    # # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=1))

    # # # # Note for memory only compare extracted values



    # # Model size/layer analysis with hooks --------------------------------------
    # No of parameters
    # from torchsummary import summary
    # summary(model.model,input_size=(3,416,416))
    # dummy_input = [{'image': torch.rand(3,100,100).to(device)}]
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

    print_layers = False
    if print_layers:
        with open('layer_names_info.txt', 'w') as f:
            for n in info_names:
                f.write(str(n)+"\n")
    
    # for n in range(len(info_sz)):
    #     if info_sz[n] != setting_list[n][1]:
    #         print(info_sz[n], setting_list[n])
    #     else:
    #         print('ok')



    # # Count parameters in Conv layers: -------------------------------
    # cnt_act = 0
    # cnt_conv = 0
    # cnt_conv_params = 0

    # for nm, m in model.model.named_modules(): 
    #     print(nm, type(m), cnt_conv)
    #     if type(m) in [torch.nn.Conv2d]: 
    #         print('yes', cnt_conv)
    #         if cnt_conv == 42:
    #             print()
    #         cnt_conv += 1
    #     if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]: 
    #         # print('yes act')
    #         cnt_act += 1

    # print('compare', cnt_act, cnt_conv)
    # print('done')

if __name__ == "__main__":
    main(sys.argv)

