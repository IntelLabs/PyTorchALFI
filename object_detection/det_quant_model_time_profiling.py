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
# from alficore.ptfiwrap_utils.hook_functions import set_quantiles_hooks, set_features_all_hooks
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr, assign_val_train
# from alficore.dataloader.objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog
from det_quant_test_auto2 import set_up_mmdet_model, save_to_nwstore, set_state
from det_quant_test_img_auto1 import set_up_img_model
from torch.profiler import profile, record_function, ProfilerActivity

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


def get_minimal_fts(model_name, dataset_name):
    d = model_name + "_" + dataset_name
    from check_dt_train_results_ft_imp import load_json_indiv, filter_alts, get_sep_lists_lay_qu
    fl_names = "unq2" #"imp", red_by_layer
    pth = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/'
    data = load_json_indiv(pth + "dt_train_ft_" + fl_names + "_" + d + ".json")

    for x in list(data.keys()):

        # Basic checks ----------------------------------
        assert len(list(data.keys()))==1, 'More than one model per file, is that intended?'
        #in case there is an empty ft key:
        if len(data[x]['min_alternatives'])==0:
            print('empty, skipping')
            continue

        # Prepare data ---------------------------------
        alts = data[x]['min_alternatives']

        # Eliminate doubles:
        alts = filter_alts(alts)
        # alts = alts[:nr_sel] #eg take first ten only

        alts_fts = [n['fts'] for n in alts]
        # alts_p = [n['p_cls'] for n in alts]
        # alts_r = [n['r_cls'] for n in alts]
        print('features no:', [len(x) for x in alts_fts])

        lays_all_sorted, qus_all_sorted = get_sep_lists_lay_qu(alts_fts)
        return lays_all_sorted, qus_all_sorted



def set_quantiles_hooks_customized(net, lays_min, qus_min):
    """
    Sets hooks into the entire model. Hooks have individual bounds but will all write to the same list act_list.
    :param net: pytorch model
    :param bnds: list of bounds
    :param resil: type of resilience method (currently only "ranger")
    :param mitigation: flag whether or not Ranger mitigation should be applied, true by default.
    :param detector: If activated the number of activated ranger layer per batch is given as output.
    :return: act_list, list of saved information from Range_detector hooks.
    :return: hook_handles_out, list of handles to be cleared later
    :return: hook_list, list of hooks to be cleared later
    """

    hook_handles_out = []
    hook_list = []
    cnt = 0
    cnt_match = 0
    for _, m in net.named_modules():
        if type(m) in [torch.nn.Conv2d]:
            if cnt in lays_min: # torch.nn.Linear
                act_hook = Range_detector_quantiles_customized(qus_min[cnt_match]) #bnds[cnt])
                handle_out = m.register_forward_hook(act_hook)
                hook_handles_out.append(handle_out)
                hook_list.append(act_hook)
                print('hook set. type', type(m), m, 'layer', lays_min[cnt_match], qus_min[cnt_match])
                cnt_match += 1
            cnt += 1

    return hook_handles_out, hook_list


def set_quantiles_hooks(net):
    """
    Sets hooks into the entire model. Hooks have individual bounds but will all write to the same list act_list.
    :param net: pytorch model
    :param bnds: list of bounds
    :param resil: type of resilience method (currently only "ranger")
    :param mitigation: flag whether or not Ranger mitigation should be applied, true by default.
    :param detector: If activated the number of activated ranger layer per batch is given as output.
    :return: act_list, list of saved information from Range_detector hooks.
    :return: hook_handles_out, list of handles to be cleared later
    :return: hook_list, list of hooks to be cleared later
    """

    hook_handles_out = []
    hook_list = []
    cnt = 0
    for _, m in net.named_modules():
        if type(m) in [torch.nn.Conv2d]: # torch.nn.Linear
            act_hook = Range_detector_quantiles() #bnds[cnt])
            handle_out = m.register_forward_hook(act_hook)
            hook_handles_out.append(handle_out)
            hook_list.append(act_hook)
            cnt += 1
            # print('type', type(m), m)

    return hook_handles_out, hook_list


class Range_detector_quantiles:
    def __init__(self):
        self.quant = []
        self.qu_to_monitor = [0,10,20,30,40,50,60,70,80,90,100]
        self.quantile_pre_summation = True

    def __call__(self, module, module_in, module_out):
        
        # print('before qu', torch.cuda.max_memory_allocated())
        # if fsum applied first(?):
        if self.quantile_pre_summation and len(module_out.shape) > 2:
            dims_sum = np.arange(2,len(module_out.shape)).tolist()
            tnsr = torch.sum(module_out, dim=dims_sum)
        else:
            tnsr = module_out

        tnsr = torch.flatten(tnsr, start_dim=1, end_dim=- 1) #flatten all except for batch nr
        quants_all = torch.quantile(tnsr, torch.tensor(np.array(self.qu_to_monitor)/100., device=tnsr.device, dtype=tnsr.dtype), dim=1)
        # quants_all = tnsr #for alt time measurement

        # for x in range(len(quants_all)): #assign all q-variables (q10, q20 etc.)
        #     globals()[f"q{x}"] = quants_all[x].cpu().numpy()
        # q0, q10, q20, q25, q30, q40, q50, q60, q70, q75, q80, q90, q100
        # lst = np.vstack([q0.cpu().numpy(), q10.cpu().numpy(), q20.cpu().numpy(), q25.cpu().numpy(), q30.cpu().numpy(), q40.cpu().numpy(), q50.cpu().numpy(), q60.cpu().numpy(), q70.cpu().numpy(), q75.cpu().numpy(), q80.cpu().numpy(), q90.cpu().numpy(), q100.cpu().numpy()])
        # lst = np.vstack(quants_all).T.tolist()
        # self.quant.extend(lst.T.tolist())

        self.quant.extend(np.vstack(quants_all.cpu().detach().numpy()).T.tolist())
        
        # print('after qu', torch.cuda.max_memory_allocated())
        return module_out

    def clear(self):
        self.quant = []


class Range_detector_quantiles_customized:
    def __init__(self, qus_min_ind):
        self.quant = []
        self.qu_to_monitor = qus_min_ind #[0,10,20,30,40,50,60,70,80,90,100]
        self.quantile_pre_summation = True

    def __call__(self, module, module_in, module_out):
        
        # print('before qu', torch.cuda.max_memory_allocated())
        # if fsum applied first(?):
        if self.quantile_pre_summation and len(module_out.shape) > 2:
            dims_sum = np.arange(2,len(module_out.shape)).tolist()
            tnsr = torch.sum(module_out, dim=dims_sum)
        else:
            tnsr = module_out

        tnsr = torch.flatten(tnsr, start_dim=1, end_dim=- 1) #flatten all except for batch nr
        quants_all = torch.quantile(tnsr, torch.tensor(np.array(self.qu_to_monitor)/100., device=tnsr.device, dtype=tnsr.dtype), dim=1)
        # quants_all = tnsr #for alt time measurement

        self.quant.extend(np.vstack(quants_all.cpu().detach().numpy()).T.tolist())
        
        # print('after qu', torch.cuda.max_memory_allocated())
        return module_out

    def clear(self):
        self.quant = []


def clean_quantiles_hooks(hook_handles, hook_list):
    quant_list = []
    try:
        if hook_list == []:
            quant_list = []
        else:
            quant_list = [n.quant for n in hook_list]

        for i in range(len(hook_handles)):
            hook_handles[i].remove()
            hook_list[i].clear()

    except ValueError:
        print("Oops!  That was no valid bounds. Check the bounds and Try again...")

    return quant_list



def set_fmaps_hooks(net):
    """
    Sets hooks into the entire model. Hooks have individual bounds but will all write to the same list act_list.
    :param net: pytorch model
    :param bnds: list of bounds
    :param resil: type of resilience method (currently only "ranger")
    :param mitigation: flag whether or not Ranger mitigation should be applied, true by default.
    :param detector: If activated the number of activated ranger layer per batch is given as output.
    :return: act_list, list of saved information from Range_detector hooks.
    :return: hook_handles_out, list of handles to be cleared later
    :return: hook_list, list of hooks to be cleared later
    """

    hook_handles_out = []
    hook_list = []
    cnt = 0
    for _, m in net.named_modules():
        if type(m) in [torch.nn.Conv2d]: # torch.nn.Linear
            act_hook = Range_detector_fmaps() #bnds[cnt])
            handle_out = m.register_forward_hook(act_hook)
            hook_handles_out.append(handle_out)
            hook_list.append(act_hook)
            cnt += 1
            # print('type', type(m), m)

    return hook_handles_out, hook_list


class Range_detector_fmaps:
    def __init__(self):
        self.quant = []
        # self.qu_to_monitor = qus_min_ind #[0,10,20,30,40,50,60,70,80,90,100]
        self.quantile_pre_summation = True

    def __call__(self, module, module_in, module_out):
        
        # print('before qu', torch.cuda.max_memory_allocated())
        # if fsum applied first(?):
        if self.quantile_pre_summation and len(module_out.shape) > 2:
            dims_sum = np.arange(2,len(module_out.shape)).tolist()
            tnsr = torch.sum(module_out, dim=dims_sum)
        else:
            tnsr = module_out

        tnsr = torch.flatten(tnsr, start_dim=1, end_dim=- 1) #flatten all except for batch nr
        # quants_all = torch.quantile(tnsr, torch.tensor(np.array(self.qu_to_monitor)/100., device=tnsr.device, dtype=tnsr.dtype), dim=1)
        # quants_all = tnsr #for alt time measurement
        quants_all = tnsr

        self.quant.extend(np.vstack(quants_all.cpu().detach().numpy()).T.tolist())
        
        # print('after qu', torch.cuda.max_memory_allocated())
        return module_out

    def clear(self):
        self.quant = []


def get_mean_errs(cpu_total_all, gpu_total_all):
    cpu_total_avg = np.mean(cpu_total_all)
    cpu_total_min, cpu_total_max = np.min(cpu_total_all), np.max(cpu_total_all)
    cpu_total_err = np.std(cpu_total_all)*1.96/np.sqrt(len(cpu_total_all))
    gpu_total_avg = np.mean(gpu_total_all)
    gpu_total_min, gpu_total_max = np.min(gpu_total_all), np.max(gpu_total_all)
    gpu_total_err = np.std(gpu_total_all)*1.96/np.sqrt(len(gpu_total_all))
    # return cpu_total_avg, cpu_total_min, cpu_total_max, cpu_total_err, gpu_total_avg, gpu_total_min, gpu_total_max, gpu_total_err
    cpu_str = str(cpu_total_avg) + ", " + str(cpu_total_min)+ ", " + str(cpu_total_max) + ", " + str(cpu_total_err)
    gpu_str = str(gpu_total_avg) + ", " + str(gpu_total_min)+ ", " + str(gpu_total_max) +  ", " + str(gpu_total_err)
    return cpu_str, gpu_str


def profile_time(model, hooks_type, N_attempts, device, dummy_input, lays_min=None, qus_min=None):
    cpu_total_all = []
    gpu_total_all = []
    for j in range(N_attempts):
        print('run', j)
        # # # a = torch.cuda.max_memory_allocated(device=device) #in bytes
        # # # hook_list, hook_handles_list, setting_list = set_hooks_info(model)
        # # # hook_handles_quant, hook_list_quant = set_quantiles_hooks(model)
        # # # hook_handles_quant, hook_list_quant = set_features_all_hooks(model)
        if hooks_type == "quantiles":
            hook_handles_quant, hook_list_quant = set_quantiles_hooks_customized(model, lays_min, qus_min) #our approach
        elif hooks_type == "quantiles_all":
            hook_handles_quant, hook_list_quant = set_quantiles_hooks(model) #our approach full
        elif hooks_type == 'fmaps':
            hook_handles_quant, hook_list_quant = set_fmaps_hooks(model) #Schorn
        # else:
        #     print('no hooks set.')

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            with record_function("model_inference"):
                for n in range(dummy_input.shape[0]):
                    # dummy_input = torch.rand(N_bs,3,100,100).to(device)
                    model(dummy_input[n])
        
        if hooks_type is not None: #== "quantiles" or hooks_type == "quantiles_all" orhooks_type == "fmaps":
            _ = clean_quantiles_hooks(hook_handles_quant, hook_list_quant)

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1))
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=1))


        cpu_total_us = prof.profiler.self_cpu_time_total #in us
        gpu_total = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=1).split()[-1]
        if gpu_total[-2:] == 'ms':
            gpu_total_us = int(float(gpu_total[:-2])*1e3)
        elif gpu_total[-2:] != 'ms' and gpu_total[-1:] == 's':
            gpu_total_us = int(float(gpu_total[:-1])*1e6)

        print(cpu_total_us, gpu_total_us)
        cpu_total_all.append(cpu_total_us)
        gpu_total_all.append(gpu_total_us)
        # print('done')

    return cpu_total_all[1:], gpu_total_all[1:] #remove first one since its always slower

def main(argv):
    """
    Workflow: 
    1. Run this file for automated experiments, additionally for range extraction. Results are in nwstore or local.
    2. Run quantiles_extract_features_plot3.py for quantile extraction and save it to nwstore.
    3. Run train_detection_model_LR3.py or ..._DT2.py or ..._fcc1.py to train anomaly detector.
    """

    # models = ['yolo', 'yolo', 'ssd', 'ssd', 'retina', 'retina', 'resnet', 'alexnet']
    # datasets = ['coco', 'kitti', 'coco', 'kitti', 'coco', 'kitti', 'imagenet', 'imagenet']
    models = ['retina', 'retina', 'resnet', 'alexnet']
    datasets = ['coco', 'kitti', 'imagenet', 'imagenet']

    for x in range(len(models)):
        # Define dataset and model specifications
        ####################################################################
        dataset_name = datasets[x]
        model_name = models[x]
        # dataset_name = 'coco' #'coco', kitti, imagenet
        # model_name = 'retina' #'yolo', 'retina_net', 'ssd', resnet, alexnet

        batch_size = 1 #batchsize for neurons
        num_faults = 1 #faults
        num_runs = 1 #number of runs #500
        sample_size = 10 #nr of images (sequential)
        dataset_type = 'val' #'train', 'val'

        save_to_nw = False #Save directly to nw store?


        N_imgs = 100
        N_bs = 10
        N_attempts = 100

        # TODO: Issues with retina_net:
        # - run with smaller bs (10), but >1 in general ok
        # - ftrace_monitoring not done yet
        # - only layer empty for monitoring, will be skipped
        # - use quantiles without presum fails (too large layers?)
        ####################################################################

        # Set device ---------------------------------------------------------
        cuda_device = 0
        device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu") 
        # device = 'cpu'
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
        if "alexnet" in model_name.lower() or "resnet" in model_name.lower():
            model = set_up_img_model(model_name, dataset_name, device)
        else:
            model = set_up_mmdet_model(model_name, dataset_name, device)

        lays_all_sorted, qus_all_sorted = get_minimal_fts(model_name, dataset_name)
        lays_min, qus_min = lays_all_sorted[0], qus_all_sorted[0]


        # Profiling: ---------------------------------------------------------
        # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        dummy_input = torch.rand(int(N_imgs/N_bs), N_bs,3,100,100).to(device)

        cpu_total_all, gpu_total_all = profile_time(model, None, N_attempts, device, dummy_input)
        cpu_str_nohooks, gpu_str_nohooks = get_mean_errs(cpu_total_all, gpu_total_all)
        print('done: no hooks')

        cpu_total_all, gpu_total_all = profile_time(model, "quantiles", N_attempts, device, dummy_input, lays_min, qus_min)
        cpu_str_quantiles, gpu_str_quantiles = get_mean_errs(cpu_total_all, gpu_total_all)
        print('done: quantiles')

        cpu_total_all, gpu_total_all = profile_time(model, "quantiles_all", N_attempts, device, dummy_input)
        cpu_str_quantiles2, gpu_str_quantiles2 = get_mean_errs(cpu_total_all, gpu_total_all)
        print('done: quantiles2')

        cpu_total_all, gpu_total_all = profile_time(model, "fmaps", N_attempts, device, dummy_input)
        cpu_str_fmaps, gpu_str_fmaps = get_mean_errs(cpu_total_all, gpu_total_all)
        print('done: fmaps')

        

        # Save to file
        save_pth = '/home/fgeissle/fgeissle_ranger/object_detection/quantile_detection_data/'
        with open(save_pth + 'time_profiling_' + model_name + '_' + dataset_name + '.txt', 'w') as f:
            # f.write(str(cpu_total_us) + "\n" + str(gpu_total_us) + "\n" + str(N_imgs) + "\n" + str(N_bs) + "\n" + str(device))
            f.write(cpu_str_nohooks + "\n" + gpu_str_nohooks + "\n" + cpu_str_quantiles + "\n" + gpu_str_quantiles + "\n" \
                + cpu_str_quantiles2 + "\n" + gpu_str_quantiles2 + "\n" + cpu_str_fmaps + "\n" + gpu_str_fmaps + "\n" + str(N_imgs) + "\n" + str(N_bs) + "\n" + str(N_attempts) + "\n" + str(device))
        print('saved as', save_pth + 'time_profiling_' + model_name + '_' + dataset_name + '.txt')



if __name__ == "__main__":
    main(sys.argv)

