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

from alficore.dataloader.imagenet_loader import imagenet_Dataloader
from alficore.wrapper.ptfiwrap import ptfiwrap
from alficore.wrapper.test_error_models_imgclass import TestErrorModels_ImgClass
from util.helper_functions import getdict_ranger, get_savedBounds_minmax
from util.evaluate import extract_ranger_bounds
from util.ranger_automation import get_Ranger_protection, get_Ranger_protection_mixed, flatten_model
import pandas as pd

from resiliency_methods.Ranger import Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg, Ranger_trivial

# logging.config.fileConfig('fi.conf')
# log = logging.getLogger()
cuda_device = 0
model_name = 'resnet50'
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
    _, imagenet_dataloader, _ = imagenet_Dataloader(root='train', shuffle=False, batch_size=batch_size, transform=transform,
                                                    sampling=True, sampleN=sample_Percentage)
    net_for_bounds = flatten_model(resnet50) #to make Relus explicit
    act_input, act_output = extract_ranger_bounds(imagenet_dataloader, net_for_bounds, ranger_file_name, dataset_name) # gets also saved automatically
    print('check Ranger input', act_input)
    print('check Ranger output', act_output)
    sys.exit()


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

    # Model and dataset: ---------------------------------------------------------
    dataset_name = 'imagenet'
    
    resnet50 = models.resnet50(pretrained=True, progress=True)
    resnet50 = resnet50.to(device)

    # Ranger bounds: ---------------------------------------------------------
    gen_ranger_bounds = False
    ranger_file_name = 'Resnet50_bounds_ImageNet_train20p_act'
    if gen_ranger_bounds:
        print('New bounds to be saved as:', ranger_file_name)
        get_Ranger_bounds(resnet50, 10, ranger_file_name, dataset_name, sample_Percentage=20)
    else:
        print('Bounds loaded:', ranger_file_name)


    # Inference runs: ---------------------------------------------------------
    apply_ranger = False #TODO:
    # resil = [Ranger, Ranger_Clip, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg]
    resil = [None]
    
    # replace with pytorchfi wrapper code
    # batchsize is set in scenarios/default.yml -> ptf_batch_size
    num_faults = [2]
    # fault_files = ["/home/qutub/PhD/git_repos/intel_gitlab_repos/nan_debug/logs/fault_rates_resnet50_1_500.bin"] #TODO: create new
    # fault_files = ['/home/fgeissle/Desktop/ranger/logs/fault_rates_resnet50_210706_184831_VPU_1000_weights.bin']
    fault_files = [None]

    save_fault_file_dir = 'result_files/test/'
    yml_file = 'default_neurons.yml'
    inj_policy = None # choose fault injection policy None (as before), 'per_batch' or 'per_epoch'

    for id, _num_faults in enumerate(num_faults):
        for _resil in resil:
            if apply_ranger:
                bnds = get_savedBounds_minmax('./bounds/' + ranger_file_name + '.txt')
                # Change network architecture to add Ranger
                net_for_protection = resnet50
                protected_resnet50, _ = get_Ranger_protection(net_for_protection, bnds, resil=_resil)
                protected_resnet50 = protected_resnet50.to(device)

                resnet50_ErrorModel = TestErrorModels_ImgClass(model=resnet50, resil_model=protected_resnet50, model_name=model_name, device=device,
                                    dataset='imagenet', store_json=True, resil_method=_resil.__name__, config_location=yml_file)

            else:
                resnet50_ErrorModel = TestErrorModels_ImgClass(model=resnet50, resil_model=None, model_name=model_name, device=device,
                                    dataset='imagenet', store_json=True, config_location=yml_file)

            resnet50_ErrorModel.test_rand_ImgClass_SBFs_inj(num_faults=_num_faults, fault_file=fault_files[id], save_fault_file_dir=save_fault_file_dir, inj_policy=inj_policy)


if __name__ == "__main__":
    # ctx = mp.get_context("spawn")
    # ctx.set_start_method('spawn')onc
    main(sys.argv)
