# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import pickle
import logging
import os
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from contextlib import ExitStack
import yaml
import pandas as pd
from datetime import datetime
import time
from os.path import dirname as up
import json
import torchvision
from copy import deepcopy
from alficore.ptfiwrap_utils.hook_functions import set_ranger_hooks_v3, set_simscore_hooks, set_nan_inf_hooks, run_nan_inf_hooks, run_simscore_hooks, set_quantiles_hooks, set_feature_trace_hooks
from alficore.ptfiwrap_utils.utils import read_yaml
from alficore.wrapper.ptfiwrap import ptfiwrap
# from alficore.resiliency_methods.ranger import Ranger, Ranger_trivial, Ranger_BackFlip, Ranger_Clip, Ranger_FmapAvg, Ranger_FmapRescale

resil_methods = {}#"ranger": Ranger, "ranger_trivial": Ranger_trivial, "ranger_backFlip": Ranger_BackFlip, "ranger_clip": Ranger_Clip, "ranger_FmapAvg": Ranger_FmapAvg, "ranger_FmapRescale": Ranger_FmapRescale}


class TestErrorModels_ImgClass:
    def __init__(self, model=None, resil_model=None, **kwargs):
        self.resil_name  = kwargs.get("resil_name", None)
        self.logger      = logging.getLogger(self.__class__.__name__)
        self.ORIG_MODEL  = model
        self.ORIG_MODEL.to(self.device)
        if resil_model is None and self.resil_name is not None: #alternative: assign resil model by resil_name
            if self.resil_name and not(self.resil_name.lower() in [name.lower() for name in list(resil_methods.keys())]):
                raise NotImplementedError("Resil method {} is not implemented or not imported!".format(self.resil_name))
            self.RESIL_MODEL = self.ORIG_MODEL
            self.RESIL_MODEL.to(self.device)
        elif resil_model is None and self.resil_name is None:
            self.RESIL_MODEL = None
        elif resil_model is not None:
            self.RESIL_MODEL = resil_model
        self.__update_attributes(**kwargs)

    def __update_attributes(self, **kwargs):
        """
        load the kwargs into class attributes
        TODO: cfg file should be used if the list is getting longer
        """
        now = datetime.now()
        self.exp_day_info = now.strftime("%d/%m/%Y %H:%M:%S")
        self.time_based_uuid = time.strftime("%Y%m%d-%H%M%S")
        self.orig_model_run  = kwargs.get("orig_model_run", False)
        self.resil_model_run = kwargs.get("resil_model_run", False)
        self.evaluator_type  = kwargs.get("evaluator_type", 'coco')
        self.dl_attr         = kwargs.get("dl_attr")
        self.ranger_bounds   = kwargs.get("ranger_bounds", [])
        self.disable_FI = kwargs.get("disable_FI", False)
        self.resil_name  = kwargs.get("resil_name", None)
        # self.resil_name  = self.resil_name.lower() if self.resil_name else pass
        self.copy_yml_scenario = kwargs.get("copy_yml_scenario", False)
        self.resume_dir      = kwargs.get("resume_dir", None) 
        self.num_faults = kwargs.get("num_faults", 1)
        self.fault_file = kwargs.get("fault_file", None)
        self.num_runs   = kwargs.get("num_runs", 1)
        self.resume_inj = kwargs.get("resume_inj", False)
        self.resume_epoch = 0

        if self.ORIG_MODEL is not None:
            self.ORIG_MODEL.eval()
            self.orig_model_run = True
            self.orig_model_FI_run = self.orig_model_run & (not self.disable_FI)
        if self.RESIL_MODEL is not None:
            self.RESIL_MODEL.eval()
            self.resil_model_run = True
            self.resil_model_FI_run = self.resil_model_run & (not self.disable_FI)

        """
        Features to monitor
        """
        self.ranger_detector    = kwargs.get("ranger_detector", True)
        self.inf_nan_monitoring = kwargs.get("inf_nan_monitoring", True)
        self.sim_score          = kwargs.get("sim_score", False) 
        self.activation_trace   = kwargs.get("activation_trace", False)
        self.quant_extr         = kwargs.get("quant_monitoring", False)
        self.ftrace_extr = kwargs.get("ftrace_monitoring", False)
        if self.quant_extr:
            self.quant_list_corr = []
            self.quant_list_resil = []
            self.quant_list_resil_corr = []
        if self.ftrace_extr:
            self.ftrace_list_corr = []
            self.ftrace_list_resil = []
            self.ftrace_list_resil_corr = []
        if self.ranger_detector:
            assert self.ranger_bounds is not None, "Ranger detector is True, but bounds are not set to be used by the detector"
            ## recording ranger activations
            self.corr_ranger_actvns = [] if self.orig_model_FI_run else None
            self.resil_ranger_actvns = [] if self.resil_model_run else None 
            self.resil_corr_ranger_actvns = [] if self.resil_model_FI_run else None 
        if self.sim_score is not None:
            self.orig_penulLayer = None
            self.corr_penulLayer = None
            self.resil_penulLayer = None
            self.resil_corr_penulLayer = None
        if self.activation_trace is not None:
            self.orig_activation_trace = None
            self.corr_activation_trace = None
            self.resil_activation_trace = None
            self.resil_corr_activation_trace = None
        if self.inf_nan_monitoring:
            self.inf_nan_monitoring_init()


        ## end Features to monitor
        self.model_eval_method = kwargs.get("eval_method", ['coco',])
        self.config_location   = kwargs.get("config_location", None)
        self.device            = kwargs.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._outputdir       = kwargs.get("result_files", None)
        self.num_faults        = kwargs.get("num_faults", None)
        self.num_runs          = kwargs.get("num_runs", None)
        self.model_name        = kwargs.get("model_name", None)

        # self.exp_type = kwargs.get("exp_type", 'hwfault') #hwfault, Gaussian_blur, Gaussian_noise, 
        # self.corr_img = not (self.exp_type == 'hwfault')
        # self.mod_img = None

        self.copied_scenario_data = {}

        self.dl_attr.dl_dataset_name = self.dl_attr.dl_dataset_name.lower()
        if self.orig_model_FI_run :
            self.model_wrapper = ptfiwrap(model=self.ORIG_MODEL, device=self.device, config_location=self.config_location, scenario_data=self.copied_scenario_data, create_runset=False)
        if self.resil_model_FI_run:
            self.resil_wrapper = ptfiwrap(model=self.RESIL_MODEL, device=self.device, config_location=self.config_location, scenario_data=self.copied_scenario_data, create_runset=False)
        self.reference_wrapper       = self.model_wrapper if self.orig_model_FI_run else \
                                            self.resil_wrapper if self.resil_model_FI_run else None
        self.model_scenario          = self.reference_wrapper.get_scenario() if not self.disable_FI else None
        self.reference_parser        = self.reference_wrapper.parser if not self.disable_FI else None
        self.dl_attr.dl_batch_size   = self.reference_parser.ptf_batch_size if not self.dl_attr.dl_batch_size else self.dl_attr.dl_batch_size

        self.scenario_is_modified = False
        if self._outputdir is None :
            if self.disable_FI or not self.reference_parser:
                self._outputdir = os.path.join(os.getcwd(), 'result_files', 'output_{}'.format(self.dl_attr.dl_dataset_name))
            else:
                self._outputdir = self.reference_parser.save_fault_file_dir
        else:
            self._outputdir = os.path.join(os.getcwd(), self._outputdir, 'output_{}'.format(self.dl_attr.dl_dataset_name))
        self.outputdir = self._outputdir

        if self.disable_FI:
            self.num_runs = 1

        if self.copy_yml_scenario and not self.disable_FI:
            try:
                self.model_scenario_file = list(Path(os.path.dirname(self.fault_file)).glob('**/*.yml'))[0]
                self.copied_scenario_data = read_yaml(self.model_scenario_file)[0]
                self.copy_yml_attr(self.copied_scenario_data)
                if (not (self.model_scenario["inj_policy"] == "per_batch") and self.model_scenario["rnd_mode"] == "weights"):
                    self.model_scenario["ptf_batch_size"] = self.dl_attr.dl_batch_size
                else:
                    self.dl_attr.dl_batch_size = self.model_scenario["ptf_batch_size"]
            except ValueError:
                print("Oops! The given fault file is None and you have asked the FI tool to copy the scenario file which should include valid fault file. \
                            You can disable the 'copy_yml_scenario' and let the FI tool generate a new random fault file")
        elif not self.disable_FI:
            self.model_scenario["ptf_batch_size"] = self.dl_attr.dl_batch_size

    def __getattr__(self, item):
            return None

    def inf_nan_monitoring_init(self, model=None):
        if model == "orig" or model is None:
            if self.orig_model_FI_run:
                self.nan_flag_image_corr_model = []
                self.nan_inf_flag_image_corr_model = []
                self.nan_inf_overall_layers_image_corr_model = []
                self.nan_inf_first_occurrence_image_corr_model = []
        if model == "resil" or model is None:
            if self.resil_model_run:
                self.nan_flag_image_resil_model = []
                self.nan_inf_flag_image_resil_model = []
                self.nan_inf_overall_layers_image_resil_model = []
                self.nan_inf_first_occurrence_image_resil_model = []
                if self.resil_model_FI_run:
                    self.nan_flag_image_resil_corr_model = []
                    self.nan_inf_flag_image_resil_corr_model = []
                    self.nan_inf_overall_layers_image_resil_corr_model = []
                    self.nan_inf_first_occurrence_image_resil_corr_model = []

    def __check_resume_inj_status(self, model_type, epoch=None):
        # if not epoch:
        #     epoch = self.curr_epoch
        if self.resume_inj and self.fault_file is not None:
            output_path = os.path.join(self.outputdir, self.dataset_name, model_type, 'epochs', str(epoch))
            if os.path.exists(output_path):
                self.logger.info("resume check {}: Skipping evalution for epoch {}".format(model_type, epoch))
                return True
            else:
                return False

    def __ptfi_dataloader(self):
        if self.dl_attr.dl_dataset_name == 'imagenet':
            from alficore.dataloader.imagenet_loader import Imagenet_dataloader
            self.dataloader = Imagenet_dataloader(dl_attr=self.dl_attr)
        elif self.dl_attr.dl_dataset_name == 'coco':
            from alficore.dataloader.coco_loader import CoCo_dataloader
            ##TODO: adapt dataloaders to receive dl_attr as function arguments
            self.dataloader = CoCo_dataloader(dataset_type='val', batch_size=self.batch_size, device=self.device, sampleN=self.dl_sampleN, shuffle=self.dl_shuffle)
        elif self.dl_attr.dl_dataset_name == 'mnist':
            from alficore.dataloader.mnist_loader import MNIST_dataloader
            self.dataloader = MNIST_dataloader(dl_attr=self.dl_attr)

    def get_nan_inf_columns(self):
        output_columns = []
        output_values  = []

        nan_inf_corr_model_columns = ['nan_flag_corr_model', 'nan_or_inf_flag_corr_model', 'Nan_or_inf_layers_corr_model', 'Nan_inf_first_occurrence_corr_model']
        nan_inf_resil_model_columns = ['nan_flag_resil_model', 'nan_or_inf_flag_resil_model', 'Nan_or_inf_layers_resil_model', 'Nan_inf_first_occurrence_resil_model']
        nan_inf_resil_corr_model_columns = ['nan_flag_resil_corr_model', 'nan_or_inf_flag_resil_corr_model', 'Nan_or_inf_layers_resil_corr_model', 'Nan_inf_first_occurrence_resil_corr_model']
        if self.resil_model_FI_run:
            if self.orig_model_FI_run:
                if self.inf_nan_monitoring:
                    output_columns.extend(nan_inf_corr_model_columns)
                    output_columns.extend(nan_inf_resil_model_columns)
                    output_columns.extend(nan_inf_resil_corr_model_columns)
                    output_values = [self.nan_flag_image_corr_model, self.nan_inf_flag_image_corr_model, self.nan_inf_overall_layers_image_corr_model, self.nan_inf_first_occurrence_image_corr_model,\
                                        self.nan_flag_image_resil_model, self.nan_inf_flag_image_resil_model, self.nan_inf_overall_layers_image_resil_model, self.nan_inf_first_occurrence_image_resil_model, \
                                            self.nan_flag_image_resil_corr_model, self.nan_inf_flag_image_resil_corr_model, self.nan_inf_overall_layers_image_resil_corr_model, self.nan_inf_first_occurrence_image_resil_corr_model]
            else:
                if self.inf_nan_monitoring:
                    output_columns.extend(nan_inf_resil_model_columns)
                    output_columns.extend(nan_inf_resil_corr_model_columns)
                    output_values = [self.nan_flag_image_resil_model, self.nan_inf_flag_image_resil_model, self.nan_inf_overall_layers_image_resil_model, self.nan_inf_first_occurrence_image_resil_model, \
                                        self.nan_flag_image_resil_corr_model, self.nan_inf_flag_image_resil_corr_model, self.nan_inf_overall_layers_image_resil_corr_model, self.nan_inf_first_occurrence_image_resil_corr_model]
        elif self.orig_model_FI_run:
            if self.inf_nan_monitoring:
                output_columns.extend(nan_inf_corr_model_columns)
                output_values = [self.nan_flag_image_corr_model, self.nan_inf_flag_image_corr_model, self.nan_inf_overall_layers_image_corr_model, self.nan_inf_first_occurrence_image_corr_model]
        return output_columns, output_values

    def __per_epoch_sanity_checks(self, num_runs):
        ## TODO: Check it is working for 1 fault, but not working for multiple faults
        if not self.disable_FI:
            num_runs = num_runs + 1 - (self.resume_epoch if self.resume_inj else 0)  ## coz _epoch is 0 indexed based
            print("current epoch: {}".format(num_runs))
            if self.reference_parser.inj_policy == "per_image":
                if self.reference_parser.rnd_mode == 'neurons':
                    runset_length = num_runs*self.dataloader.dataset_length*self.num_faults
                elif self.reference_parser.rnd_mode == 'weights':
                    runset_length = num_runs*self.num_faults
            elif self.reference_parser.inj_policy == "per_epoch":
                if self.reference_parser.rnd_mode == 'neurons':
                    runset_length = num_runs*self.num_faults*self.dataloader.dataset_length
                elif self.reference_parser.rnd_mode == 'weights':
                    runset_length = num_runs*self.num_faults
            elif self.reference_parser.inj_policy == 'per_batch':
                runset_length = int(np.ceil(num_runs*self.dataloader.dataset_length*self.num_faults/self.dl_attr.dl_batch_size))

            used_models = []
            if self.orig_model_FI_run:
                used_models.append('corr')
            if self.resil_model_FI_run:
                used_models.append('{} corr'.format(self.resil_name))
            for used_model in used_models:
                if used_model == 'corr':
                    runset = self.model_wrapper.runset_updated
                    bit_flip_monitor = self.model_wrapper.pytorchfi.bit_flips_monitor[:runset_length]
                    bit_flips_direc = self.model_wrapper.pytorchfi.bit_flips_direc[:runset_length]
                    if self.ranger_detector and (self.resil_name or self.RESIL_MODEL):
                        ranger_actvns = self.corr_ranger_actvns
                elif used_model == '{} corr'.format(self.resil_name):
                    runset = self.resil_wrapper.runset_updated
                    bit_flip_monitor = self.resil_wrapper.pytorchfi.bit_flips_monitor[:runset_length]
                    bit_flips_direc = self.resil_wrapper.pytorchfi.bit_flips_direc[:runset_length]
                    if self.ranger_detector and (self.resil_name or self.RESIL_MODEL):
                        ranger_actvns = self.resil_corr_ranger_actvns
                    if self.ranger_detector and (self.resil_name or self.RESIL_MODEL):
                        num_inferences = num_runs*self.dataloader.dataset_length
                        assert len(ranger_actvns) == num_inferences, "Epoch- {}: {} Sanity check: Ranger activations captured in {} and total number of number - \
                            {} dont match".format(self.curr_epoch, used_model, len(ranger_actvns), num_inferences)
                # if self.corr_img:
                #     assert (bit_flip_monitor == None).all() and (bit_flips_direc == None).all(), "Mode is image corruption but bit flips have been found too!"
                else:
                    if self.reference_parser.inj_policy == "per_image" or self.reference_parser.inj_policy == "per_batch":
                        if self.reference_parser.rnd_mode == 'neurons':
                            assert (bit_flip_monitor == self.reference_wrapper.runset_updated[6,:][:runset_length]).all(), "Epoch- {}: {} Sanity check: bit flips monitored is not matching with the actual runset".format(num_runs, used_model)
                        elif self.reference_parser.rnd_mode == 'weights':
                            assert (bit_flip_monitor == self.reference_wrapper.runset[6,:][:runset_length]).all(), "Epoch- {}: {} Sanity check: bit flips monitored is not matching with the actual runset".format(num_runs, used_model)
                    elif self.reference_parser.inj_policy=='per_epoch':
                        if self.reference_parser.rnd_mode == 'neurons':
                            assert (bit_flip_monitor == self.reference_wrapper.runset_updated[6,:][:runset_length]).all(), "Epoch- {}: {} Sanity check: bit flips monitored is not matching with the actual runset".format(num_runs, used_model)
                            assert (bit_flip_monitor[0:len(bit_flip_monitor):self.dataloader.dataset_length][:num_runs] == self.reference_wrapper.runset[6,:][:num_runs]).all(), "Epoch- {}: {} Sanity check: bit flips monitored is not matching with the actual runset".format(num_runs, used_model)
                        elif self.reference_parser.rnd_mode == 'weights':
                            # assert (bit_flip_monitor == self.reference_wrapper.runset_updated[6,:][0:self.reference_wrapper.runset_updated.shape[1]:self.dataloader.dataset_length][:runset_length]).all(), "Epoch- {}: {} Sanity check: bit flips monitored is not matching with the actual runset".format(num_runs, used_model)
                            assert (bit_flip_monitor == self.reference_wrapper.runset[6,:][:num_runs]).all(), "Epoch- {}: {} Sanity check: bit flips monitored is not matching with the actual runset".format(num_runs, used_model)

                assert len(bit_flip_monitor[bit_flip_monitor==None]) == 0, "Epoch- {}: {} Sanity check: bit flips monitored - {} and total number of bit flips captured - \
                    {} dont match".format(num_runs, used_model, len(bit_flip_monitor), len(bit_flip_monitor)-len(bit_flip_monitor[bit_flip_monitor==None]))
                assert len(bit_flips_direc[bit_flips_direc==None]) == 0, "Epoch- {}: {} Sanity check: bit flips monitored - {} and total number of bit flips captured - \
                    {} dont match".format(num_runs, used_model, len(bit_flips_direc), len(bit_flips_direc)-len(bit_flips_direc[bit_flips_direc==None]))

                assert len(bit_flip_monitor) == len(bit_flips_direc), "Epoch- {}: {} Sanity check: bit flips monitored - {} and total number of bit flips captured - \
                    {} dont match".format(num_runs, used_model, len(bit_flip_monitor), len(bit_flips_direc))
                assert runset_length == len(bit_flips_direc), "Epoch- {}: {} Sanity check: runset len after tiling (if inj policy is per epoch or per batch) - \
                    {} and total number of bit flips captured - {} dont match".format(num_runs, used_model, runset_length, len(bit_flips_direc))

    
    def __clean_ranger_hooks(self, hook_handles, hook_list):
        activated = [] #protection layers activated in one image batch!
        try:
            
            if hook_list == []:
                activated = None
            else:
                acts = [n.act for n in hook_list]
                lens = np.unique([len(x) for x in acts])
                if len(lens) > 1 or (len(lens) == 1 and lens[0] != self.dataloader.curr_batch_size): #multi-use
                    activated = [np.sum([np.sum(n) for n in acts])]
                else:
                    activated = np.sum(np.array(acts),0) #sum over layers

            for i in range(len(hook_handles)):
                hook_handles[i].remove()
                hook_list[i].clear()

        except ValueError:
            print("Oops!  That was no valid bounds. Check the bounds and Try again...")
        return activated


        
    def __clean_quantiles_hooks(self, hook_handles, hook_list):
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


    def __save_model_scenario(self):
        if not self.copy_yml_scenario and not self.disable_FI:
            model_scenario_yml = os.path.join(self.outputdir, self.dataset_name, self.func + '_' +
                    str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_' + self.dl_attr.dl_dataset_name + '_model_scenario.yml')
            data = self.model_scenario
            data['resil_name']        = self.resil_name
            data['sensor_channels']   = self.sensor_channels
            data['dataset']           = self.dataset_name
            data['ranger_detector']   = self.ranger_detector
            data['model_name']        = self.model_name
            data['dl_sampleN']        = self.dl_attr.dl_sampleN
            data['date_time']         = self.exp_day_info
            data['uuid']              = str(self.time_based_uuid)
            data['dl_shuffle']        = self.dl_attr.dl_shuffle
            
            with open(model_scenario_yml, 'w') as outfile:
                yaml.dump(data, outfile, default_flow_style=False)

    def attach_hooks(self, model, resil=None, **kwargs):
        # corr_img = kwargs.get('corr_img', False) #this way to distinguish orig and corr models
        nan_dict_corr, inf_dict_corr, detected_activations, penulLayer, quant_list, ftrace_list = None, None, None, None, None, None
        if self.inf_nan_monitoring:
            save_nan_inf, hook_handles_nan_inf, hook_layer_names = set_nan_inf_hooks(model)
        if self.ranger_detector and resil is not None:
            hook_handles_acts, hook_list = set_ranger_hooks_v3(model, self.ranger_bounds, resil=resil, detector=self.ranger_detector)
        if self.sim_score:
            save_penult_layer, hook_handles_penult_layer, _ = set_simscore_hooks(model, self.model_name)
        if self.quant_extr:
            hook_handles_quant, hook_list_quant = set_quantiles_hooks(model)
        if self.ftrace_extr:
            hook_handles_ftrace, hook_list_ftrace = set_feature_trace_hooks(model)
        # if corr_img:
        #     assert self.exp_type in ["hwfault", 'Gaussian_blur', 'Gaussian_noise'], 'exp_type is unknown, should be one of ["hwfault", "Gaussian_blur", "Gaussian_noise"].'

        #     orig_img = deepcopy(self.dataloader.images) #save uncorrupted image from dataloader for later
            
        #     if self.mod_img is None: #if self.mod_image is None create mod image, otherwise use self.mod_image
        #         # Blur
        #         if self.exp_type == "Gaussian_blur":
        #             self.mod_img = torchvision.transforms.GaussianBlur((21,21), sigma=(3,3))(orig_img)

        #         # Noise
        #         elif self.exp_type == "Gaussian_noise":
        #             noise = (10**0.5)*torch.randn(orig_img.shape) #10 leads to 40% SDC, 1 to 17% SDC
        #             self.mod_img = (orig_img + noise).to(torch.uint8)
        #             # self.mod_img = orig_img #TODO TODO: for blank run (activation check)
        

        #     self.dataloader.data[0]['image'] = self.mod_img
        #     output = model(self.dataloader.data)
        #     self.dataloader.data[0]['image'] = orig_img #put back original image for next use
        # else:
        output = model(self.dataloader.images)
        
        if self.inf_nan_monitoring:
            nan_dict_corr, inf_dict_corr = run_nan_inf_hooks(save_nan_inf, hook_handles_nan_inf, hook_layer_names)
        if self.ranger_detector and resil is not None:
            detected_activations = self.__clean_ranger_hooks(hook_handles_acts, hook_list)
        if self.sim_score:
            penulLayer = run_simscore_hooks(save_penult_layer, hook_handles_penult_layer)
        if self.quant_extr:
            quant_list = self.__clean_quantiles_hooks(hook_handles_quant, hook_list_quant)
            if quant_list != []:
                quant_list = np.swapaxes(np.array(quant_list),0,1) #rearrange dimensions as image, layers, quantiles
        if self.ftrace_extr:
            ftrace_list = self.__clean_quantiles_hooks(hook_handles_ftrace, hook_list_ftrace)
            if ftrace_list != []:
                ftrace_list = np.swapaxes(np.array(ftrace_list),0,1) #rearrange dimensions as image, layers, quantiles

        return output, nan_dict_corr, inf_dict_corr, detected_activations, penulLayer,  quant_list, ftrace_list



    def _processing(self, orig_output):
        __TOP_RES = 5

        _orig_output = orig_output
        _orig_output = torch.unsqueeze(_orig_output, 0)
        percentage = torch.nn.functional.softmax(_orig_output, dim=1)[0] * 100 ## entropy should receive normalised probabiiities
        _, orig_index = torch.sort(_orig_output, descending=True)
        if self.compute_entropy:
            orig_model_entropy = self.entropy(percentage)
        else:
            orig_model_entropy = None

        if self.device.type == 'cuda':
            orig_perct = np.round(percentage[orig_index[0][:__TOP_RES]].cpu().detach().numpy(), decimals=2)
            orig_index = orig_index[0][:__TOP_RES].cpu().detach().numpy()
        else:
            orig_perct = np.round(percentage[orig_index[0][:__TOP_RES]].detach().numpy(), decimals=2)
            orig_index = orig_index[0][:__TOP_RES].detach().numpy()

        return orig_model_entropy, orig_perct, orig_index

    def __run_inference_orig_model(self):
        self.ORIG_MODEL.eval()
        start = time.time()
        orig_output = self.ORIG_MODEL(self.dataloader.images)
        self.time_orig = time.time() - start
        # len(orig_output)
        return orig_output

    def __run_inference_corr_model(self):
        start = time.time()
        self.CORR_MODEL.eval()
        corr_output, nan_dict_corr, inf_dict_corr, detected_activations, penulLayer, quant_list, ftrace_list = self.attach_hooks(self.CORR_MODEL, resil='ranger_trivial')
        if self.inf_nan_monitoring:
            self.nan_flag_image_corr_model = nan_dict_corr['flag'] #Flag true or false per image depending if nan found at any layer
            self.nan_inf_flag_image_corr_model = [nan_dict_corr['flag'][h] or inf_dict_corr['flag'][h] for h in range(len(inf_dict_corr['flag']))]
            self.nan_inf_overall_layers_image_corr_model = [np.unique(nan_dict_corr['overall'][i] + inf_dict_corr['overall'][i]).tolist() for i in range(len(nan_dict_corr['overall']))]
            self.nan_inf_first_occurrence_image_corr_model = [i for i in nan_dict_corr['first_occur_compare']]
        if self.ranger_detector:
            self.corr_ranger_actvns.extend(detected_activations)
        if self.quant_extr:
            self.quant_list_corr.extend(quant_list)
        if self.ftrace_extr:
            self.ftrace_list_corr.extend(ftrace_list)
        if self.sim_score:
            corr_penulLayer =  torch.cat([x.unsqueeze(0) for x in penulLayer])
            if self.corr_penulLayer is not None:
                self.corr_penulLayer = torch.hstack([self.corr_penulLayer, corr_penulLayer])
            else:
                self.corr_penulLayer = torch.hstack([corr_penulLayer])
        if self.activation_trace:
            trace_output = trace_output if trace_output else []
            if self.corr_activation_trace is not None:
                self.corr_activation_trace = torch.cat((self.corr_activation_trace, trace_output), 0)
            else:
                self.corr_activation_trace = trace_output

        self.time_corr = time.time() - start
        # len_output = len(corr_output)
        return corr_output 
            
    def __run_inference_resil_model(self):
        self.RESIL_MODEL.eval()
        start = time.time()
        resil_output, nan_dict_resil, inf_dict_resil, detected_activations, penulLayer, quant_list, ftrace_list = self.attach_hooks(self.RESIL_MODEL, resil=self.resil_name.lower())
        if self.inf_nan_monitoring:
            self.nan_flag_image_resil_model = nan_dict_resil['flag'] #Flag true or false per image depending if nan found at any layer
            self.nan_inf_flag_image_resil_model = [nan_dict_resil['flag'][h] or inf_dict_resil['flag'][h] for h in range(len(inf_dict_resil['flag']))]
            self.nan_inf_overall_layers_image_resil_model = [np.unique(nan_dict_resil['overall'][i] + inf_dict_resil['overall'][i]).tolist() for i in range(len(nan_dict_resil['overall']))]
            self.nan_inf_first_occurrence_image_resil_model = [i for i in nan_dict_resil['first_occur_compare']]
        if self.ranger_detector:
            self.resil_ranger_actvns.extend(detected_activations)
        if self.quant_extr:
            self.quant_list_resil.extend(quant_list)
        if self.ftrace_extr:
            self.ftrace_list_resil.extend(ftrace_list)
        if self.sim_score:
            resil_penulLayer =  torch.cat([x.unsqueeze(0) for x in penulLayer])
            if self.resil_penulLayer is not None:
                self.resil_penulLayer = torch.hstack([self.resil_penulLayer, resil_penulLayer])
            else:
                self.resil_penulLayer = torch.hstack([resil_penulLayer])
        if self.activation_trace:
            trace_output = trace_output if trace_output else []
            if self.resil_activation_trace is not None:
                self.resil_activation_trace = torch.cat((self.resil_activation_trace, trace_output), 0)
            else:
                self.resil_activation_trace = trace_output
        resil_output = self.RESIL_MODEL(self.dataloader.images)
        self.time_resil = time.time() - start
        # len_output = len(resil_output)
        return resil_output

    def __run_inference_resil_corr_model(self):
        start = time.time()
        # if self.corr_img:
        #     self.RESIL_CORR_MODEL = self.RESIL_MODEL
        self.RESIL_CORR_MODEL.eval()
        resil_corr_output, nan_dict_resil_corr, inf_dict_resil_corr, detected_activations, penulLayer, quant_list, ftrace_list = self.attach_hooks(self.RESIL_CORR_MODEL, resil=self.resil_name.lower())

        if self.inf_nan_monitoring:
            self.nan_flag_image_resil_corr_model = nan_dict_resil_corr['flag'] #Flag true or false per image depending if nan found at any layer
            self.nan_inf_flag_image_resil_corr_model = [nan_dict_resil_corr['flag'][h] or inf_dict_resil_corr['flag'][h] for h in range(len(inf_dict_resil_corr['flag']))]
            self.nan_inf_overall_layers_image_resil_corr_model = [np.unique(nan_dict_resil_corr['overall'][i] + inf_dict_resil_corr['overall'][i]).tolist() for i in range(len(nan_dict_resil_corr['overall']))]
            self.nan_inf_first_occurrence_image_resil_corr_model = [i for i in nan_dict_resil_corr['first_occur_compare']]
        if self.ranger_detector:
            self.resil_corr_ranger_actvns.extend(detected_activations)
        if self.quant_extr:
            self.quant_list_resil_corr.extend(quant_list)
        if self.ftrace_extr:
            self.ftrace_list_resil_corr.extend(ftrace_list)
        if self.sim_score:
            resil_corr_penulLayer =  torch.cat([x.unsqueeze(0) for x in penulLayer])
            if self.resil_corr_penulLayer is not None:
                self.resil_corr_penulLayer = torch.hstack([self.resil_corr_penulLayer, resil_corr_penulLayer])
            else:
                self.resil_corr_penulLayer = torch.hstack([resil_corr_penulLayer])

        if self.activation_trace:
            trace_output = trace_output if trace_output else []
            if self.resil_corr_activation_trace is not None:
                self.resil_corr_activation_trace = torch.cat((self.resil_corr_activation_trace, trace_output), 0)
            else:
                self.resil_corr_activation_trace = trace_output
        self.time_corr_resil = time.time() - start
        # len_output = len(resil_corr_output)
        return resil_corr_output


    def __run_inference(self):

        with torch.no_grad():
            if self.orig_model_run:
                if self.golden_epoch:
                    orig_output = self.__run_inference_orig_model()
                if self.orig_model_FI_run:
                    corr_output = self.__run_inference_corr_model()

            if self.resil_model_run:
                if self.golden_epoch:
                    resil_output = self.__run_inference_resil_model()
                if self.resil_model_FI_run:
                    resil_corr_output = self.__run_inference_resil_corr_model()


            for i in range(len(orig_output)):
                # Processing:
                if self.golden_epoch:
                    if self.orig_model_run:
                        orig_model_entropy, orig_perct, orig_index = self._processing(orig_output[i])
                    if self.resil_model_run:
                        resil_model_entropy, resil_perct, resil_output_index = self._processing(resil_output[i])
                if self.orig_model_FI_run:
                    corr_model_entropy, corrupt_perct, corrupt_op_index = self._processing(corr_output[i])
                if self.resil_model_FI_run:
                    resil_corr_model_entropy, resil_corr_perct, resil_corr_output_index = self._processing(resil_corr_output[i])

                # Labels
                if self.device.type == 'cuda':
                    label = self.dataloader.labels[i].cpu().detach().numpy()
                else:
                    label = self.dataloader.labels[i].detach().numpy()

                if not self.disable_FI:
                    bit_flip_pos = self.model_wrapper.get_fault_for_image(i)
                    if self.reference_parser.rnd_mode == 'weights':
                        layer, channel_in, channel_out, channel, height, width, bit = bit_flip_pos
                    elif self.reference_parser.rnd_mode == 'neurons':
                        _batch, layer, channel_out, channel, height, width, bit = bit_flip_pos
                        channel_in = None

                if self.golden_epoch:
                    if self.resil_model_run and self.orig_model_run:
                        df_i = [self.dataloader.image_path[i], label, orig_index, resil_output_index, orig_perct, resil_perct, self.time_orig, self.time_resil]
                    elif self.resil_model_run and not(self.orig_model_run):
                        df_i = [self.dataloader.image_path[i], label, resil_output_index, resil_perct, self.time_resil]
                    elif self.orig_model_run and not(self.resil_model_run):
                        df_i = [self.dataloader.image_path[i], label, orig_index, orig_perct, self.time_orig]
                    if self.compute_entropy:
                        if self.orig_model_run:
                            df_i.extend([orig_model_entropy])
                        if self.resil_model_run:
                            df_i.extend([resil_model_entropy])

                    if self.inf_nan_monitoring:
                        if self.resil_model_run:
                            nan_inf_flags = [self.nan_flag_image_resil_model[i], self.nan_inf_flag_image_resil_model[i], self.nan_inf_overall_layers_image_resil_model[i], self.nan_inf_first_occurrence_image_resil_model[i]]
                        else:
                            nan_inf_flags = [None, None, None, None]

                        df_i.extend(nan_inf_flags)
                    if self.golden_model_outputs is None:
                        self.golden_model_outputs = pd.DataFrame([df_i], columns = self.get_orig_output_columns())
                    else:
                        df_i = pd.DataFrame([df_i], columns = self.get_orig_output_columns())
                        self.golden_model_outputs = pd.concat([self.golden_model_outputs, df_i], ignore_index = True)

                if self.resil_model_FI_run:
                    if self.orig_model_FI_run:
                        if self.inf_nan_monitoring:
                            df_i = [self.dataloader.image_path[i], label, corrupt_op_index, resil_corr_output_index,
                                    corrupt_perct, resil_corr_perct, layer, channel_in, channel_out, channel, height, width, bit,
                                    self.time_corr, self.time_corr_resil, self.nan_flag_image_corr_model[i], self.nan_inf_flag_image_corr_model[i], np.array(self.nan_inf_overall_layers_image_corr_model[i])[:1], np.array(self.nan_inf_first_occurrence_image_corr_model[i])[:1], 
                                    self.nan_flag_image_resil_corr_model[i], self.nan_inf_flag_image_resil_corr_model[i], np.array(self.nan_inf_overall_layers_image_resil_corr_model[i])[:1], np.array(self.nan_inf_first_occurrence_image_resil_corr_model[i])[:1]]
                        else:
                            df_i = [self.dataloader.image_path[i], label, corrupt_op_index, resil_corr_output_index,
                                    corrupt_perct, resil_corr_perct, layer, channel_in, channel_out, channel, height, width, bit,
                                    self.time_corr, self.time_corr_resil]
                    else:
                        if self.inf_nan_monitoring:
                            df_i = [self.dataloader.image_path[i], label, resil_corr_output_index,
                                    resil_corr_perct, layer, channel_in, channel_out, channel, height, width, bit,
                                    self.time_corr, self.time_corr_resil, self.nan_flag_image_resil_corr_model[i], self.nan_inf_flag_image_resil_corr_model[i], 
                                    self.nan_inf_overall_layers_image_resil_corr_model[i], self.nan_inf_first_occurrence_image_resil_corr_model[i]]
                        else:
                            df_i = [self.dataloader.image_path[i], label, resil_corr_output_index,
                                    resil_corr_perct, layer, channel_in, channel_out, channel, height, width, bit, self.time_corr_resil]
                elif self.orig_model_FI_run:
                    if self.inf_nan_monitoring:
                        df_i = [self.dataloader.image_path[i], label, corrupt_op_index, corrupt_perct,
                                layer, channel_in, channel_out, channel, height, width, bit, self.time_corr,
                                self.nan_flag_image_corr_model[i], self.nan_inf_flag_image_corr_model[i], np.array(self.nan_inf_overall_layers_image_corr_model[i])[:1], np.array(self.nan_inf_first_occurrence_image_corr_model[i])[:1]]

                    else:
                        df_i = [self.dataloader.image_path[i], label, corrupt_op_index, corrupt_perct,
                                layer, channel_in, channel_out, channel, height, width, bit, self.time_corr]
                if self.compute_entropy:
                    if self.orig_model_run:
                        df_i.extend([corr_model_entropy])
                    if self.resil_corr_model_run:
                        df_i.extend([resil_corr_model_entropy])

                if not self.disable_FI:
                    if self.model_outputs is None:
                        self.model_outputs = pd.DataFrame([df_i], columns = self.get_corr_output_columns())
                    else:
                        df_i = pd.DataFrame([df_i], columns = self.get_corr_output_columns())
                        self.model_outputs = pd.concat([self.model_outputs, df_i], ignore_index = True)

    def __save_ACTtrace(self):
        if self.golden_epoch:
            fault_bin_file = os.path.join(self.outputdir, self.dataset_name, 'orig_model', 'epochs', str(self.curr_epoch), 'orig_golden_ACTtrace.bin')
            print('storing activation trace info - {} into {}'.format(self.func, fault_bin_file))
            os.makedirs(os.path.dirname(fault_bin_file), exist_ok=True)
            f = open(fault_bin_file, 'wb')
            pickle.dump(self.orig_activation_trace, f)
            self.orig_trace = None

            fault_bin_file = os.path.join(self.outputdir, self.dataset_name, '{}_model'.format(self.resil_name), 'epochs', str(self.curr_epoch), 'resil_golden_ACTtrace.bin')
            print('storing activation trace info - {} into {}'.format(self.func, fault_bin_file))
            f = open(fault_bin_file, 'wb')
            pickle.dump(self.resil_activation_trace, f)
            self.resil_trace = None
            return None

        ## corr penultimalte layer info
        if self.orig_model_run:
            fault_bin_file = os.path.join(self.outputdir, self.dataset_name, 'corr_model', 'epochs', str(self.curr_epoch), 'corr_ACTtrace.bin')
            os.makedirs(os.path.dirname(fault_bin_file), exist_ok=True)
            print('storing activation trace info - {} into {}'.format(self.func, fault_bin_file))
            fault_bin = torch.cat((self.corr_activation_trace), 0)
            f = open(fault_bin_file, 'wb')
            pickle.dump(fault_bin, f)
            self.corr_activation_trace = None

        ## resil corr penultimalte layer info
        if self.resil_model_run:
            fault_bin_file = os.path.join(self.outputdir, self.dataset_name, '{}_corr_model'.format(self.resil_name), 'epochs', str(self.curr_epoch), 'resil_corr_ACTtrace.bin')
            os.makedirs(os.path.dirname(fault_bin_file), exist_ok=True)
            print('storing activation trace info - {} into {}'.format(self.func, fault_bin_file))
            fault_bin = torch.cat((self.resil_corr_activation_trace), 0)
            f = open(fault_bin_file, 'wb')
            pickle.dump(fault_bin, f)
            self.resil_corr_trace = None
        return None


    def __save_quantiles(self):
        # Save quantile info
        quant_dict = {'corr': [n.tolist() for n in self.quant_list_corr], 'resil': [n.tolist() for n in self.quant_list_resil], 'resil_corr': [n.tolist() for n in self.quant_list_resil_corr]}
        file_name = os.path.join(self.outputdir, self.dataset_name, self.func + '_' + str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_quantiles.json')

        with open(file_name, 'w') as outfile:
            json.dump(quant_dict, outfile)
        print('saved quantiles under', file_name)


    def __save_ftraces(self):
        # Save quantile info
        quant_dict = {'corr': [n.tolist() for n in self.ftrace_list_corr], 'resil': [n.tolist() for n in self.ftrace_list_resil], 'resil_corr': [n.tolist() for n in self.ftrace_list_resil_corr]}
        file_name = os.path.join(self.outputdir, self.dataset_name, self.func + '_' + str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_ftraces.json')

        with open(file_name, 'w') as outfile:
            json.dump(quant_dict, outfile)
        print('saved ftraces under', file_name)

    def __runset_attr_adjust(self, attr):
        dataset_size = self.reference_wrapper.dataset_size
        batch_size   = self.reference_wrapper.batch_size
        batches = int(np.ceil(self.num_runs*dataset_size/batch_size))
        max_faults_per_image = self.reference_wrapper.max_faults_per_image
        _attr = np.tile(attr[0:max_faults_per_image], (1, batch_size))
        curr_dataset_size = batch_size
        curr_batch_size   = batch_size
        for i in range(1, batches):
            if curr_dataset_size + curr_batch_size  > dataset_size:
                curr_batch_size = dataset_size - curr_dataset_size
                curr_batch_size = curr_batch_size if curr_batch_size>0 else batch_size
                curr_dataset_size = 0
            else:
                curr_batch_size = batch_size
                curr_dataset_size = curr_dataset_size + curr_batch_size
            tile_attr = np.tile(attr[i*max_faults_per_image:i*max_faults_per_image+max_faults_per_image],(1, curr_batch_size))
            _attr = np.hstack([_attr, tile_attr])
        return _attr

    def __save_fault_file(self):
        if not self.disable_FI:

            if self.reference_parser.rnd_mode == 'neurons':
                runset_length = self.num_runs*self.dataloader.dataset_length*self.num_faults
            elif self.reference_parser.rnd_mode == 'weights':
                if self.reference_parser.inj_policy == 'per_epoch':
                    runset_length = self.num_runs*self.num_faults
                elif self.reference_parser.inj_policy == 'per_batch':
                    runset_length = int(np.ceil(self.num_runs*self.dataloader.dataset_length*self.num_faults/self.dl_attr.dl_batch_size))

            def add_ranger_detector(runset, ranger_actvns, used_model):
                if self.ranger_detector:
                    num_inferences =  self.num_runs*self.dataloader.dataset_length
                    assert len(ranger_actvns) == num_inferences, "{} Sanity check: Ranger activations captured in {} and total number of number - \
                        {} dont match".format(used_model, len(ranger_actvns), num_inferences)
                    runset = np.vstack([runset, np.array(ranger_actvns)])
                return runset

            """
            bitflip_attr updates the flip direction and bit flip position if bit flip type is either using bound range or weighted method
            also add ranger detector information.
            ## TODO: works for neurons; need to check if this works for weights
            fault_bin column = [[original fault location 7 elements], bit-flip direction (monitor), orig value, corr value, total ranger_detections - IF APPLIED]
            # --- [original fault location 7 elements] => Meaning for NEURON injection: --- #
                # 1. batchnumber (used in: conv2d,conv3d)
                # 2. layer (everywhere)
                # 3. channel (used in: conv2d,conv3d)
                # 4. depth (used in: conv3d)
                # 5. height (everywhere)
                # 6. width (everywhere)
                # 7. value (everywhere)
            # --- [original fault location 7 elements] => Meaning for WEIGHT injection: --- #
                # 1. layer (everywhere)
                # 2. Kth filter (everywhere)
                # 3. channel(used in: conv2d, conv3d)
                # 4. depth (used in: conv3d)
                # 5. height (everywhere)
                # 6. width (everywhere)
                # 7. value (everywhere)
            """
            used_models = []
            if self.orig_model_run:
                used_models.append('corr')
            if self.resil_model_run:
                used_models.append('{} corr'.format(self.resil_name))
            for used_model in used_models:
                if used_model == 'corr':
                    runset = self.model_wrapper.runset_updated
                    # runset = runset[:,:runset_length]
                    bit_flip_monitor = self.model_wrapper.pytorchfi.bit_flips_monitor
                    value_monitor    = self.model_wrapper.pytorchfi.value_monitor
                    bit_flips_direc = self.model_wrapper.pytorchfi.bit_flips_direc
                    ranger_actvns = self.corr_ranger_actvns
                elif used_model == '{} corr'.format(self.resil_name):
                    runset = self.resil_wrapper.runset_updated
                    bit_flip_monitor = self.resil_wrapper.pytorchfi.bit_flips_monitor
                    value_monitor    = self.resil_wrapper.pytorchfi.value_monitor
                    bit_flips_direc = self.resil_wrapper.pytorchfi.bit_flips_direc
                    ranger_actvns = self.resil_corr_ranger_actvns
                assert runset_length == len(bit_flips_direc), "{} Sanity check: runset len after tiling (if inj policy is per epoch or per batch) - \
                    {} and total number of bit flips captured - {} dont match".format(used_model, runset_length, len(bit_flips_direc))
                assert len(bit_flip_monitor) == len(bit_flips_direc), "{} Sanity check: bit flips monitored - {} and total number of bit flips captured - \
                    {} dont match".format(used_model, bit_flip_monitor, bit_flips_direc)
        
                # if self.corr_img:
                #     bit_flip_monitor = -1*np.ones(len(bit_flip_monitor)) #convert it to -1 for saving to fault-file
                #     bit_flips_direc = -1*np.ones(len(bit_flips_direc))

                ## updating the bit information which was actually flipped and not the copy of runset bits
                if self.reference_parser.rnd_mode == 'neurons':
                    runset[-1] = bit_flip_monitor
                    runset = np.vstack([runset, bit_flips_direc])
                    runset = np.vstack([runset, value_monitor[0]]) ## orig value
                    runset = np.vstack([runset, value_monitor[1]]) ## corr value
                    runset = add_ranger_detector(runset, ranger_actvns, used_model)
                elif self.reference_parser.rnd_mode == 'weights':
                    if self.reference_parser.inj_policy == 'per_epoch':
                        runset[-1] = np.repeat(bit_flip_monitor, self.model_scenario["dataset_size"])
                        bit_flips_direc = np.repeat(bit_flips_direc, self.model_scenario["dataset_size"])
                        orig_value = np.repeat(value_monitor[0], self.model_scenario["dataset_size"])
                        corr_value = np.repeat(value_monitor[1], self.model_scenario["dataset_size"])
                    elif self.reference_parser.inj_policy == 'per_batch':
                        runset[-1] = self.__runset_attr_adjust(bit_flip_monitor)
                        bit_flips_direc = self.__runset_attr_adjust(bit_flips_direc)
                        orig_value = self.__runset_attr_adjust(value_monitor[0])
                        corr_value = self.__runset_attr_adjust(value_monitor[1])

                    runset = np.vstack([runset, bit_flips_direc])
                    runset = np.vstack([runset, orig_value]) ## orig value
                    runset = np.vstack([runset, corr_value]) ## corr value
                    runset = add_ranger_detector(runset, ranger_actvns, used_model)

                if self.reference_parser.rnd_value_type in ["bitflip_weighted"]:
                    fault_bin = os.path.join(self.outputdir, self.dataset_name, self.func + '_' + used_model + '_' +
                        str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_' + self.dl_attr.dl_dataset_name + '_fault_locs_overwritten.bin')
                else:
                    fault_bin = os.path.join(self.outputdir, self.dataset_name, self.func + '_' +   used_model + '_' +
                        str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_' + self.dl_attr.dl_dataset_name + '_updated_rs_fault_locs.bin')

                dirname = os.path.dirname(fault_bin)
                os.makedirs(os.path.dirname(fault_bin), exist_ok=True)
                f = open(fault_bin, 'wb')
                pickle.dump(runset, f)
                f.flush()
                f.close()
                print('{}: saving fault file with bit flip direction info with {} fault runset length and total {} bit flips into {}'.format(used_model, len(runset[0,:]), len(bit_flips_direc), f.name ))

        if not self.copy_yml_scenario and not self.disable_FI:
            fault_bin = os.path.join(self.outputdir, self.dataset_name, self.func + '_' +
                str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_' + self.dl_attr.dl_dataset_name + '_fault_locs.bin')
            self.model_scenario['fi_logfile'] = os.path.basename(fault_bin)
            runset = self.reference_wrapper.runset
            os.makedirs(os.path.dirname(fault_bin), exist_ok=True)
            f = open(fault_bin, 'wb')
            pickle.dump(runset, f)
            f.flush()
            f.close()

        if self.resil_model_run:
            resil_detections_file = os.path.join(self.outputdir, self.dataset_name, self.func + '_' +
                str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_' + self.dl_attr.dl_dataset_name + '_{}_detections.bin'.format(self.resil_name))
            resil_detections = self.resil_ranger_actvns
            os.makedirs(os.path.dirname(resil_detections_file), exist_ok=True)
            f = open(resil_detections_file, 'wb')
            pickle.dump(resil_detections, f)
            f.flush()
            f.close()

    def save_results(self, image_fp=False, golden_epoch=False):
        if image_fp:
            try:
                dataset_fp = pd.DataFrame({'dataset fp indx': self.dataloader.datasetfp})
            except: #if empty
                dataset_fp = pd.DataFrame({'dataset fp indx': self.dataloader.datasetfp}, index=[0])
            dataset_fp.to_csv(self.image_fp, index=True)
        if golden_epoch:
            print('storing golden resiliency output {} into {}'.format(self.func, self.fi_result_file + '_golden.csv'))
            if not os.path.exists(os.path.dirname(self.fi_result_file + '_golden.csv')):
                os.makedirs(os.path.dirname(self.fi_result_file + '_golden.csv'))
            self.golden_model_outputs.to_csv(self.fi_result_file + '_golden.csv', index=False)
            self.golden_model_outputs = None
            return None
        elif not self.disable_FI:
            if (os.path.exists(self.fi_result_file)):
                model_outputs_df = pd.read_csv(self.fi_result_file + '.csv')
                self.model_outputs = pd.concat([model_outputs_df, self.model_outputs], ignore_index=True)
                print('storing ranger resiliency - {} into {}'.format(self.func, self.fi_result_file + '.csv'))
                self.model_outputs.to_csv(self.fi_result_file + '_corr.csv', index=False, mode='w+')
                self.model_outputs = None
            else:
                print('storing ranger resiliency - {} into {}'.format(self.func, self.fi_result_file + '.csv'))
                if not os.path.exists(os.path.dirname(self.fi_result_file + '.csv')):
                    os.makedirs(os.path.dirname(self.fi_result_file + '.csv'))
                self.model_outputs.to_csv(self.fi_result_file + '_corr.csv', index=False)
                self.model_outputs = None
            return None


    def get_orig_output_columns(self):
        output_columns = []
        nan_inf_resil_model_columns = ['nan_flag_resil_model', 'nan_or_inf_flag_resil_model', 'Nan_or_inf_layers_resil_model', 'Nan_inf_first_occurrence_resil_model']
        if self.resil_model_run and self.orig_model_run:
            output_columns = [ 'file_path', 'gnd_label', 'orig output index - top5',
                'resil output index - top5', 'orig output prob - top5', 'resil output prob - top5', 
                'time - orig model','time - {} model'.format(self.resil_name)]
        elif self.resil_model_run and not(self.orig_model_run):
            output_columns = [ 'file_path', 'gnd_label', 'resil output index - top5', 'resil output prob - top5', 'time - {} model'.format(self.resil_name)]
        elif self.orig_model_run and not(self.resil_model_run):
            output_columns = [ 'file_path', 'gnd_label', 'orig output index - top5', 'orig output prob - top5', 'time - orig model']
        if self.compute_entropy:
            if self.orig_model_run:
                output_columns.extend(['orig_entropy'])
            if self.resil_model_run:
                output_columns.extend(['resil_entropy'])
        if self.inf_nan_monitoring:
            output_columns.extend(nan_inf_resil_model_columns)
        return output_columns

    def get_corr_output_columns(self):
        output_columns = []
        nan_inf_corr_model_columns = ['nan_flag_corr_model', 'nan_or_inf_flag_corr_model', 'Nan_or_inf_layers_corr_model', 'Nan_inf_first_occurrence_corr_model']
        nan_inf_resil_model_columns = ['nan_flag_resil_corr_model', 'nan_or_inf_flag_resil_corr_model', 'Nan_or_inf_layers_resil_corr_model', 'Nan_inf_first_occurrence_resil_corr_model']
        if self.resil_model_FI_run:
            if self.orig_model_FI_run:
                output_columns = [ 'file_path', 'gnd_label', 'corr output index - top5',
                    'resil corr output index - top5', 'corr output prob - top5', 'resil corr output prob - top5', 'layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit', 'time - corr model',
                    'time - {} corr model'.format(self.resil_name)]
                if self.inf_nan_monitoring:
                    output_columns.extend(nan_inf_corr_model_columns)
                    output_columns.extend(nan_inf_resil_model_columns)
            else:
                output_columns = [ 'file_path', 'gnd_label', 'resil corr output index - top5', 'resil corr output prob - top5', 'layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit',
                    'time - {} corr model'.format(self.resil_name)]
                if self.inf_nan_monitoring:
                    output_columns.extend(nan_inf_resil_model_columns)
        elif self.orig_model_FI_run:
            output_columns = ['file_path', 'gnd_label', 'corr output index - top5',\
            'corr output prob - top5', 'layer', 'channel_in', 'channel_out', '3D channel',\
            'height', 'width', 'bit', 'time - corr model']
            if self.inf_nan_monitoring:
                output_columns.extend(nan_inf_corr_model_columns)
        if self.compute_entropy:
            if self.orig_model_run:
                output_columns.extend(['corr_entropy'])
            if self.resil_model_run:
                output_columns.extend(['resil_entropy'])
        return output_columns

    def entropy(self, prob):
        prob = prob / prob.sum(0).expand_as(prob)
        prob = prob.cpu().detach().numpy() if 'cuda' in prob.device.type else prob.detach().numpy()
        return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)
    def save_simscore(self, golden=False):
        if golden:
            fault_bin_file = os.path.join(os.path.dirname(self.fi_result_file), 'orig_golden_ACTtrace.bin')
            print('storing penulLayer info - {} into {}'.format(self.func, fault_bin_file))
            os.makedirs(os.path.dirname(fault_bin_file), exist_ok=True)
            f = open(fault_bin_file, 'wb')
            pickle.dump(self.orig_penulLayer, f)
            self.orig_penulLayer = None

            fault_bin_file = os.path.join(os.path.dirname(self.fi_result_file), 'resil_golden_ACTtrace.bin')
            print('storing penulLayer info - {} into {}'.format(self.func, fault_bin_file))
            f = open(fault_bin_file, 'wb')
            pickle.dump(self.resil_penulLayer, f)
            self.resil_penulLayer = None
            return None

        ## corr penultimalte layer info
        fault_bin_file = os.path.join(os.path.dirname(self.fi_result_file), 'corr_ACTtrace.bin')
        os.makedirs(os.path.dirname(fault_bin_file), exist_ok=True)
        if (os.path.exists(fault_bin_file)):
            fault_bin = read_faultbin_file(fault_bin_file)
            fault_bin = torch.cat([fault_bin, self.corr_penulLayer])
            print('storing penulLayer info - {} into {}'.format(self.func, fault_bin_file))
            f = open(fault_bin_file, 'wb')
            pickle.dump(fault_bin, f)
        else:
            print('storing penulLayer info - {} into {}'.format(self.func, fault_bin_file))
            fault_bin = torch.cat([self.corr_penulLayer])
            f = open(fault_bin_file, 'wb')
            pickle.dump(fault_bin, f)
        self.corr_penulLayer = None

        ## resil corr penultimalte layer info
        fault_bin_file = os.path.join(os.path.dirname(self.fi_result_file), 'resil_corr_ACTtrace.bin')
        if (os.path.exists(fault_bin_file)):
            fault_bin = read_faultbin_file(fault_bin_file)
            fault_bin = torch.cat([fault_bin, self.resil_corr_penulLayer])
            print('storing penulLayer info - {} into {}'.format(self.func, fault_bin_file))
            f = open(fault_bin_file, 'wb')
            pickle.dump(fault_bin, f)
        else:
            print('storing penulLayer info - {} into {}'.format(self.func, fault_bin_file))
            fault_bin = torch.cat([self.resil_corr_penulLayer])
            f = open(fault_bin_file, 'wb')
            pickle.dump(fault_bin, f)
        self.resil_corr_penulLayer = None
        return None

    def save_monitored_features(self, golden_epoch=False):
        if self.activation_trace:
            self.__save_ACTtrace()
        if self.sim_score:
            # self.save_simscore(golden=golden)
            pass
        if self.quant_extr:
            self.__save_quantiles()
        if self.ftrace_extr:
            self.__save_ftraces()

    def inject_faults(self):
        if self.orig_model_run:
            if not self.disable_FI:
                self.CORR_MODEL = next(self.orig_models_fault_iter)
                self.CORR_MODEL.to(self.device)
            else: 
                self.CORR_MODEL = None
        if self.resil_model_run:
            if not self.disable_FI:
                self.RESIL_CORR_MODEL = next(self.resil_models_fault_iter)
                self.RESIL_CORR_MODEL.to(self.device)
            else:
                self.RESIL_CORR_MODEL = None

    def copy_yml_attr(self, data):
        for key in list(data.keys()):
            self.model_scenario[key] = data[key]

    def set_ptfi_batch_pointer(self):
        if self.reference_parser.rnd_mode == "neurons":
            if self.reference_parser.inj_policy == "per_image" or self.reference_parser.inj_policy == 'per_batch':
                ptfi_batch_pointer = self.curr_epoch*self.dataloader.dataset_length*self.num_faults + self.dataloader.datagen_iter_cnt
            elif self.reference_parser.inj_policy == "per_epoch":
                ptfi_batch_pointer = self.curr_epoch*self.dataloader.dataset_length*self.num_faults + self.dataloader.datagen_iter_cnt
            if self.orig_model_FI_run:
                self.model_wrapper.pytorchfi.ptfi_batch_pointer = ptfi_batch_pointer
            if self.resil_model_FI_run:
                self.resil_wrapper.pytorchfi.ptfi_batch_pointer = ptfi_batch_pointer
        if self.reference_parser.rnd_mode == "weights":
            if self.reference_parser.inj_policy == "per_epoch":
                ptfi_batch_pointer = self.curr_epoch
            if self.reference_parser.inj_policy == 'per_batch':
                ptfi_batch_pointer = int(np.ceil(self.curr_epoch*self.dataloader.dataset_length*(1/self.dl_attr.dl_batch_size))) + int(np.ceil(self.dataloader.datagen_iter_cnt/(self.dl_attr.dl_batch_size or 1)))
            if self.orig_model_FI_run:
                self.model_wrapper.pytorchfi.ptfi_batch_pointer = ptfi_batch_pointer
            if self.resil_model_FI_run:
                self.resil_wrapper.pytorchfi.ptfi_batch_pointer = ptfi_batch_pointer

    def set_FI_attributes(self):
        if not self.disable_FI:
            self.scenario_is_modified = True
            if self.num_faults:
                self.model_scenario["max_faults_per_image"] = self.num_faults
            else:
                self.num_faults = self.reference_wrapper.parser.max_faults_per_image
            if not self.copy_yml_scenario:
                if self.num_runs:
                    self.model_scenario["num_runs"] = self.num_runs
                else:
                    num_runs = self.reference_wrapper.parser.num_runs
                    self.num_runs = num_runs
                self.bit_range = str(self.reference_parser.rnd_bit_range) if self.reference_parser.rnd_bit_range else '[0_{}]'.format(self.reference_parser.rnd_value_bits)
                self.bit_range = self.bit_range.replace(" ", "")
                if self.resume_inj:
                    ## not fully developed ## under development
                    ## TODO: while resuming copy the bit flip direction and fault locations from fault files of corrupt and ranger fault files.
                    self.outputdir = self.resume_dir
                else:
                    self.outputdir  = os.path.join(self._outputdir, '{}_{}_trials'.format(
                        self.model_name, self.num_runs), '{}_injs'.format(self.reference_parser.rnd_mode), '{}'.format(self.reference_parser.inj_policy), 'imClass_{}_{}_faults_{}_bits'.format(self.time_based_uuid, self.num_faults, self.bit_range))

            if self.fault_file and self.copy_yml_scenario:
                assert up(self.fault_file) == up(self.model_scenario_file), "Fault file and scenario file don't belong to same experiment folder"
                self.model_scenario["read_from_file"] = self.fault_file
            self.model_scenario["dataset_size"] = self.dataloader.dataset_length

            if self.dataloader.dataset_length < self.reference_parser.ptf_batch_size:
                self.reference_parser.ptf_batch_size = self.dataloader.dataset_length
                self.dl_attr.dl_batch_size = self.dataloader.dataset_length
            if self.scenario_is_modified:
                self.reference_wrapper.set_scenario(self.model_scenario, create_runset=True)
                if self.orig_model_run and self.resil_model_run:
                    self.resil_wrapper.set_scenario(self.model_scenario, create_runset=False)
                    self.resil_wrapper.runset = self.reference_wrapper.runset
                    self.resil_wrapper.runset_updated = self.reference_wrapper.runset_updated
                ## load updated scenario; to be used to store model scenario as dictionary in the form json or yml 
                self.model_scenario = self.reference_wrapper.get_scenario()

            if self.copy_yml_scenario:
                outputdir = up(up(up(list(Path(os.path.dirname(self.fault_file)).glob('**/*.yml'))[0])))
                self.num_runs = self.reference_wrapper.parser.num_runs
                self.bit_range = str(self.reference_parser.rnd_bit_range) if self.reference_parser.rnd_bit_range else '[0_{}]'.format(self.reference_parser.rnd_value_bits)
                self.bit_range = self.bit_range.replace(" ", "")
                if self.resume_inj:
                    self.outputdir = self.resume_dir
                else:
                    self.outputdir  = outputdir

            self.resume_epoch = 0
            self.resume_pointer = 0
            if self.resume_inj:
                for _epoch in range(self.num_runs):
                    if self.__check_resume_inj_status(model_type='corr_model', epoch=_epoch):
                        self.resume_epoch = _epoch + 1
                        continue
                    else:
                        print('resuming the experiment from {} epoch'.format(self.resume_epoch))
                        break
                if self.reference_parser.inj_policy == "per_image":
                    if self.reference_parser.rnd_mode == 'neurons':
                        self.resume_pointer = self.resume_epoch*self.dataloader.dataset_length*self.num_faults
                    elif self.reference_parser.rnd_mode == 'weights':
                        self.resume_pointer = self.resume_epoch*self.num_faults
                elif self.reference_parser.inj_policy == "per_epoch":
                    self.resume_pointer = self.resume_epoch*self.num_faults
                elif self.reference_parser.inj_policy == 'per_batch':
                    self.resume_pointer = int(np.ceil(self.resume_epoch*self.dataloader.dataset_length*self.num_faults/self.dl_attr.dl_batch_size))
            if self.orig_model_FI_run:
                self.orig_models_fault_iter = self.model_wrapper.get_fimodel_iter(resume_pointer=self.resume_pointer)
            if self.resil_model_FI_run:
                self.resil_models_fault_iter = self.resil_wrapper.get_fimodel_iter(resume_pointer=self.resume_pointer)
        self.dataset_name = self.dl_attr.dl_dataset_name + '/' + self.dl_attr.dl_dataset_type
        self.fi_result_file = os.path.join(self.outputdir, self.dataset_name, self.func + '_' + 
            str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_' + self.dl_attr.dl_dataset_name + '_results')

        self.image_fp = os.path.join(self.outputdir, self.dataset_name, self.func + '_' + 
            str(self.num_faults) + '_' + str(self.num_runs) + '_' + str(self.dl_attr.dl_batch_size) + 'bs' + '_' + self.dl_attr.dl_dataset_name + '_image_fp.csv')

        if not os.path.exists(os.path.dirname(self.fi_result_file)):
            os.makedirs(os.path.dirname(self.fi_result_file))


    def __copy_dl_attr(self):
        """
        copies dl_attr class attr to the scenario dictionary.
        """
        if self.copy_yml_scenario:
            if not self.disable_FI:
                for dl_key in self.dl_attr.__dict__.keys():
                    if dl_key in self.copied_scenario_data:
                        if not dl_key == "dl_batch_size":
                            setattr(self.dl_attr, dl_key, self.copied_scenario_data[dl_key])
                    else:
                        self.copied_scenario_data[dl_key] = getattr(self.dl_attr, dl_key)
            else:
                for dl_key in self.dl_attr.__dict__.keys():
                    dl_value = getattr(self.dl_attr, dl_key)
                    self.copied_scenario_data[dl_key] = dl_value

    def test_rand_ImgClass_SBFs_inj(self):
        """
        Injected random single bit flips onto object detection DNNs
        loads the faults and injects into the models
        during inference on original and corrupt model
        all the models will be associated with chosen evaluators
        after each batch of inference model's evalutors processs(inputs, outputs) will be called
        to store epoch's interim results in json file.
        """
        self.inj_value_type = self.reference_wrapper.value_type if not self.disable_FI else "with disable FI"
        self.func = '{}_test_random_sbf_{}_inj'.format(self.model_name, self.reference_parser.rnd_mode)
        self.__copy_dl_attr()
        self.__ptfi_dataloader()
        self.set_FI_attributes()
        self.golden_epoch = True
        self.__save_model_scenario()
        self.bit_range = self.bit_range if self.bit_range else 0
        for _epoch in tqdm(range(self.resume_epoch, self.num_runs), desc="injecting {} faults with {} inj policy - {} runs, {} faults & {} bit_range".format(self.inj_value_type, \
                self.reference_parser.inj_policy if not self.disable_FI else 'no injection', \
                    self.num_runs, self.num_faults if not self.disable_FI else '0', \
                        self.bit_range if not self.disable_FI else '')):
            self.curr_epoch = _epoch
            self.dataloader.datagen_reset()
            if not self.disable_FI:
                self.set_ptfi_batch_pointer()
            with ExitStack() as stack:
                stack.enter_context(torch.no_grad())
                if self.disable_FI:
                    while self.dataloader.data_incoming:
                        self.dataloader.datagen_itr()
                        self.__run_inference()
                    self.dataloader.datagen_reset()
                else:
                    if self.reference_parser.inj_policy == "per_image":
                            while self.dataloader.data_incoming:
                                if self.reference_parser.rnd_mode == 'neurons':
                                    self.inject_faults()
                                self.dataloader.datagen_itr()
                                self.__run_inference()
                                self.set_ptfi_batch_pointer()
                                # self.mod_img = None
                            self.dataloader.datagen_reset()
                    elif self.reference_parser.inj_policy == "per_epoch":  #here eval todo
                        self.inject_faults()
                        while self.dataloader.data_incoming:
                            self.dataloader.datagen_itr()
                            self.__run_inference()
                            self.set_ptfi_batch_pointer()
                            # self.mod_img = None
                            # print('in while loop; dataloader: {}'.format(self.dataloader.datagen_iter_cnt))
                        self.dataloader.datagen_reset()
                    elif self.reference_parser.inj_policy == 'per_batch':
                        while self.dataloader.data_incoming:
                            # if self.reference_parser.rnd_mode == 'neurons':
                            self.inject_faults()
                            self.dataloader.datagen_itr()
                            self.__run_inference()
                            self.set_ptfi_batch_pointer()
                            # self.mod_img = None
                        self.dataloader.datagen_reset()
                    else:
                        print("Parameter inj_policy invalid, should be 'per_image', 'per_batch', or 'per_epoch'")
            
            self.__per_epoch_sanity_checks(num_runs=_epoch)
        if self.golden_epoch:
            self.golden_epoch = False
            self.save_monitored_features(golden_epoch=True)
            self.save_results(image_fp=False, golden_epoch=True)
            # if not self.corr_img: 
            self.ORIG_MODEL = None
            self.RESIL_MODEL = None
            self.ORIG_MODEL_eval = None
            self.RESIL_MODEL_eval = None
        self.save_results()
        self.save_monitored_features()
        self.__save_fault_file()