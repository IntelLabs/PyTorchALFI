"""
pytorchfi.core contains the core functionality for fault injections.
"""

import copy
import logging

import torch
import torch.nn as nn
import numpy as np
import gc
logging.getLogger('numba').setLevel(logging.WARNING)

class fault_injection:
    def __init__(self, model, model_attr_parsed, **kwargs):
        # logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")
        
        self.imageH    = model_attr_parsed.ptf_H
        self.imageW    = model_attr_parsed.ptf_W
        batch_size     = model_attr_parsed.ptf_batch_size
        self.imageC    = model_attr_parsed.ptf_C
        self.imageClip = model_attr_parsed.ptf_D
        
        self.ORIG_MODEL          = None
        self.CORRUPTED_MODEL     = None
        self._BATCH_SIZE         = -1
        self._RUNTIME_BATCH_SIZE = -1
        self.device              = kwargs.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.CUSTOM_INJECTION    = False
        self.INJECTION_FUNCTION  = None

        self.CORRUPT_BATCH     = -1
        self.CORRUPT_CONV      = -1
        self.CORRUPT_CONV_NAME = ""
        self.CORRUPT_CLIP      = -1
        self.CORRUPT_C         = -1
        self.CORRUPT_H         = -1
        self.CORRUPT_W         = -1
        self.CORRUPT_VALUE     = None

        self.CURRENT_CONV         = 0
        self.INPUT_SIZE           = []
        self.OUTPUT_SIZE          = []
        self.NEURON_NUM           = 0
        self.WEIGHT_NUM           = 0
        self.LAYER_NEURON_WEIGHTS = []
        self.LAYER_WEIGHT_WEIGHTS = []
        self.INPUT_LOOKUP         = []
        self.OUTPUT_LOOKUP        = []
        self.CORRUPT_SIZE         = []
        self.CORRUPT_INPUT_SIZE   = []
        self.HANDLES              = []

        self.LAYER_TYPE_LeakyRelu   = False
        self.LAYER_TYPE_CONV2D      = False
        self.LAYER_TYPE_FCC         = False
        self.LAYER_TYPE_CONV3D      = False
        self.LAYER_TYPE_ATT         = False
        self.LAYER_TYPE_ATT_EMBED   = False
        self.LAYER_TYPE_BatchNorm   = False
        
        self.RANDOM_LAYER_SELECTED     = model_attr_parsed.rnd_layer_selected
        self.RANDOM_LAYER_SELECTED_NUM = model_attr_parsed.rnd_layer_selected_num
        self.LAYER_TYPES               = model_attr_parsed.layer_types
        self.LAYER_TYPES_NAMED         = model_attr_parsed.layer_types_named
        self.LAYER_TYPES_EXCLUDE       = model_attr_parsed.layer_types_exclude
        self.model_attr_parsed         = model_attr_parsed

        if "conv2d" in self.LAYER_TYPES:
            self.LAYER_TYPE_CONV2D = True
        if "conv3d" in self.LAYER_TYPES:
            self.LAYER_TYPE_CONV3D = True
        if "fcc" in self.LAYER_TYPES:
            self.LAYER_TYPE_FCC = True
        if "leakyRelu" in self.LAYER_TYPES:
            self.LAYER_TYPE_LeakyRelu = True
        if "batchnorm" in self.LAYER_TYPES:
            self.LAYER_TYPE_BatchNorm = True
        if "attn_fcc" in self.LAYER_TYPES:
            self.LAYER_TYPE_ATT = True
        if "embedding" in self.LAYER_TYPES:
            self.LAYER_TYPE_ATT_EMBED = True

        self.use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)
        model_dtype = next(model.parameters()).dtype

        self.ORIG_MODEL = model
        self.ORIG_MODEL = self.ORIG_MODEL.to(self.device)
        self._BATCH_SIZE = batch_size
        input_num =  kwargs.get("input_num", 1)
        self._INPUT_NUM = input_num

        handles = []
        self._named_id = 0
        self._input_named_id = 0
        k = 0
        Handle_names = []
        for name, param in self.ORIG_MODEL.named_modules():
            if self.__layer_check(param, name):
                # print(name, param) #TODO:
                param.new_name = name
                Handle_names.append(name)
                param.new_id = []
                param.input_new_id = []
                if self.model_attr_parsed.fault_injec_hook_type == "input":
                    handles.append(param.register_forward_pre_hook(self._save_input_size))
                else:
                    handles.append(param.register_forward_hook(self._save_output_size))
                k += 1
                # print(name, param)


        # device = "cuda" if self.use_cuda else None
        #device = torch.device("cuda:{}".format(self.cuda_device) if torch.cuda.is_available() else "cpu")
        # support for 3d conv
        # this seems to trigger the call of _save_output_size
        if not self.imageClip == -1:
            _dummyTensor = torch.randn(
                self._BATCH_SIZE, self.imageC, self.imageClip, self.imageH, self.imageW, dtype=model_dtype, device=self.device
            )
        else:
            _dummyTensor = torch.randn(
                max(self._BATCH_SIZE,1), max(self.imageC,1), max(self.imageH,1), max(self.imageW,1), dtype=model_dtype, device=self.device
            )
        # TODO this is not flexible at all, need to find a better method

        if self._INPUT_NUM == 3:
            self.ORIG_MODEL(_dummyTensor, _dummyTensor, _dummyTensor)
        elif self._INPUT_NUM == 2:
            self.ORIG_MODEL(_dummyTensor, _dummyTensor)
        else:
            self.ORIG_MODEL(_dummyTensor, dummy=True)
            # self.ORIG_MODEL(_dummyTensor)

        for i in range(len(handles)):
            handles[i].remove()

        # store weight size once
        self.WEIGHT_NUM, self.LAYER_WEIGHT_WEIGHTS = self.__get_weight_size()

        # store neuron size only once
        self.NEURON_NUM = np.sum([np.prod(line) for line in self.OUTPUT_SIZE])
        self.LAYER_NEURON_WEIGHTS= [np.prod(line)/self.NEURON_NUM for line in self.OUTPUT_SIZE]
        
        self.INPUT_NEURON_NUM = np.sum([np.prod(line) for line in self.INPUT_SIZE])
        self.INPUT_LAYER_NEURON_WEIGHTS= [np.prod(line)/self.INPUT_NEURON_NUM for line in self.INPUT_SIZE]
        # i=0
        # logging.info("Model output size")
        # for row in self.OUTPUT_SIZE:
        #     logging.info("{} {}".format(i,row))
        #     i += 1

    def __layer_check(self, param, name):
        ret = False
        if self.LAYER_TYPES_NAMED:
            check_layer_types_named = any([layer_types_named in name for layer_types_named in self.LAYER_TYPES_NAMED])
            ## filtering NonDynamicallyQuantizableLinear layers:
            check_layer_types_named = check_layer_types_named and not('out_proj' in name)
            if isinstance(param, nn.Linear) and self.LAYER_TYPE_ATT and check_layer_types_named:
                ret = True
            if isinstance(param, nn.modules.sparse.Embedding) and self.LAYER_TYPE_ATT_EMBED and check_layer_types_named:
                ret = True
            if isinstance(param, nn.Linear) and self.LAYER_TYPE_FCC and check_layer_types_named:
                ret = True
        else:
            if isinstance(param, nn.Conv2d) and self.LAYER_TYPE_CONV2D:
                ret = True
            elif isinstance(param, nn.Conv3d) and self.LAYER_TYPE_CONV3D:
                ret = True
            elif isinstance(param, nn.Linear) and self.LAYER_TYPE_FCC:
                ret = True
            elif isinstance(param, nn.modules.activation.LeakyReLU) and self.LAYER_TYPE_LeakyRelu:
                ret = True
            elif isinstance(param, nn.modules.batchnorm.BatchNorm2d) and self.LAYER_TYPE_BatchNorm:
                ret = True
        if self.LAYER_TYPES_EXCLUDE:
            check_layer_types_exclude = any([layer_types_exclude in name for layer_types_exclude in self.LAYER_TYPES_EXCLUDE])
            if check_layer_types_exclude:
                ret = False
        return ret

    def __get_weight_size(self):
        total_size = 0
        size_arr = []
        cnt = 0
        cnt2 = 0
        for module_name, module in self.get_original_model().named_modules():
            if self.__layer_check(module, module_name):
                for name, param in module.named_parameters():
                    # print(module, name, param.numel())
                    cnt += 1
                    if "weight" in name:
                        # print(module, name, param.numel())
                        cnt2 += 1
                        x = param.numel()
                        total_size += x
                        size_arr.append(x)
        for i in range(len(size_arr)):
            size_arr[i] = size_arr[i] / total_size

        # print(cnt, cnt2) 
        return total_size, size_arr

    def fi_reset(self):
        self._fi_state_reset()
        self.CORRUPTED_MODEL = None
        logging.info("Reset fault injector")

    def _fi_state_reset(self):
        (
            self.CURRENT_CONV,
            self.CORRUPT_BATCH,
            self.CORRUPT_CONV,
            self.CORRUPT_CLIP,
            self.CORRUPT_C,
            self.CORRUPT_H,
            self.CORRUPT_W,
            self.CORRUPT_VALUE,
        ) = (
            0,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            None,
        )

        for i in range(len(self.HANDLES)):
            self.HANDLES[i].remove()

    def declare_weight_fi(self, **kwargs):
        self._fi_state_reset()
        CUSTOM_INJECTION = False
        CUSTOM_FUNCTION = False
        corrupt_value =0
        # TODO add dimension for 3d conv
        if kwargs:
            if "function" in kwargs:
                CUSTOM_INJECTION, INJECTION_FUNCTION = True, kwargs.get("function")
                self.CORRUPT_CONV = kwargs.get("conv_num", -1)
                self.CORRUPT_CLIP = kwargs.get("clip", -1)
                self.CORRUPT_K = kwargs.get("k", -1)
                self.CORRUPT_C = kwargs.get("c", -1)
                self.CORRUPT_H = kwargs.get("h", -1)
                self.CORRUPT_W = kwargs.get("w", -1)
            else:
                self.CORRUPT_CONV = kwargs.get("conv_num", -1)
                self.CORRUPT_CLIP = kwargs.get("clip", -1)
                self.CORRUPT_K = kwargs.get("k", -1)
                self.CORRUPT_C = kwargs.get("c", -1)
                self.CORRUPT_H = kwargs.get("h", -1)
                self.CORRUPT_W = kwargs.get("w", -1)
                self.CORRUPT_VALUE = kwargs.get("value", None)
        if kwargs:
            if "function" in kwargs:
                CUSTOM_INJECTION, CUSTOM_FUNCTION = True, kwargs.get("function")
                corrupt_layer = kwargs.get("conv_num", -1)
                corrupt_k = kwargs.get("k", -1)
                corrupt_c = kwargs.get("c", -1)
                corrupt_clip = kwargs.get("clip", -1)
                corrupt_kH = kwargs.get("h", -1)
                corrupt_kW = kwargs.get("w", -1)

            else:
                corrupt_layer = kwargs.get("conv_num", -1)
                corrupt_k = kwargs.get("k", -1)
                corrupt_clip = kwargs.get("clip", -1)
                corrupt_c = kwargs.get("c", -1)
                corrupt_kH = kwargs.get("h", -1)
                corrupt_kW = kwargs.get("w", -1)
                corrupt_value = kwargs.get("value", -1)
        else:
            raise ValueError("Please specify an injection or injection function")

        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)
        gc.collect()
        self.CORRUPTED_MODEL = self.CORRUPTED_MODEL.to(self.device)
        if type(corrupt_layer) == list:
            modules =  self.CORRUPTED_MODEL.named_modules()
            plist = nn.ParameterList()
            for name, module in modules:
                if self.__layer_check(module, name):
                    for name, param in module.named_parameters():
                        # put all parameters in an indexed list
                        if "weight" in name:
                            plist.append(param)
                            #print(param.data)

            iter = len(corrupt_layer)
            #logging.info("injecting {} faults into weights".format(iter))
            #logging.info("###############################################")
            for i in range(iter):
                # 3d conv
                if corrupt_clip[i] >= 0 and corrupt_k[i] >= 0 and corrupt_c[
                    i] >= 0 and self.LAYER_TYPE_CONV3D:
                    corrupt_idx = [corrupt_k[i], corrupt_c[i], corrupt_clip[i], corrupt_kH[i],
                                   corrupt_kW[i]]
                elif corrupt_k[i] >= 0 and corrupt_c[i] >= 0:  # for conv layer
                    corrupt_idx = [corrupt_k[i], corrupt_c[i], corrupt_kH[i], corrupt_kW[i]]
                else:  # for fcc layer
                    corrupt_idx = [corrupt_kH[i], corrupt_kW[i]]

                if self.CORRUPT_CLIP[i] >= 0 and self.CORRUPT_K[i] >= 0 and self.CORRUPT_C[
                    i] >= 0 and self.LAYER_TYPE_CONV3D:
                    corrupt_idx = [self.CORRUPT_K[i], self.CORRUPT_C[i], self.CORRUPT_CLIP[i], self.CORRUPT_H[i],
                                   self.CORRUPT_W[i]]
                elif self.CORRUPT_K[i] >= 0 and self.CORRUPT_C[i] >= 0:  # for conv layer
                    corrupt_idx = [self.CORRUPT_K[i], self.CORRUPT_C[i], self.CORRUPT_CLIP[i], self.CORRUPT_H[i]]
                else:  # for fcc layer
                    corrupt_idx = [self.CORRUPT_H[i], self.CORRUPT_W[i]]

                curr_layer = 0

                corrupt_idx = (
                    tuple(corrupt_idx)
                    if isinstance(corrupt_idx, list)
                    else corrupt_idx
                )
                #logging.info("injecting into index {}".format(corrupt_idx))

                orig_value = plist[corrupt_layer[i]].data[corrupt_idx].item()
                self.ptfi_batch_pointer_curr = self.ptfi_batch_pointer + i
                if CUSTOM_INJECTION:
                    #corrupt_value = CUSTOM_FUNCTION(param.data, corrupt_idx)
                    corrupt_value = CUSTOM_FUNCTION(orig_value)
                # logging.info("changed value: {}\n".format(corrupt_value))
                plist[corrupt_layer[i]].data[corrupt_idx] = corrupt_value
                # logging.info("Weight Injection")
                # logging.info("Layer index: %s" % corrupt_layer)
                # logging.info("Module: %s" % name)
                # logging.info("Original value: %s" % orig_value)
                # logging.info("Injected value: %s" % corrupt_value)
        else:

            # 3d conv
            if corrupt_clip >= 0 and corrupt_k >= 0 and corrupt_c >= 0 and self.LAYER_TYPE_CONV3D:
                corrupt_idx = [corrupt_k, corrupt_c, corrupt_clip, corrupt_kH, corrupt_kW]
            elif corrupt_k >= 0 and corrupt_c >= 0:  # for conv layer
                corrupt_idx = [corrupt_k, corrupt_c, corrupt_kH, corrupt_kW]
            else:  # for fcc layer
                corrupt_idx = [corrupt_kH, corrupt_kW]

            if self.CORRUPT_CLIP >= 0 and self.CORRUPT_K >= 0 and self.CORRUPT_C >= 0 and self.LAYER_TYPE_CONV3D:
                corrupt_idx = [self.CORRUPT_K, self.CORRUPT_C, self.CORRUPT_CLIP, self.CORRUPT_H, self.CORRUPT_W]
            elif self.CORRUPT_K >= 0 and self.CORRUPT_C >= 0:  # for conv layer
                corrupt_idx = [self.CORRUPT_K, self.CORRUPT_C, self.CORRUPT_CLIP, self.CORRUPT_H]
            else:  # for fcc layer
                corrupt_idx = [self.CORRUPT_H, self.CORRUPT_W]

            curr_layer = 0

            modules = self.CORRUPTED_MODEL.named_modules()
            tmp_param_size = []
            for name, module in modules:
                if self.__layer_check(module, name):
                    for name, param in module.named_parameters():
                        #print("param size: {}".format(param.size()))
                        tmp_param_size.append(list(param.size()))
                        if curr_layer == corrupt_layer and "weight" in name:
                            corrupt_idx = (
                                tuple(corrupt_idx)
                                if isinstance(corrupt_idx, list)
                                else corrupt_idx
                            )
                            orig_value = param.data[corrupt_idx].item()
                            self.ptfi_batch_pointer_curr = self.ptfi_batch_pointer
                            if CUSTOM_INJECTION:
                                # corrupt_value = CUSTOM_FUNCTION(param.data, corrupt_idx)
                                corrupt_value = CUSTOM_FUNCTION(orig_value)
                            logging.debug("changed value: {}".format(corrupt_value))
                            param.data[corrupt_idx] = corrupt_value

                            logging.debug("Weight Injection")
                            logging.debug("Layer index: %s" % corrupt_layer)
                            logging.debug("Module: %s" % name)
                            logging.debug("Original value: %s" % orig_value)
                            logging.debug("Injected value: %s" % corrupt_value)

                    curr_layer += 1
            """ print("Relevant param size of corrupted model")
            for tmp in tmp_param_size:
                print("{}".format(tmp)) """
        return self.CORRUPTED_MODEL

    def declare_neuron_fi(self, **kwargs):
        self._fi_state_reset()
        CUSTOM_INJECTION = False
        INJECTION_FUNCTION = False
        self.layer_call_counter=0
        if kwargs:
            self.CORRUPT_CONV = kwargs.get("conv_num", -1)
            self.CORRUPT_BATCH = kwargs.get("batch", -1)
            self.CORRUPT_CLIP = kwargs.get("clip", -1)
            self.CORRUPT_C = kwargs.get("c", -1)
            self.CORRUPT_H = kwargs.get("h", -1)
            self.CORRUPT_W = kwargs.get("w", -1)
            if "function" in kwargs:
                CUSTOM_INJECTION, INJECTION_FUNCTION = True, kwargs.get("function")
                CUSTOM_INJECTION, PRE_HOOK_INJECTION_FUNCTION = True, kwargs.get("pre_hook_function")
            else:
                self.CORRUPT_VALUE = kwargs.get("value", None)

            # logging.info("Declaring Specified Neuron Fault Injector")
            # logging.info("Convolution: %s" % self.CORRUPT_CONV)
            # logging.info("Batch, Clip, C, H, W:")
            # logging.info(
            #     "%s \n %s \n %s \n %s \n %s"
            #     % (
            #         self.CORRUPT_BATCH,
            #         self.CORRUPT_CLIP,
            #         self.CORRUPT_C,
            #         self.CORRUPT_H,
            #         self.CORRUPT_W,
            #     )
            # )
        else:
            raise ValueError("Please specify an injection or injection function")

        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)
        gc.collect()
        corrupt_conv = [self.CORRUPT_CONV] if not isinstance(self.CORRUPT_CONV, list) else self.CORRUPT_CONV
        for name, param in self.CORRUPTED_MODEL.named_modules():
            if self.__layer_check(param, name):
            #if isinstance(param, nn.Conv3d):
                # name, param = param.named_parameters()
                # print("Name: {}, Param: {}".format(name,param))
                param.new_name = name
                if self.model_attr_parsed.fault_injec_hook_type == "input":
                    if self.model_attr_parsed.rnd_value_type == 'bitflip_bounds':
                        hook = PRE_HOOK_INJECTION_FUNCTION if CUSTOM_INJECTION else self._set_value
                        self.HANDLES.append(param.register_forward_pre_hook(hook))
                    else:
                        if len(set(param.input_new_id).intersection(set(corrupt_conv))) > 0:
                            hook = PRE_HOOK_INJECTION_FUNCTION if CUSTOM_INJECTION else self._set_value
                            self.HANDLES.append(param.register_forward_pre_hook(hook))                   
                else:
                    if self.model_attr_parsed.rnd_value_type == 'bitflip_bounds':
                        hook = INJECTION_FUNCTION if CUSTOM_INJECTION else self._set_value
                        self.HANDLES.append(param.register_forward_hook(hook))
                    else:
                        if len(set(param.new_id).intersection(set(corrupt_conv))) > 0:
                            hook = INJECTION_FUNCTION if CUSTOM_INJECTION else self._set_value
                            self.HANDLES.append(param.register_forward_hook(hook))
                #print(id(param))

        return self.CORRUPTED_MODEL

    def assert_inj_bounds_input(self, **kwargs):
        if type(self.CORRUPT_CONV) == list:
            index = kwargs.get("index", -1)
            assert (
                self.CORRUPT_CONV[index] >= 0
                and self.CORRUPT_CONV[index] < self.get_total_conv()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH[index] >= 0
                and self.CORRUPT_BATCH[index] < self._RUNTIME_BATCH_SIZE
            ), "Invalid batch!"
            #print("Index: {}, CORRUPT_C: {}, INPUT_SIZE: {}".format(
            #    index,self.CORRUPT_C[index], self.INPUT_SIZE[self.CORRUPT_CONV[index]][iter]))
            assert (
                self.CORRUPT_C[index] == -1
                or (self.CORRUPT_C[index] >= 0
                and self.CORRUPT_C[index]
                < self.INPUT_SIZE[self.CORRUPT_CONV[index]][1])
            ), "Invalid C!; Index: {}, CORRUPT_C: {}, INPUT_SIZE: {}".format(
                index,self.CORRUPT_C[index], self.INPUT_SIZE[self.CORRUPT_CONV[index]][1])
            if self.LAYER_TYPE_CONV3D:
                assert (
                    self.CORRUPT_CLIP[index] >= 0
                    and self.CORRUPT_CLIP[index]
                    < self.INPUT_SIZE[self.CORRUPT_CONV[index]][2]
                ), "Invalid Clip!"
            assert (
                self.CORRUPT_H[index] >= 0
                and self.CORRUPT_H[index]
                < self.INPUT_SIZE[self.CORRUPT_CONV[index]][-2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W[index] >= 0
                and self.CORRUPT_W[index]
                < self.INPUT_SIZE[self.CORRUPT_CONV[index]][-1]
            ), "Invalid W!"
        else:
            assert (
                self.CORRUPT_CONV >= 0 and self.CORRUPT_CONV < self.get_total_conv()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH >= 0 and self.CORRUPT_BATCH < self._RUNTIME_BATCH_SIZE
            ), "Invalid batch!"
            assert (
                self.CORRUPT_C == -1
                or (self.CORRUPT_C >= 0
                and self.CORRUPT_C < self.INPUT_SIZE[self.CORRUPT_CONV][1])
            ), "Invalid C!"
            if self.LAYER_TYPE_CONV3D:
                assert (
                        self.CORRUPT_CLIP >= 0
                        and self.CORRUPT_CLIP < self.INPUT_SIZE[self.CORRUPT_CONV][2]
                ), "Invalid Clip!"
            assert (
                self.CORRUPT_H >= 0
                and self.CORRUPT_H < self.INPUT_SIZE[self.CORRUPT_CONV][-2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W >= 0
                and self.CORRUPT_W < self.INPUT_SIZE[self.CORRUPT_CONV][-1]
            ), "Invalid W!"
            
    def assert_inj_bounds(self, **kwargs):
        if type(self.CORRUPT_CONV) == list:
            index = kwargs.get("index", -1)
            assert (
                self.CORRUPT_CONV[index] >= 0
                and self.CORRUPT_CONV[index] < self.get_total_conv()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH[index] >= 0
                and self.CORRUPT_BATCH[index] < self._RUNTIME_BATCH_SIZE
            ), "Invalid batch!"
            #print("Index: {}, CORRUPT_C: {}, OUTPUT_SIZE: {}".format(
            #    index,self.CORRUPT_C[index], self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][iter]))
            assert (
                self.CORRUPT_C[index] == -1
                or (self.CORRUPT_C[index] >= 0
                and self.CORRUPT_C[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][1])
            ), "Invalid C!; Index: {}, CORRUPT_C: {}, OUTPUT_SIZE: {}".format(
                index,self.CORRUPT_C[index], self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][1])
            if self.LAYER_TYPE_CONV3D:
                assert (
                    self.CORRUPT_CLIP[index] >= 0
                    and self.CORRUPT_CLIP[index]
                    < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][2]
                ), "Invalid Clip!"
            assert (
                self.CORRUPT_H[index] >= 0
                and self.CORRUPT_H[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][-2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W[index] >= 0
                and self.CORRUPT_W[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][-1]
            ), "Invalid W!"
        else:
            assert (
                self.CORRUPT_CONV >= 0 and self.CORRUPT_CONV < self.get_total_conv()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH >= 0 and self.CORRUPT_BATCH < self._RUNTIME_BATCH_SIZE
            ), "Invalid batch!"
            assert (
                self.CORRUPT_C == -1
                or (self.CORRUPT_C >= 0
                and self.CORRUPT_C < self.OUTPUT_SIZE[self.CORRUPT_CONV][1])
            ), "Invalid C!"
            if self.LAYER_TYPE_CONV3D:
                assert (
                        self.CORRUPT_CLIP >= 0
                        and self.CORRUPT_CLIP < self.OUTPUT_SIZE[self.CORRUPT_CONV][2]
                ), "Invalid Clip!"
            assert (
                self.CORRUPT_H >= 0
                and self.CORRUPT_H < self.OUTPUT_SIZE[self.CORRUPT_CONV][-2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W >= 0
                and self.CORRUPT_W < self.OUTPUT_SIZE[self.CORRUPT_CONV][-1]
            ), "Invalid W!"

    def _set_value(self, module, input, output):
        if type(self.CORRUPT_CONV) == list:
            inj_list = list(
                filter(
                    lambda x: self.CORRUPT_CONV[x] == self.get_curr_conv(),
                    range(len(self.CORRUPT_CONV)),
                )
            )
            # print('inj_list', inj_list, self.CORRUPT_CONV, self.get_curr_conv()) #TODO:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                if max(self.CORRUPT_CLIP) >= 0:
                    logging.info(
                        "Original value at [%d][%d][%d][%d][%d]: %f"
                        % (
                            self.CORRUPT_BATCH[i],
                            self.CORRUPT_CLIP[i],
                            self.CORRUPT_C[i],
                            self.CORRUPT_H[i],
                            self.CORRUPT_W[i],
                            output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                                self.CORRUPT_H[i]
                            ][self.CORRUPT_W[i]]
                        )
                    )
                else:
                    logging.info(
                        "Original value at [%d][%d][%d][%d]: %f"
                        % (
                            self.CORRUPT_BATCH[i],
                            self.CORRUPT_C[i],
                            self.CORRUPT_H[i],
                            self.CORRUPT_W[i],
                            output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                                self.CORRUPT_H[i]
                            ][self.CORRUPT_W[i]]
                        )
                    )
                # TODO verify for fcc layer
                logging.info("Changing value to %d" % self.CORRUPT_VALUE[i])
                val = torch.tensor(float(self.CORRUPT_VALUE[i]))
                if max(self.CORRUPT_C) >= 0:
                    if max(self.CORRUPT_CLIP) >= 0: # 3dconv layer
                        # print(list(output.size()))
                        # print("{}".format(self.CORRUPT_BATCH))
                        real_batch =list(output.size())[0]
                        if self.CORRUPT_BATCH[i] >= real_batch:
                            self.CORRUPT_BATCH[i] = real_batch - 1
                        output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_CLIP[i]][self.CORRUPT_H[i]]
                        [self.CORRUPT_W[i]] = val
                    else: #2DConv layer
                        output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_H[i]][
                            self.CORRUPT_W[i] 
                        ] = val
                else:
                    output[self.CORRUPT_H[i]][self.CORRUPT_W[i]] = val
        else:
            self.assert_inj_bounds()
            if self.get_curr_conv() == self.CORRUPT_CONV:
                if self.CORRUPT_CLIP >= 0:
                    logging.info(
                        "Original value at [%d][%d][%d][%d][%d]: %f"
                        % (
                            self.CORRUPT_BATCH,
                            self.CORRUPT_CLIP,
                            self.CORRUPT_C,
                            self.CORRUPT_H,
                            self.CORRUPT_W,
                            float(output[self.CORRUPT_BATCH][self.CORRUPT_CLIP][self.CORRUPT_C][self.CORRUPT_H][
                                self.CORRUPT_W
                            ])
                        )
                    )
                else:
                    logging.info(
                        "Original value at [%d][%d][%d][%d]: %f"
                        % (
                            self.CORRUPT_BATCH,
                            self.CORRUPT_C,
                            self.CORRUPT_H,
                            self.CORRUPT_W,
                            float(output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                                self.CORRUPT_W
                            ])
                        )
                    )
                logging.info("Changing value to %d" % self.CORRUPT_VALUE)
                d_type = self.CORRUPT_VALUE.dtype
                val = torch.tensor(float(self.CORRUPT_VALUE))
                if self.CORRUPT_CLIP >= 0 and self.CORRUPT_C >= 0 and self.LAYER_TYPE_CONV3D:
                    output[self.CORRUPT_BATCH][self.CORRUPT_CLIP][self.CORRUPT_C][self.CORRUPT_H][
                        self.CORRUPT_W
                    ] = val
                elif self.CORRUPT_C >= 0:
                    output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                        self.CORRUPT_W
                    ] = val
                else: # todo verify
                    output[self.CORRUPT_H][self.CORRUPT_W] = val
        #logging.debug("calling UpdateConv")
        self.updateConv()

    def _save_output_size(self, module, input, output):
        #print("Output size: {}".format(output.size()))
        # list_pa = list(net.named_parameters())
        # list_weights_ranger = [list_pa[i][0] for i in range(len(list_pa))]
        # dict_vgg_ranger = dict(zip(list_weights_ranger, list(dict_vgg.values())))
        check_list = [11, 13, 16, 20, 13]
        name=module.new_name
        module.new_id.append(self._named_id)
        layer_size = list(output.size())
        if self._RUNTIME_BATCH_SIZE < layer_size[0]:
            self._RUNTIME_BATCH_SIZE = layer_size[0]
        self.OUTPUT_SIZE.append(layer_size)
        self.OUTPUT_LOOKUP.append(name)
        self._named_id +=1

    def _save_input_size(self, module, input):
        #print("Output size: {}".format(output.size()))
        # list_pa = list(net.named_parameters())
        # list_weights_ranger = [list_pa[i][0] for i in range(len(list_pa))]
        # dict_vgg_ranger = dict(zip(list_weights_ranger, list(dict_vgg.values())))
        name=module.new_name
        module.input_new_id.append(self._input_named_id)
        layer_size = list(input[0].size())
        if self._RUNTIME_BATCH_SIZE < layer_size[0]:
            self._RUNTIME_BATCH_SIZE = layer_size[0]
        self.INPUT_SIZE.append(layer_size)
        self.INPUT_LOOKUP.append(name)
        self._input_named_id +=1
        
    def _save_corrupt_size(self, module, input, output):
        #print("Output size: {}".format(output.size()))
        self.CORRUPT_SIZE.append(list(output.size()))

    def _save_corrupt_input_size(self, module, input):
        #print("Output size: {}".format(output.size()))
        self.CORRUPT_INPUT_SIZE.append(list(input[0].size()))

    def get_layer_neuron_weights(self):
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            return self.INPUT_LAYER_NEURON_WEIGHTS
        else:
            return self.LAYER_NEURON_WEIGHTS

    def get_layer_weight_weights(self):
        return self.LAYER_WEIGHT_WEIGHTS

    def get_original_model(self):
        return self.ORIG_MODEL

    def get_corrupted_model(self):
        return self.CORRUPTED_MODEL
    
    def get_output_size(self):
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            return self.INPUT_SIZE
        else:
            return self.OUTPUT_SIZE

    def updateConv(self, value=1):
        self.CURRENT_CONV += value

    def reset_curr_conv(self):
        self.CURRENT_CONV = 0

    def set_corrupt_conv(self, value):
        self.CORRUPT_CONV = value
#        self.CORRUPT_CONV_NAME = self.OUTPUT_LOOKUP[value]

    def get_curr_conv(self):
        return self.CURRENT_CONV

    def get_corrupt_conv(self):
        return self.CORRUPT_CONV

    def get_total_batches(self):
        return self._RUNTIME_BATCH_SIZE
    
    def get_random_layer_selected(self):
        return self.RANDOM_LAYER_SELECTED_NUM
        
    def get_total_conv(self):
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            return len(self.INPUT_SIZE)
        else:
            return len(self.OUTPUT_SIZE)

    def get_fmaps_num(self, layer):
        # todo check what happens for fcc layer
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            if len(self.INPUT_SIZE[layer]) >= 4:
                return self.INPUT_SIZE[layer][1]
            else:
                return int(-1)
        else:
            if len(self.OUTPUT_SIZE[layer]) >= 4:
                return self.OUTPUT_SIZE[layer][1]
            else:
                return int(-1)

    def get_neuron_num(self):
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            return self.INPUT_NEURON_NUM
        else:
            return self.NEURON_NUM

    def get_weight_num(self):
        return self.WEIGHT_NUM
        
    def get_fmaps_D(self, layer):
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            if len(self.INPUT_SIZE[layer]) == 5:
                return self.INPUT_SIZE[layer][2]
            else:
                return int(-1)        
        else:
            if len(self.OUTPUT_SIZE[layer]) == 5:
                return self.OUTPUT_SIZE[layer][2]
            else:
                return int(-1)

    def get_fmaps_H(self, layer):
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            return self.INPUT_SIZE[layer][-2]
        else:
            return self.OUTPUT_SIZE[layer][-2]

    def get_fmaps_W(self, layer):
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            return self.INPUT_SIZE[layer][-1]
        else:
            return self.OUTPUT_SIZE[layer][-1]

    def get_fmap_HW(self, layer):
        if self.model_attr_parsed.fault_injec_hook_type == "input":
            return (self.get_input_fmaps_H(layer), self.get_input_fmaps_W(layer))
        else:
            return (self.get_fmaps_H(layer), self.get_fmaps_W(layer))


class fault_injection_classic:
    def __init__(self, model, h, w, batch_size, **kwargs):
        # logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")
        self.ORIG_MODEL = None
        self.CORRUPTED_MODEL = None
        self._BATCH_SIZE = -1
        self._RUNTIME_BATCH_SIZE = -1
        self.device = kwargs.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.classic = kwargs.get("classic", False)
        self.CUSTOM_INJECTION = False
        self.INJECTION_FUNCTION = None

        self.CORRUPT_BATCH = -1
        self.CORRUPT_CONV = -1
        self.CORRUPT_CONV_NAME = ""
        self.CORRUPT_CLIP = -1
        self.CORRUPT_C = -1
        self.CORRUPT_H = -1
        self.CORRUPT_W = -1
        self.CORRUPT_VALUE = None

        self.CURRENT_CONV = 0
        self.OUTPUT_SIZE = []
        self.NEURON_NUM = 0
        self.WEIGHT_NUM = 0
        self.LAYER_NEURON_WEIGHTS = []
        self.LAYER_WEIGHT_WEIGHTS = []
        self.OUTPUT_LOOKUP = []
        self.CORRUPT_SIZE = []
        self.HANDLES = []

        self.imageC = kwargs.get("c", 3)
        self.imageClip = kwargs.get("clip", -1)
        self.imageH = h
        self.imageW = w

        self.LAYER_TYPE_CONV2D = False
        self.LAYER_TYPE_FCC = False
        self.LAYER_TYPE_CONV3D = False
        self.LAYER_TYPES = kwargs.get("layer_types",("conv2d"))
        if "conv2d" in self.LAYER_TYPES:
            self.LAYER_TYPE_CONV2D = True
        if "conv3d" in self.LAYER_TYPES:
            self.LAYER_TYPE_CONV3D = True
        if "fcc" in self.LAYER_TYPES:
            self.LAYER_TYPE_FCC = True

        self.LAYER_TYPES = ["attn_fcc"]
        self.LAYER_TYPES_NAMED = ["attn"]
        self.LAYER_TYPE_ATT = False
        self.LAYER_TYPE_ATT_EMBED = False
        self.LAYER_TYPES_EXCLUDE = None

        if "attn_fcc" in self.LAYER_TYPES:
            self.LAYER_TYPE_ATT = True
        if "embedding" in self.LAYER_TYPES:
            self.LAYER_TYPE_ATT_EMBED = True
            
        self.use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)
        model_dtype = next(model.parameters()).dtype

        self.ORIG_MODEL = model
        self.ORIG_MODEL = self.ORIG_MODEL.to(self.device)
        self._BATCH_SIZE = batch_size
        input_num =  kwargs.get("input_num", 1)
        self._INPUT_NUM = input_num

        handles = []
        self._named_id = 0
        k = 0
        layer_counter = 0
        self.layer_counter_list = []
        for layer_id, (module_name, module) in enumerate(self.get_original_model().named_modules()):
            layer_check=False
            if self.classic:
                if self.__layer_check_classic(module):
                    layer_check=True
            else:
                if self.__layer_check(module, module_name):
                    layer_check=True
            if layer_check:
                # print(name, param) #TODO:
                module.new_name = module_name
                module.new_id = []
                handles.append(module.register_forward_hook(self._save_output_size))
                k += 1
                layer_info = [layer_id, layer_counter, module_name]
                layer_counter+=1
                self.layer_counter_list.append(layer_info)
                # print(name, param)
                
        # device = "cuda" if self.use_cuda else None
        #device = torch.device("cuda:{}".format(self.cuda_device) if torch.cuda.is_available() else "cpu")
        # support for 3d conv
        # this seems to trigger the call of _save_output_size
        if not self.imageClip == -1:
            _dummyTensor = torch.randn(
                self._BATCH_SIZE, self.imageC, self.imageClip, self.imageH, self.imageW, dtype=model_dtype, device=self.device
            )
        else:
            _dummyTensor = torch.randn(
                max(self._BATCH_SIZE,1), max(self.imageC,1), max(self.imageH,1), max(self.imageW,1), dtype=model_dtype, device=self.device
            )
        # TODO this is not flexible at all, need to find a better method

        if self._INPUT_NUM == 3:
            self.ORIG_MODEL(input=[_dummyTensor, _dummyTensor, _dummyTensor], dummy=True)
        elif self._INPUT_NUM == 2:
            self.ORIG_MODEL(input=[_dummyTensor, _dummyTensor], dummy=True)
        else:
            self.ORIG_MODEL(input=_dummyTensor, dummy=True)

        for i in range(len(handles)):
            handles[i].remove()

        # store weight size once
        self.WEIGHT_NUM, self.LAYER_WEIGHT_WEIGHTS = self.__get_weight_size()

        # store neuron size only once
        self.NEURON_NUM = np.sum([np.prod(line) for line in self.OUTPUT_SIZE])
        self.LAYER_NEURON_WEIGHTS= [np.prod(line)/self.NEURON_NUM for line in self.OUTPUT_SIZE]
        # i=0
        # logging.info("Model output size")
        # for row in self.OUTPUT_SIZE:
        #     logging.info("{} {}".format(i,row))
        #     i += 1

    def __layer_check(self, param, name):
        ret = False
        if self.LAYER_TYPES_NAMED:
            check_layer_types_named = any([layer_types_named in name for layer_types_named in self.LAYER_TYPES_NAMED])
            ## filtering NonDynamicallyQuantizableLinear layers:
            check_layer_types_named = check_layer_types_named and not('out_proj' in name)
            # check_layer_types_named = check_layer_types_named
            if isinstance(param, nn.Linear) and self.LAYER_TYPE_ATT and check_layer_types_named:
                ret = True
            if isinstance(param, nn.modules.sparse.Embedding) and self.LAYER_TYPE_ATT_EMBED and check_layer_types_named:
                ret = True
            if isinstance(param, nn.Linear) and self.LAYER_TYPE_FCC and check_layer_types_named:
                ret = True
        else:
            if isinstance(param, nn.Conv2d) and self.LAYER_TYPE_CONV2D:
                ret = True
            elif isinstance(param, nn.Conv3d) and self.LAYER_TYPE_CONV3D:
                ret = True
            elif isinstance(param, nn.Linear) and self.LAYER_TYPE_FCC:
                ret = True
            elif isinstance(param, nn.modules.activation.LeakyReLU) and self.LAYER_TYPE_LeakyRelu:
                ret = True
            elif isinstance(param, nn.modules.batchnorm.BatchNorm2d) and self.LAYER_TYPE_BatchNorm:
                ret = True
        if self.LAYER_TYPES_EXCLUDE:
            check_layer_types_exclude = any([layer_types_exclude in name for layer_types_exclude in self.LAYER_TYPES_EXCLUDE])
            if check_layer_types_exclude:
                ret = False
        return ret
    
    def __layer_check_classic(self,param):
        ret = False

        if isinstance(param, nn.Conv2d) and self.LAYER_TYPE_CONV2D:
            ret = True
        if isinstance(param, nn.Conv3d) and self.LAYER_TYPE_CONV3D:
            ret = True
        if isinstance(param, nn.Linear) and self.LAYER_TYPE_FCC:
            ret = True
        return ret
    
    def __get_weight_size(self):
        total_size = 0
        size_arr = []
        cnt = 0
        cnt2 = 0
        for module_name, module in self.get_original_model().named_modules():
            if self.__layer_check(module, module_name):
                for name, param in module.named_parameters():
                    # print(module, name, param.numel())
                    cnt += 1
                    if "weight" in name:
                        # print(module, name, param.numel())
                        cnt2 += 1
                        x = param.numel()
                        total_size += x
                        size_arr.append(x)
        for i in range(len(size_arr)):
            size_arr[i] = size_arr[i] / total_size

        # print(cnt, cnt2) 
        return total_size, size_arr

    def fi_reset(self):
        self._fi_state_reset()
        self.CORRUPTED_MODEL = None
        logging.info("Reset fault injector")

    def _fi_state_reset(self):
        (
            self.CURRENT_CONV,
            self.CORRUPT_BATCH,
            self.CORRUPT_CONV,
            self.CORRUPT_CLIP,
            self.CORRUPT_C,
            self.CORRUPT_H,
            self.CORRUPT_W,
            self.CORRUPT_VALUE,
        ) = (
            0,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            None,
        )

        for i in range(len(self.HANDLES)):
            self.HANDLES[i].remove()
        return None

    def declare_weight_fi(self, **kwargs):
        CUSTOM_INJECTION = False
        CUSTOM_FUNCTION = False
        self._fi_state_reset()
        corrupt_value = 0
        # TODO add dimension for 3d conv

        if kwargs:
            if "function" in kwargs:
                CUSTOM_INJECTION, CUSTOM_FUNCTION = True, kwargs.get("function")
                corrupt_layer = kwargs.get("conv_num", -1)
                corrupt_k = kwargs.get("k", -1)
                corrupt_c = kwargs.get("c", -1)
                corrupt_clip = kwargs.get("clip", -1)
                corrupt_kH = kwargs.get("h", -1)
                corrupt_kW = kwargs.get("w", -1)
                corrupt_bit = kwargs.get("bit", -1)

            else:
                corrupt_layer = kwargs.get("conv_num", -1)
                corrupt_k = kwargs.get("k", -1)
                corrupt_clip = kwargs.get("clip", -1)
                corrupt_c = kwargs.get("c", -1)
                corrupt_kH = kwargs.get("h", -1)
                corrupt_kW = kwargs.get("w", -1)
                corrupt_value = kwargs.get("value", -1)
        else:
            raise ValueError("Please specify an injection or injection function")

        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)
        gc.collect()
        self.CORRUPTED_MODEL = self.CORRUPTED_MODEL.to(self.device)
        if type(corrupt_layer) == list:
            modules =  self.CORRUPTED_MODEL.named_modules()
            plist = nn.ParameterList()
            for module_name, module in modules:
                layer_check=False
                if self.classic:
                    if self.__layer_check_classic(module):
                        layer_check=True
                else:
                    if self.__layer_check(module, module_name):
                        layer_check=True
                if layer_check:
                    for name, param in module.named_parameters():
                        # put all parameters in an indexed list
                        if "weight" in name:
                            plist.append(param)
                            #print(param.data)    
            iter = len(corrupt_layer)
            #logging.info("injecting {} faults into weights".format(iter))
            #logging.info("###############################################")
            for i in range(iter):
                # 3d conv
                if corrupt_clip[i] >= 0 and corrupt_k[i] >= 0 and corrupt_c[
                    i] >= 0 and self.LAYER_TYPE_CONV3D:
                    corrupt_idx = [corrupt_k[i], corrupt_c[i], corrupt_clip[i], corrupt_kH[i],
                                   corrupt_kW[i]]
                elif corrupt_k[i] >= 0 and corrupt_c[i] >= 0:  # for conv layer
                    corrupt_idx = [corrupt_k[i], corrupt_c[i], corrupt_kH[i], corrupt_kW[i]]
                else:  # for fcc layer
                    corrupt_idx = [corrupt_kH[i], corrupt_kW[i]]
                curr_layer = 0

                corrupt_idx = (
                    tuple(corrupt_idx)
                    if isinstance(corrupt_idx, list)
                    else corrupt_idx
                )
                #logging.info("injecting into index {}".format(corrupt_idx))

                orig_value = plist[corrupt_layer[i]].data[corrupt_idx].item()

                if CUSTOM_INJECTION:
                    #corrupt_value = CUSTOM_FUNCTION(param.data, corrupt_idx)
                    corrupt_value = CUSTOM_FUNCTION(orig_value, corrupt_bit)
                # logging.info("changed value: {}\n".format(corrupt_value))
                plist[corrupt_layer[i]].data[corrupt_idx] = corrupt_value
                # logging.info("Weight Injection")
                # logging.info("Layer index: %s" % corrupt_layer)
                # logging.info("Module: %s" % name)
                # logging.info("Original value: %s" % orig_value)
                # logging.info("Injected value: %s" % corrupt_value)
        else:
            # 3d conv
            if corrupt_clip >= 0 and corrupt_k >= 0 and corrupt_c >= 0 and self.LAYER_TYPE_CONV3D:
                corrupt_idx = [corrupt_k, corrupt_c, corrupt_clip, corrupt_kH, corrupt_kW]
            elif corrupt_k >= 0 and corrupt_c >= 0:  # for conv layer
                corrupt_idx = [corrupt_k, corrupt_c, corrupt_kH, corrupt_kW]
            else:  # for fcc layer
                corrupt_idx = [corrupt_kH, corrupt_kW]
            curr_layer = 0

            modules = self.CORRUPTED_MODEL.named_modules()
            tmp_param_size = []
            for module_name, module in modules:
                layer_check=False
                if self.classic:
                    if self.__layer_check_classic(module):
                        layer_check=True
                else:
                    if self.__layer_check(module, module_name):
                        layer_check=True
                if layer_check:
                    for name, param in module.named_parameters():
                        #print("param size: {}".format(param.size()))
                        tmp_param_size.append(list(param.size()))
                        if curr_layer == corrupt_layer and "weight" in name:
                            corrupt_idx = (
                                tuple(corrupt_idx)
                                if isinstance(corrupt_idx, list)
                                else corrupt_idx
                            )
                            orig_value = param.data[corrupt_idx]
                            if CUSTOM_INJECTION:
                                # corrupt_value = CUSTOM_FUNCTION(param.data, corrupt_idx)
                                corrupt_value = CUSTOM_FUNCTION(orig_value, corrupt_bit)
                            logging.debug("changed value: {}".format(corrupt_value))
                            param.data[corrupt_idx] = corrupt_value

                            logging.debug("Weight Injection")
                            logging.debug("Layer index: %s" % corrupt_layer)
                            logging.debug("Module: %s" % name)
                            logging.debug("Original value: %s" % orig_value)
                            logging.debug("Injected value: %s" % corrupt_value)

                    curr_layer += 1
            """ print("Relevant param size of corrupted model")
            for tmp in tmp_param_size:
                print("{}".format(tmp)) """
        return self.CORRUPTED_MODEL

    def declare_neuron_fi(self, **kwargs):
        self._fi_state_reset()
        CUSTOM_INJECTION = False
        INJECTION_FUNCTION = False

        if kwargs:
            if "function" in kwargs:
                CUSTOM_INJECTION, INJECTION_FUNCTION = True, kwargs.get("function")
                self.CORRUPT_CONV = kwargs.get("conv_num", -1)
                self.CORRUPT_BATCH = kwargs.get("batch", -1)
                self.CORRUPT_CLIP = kwargs.get("clip", -1)
                self.CORRUPT_C = kwargs.get("c", -1)
                self.CORRUPT_H = kwargs.get("h", -1)
                self.CORRUPT_W = kwargs.get("w", -1)
            else:
                self.CORRUPT_CONV = kwargs.get("conv_num", -1)
                self.CORRUPT_BATCH = kwargs.get("batch", -1)
                self.CORRUPT_CLIP = kwargs.get("clip", -1)
                self.CORRUPT_C = kwargs.get("c", -1)
                self.CORRUPT_H = kwargs.get("h", -1)
                self.CORRUPT_W = kwargs.get("w", -1)
                self.CORRUPT_VALUE = kwargs.get("value", None)

            # logging.info("Declaring Specified Neuron Fault Injector")
            # logging.info("Convolution: %s" % self.CORRUPT_CONV)
            # logging.info("Batch, Clip, C, H, W:")
            # logging.info(
            #     "%s \n %s \n %s \n %s \n %s"
            #     % (
            #         self.CORRUPT_BATCH,
            #         self.CORRUPT_CLIP,
            #         self.CORRUPT_C,
            #         self.CORRUPT_H,
            #         self.CORRUPT_W,
            #     )
            # )
        else:
            raise ValueError("Please specify an injection or injection function")

        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)
        gc.collect()
        corrupt_conv = [self.CORRUPT_CONV] if not isinstance(self.CORRUPT_CONV, list) else self.CORRUPT_CONV
        for name, param in self.CORRUPTED_MODEL.named_modules():
            if self.__layer_check(param):
            #if isinstance(param, nn.Conv3d):
                # name, param = param.named_parameters()
                # print("Name: {}, Param: {}".format(name,param))
                param.new_name = name
                if len(set(param.new_id).intersection(set(corrupt_conv))) > 0:
                    hook = INJECTION_FUNCTION if CUSTOM_INJECTION else self._set_value
                    self.HANDLES.append(param.register_forward_hook(hook))
                #print(id(param))

        return self.CORRUPTED_MODEL

    def assert_inj_bounds(self, **kwargs):
        if type(self.CORRUPT_CONV) == list:
            index = kwargs.get("index", -1)
            assert (
                self.CORRUPT_CONV[index] >= 0
                and self.CORRUPT_CONV[index] < self.get_total_conv()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH[index] >= 0
                and self.CORRUPT_BATCH[index] < self._RUNTIME_BATCH_SIZE
            ), "Invalid batch!"
            #print("Index: {}, CORRUPT_C: {}, OUTPUT_SIZE: {}".format(
            #    index,self.CORRUPT_C[index], self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][iter]))
            assert (
                self.CORRUPT_C[index] == -1
                or (self.CORRUPT_C[index] >= 0
                and self.CORRUPT_C[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][1])
            ), "Invalid C!; Index: {}, CORRUPT_C: {}, OUTPUT_SIZE: {}".format(
                index,self.CORRUPT_C[index], self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][1])
            if self.LAYER_TYPE_CONV3D:
                assert (
                    self.CORRUPT_CLIP[index] >= 0
                    and self.CORRUPT_CLIP[index]
                    < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][2]
                ), "Invalid Clip!"
            assert (
                self.CORRUPT_H[index] >= 0
                and self.CORRUPT_H[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][-2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W[index] >= 0
                and self.CORRUPT_W[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][-1]
            ), "Invalid W!"
        else:
            assert (
                self.CORRUPT_CONV >= 0 and self.CORRUPT_CONV < self.get_total_conv()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH >= 0 and self.CORRUPT_BATCH < self._RUNTIME_BATCH_SIZE
            ), "Invalid batch!"
            assert (
                self.CORRUPT_C == -1
                or (self.CORRUPT_C >= 0
                and self.CORRUPT_C < self.OUTPUT_SIZE[self.CORRUPT_CONV][1])
            ), "Invalid C!"
            if self.LAYER_TYPE_CONV3D:
                assert (
                        self.CORRUPT_CLIP >= 0
                        and self.CORRUPT_CLIP < self.OUTPUT_SIZE[self.CORRUPT_CONV][2]
                ), "Invalid Clip!"
            assert (
                self.CORRUPT_H >= 0
                and self.CORRUPT_H < self.OUTPUT_SIZE[self.CORRUPT_CONV][-2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W >= 0
                and self.CORRUPT_W < self.OUTPUT_SIZE[self.CORRUPT_CONV][-1]
            ), "Invalid W!"

    def _set_value(self, module, input, output):
        if type(self.CORRUPT_CONV) == list:
            inj_list = list(
                filter(
                    lambda x: self.CORRUPT_CONV[x] == self.get_curr_conv(),
                    range(len(self.CORRUPT_CONV)),
                )
            )
            # print('inj_list', inj_list, self.CORRUPT_CONV, self.get_curr_conv()) #TODO:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                if max(self.CORRUPT_CLIP) >= 0:
                    logging.info(
                        "Original value at [%d][%d][%d][%d][%d]: %f"
                        % (
                            self.CORRUPT_BATCH[i],
                            self.CORRUPT_CLIP[i],
                            self.CORRUPT_C[i],
                            self.CORRUPT_H[i],
                            self.CORRUPT_W[i],
                            output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                                self.CORRUPT_H[i]
                            ][self.CORRUPT_W[i]]
                        )
                    )
                else:
                    logging.info(
                        "Original value at [%d][%d][%d][%d]: %f"
                        % (
                            self.CORRUPT_BATCH[i],
                            self.CORRUPT_C[i],
                            self.CORRUPT_H[i],
                            self.CORRUPT_W[i],
                            output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                                self.CORRUPT_H[i]
                            ][self.CORRUPT_W[i]]
                        )
                    )
                # TODO verify for fcc layer
                logging.info("Changing value to %d" % self.CORRUPT_VALUE[i])
                val = torch.tensor(float(self.CORRUPT_VALUE[i]))
                if max(self.CORRUPT_C) >= 0:
                    if max(self.CORRUPT_CLIP) >= 0: # 3dconv layer
                        # print(list(output.size()))
                        # print("{}".format(self.CORRUPT_BATCH))
                        real_batch =list(output.size())[0]
                        if self.CORRUPT_BATCH[i] >= real_batch:
                            self.CORRUPT_BATCH[i] = real_batch - 1
                        output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_CLIP[i]][self.CORRUPT_H[i]]
                        [self.CORRUPT_W[i]] = val
                    else: #2DConv layer
                        output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_H[i]][
                            self.CORRUPT_W[i] 
                        ] = val
                else:
                    output[self.CORRUPT_H[i]][self.CORRUPT_W[i]] = val
        else:
            self.assert_inj_bounds()
            if self.get_curr_conv() == self.CORRUPT_CONV:
                if self.CORRUPT_CLIP >= 0:
                    logging.info(
                        "Original value at [%d][%d][%d][%d][%d]: %f"
                        % (
                            self.CORRUPT_BATCH,
                            self.CORRUPT_CLIP,
                            self.CORRUPT_C,
                            self.CORRUPT_H,
                            self.CORRUPT_W,
                            float(output[self.CORRUPT_BATCH][self.CORRUPT_CLIP][self.CORRUPT_C][self.CORRUPT_H][
                                self.CORRUPT_W
                            ])
                        )
                    )
                else:
                    logging.info(
                        "Original value at [%d][%d][%d][%d]: %f"
                        % (
                            self.CORRUPT_BATCH,
                            self.CORRUPT_C,
                            self.CORRUPT_H,
                            self.CORRUPT_W,
                            float(output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                                self.CORRUPT_W
                            ])
                        )
                    )
                logging.info("Changing value to %d" % self.CORRUPT_VALUE)
                d_type = self.CORRUPT_VALUE.dtype
                val = torch.tensor(float(self.CORRUPT_VALUE))
                if self.CORRUPT_CLIP >= 0 and self.CORRUPT_C >= 0 and self.LAYER_TYPE_CONV3D:
                    output[self.CORRUPT_BATCH][self.CORRUPT_CLIP][self.CORRUPT_C][self.CORRUPT_H][
                        self.CORRUPT_W
                    ] = val
                elif self.CORRUPT_C >= 0:
                    output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                        self.CORRUPT_W
                    ] = val
                else: # todo verify
                    output[self.CORRUPT_H][self.CORRUPT_W] = val
        #logging.debug("calling UpdateConv")
        self.updateConv()

    def _save_output_size(self, module, input, output):
        #print("Output size: {}".format(output.size()))
        # list_pa = list(net.named_parameters())
        # list_weights_ranger = [list_pa[i][0] for i in range(len(list_pa))]
        # dict_vgg_ranger = dict(zip(list_weights_ranger, list(dict_vgg.values())))
        name=module.new_name
        module.new_id.append(self._named_id)
        layer_size = list(output.size())
        if self._RUNTIME_BATCH_SIZE < layer_size[0]:
            self._RUNTIME_BATCH_SIZE = layer_size[0]
        self.OUTPUT_SIZE.append(layer_size)
        self.OUTPUT_LOOKUP.append(name)
        self._named_id +=1

    def _save_corrupt_size(self, module, input, output):
        #print("Output size: {}".format(output.size()))
        self.CORRUPT_SIZE.append(list(output.size()))

    def get_layer_neuron_weights(self):
        return self.LAYER_NEURON_WEIGHTS

    def get_layer_weight_weights(self):
        return self.LAYER_WEIGHT_WEIGHTS

    def get_original_model(self):
        return self.ORIG_MODEL

    def get_corrupted_model(self):
        return self.CORRUPTED_MODEL

    def get_output_size(self):
        return self.OUTPUT_SIZE

    def updateConv(self, value=1):
        self.CURRENT_CONV += value

    def reset_curr_conv(self):
        self.CURRENT_CONV = 0

    def set_corrupt_conv(self, value):
        self.CORRUPT_CONV = value
#        self.CORRUPT_CONV_NAME = self.OUTPUT_LOOKUP[value]

    def get_curr_conv(self):
        return self.CURRENT_CONV

    def get_corrupt_conv(self):
        return self.CORRUPT_CONV

    def get_total_batches(self):
        return self._RUNTIME_BATCH_SIZE

    def get_total_conv(self):
        return len(self.OUTPUT_SIZE)

        # # Tests:
        # if len(self.OUTPUT_LOOKUP) > len(np.unique(self.OUTPUT_LOOKUP)):
        #     print('Architecture seems to reuse layers during a single forward pass, respective layers are only counted once for fault injection.')


        # to_elim_ind = []
        # to_elim_names = []
        # for n, it in enumerate(self.OUTPUT_LOOKUP):
        #     # lay = self.OUTPUT_LOOKUP[n] #name of layer = it
        #     match_bool = [x==it for x in self.OUTPUT_LOOKUP]
        #     inds = np.arange(len(self.OUTPUT_LOOKUP))[match_bool] #indices where to find them
        #     inds_notself = inds[inds != n] #other indices then n
        #     print('check', n, it, inds, inds_notself)
        #     if inds_notself.any() and it not in to_elim_names:
        #         print("added")
        #         to_elim_ind.append([n, inds_notself])
        #         to_elim_names.append(it)
        
        # return len(np.unique(self.OUTPUT_LOOKUP))

    def get_fmaps_num(self, layer):
        # todo check what happens for fcc layer
        if len(self.OUTPUT_SIZE[layer]) >= 4:
            return self.OUTPUT_SIZE[layer][1]
        else:
            return int(-1)

    def get_neuron_num(self):
        return self.NEURON_NUM

    def get_weight_num(self):
        return self.WEIGHT_NUM

    def get_fmaps_D(self, layer):
        if len(self.OUTPUT_SIZE[layer]) == 5:
            return self.OUTPUT_SIZE[layer][2]
        else:
            return int(-1)

    def get_fmaps_H(self, layer):
        return self.OUTPUT_SIZE[layer][-2]


    def get_fmaps_W(self, layer):
        return self.OUTPUT_SIZE[layer][-1]

    def get_fmap_HW(self, layer):
        return (self.get_fmaps_H(layer), self.get_fmaps_W(layer))