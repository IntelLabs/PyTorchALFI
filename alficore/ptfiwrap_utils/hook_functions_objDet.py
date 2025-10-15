# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT
import torch
from alficore.resiliency_methods.ranger import Ranger, Ranger_trivial, Ranger_BackFlip, Ranger_Clip, Ranger_FmapAvg, Ranger_FmapRescale
from alficore.resiliency_methods.ranger_automation import My_Bottleneck_Ranger
import numpy as np
from tabulate import tabulate

# For tutorial on hooks see
#  https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

class SaveOutput:
    def __init__(self):
        self.layer_id = 0
        self.outputs = []
        self.module_names = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        self.module_names.append(module.__dict__['layer_name'])
        self.layer_id+=1

    def clear(self):
        self.layer_id = 0
        self.outputs = []
        self.module_names = []
class OutputHook:
    def __init__(self, **kwargs):
        self.outputs = []
        self.lay_info = kwargs.get("lay_info", [None, None])

    def __call__(self, module, module_in, module_out):
        # print('hook is', self.lay_info)
        # self.lay_info = self.lay_info
        self.outputs.extend(module_out)
        return module_out

    def clear(self):
        self.outputs = []
        self.lay_info = [None, None]


class SaveInput:
    def __init__(self, name=None):
        self.inputs = []
        self.name=name

    def __call__(self, module, module_in, module_out):
        self.inputs.append(module_in)
        # print('check hook', module_in[0].device.index) #check on which device the hook is set

    def clear(self):
        self.inputs = []
class SaveTraceOutput:
    def __init__(self, trace_func ='sum'):
        self.trace_func = trace_func
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        if self.trace_func == 'sum':
            trace = torch.sum(module_out, (-1, -2))
        if self.trace_func == 'mean':
            trace = torch.mean(module_out, (-1, -2))
        self.outputs.append(trace)

    def clear(self):
        self.outputs = []

def set_ranger_hooks_v2(net, resil=Ranger_trivial):
    """
    Creates an instance of the classes Save_nan_inf that collect True, False values about whether nan, infs were found in the input and ouptut of any layer.
    :param net: pytorch model
    :return: save_nan_inf, class instance of Save_nan_inf
    :return: hook_handles
    """
 
    save_acts = SaveInput()
    hook_handles = [] #handles list could be used to remove hooks later, here not used
 
    for _, m in net.named_modules():
        if type(m) == resil:# if type(m)==nn.Conv2d: ...
            handl = m.register_forward_hook(save_acts)
            hook_handles.append(handl)
            
    return save_acts, hook_handles


def set_ranger_hooks_v3(net, bnds, resil='ranger', mitigation=True, detector=True, correct_DUE=True):
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
    mitigation = False if 'trivial' in resil else mitigation
    # if ranger_hook == 'global':
    if 'global' in resil.lower():
        ranger_global = Range_detector_global(bnds, mitigation=mitigation, detector=detector, resil=resil, correct_DUE=correct_DUE)
    correct_DUE = mitigation*(True if "due" in resil.lower() else False)
    correct_DUE = True
    for _, m in net.named_modules():
        
        if 'global' in resil.lower():
            if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU, torch.nn.Linear]:
                handle_out = m.register_forward_hook(ranger_global)
                hook_handles_out.append(handle_out)
                cnt += 1
        else:
            if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]:
                act_hook = Range_detector(bnds[cnt], mitigation=mitigation, detector=detector, resil=resil, correct_DUE=correct_DUE)
                handle_out = m.register_forward_hook(act_hook)
                hook_list.append(act_hook)
                hook_handles_out.append(handle_out)
                cnt += 1

    # if ranger_hook == 'global':
    if 'global' in resil.lower():
        return hook_handles_out, ranger_global
    else:    
        return hook_handles_out, hook_list

class Range_detector_v2:
    def __init__(self, bnds, **kwargs):
        self.layer_counter = 0
        self.bnds = bnds #specific bound for that hook (format [min, max])
        self.acts_dets = [None]*len(self.bnds)
        self.monitor_mean = [None]*len(self.bnds)
        self.monitor_var = [None]*len(self.bnds)
        self.mit = kwargs.get("mitigation", False)
        self.det = kwargs.get("detector", True) #(if mit is off then nothing is saved since it is trivial)
        self.resil = kwargs.get("resil", None)
        self.correct_DUE = kwargs.get("correct_DUE", True)
        self.layer_id = kwargs.get("layer_id", -1)
        # self.lay_info = kwargs.get("lay_info", [None, None])

    def __call__(self, module, module_in, module_out):
        # print('which hook?', self.lay_info)
        module.ranger_layer_counter = self.layer_counter
        if self.det or self.mit:
            try:
                lower_bnd = self.bnds[self.layer_counter][0]
                upper_bnd = self.bnds[self.layer_counter][-1]
            except:
                print("bounds and their respective layer numbers dount align")
            if self.det:
                # Check if clamping had an effect
                module_out_clamped = torch.clamp(module_out, min=lower_bnd, max=upper_bnd)
                comp = torch.abs(torch.sub(module_out, module_out_clamped)) #difference between before and after clamping
                # self.act = torch.sum(comp>0, dim=list(range(1,len(comp.shape)))).tolist()
                # self.act.extend(torch.sum(comp>0, dim=list(range(1,len(comp.shape)))).tolist())
                self.acts_dets[self.layer_counter] = torch.sum(comp>0, dim=list(range(1,len(comp.shape)))).tolist()
                if not self.mit:
                    self.monitor_mean[self.layer_counter] = torch.mean(module_out).item()
                    self.monitor_var[self.layer_counter] = torch.var(module_out).item()
                    self.layer_counter+=1        
            if self.mit:
                if "ranger" in self.resil.lower():
                    module_out = self.correct_due(torch.clamp(module_out, min=lower_bnd, max=upper_bnd))
                    self.monitor_mean[self.layer_counter] = torch.mean(module_out).item()
                    self.monitor_var[self.layer_counter] = torch.var(module_out).item()
                    self.layer_counter+=1
                    return module_out
                elif "clip" in self.resil.lower():
                    mask1 = module_out.ge(upper_bnd)
                    mask2 = module_out.le(lower_bnd)
                    mask = torch.logical_or(mask1, mask2)
                    module_out = self.correct_due(module_out.masked_fill(mask, 0.))
                    self.monitor_mean[self.layer_counter] = torch.mean(module_out).item()
                    self.monitor_var[self.layer_counter] = torch.var(module_out).item()
                    self.layer_counter+=1
                    return module_out
                elif "FmapAvg" in self.resil.lower():
                    self.layer_counter+=1
                    if len(module_out.shape) == 4:  # will work only for conv layers
                        wh = module_out.size()[2]
                        nrf = module_out.size()[1]
                        comp = torch.abs(torch.sub(module_out, module_out_clamped)) #difference between before and after clamping
                        if comp:
                            pass                            
                    pass
                ## replacing nan and infs to 0
        else:
            return module_out
        
    def correct_due(self, module_out):
        if self.correct_DUE:
            ## correcting INFs
            module_out[module_out == float('inf')] = 0
            ## correcting NaNs
            module_out[module_out != module_out] = 0
            return module_out
        else:
            return module_out

    def clear(self):
        self.layer_counter = 0
        self.act = []
        self.bnd = [0, 0]
        self.lay_info = [None, None]
        
def set_ranger_hooks_v4(net, bnds, resil='ranger', mitigation=True, detector=True, correct_DUE=True, use_v2=True):
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
    mitigation = False if 'trivial' in resil else mitigation
    correct_DUE = mitigation*(True if "due" in resil.lower() else False)
    correct_DUE = True
    # if ranger_hook == 'global':
    if 'global' in resil.lower():
        ranger_global = Range_detector_global(bnds, mitigation=mitigation, detector=detector, resil=resil, correct_DUE=correct_DUE)
        use_v2 = False
    elif use_v2:
        act_hook = Range_detector_v2(bnds, mitigation=mitigation, detector=detector, resil=resil, correct_DUE=correct_DUE)

    for layer_id, (name, m) in enumerate(net.named_modules()):
        
        if 'global' in resil.lower():
            # ranger_global = Range_detector_global(bnds, mitigation=mitigation, detector=detector, resil=resil, correct_DUE=correct_DUE, layer_id=layer_id)
            # if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU, torch.nn.Linear]:
            if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Linear]:
                handle_out = m.register_forward_hook(ranger_global)
                hook_handles_out.append(handle_out)
                cnt += 1
                m.__dict__['layer_name'] = name
        else:
            # if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]:
            # if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU]:
            if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Linear]:
                if not use_v2:
                    act_hook = Range_detector(bnds[cnt], mitigation=mitigation, detector=detector, resil=resil, correct_DUE=correct_DUE, layer_id=layer_id)
                # act_hook = ranger_det_v2
                handle_out = m.register_forward_hook(act_hook)
                if not use_v2:
                    hook_list.append(act_hook)
                hook_handles_out.append(handle_out)
                cnt += 1
                m.__dict__['layer_name'] = name
    # if ranger_hook == 'global':
    if 'global' in resil.lower():
        return hook_handles_out, ranger_global
    else:
        if use_v2:
            return hook_handles_out, act_hook
        else:
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
        if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]:
            act_hook = Range_detector_quantiles() #bnds[cnt])
            handle_out = m.register_forward_hook(act_hook)
            hook_handles_out.append(handle_out)
            hook_list.append(act_hook)
            cnt += 1

    return hook_handles_out, hook_list


def set_feature_trace_hooks(net):
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
        if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]:
            # act_hook = Range_detector_quantiles() #bnds[cnt])
            act_hook = Range_detector_feature_trace() #bnds[cnt])
            handle_out = m.register_forward_hook(act_hook)
            hook_handles_out.append(handle_out)
            hook_list.append(act_hook)
            cnt += 1

    return hook_handles_out, hook_list


class Range_detector:
    def __init__(self, bnd, **kwargs):
        self.act = []
        self.bnd = bnd #specific bound for that hook (format [min, max])
        self.mit = kwargs.get("mitigation", False)
        self.det = kwargs.get("detector", True) #(if mit is off then nothing is saved since it is trivial)
        self.resil = kwargs.get("resil", None)
        self.correct_DUE = kwargs.get("correct_DUE", True)
        # self.lay_info = kwargs.get("lay_info", [None, None])

    def __call__(self, module, module_in, module_out):
        # print('which hook?', self.lay_info)
        if self.det:
            # Check if clamping had an effect
            module_out_clamped = torch.clamp(module_out, min=self.bnd[0], max=self.bnd[-1])
            comp = torch.abs(torch.sub(module_out, module_out_clamped)) #difference between before and after clamping
            # self.act = torch.sum(comp>0, dim=list(range(1,len(comp.shape)))).tolist()
            self.act.extend(torch.sum(comp>0, dim=list(range(1,len(comp.shape)))).tolist())
        if self.mit:
            if "ranger" in self.resil.lower():
                return self.correct_due(torch.clamp(module_out, min=self.bnd[0], max=self.bnd[-1]))
            elif "clip" in self.resil.lower():
                bnd_low = self.bnd[0]
                bnd_up = self.bnd[1]
                mask1 = module_out.ge(bnd_up)
                mask2 = module_out.le(bnd_low)
                mask = torch.logical_or(mask1, mask2)
                return self.correct_due(module_out.masked_fill(mask, 0.))
            ## replacing nan and infs to 0
        else:
            return module_out

    def correct_due(self, module_out):
        if self.correct_DUE:
            ## correcting INFs
            module_out[module_out == float('inf')] = 0
            ## correcting NaNs
            module_out[module_out != module_out] = 0
            return module_out
        else:
            return module_out
    def clear(self):
        self.act = []
        self.bnd = [0, 0]
        self.lay_info = [None, None]
    
class Range_detector_global:
    def __init__(self, bnds, **kwargs):
        self.act = []
        self.module_names = []
        self.layer_counter = 0
        self.layer_counter_debug = 0
        self.bnds = bnds #specific bound for that hook (format [min, max])
        self.acts_dets = [None]*len(self.bnds)
        self.mit = kwargs.get("mitigation", False)
        self.det = kwargs.get("detector", True) #(if mit is off then nothing is saved since it is trivial)
        self.resil = kwargs.get("resil", None)
        self.hybrid = True if "hybrid" in self.resil.lower() else False
        self.correct_DUE = kwargs.get("correct_DUE", True)
        # self.lay_info = kwargs.get("lay_info", [None, None])

    def __call__(self, module, module_in, module_out):
        # print('which hook?', self.lay_info)
        module.ranger_layer_counter = self.layer_counter
        layer_name = module.__dict__['layer_name']
        backbone_layer = True if 'backbone' in layer_name.lower() else False
        relu_layer = True if 'relu' in layer_name.lower() else False
        self.module_names.append(module.__dict__['layer_name'])
        if self.hybrid:
            hybrid_mit = "ranger" if relu_layer else "clip"
            # hybrid_mit = "clip" if relu_layer else "ranger"
            hybrid_mit = "clip" if relu_layer else "ranger"
        else:
            hybrid_mit = self.resil.lower()
        if self.det or self.mit:
            try:
                lower_bnd = self.bnds[self.layer_counter][0]
                upper_bnd = self.bnds[self.layer_counter][-1]
                self.layer_counter_debug+=1
            except:
                print("bounds and their respective layer numbers dount align")
            if self.det:
                # Check if clamping had an effect
                module_out_clamped = torch.clamp(module_out, min=lower_bnd, max=upper_bnd)
                comp = torch.abs(torch.sub(module_out, module_out_clamped)) #difference between before and after clamping
                # self.act = torch.sum(comp>0, dim=list(range(1,len(comp.shape)))).tolist()
                # self.act.extend(torch.sum(comp>0, dim=list(range(1,len(comp.shape)))).tolist())
                self.acts_dets[self.layer_counter] = torch.sum(comp>0, dim=list(range(1,len(comp.shape)))).tolist()
                if not self.mit:
                    self.layer_counter+=1        
            if self.mit:
                self.layer_counter+=1
                if "ranger" in hybrid_mit:
                    return self.correct_due(torch.clamp(module_out, min=lower_bnd, max=upper_bnd))
                elif "clip" in hybrid_mit:
                    bnd_low = lower_bnd
                    bnd_up = upper_bnd
                    mask1 = module_out.ge(bnd_up)
                    mask2 = module_out.le(bnd_low)
                    mask = torch.logical_or(mask1, mask2)
                    return self.correct_due(module_out.masked_fill(mask, 0.))
                elif "FmapAvg" in hybrid_mit:
                    pass
                    
                ## replacing nan and infs to 0
        else:
            return module_out
        
    def correct_due(self, module_out):
        if self.correct_DUE:
            ## correcting INFs
            module_out[module_out == float('inf')] = 0
            ## correcting NaNs
            module_out[module_out != module_out] = 0
            return module_out
        else:
            return module_out

    def clear(self):
        self.layer_counter = 0
        self.act = []
        self.bnd = [0, 0]
        self.lay_info = [None, None]

class Range_detector_quantiles:
    def __init__(self):
        self.quant = []

    def __call__(self, module, module_in, module_out):
        
        tnsr = torch.flatten(module_out, start_dim=1, end_dim=- 1) #flatten all except for batch nr
        q0, q10, q25, q50, q75, q100 = torch.quantile(tnsr, torch.tensor([0., 0.10, 0.25, 0.50, 0.75, 1.], device=tnsr.device), dim=1)

        lst = np.vstack([q0.cpu().numpy(), q10.cpu().numpy(), q25.cpu().numpy(), q50.cpu().numpy(), q75.cpu().numpy(), q100.cpu().numpy()])


        self.quant.extend(lst.T.tolist())
        # self.quant.extend(tnsr.cpu().numpy().tolist()) #TODO TODO: #only when all activations should be extracted


        return module_out

    def clear(self):
        self.quant = []


class Range_detector_feature_trace:
    def __init__(self):
        self.quant = []
        # self.bnd = bnd #specific bound for that hook (format [min, max])

    def __call__(self, module, module_in, module_out):
        
        if len(module_out.shape) > 2:
            dims_sum = np.arange(2,len(module_out.shape)).tolist()
            lst = torch.sum(module_out, dim=dims_sum).tolist()
        else:
            lst = module_out.tolist()

        self.quant.extend(lst)

        return module_out

    def clear(self):
        # self.bnd = [0, 0]
        self.quant = []


def run_trace_hooks(trace_output, trace_hook_handles_out):

    trace_outputs = trace_output.outputs
    trace_output.clear() # clear the hook lists, otherwise memory leakage
    for i in range(len(trace_hook_handles_out)):
        trace_hook_handles_out[i].remove()

    batch_size, _ = trace_outputs[0].shape
    total_filters = 0
    num_of_conv_layers = len(trace_outputs)
    for output in trace_outputs:
        total_filters += output.shape[1]
    
    output_trace = None
    for i in range(batch_size):
        trace_image = []
        for output in trace_outputs:
            num_filters = output[i].shape[0]
            output_image = output[i].reshape(num_filters)
            trace_image.extend([output_image])

        trace_image = torch.unsqueeze(torch.cat(trace_image, dim=0), 0)
        # output_trace.append(output_trace)
        if output_trace is None:
            output_trace = trace_image
        else:
            output_trace = torch.cat((output_trace, trace_image), 0)
    return output_trace

def set_trace_hooks_conv2d(net, trace_func='sum'):
    """
    Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
    :param net: pytorch model
    :return: save_input, class instance of SaveInput
    :return: save_output, class instance of SaveOutput
    """
    trace_output = SaveTraceOutput(trace_func=trace_func)
    trace_hook_handles_out = []

    for name, m in net.named_modules():
        if type(m) == torch.nn.Conv2d:# if type(m)==nn.Conv2d: ...
            # print('Ranger hook set')
            handle_out = m.register_forward_hook(trace_output)
            trace_hook_handles_out.append(handle_out)

    return trace_output, trace_hook_handles_out

def set_ranger_hooks(net):
    """
    Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
    :param net: pytorch model
    :return: save_input, class instance of SaveInput
    :return: save_output, class instance of SaveOutput
    """

    save_input = SaveInput()
    save_output = SaveOutput()

    hook_handles_in = {} #handles list could be used to remove hooks later, here not used
    hook_handles_out = {}
    module_names  = []

    for layer_id, (name, m) in enumerate(net.named_modules()):
    # for name, m in net.named_modules():
        # if type(m) == Ranger:# if type(m)==nn.Conv2d: ...
        # if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU, torch.nn.modules.conv.Conv2d]
        if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Linear]:
        # if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU]:
        # if type(m) in [torch.nn.modules.batchnorm.BatchNorm2d]:
            # print('Ranger hook set')
            hook_handles_in[name] = m.register_forward_hook(save_input)
            hook_handles_out[name] = m.register_forward_hook(save_output)

            # hook_handles_in.append(handle_in)
            # hook_handles_out.append(handle_out)
            module_names.append(name)
            m.__dict__['layer_name'] = name
    return save_input, save_output, hook_handles_in, hook_handles_out

def set_ranger_hooks_ReLU(net):
    """
    Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
    :param net: pytorch model
    :return: save_output, class instance of SaveInput
    :return: hook_handles_out, list of hook handles
    """

    hook_list = []
    hook_handles_out = []
    cnt = 0
    for name, m in net.named_modules():
        # print('names', name, m)
        if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]: #any type of activation function!
            # print('Ranger hook set', type(m), name)
            save_output = OutputHook(lay_info=[cnt, name])
            handle_out = m.register_forward_hook(save_output)
            hook_handles_out.append(handle_out)
            hook_list.append(save_output)
        cnt += 1

    return hook_list, hook_handles_out

def set_ranger_hooks_conv2d(net):
    """
    Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
    :param net: pytorch model
    :return: save_input, class instance of SaveInput
    :return: save_output, class instance of SaveOutput
    """

    save_input = SaveInput()
    save_output = SaveOutput()

    hook_handles_in = [] #handles list could be used to remove hooks later, here not used
    hook_handles_out = []

    for name, m in net.named_modules():
        if type(m) == torch.nn.Conv2d:# if type(m)==nn.Conv2d: ...
            # print('Ranger hook set')
            handle_in = m.register_forward_hook(save_input)
            handle_out = m.register_forward_hook(save_output)

            hook_handles_in.append(handle_in)
            hook_handles_out.append(handle_out)

    return save_input, save_output, hook_handles_in, hook_handles_out

def set_ranger_hooks_bn(net):
    """
    Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
    :param net: pytorch model
    :return: save_input, class instance of SaveInput
    :return: save_output, class instance of SaveOutput
    """

    save_input = SaveInput()
    save_output = SaveOutput()

    hook_handles_in = [] #handles list could be used to remove hooks later, here not used
    hook_handles_out = []

    for name, m in net.named_modules():
        if type(m) == torch.nn.BatchNorm2d:# if type(m)==nn.Conv2d: ...
            # print('Ranger hook set')
            handle_in = m.register_forward_hook(save_input)
            handle_out = m.register_forward_hook(save_output)

            hook_handles_in.append(handle_in)
            hook_handles_out.append(handle_out)

    return save_input, save_output, hook_handles_in, hook_handles_out

def run_with_hooks(net, images, bnds):
    """
    Creates hooks for net, executes an input image, and eliminates hooks again. Avoids GPU memory leakage issues.
    :param net: pytorch model
    :param inputs: image batch
    :return fi_outputs: (tensor) inference results
    :return activated: (list of size batchsize), nr of ranger layers that got activated in one image, collected by batch
    """

    if bnds is None or (bnds == [None, None]).all():
        fi_outputs = net(images)
        return fi_outputs, []


    save_input = SaveInput()
    save_output = SaveOutput()

    hook_handles_in = [] #handles list could be used to remove hooks later, here not used
    hook_handles_out = []

    ranger_count_check = 0
    for name, m in net.named_modules():
        if type(m) == Ranger:# if type(m)==nn.Conv2d: ...
            # print('Ranger hook set')
            ranger_count_check += 1
            handle_in = m.register_forward_hook(save_input)
            handle_out = m.register_forward_hook(save_output)

            hook_handles_in.append(handle_in)
            hook_handles_out.append(handle_out)

    # print('check nr rangers', ranger_count_check, len(save_input.inputs))
    fi_outputs = net(images) 
    # print('check nr rangers2', len(save_input.inputs)) 

    # Check activations ---------------------------------------
    act_in, act_out = get_max_min_lists(save_input.inputs, save_output.outputs) 
    save_input.clear()  # clear the hook lists, otherwise memory leakage
    save_output.clear()

    for i in range(len(hook_handles_in)):
        hook_handles_in[i].remove()
        hook_handles_out[i].remove()

    activated = [] #Ranger layers activated in one image batch!
    for n in range(len(act_in)): #images
        act_layers = 0
        
        for ran in range(len(act_in[n])):
            # if (act_in[n][ran, 0] < act_out[n][ran, 0]) or (act_in[n][ran, 1] > act_out[n][ran, 1]): #todo: just different or >, <?
            if (act_in[n][ran, 0] < bnds[ran, 0]) or (act_in[n][ran, 1] > bnds[ran, 1]): #todo: just different or >, <?
                act_layers += 1
                # print('compare: image:', n, 'rlayer:', ran, act_in[n][ran], act_out[n][ran]) #debugging
        activated.append(act_layers)
    # --------------------------------------------------------

    return fi_outputs, activated

def get_max_min_lists_in(activations_in):
    """
    Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
    :param activations_in: list of tuple of tensors
    :param activations_out: list of tensors
    :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
    :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
    """
    batch_nr = activations_in[0][0].size()[0]
    nr_rangers = len(activations_in)
    activations_in2 = []

    for b in range(batch_nr): #walk through a batch, i.e. through images
        ranger_list_in = []

        for r in range(nr_rangers):

            rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
            rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
            ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])

        activations_in2.append(ranger_list_in)

    return np.array(activations_in2)

def get_mean_var(activations_out):
    batch_nr = activations_out[0].shape[0]
    nr_rangers = len(activations_out)
    activations_in2 = []
    mean_var_lists = []

    for b in range(batch_nr): #walk through a batch (here usually just 1)
        mean_var_list_out = []
        for r in range(nr_rangers): #walk through a layer
            mean = torch.mean(activations_out[r][b]).tolist()
            var = torch.var(activations_out[r][b]).tolist()
            mean_var_list_out.append([mean, var])
        mean_var_lists.append(mean_var_list_out)
    mean_var_lists = np.array(mean_var_lists)
    return np.squeeze(mean_var_lists)

def get_max_min_lists(activations_in, activations_out):
    """
    Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
    :param activations_in: list of tuple of tensors
    :param activations_out: list of tensors
    :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
    :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
    """
    batch_nr = activations_in[0][0].size()[0]
    nr_rangers = len(activations_in)
    activations_in2 = []
    activations_out2 = []

    for b in range(batch_nr): #walk through a batch, i.e. through images
        ranger_list_in = []
        ranger_list_out = []

        for r in range(nr_rangers):

            rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
            rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
            ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])

            rmax_perIm_out = torch.max(activations_out[r][b]).tolist()
            rmin_perIm_out = torch.min(activations_out[r][b]).tolist()
            ranger_list_out.append([rmin_perIm_out, rmax_perIm_out])
            # if rmax_perIm_in > 100:
            #     print('in getfct', rmax_perIm_in, rmax_perIm_out) #todo

        activations_in2.append(ranger_list_in)
        activations_out2.append(ranger_list_out)

    return np.array(activations_in2), np.array(activations_out2)

    
## Simplified hook on all layers -----------------------------------------------------------------------------
class Save_nan_inf:
    """
    Inputs is list with dims: nr of layers, nr of images in batch, 2 for nan, inf.
    """
    def __init__(self):
        self.outputs = []
        self.hooked_layers = []
        self.unhooked_layers = []

    def __call__(self, module, module_in, module_out):
        """
        Sequential and Bottleneck containers input/output is not considered here (skipped) for clearer monitoring.
        """
        if isinstance(module_out, torch.Tensor):
            output_nan_flags = module_out.isnan() # outgoing tensor not wrapped
            output_inf_flags = module_out.isinf()
            dims = len(output_nan_flags.shape)
            shape = output_nan_flags.shape
            if shape[0]>shape[1]:
                output_inf_flags_reshape = output_inf_flags.permute(1,0,-1)
                output_nan_flags_reshape = output_nan_flags.permute(1,0,-1)
                monitors = [(torch.sum(output_nan_flags_reshape, dim=tuple(range(1,dims))) > 0).cpu().numpy(), (torch.sum(output_inf_flags_reshape, dim=tuple(range(1,dims))) > 0).cpu().numpy()]
            else:    
                monitors = [(torch.sum(output_nan_flags, dim=tuple(range(1,dims))) > 0).cpu().numpy(), (torch.sum(output_inf_flags, dim=tuple(range(1,dims))) > 0).cpu().numpy()]
            self.outputs.append(monitors)
            # if module.__module__ not in self.hooked_layers:
            self.hooked_layers.append(module.__module__)
        else:
            # if module.__module__ not in self.unhooked_layers:
            self.unhooked_layers.append(module.__module__)
            

    def clear(self):
        self.outputs = []
        self.hooked_layers = []
        self.unhooked_layers = []


class Save_penult_layer:
    """
    Inputs is list with dims: nr of layers, nr of images in batch, 2 for nan, inf.
    """
    def __init__(self):
        # self.inputs = []
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        """
        Sequential and Bottleneck containers input/output is not considered here (skipped) for clearer monitoring.
        """
        self.outputs.append(module_in[0])
        
    def clear(self):
        # self.inputs = []
        self.outputs = []

def set_simscore_hooks(net, model_name):
    """
    Creates an instance of the classes Save_nan_inf that collect True, False values about whether nan, infs were found in the input and ouptut of any layer.
    :param net: pytorch model
    :return: save_nan_inf, class instance of Save_nan_inf
    :return: hook_handles
    """
    save_penult_layer = Save_penult_layer()
    hook_handles = [] #handles list could be used to remove hooks later, here not used
    hook_layer_names = []

    penultimate_layer = None
    if 'alexnet' in model_name:
        # penultimate_layer = ['classifier.3', 'classifier.4', 'classifier.6', '57', '58', '61']
        penultimate_layer = ['classifier.5', '31']
        for layer_name, m in net.named_modules():
            if layer_name in penultimate_layer:
                # cnt += 1
                # print(cnt, type(m))
                handle_in_out = m.register_forward_hook(save_penult_layer)
                hook_handles.append(handle_in_out)
                hook_layer_names.append(m.__module__)

        return save_penult_layer, hook_handles, hook_layer_names
    if 'vgg' in model_name:
        penultimate_layer = ['classifier.6', '61']
        for layer_name, m in net.named_modules():
            if layer_name in penultimate_layer:
                handle_in_out = m.register_forward_hook(save_penult_layer)
                hook_handles.append(handle_in_out)
                hook_layer_names.append(m.__module__)

        return save_penult_layer, hook_handles, hook_layer_names
    if 'resnet' in model_name:
        # penultimate_layer = ['classifier.3', 'classifier.4', 'classifier.6', '57', '58', '61']
        penultimate_layer = ['fc', '42']
        for layer_name, m in net.named_modules():
            if layer_name in penultimate_layer:
                # cnt += 1
                # print(cnt, type(m))
                handle_in_out = m.register_forward_hook(save_penult_layer)
                hook_handles.append(handle_in_out)
                hook_layer_names.append(m.__module__)

        return save_penult_layer, hook_handles, hook_layer_names


def set_nan_inf_hooks(net):
    """
    Creates an instance of the classes Save_nan_inf that collect True, False values about whether nan, infs were found in the input and ouptut of any layer.
    :param net: pytorch model
    :return: save_nan_inf, class instance of Save_nan_inf
    :return: hook_handles
    """
    save_nan_inf = Save_nan_inf()
    hook_handles = [] #handles list could be used to remove hooks later, here not used
    hook_layer_names = []
    for _, m in net.named_modules():
        if np.all([x not in str(type(m)) for x in ['Sequential', 'ModuleList']]):
        # if 'torch.nn.' in str(type(m)):
            handle_in_out = m.register_forward_hook(save_nan_inf)
            hook_handles.append(handle_in_out)
            hook_layer_names.append(m.__module__)
            # print('hook set for layer', type(m)) #

    return save_nan_inf, hook_handles, hook_layer_names

    
def run_with_debug_hooks_v3(net, image, bnds, ranger_activity, nan_inf_activity, resil=Ranger_trivial):


    nan_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
        'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False}
    inf_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
        'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False}

    activated = [] #protection layers activated in one image batch!

    # Set naninf hooks
    if nan_inf_activity:
        save_nan_inf, hook_handles = set_nan_inf_hooks(net)

    # set ranger hooks
    if ranger_activity and (bnds is not None or (bnds != [None, None]).all()):
        save_acts, hook_handles_act = set_ranger_hooks_v2(net, resil=resil)

    corrupted_output = net(image)

    # Save naninf activations
    if nan_inf_activity:
        # nan_inf_in, nan_inf_out = save_nan_inf.inputs, save_nan_inf.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
        nan_inf_out = save_nan_inf.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
        save_nan_inf.clear() #clear the hook lists, otherwise memory leakage
        for i in range(len(hook_handles)):
            hook_handles[i].remove()


        # Process nan
        nan_all_layers = np.array(nan_inf_out)[:, :, 0] #rows are layers, cols are images
        nan_dict["overall"] = [np.where(nan_all_layers[:, u] == True)[0].tolist() for u in range(len(nan_all_layers[0]))] #former in only
        nan_dict['flag'] = [x != [] for x in nan_dict['overall']]

        #Process inf
        inf_all_layers = np.array(nan_inf_out)[:, :, 1]
        inf_dict["overall"] = [np.where(inf_all_layers[:, u] == True)[0].tolist() for u in range(len(inf_all_layers[0]))]
        inf_dict['flag'] = [x != [] for x in inf_dict['overall']]

        # Spot first nan, inf
        spots = True
        nan_dict['error_cause'] = []

        if spots:
            inf_mins_out = [np.min(u) if u !=[] else 1000 for u in inf_dict['overall']]
            nan_mins_out = [np.min(u) if u !=[] else 1000 for u in nan_dict['overall']]
            # comp = [[inf_mins_in[x], nan_mins_in[x], inf_mins_out[x], nan_mins_out[x]] for x in range(len(inf_mins_in))]
            comp = [[inf_mins_out[x], nan_mins_out[x]] for x in range(len(inf_mins_out))]
            comp_ind = [np.where(n==np.min(n))[0].tolist() if np.min(n) < 1000 else [] for n in comp]

            layer_list = list(net.named_modules())
            layer_list_noseq = []
            for name, m in layer_list:
                if type(m) not in [torch.nn.Sequential, My_Bottleneck_Ranger]:
                    layer_list_noseq.append([name, m])
            layer_list = layer_list_noseq

            info_imgs = []
            for i in range(len(comp_ind)): #for different images
                if comp_ind[i] == []:
                    info = []
                else:   
                    lay = 1000
                    # pos = ''
                    tp = ''

                    if 0 in comp_ind[i]:
                        lay = comp[i][0]
                        # pos = pos  + 'in'
                        tp = tp + 'Inf'
                        
                    if 1 in comp_ind[i]:
                        lay = comp[i][1]
                        # pos = pos  + 'in'
                        tp = tp + 'Nan'

                    info = [lay, type(layer_list[lay][1]).__name__, tp]

                info_imgs.append(info)
            nan_dict['error_cause'] = info_imgs

   # Save ranger activations
    if ranger_activity and (bnds is not None or (bnds != [None, None]).all()):
        act_in = get_max_min_lists_in(save_acts.inputs) 
        save_acts.clear()  # clear the hook lists, otherwise memory leakage
        for i in range(len(hook_handles_act)):
            hook_handles_act[i].remove()

        for n in range(len(act_in)): #images
            act_layers = 0
            for ran in range(len(act_in[n])):
                # if (act_in[n][ran, 0] < act_out[n][ran, 0]) or (act_in[n][ran, 1] > act_out[n][ran, 1]): #todo: just different or >, <?
                if (act_in[n][ran, 0] < bnds[ran, 0]) or (act_in[n][ran, 1] > bnds[ran, 1]): #todo: just different or >, <?
                    act_layers += 1
                    # print('compare: image:', n, 'rlayer:', ran, act_in[n][ran], act_out[n][ran]) #debugging
            activated.append(act_layers)    
    

    return corrupted_output, activated, nan_dict, inf_dict

def run_with_debug_hooks_v2(net, image):

    nan_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
        'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False}
    inf_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
        'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False}

    save_nan_inf, hook_handles, hook_layer_names = set_nan_inf_hooks(net)

    corrupted_output = net(image)

    nan_inf_out = save_nan_inf.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
    
    save_nan_inf.clear() #clear the hook lists, otherwise memory leakage
    for i in range(len(hook_handles)):
        hook_handles[i].remove()

    # Process nan
    nan_all_layers = np.array(nan_inf_out)[:, :, 0] #rows are layers, cols are images
    nan_dict["overall"] = [np.where(nan_all_layers[:, u] == True)[0].tolist() for u in range(len(nan_all_layers[0]))] #former in only
    nan_dict['flag'] = [x != [] for x in nan_dict['overall']]


    #Process inf
    inf_all_layers = np.array(nan_inf_out)[:, :, 1]
    inf_dict["overall"] = [np.where(inf_all_layers[:, u] == True)[0].tolist() for u in range(len(inf_all_layers[0]))]
    inf_dict['flag'] = [x != [] for x in inf_dict['overall']]


    return corrupted_output, nan_dict, inf_dict

def run_nan_inf_hooks(save_nan_inf, hook_handles, layers_with_hooks):

    nan_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
        'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False, 'first_occurrence': [], 'first_occur_compare': []}
    inf_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
        'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False, 'first_occurrence': []}

    nan_inf_out = np.array(save_nan_inf.outputs) #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
    # Note: layers_with_hooks is list of names where hooks were set, save_nan_inf.hooked_layers are hooks that were actually called.
    hook_layer_names = save_nan_inf.hooked_layers
    # print('layers with hooks but no tensor output', save_nan_inf.unhooked_layers)

    save_nan_inf.clear() #clear the hook lists, otherwise memory leakage
    for i in range(len(hook_handles)):
        hook_handles[i].remove()

    # Process nan
    nan_all_layers_out = np.array(nan_inf_out)[:, 0, :]
    nan_dict["overall_out"] = [list(np.where(n==True)[0]) for n in nan_all_layers_out.T]
    nan_dict['overall'] = nan_dict["overall_out"] #no in here
    #[np.unique(nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] 
    nan_dict['flag'] = [x != [] for x in nan_dict['overall']]

    for i in range(len(nan_dict['overall'])):
        first_nan_layer_index = nan_dict['overall'][i]
        if first_nan_layer_index: #TODO:
            # print(first_nan_layer_index[0])
            if first_nan_layer_index[0] > len(hook_layer_names):
                nan_dict['first_occurrence'].append([first_nan_layer_index[0], 'nan', 'layer'])
            else:
                nan_dict['first_occurrence'].append([first_nan_layer_index[0], 'nan', hook_layer_names[first_nan_layer_index[0]]])
        else:
            nan_dict['first_occurrence'].append([])

    #Process inf 
    inf_all_layers_out = np.array(nan_inf_out)[:, 1, :]
    inf_dict["overall_out"] = [list(np.where(n==True)[0]) for n in inf_all_layers_out.T]
    inf_dict['overall'] = inf_dict["overall_out"]
    #[np.unique(inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] 
    inf_dict['flag'] = [x != [] for x in inf_dict['overall']]

    for i in range(len(inf_dict['overall'])):
        first_inf_layer_index = inf_dict['overall'][i]
        if first_inf_layer_index: #TODO:
            # print(first_inf_layer_index[0], len(hook_layer_names)) #hook layer names
            if first_inf_layer_index[0] > len(hook_layer_names):
                inf_dict['first_occurrence'].append([first_inf_layer_index[0], 'inf', 'layer'])
            else:
                inf_dict['first_occurrence'].append([first_inf_layer_index[0], 'inf', hook_layer_names[first_inf_layer_index[0]]])
        else:
            inf_dict['first_occurrence'].append([])

    for i in range(len(nan_dict['first_occurrence'])):
        nan_info = nan_dict['first_occurrence'][i]
        inf_info = inf_dict['first_occurrence'][i]
        if nan_info and inf_info:
            nan_inf_list = [nan_info, inf_info]
            if nan_info[0] == inf_info[0]:
                nan_dict['first_occur_compare'].append(nan_inf_list)
            else:
                nan_dict['first_occur_compare'].append(nan_inf_list[np.argmin([nan_info[0], inf_info[0]])])
        elif nan_info:
            nan_dict['first_occur_compare'].append(nan_info)
        elif inf_dict['first_occurrence'][i]:
            nan_dict['first_occur_compare'].append(inf_info)
        else:
            nan_dict['first_occur_compare'].append([])

    return nan_dict, inf_dict


## deprecated
def run_with_debug_hooks_simple(net, image):

    nan_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
        'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False, 'first_occurrence': [], 'first_occur_compare': []}
    inf_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
        'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False, 'first_occurrence': []}

    save_nan_inf, hook_handles, hook_layer_names = set_nan_inf_hooks(net)

    corrupted_output = net(image)

    nan_inf_out = save_nan_inf.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
    
    save_nan_inf.clear() #clear the hook lists, otherwise memory leakage
    for i in range(len(hook_handles)):
        hook_handles[i].remove()

    # Process naninf
    nan_all_layers_out = np.array(nan_inf_out)[:, :, 0]
    nan_dict["overall_out"] = [np.where(nan_all_layers_out[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_out[0]))]
    #
    # nan_dict['overall'] = [np.unique(nan_dict['overall_in'][i] + nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] #in and out combined
    nan_dict['overall'] = [np.unique(nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] 
    nan_dict['flag'] = [x != [] for x in nan_dict['overall']]
    for i in range(len(nan_dict['overall'])):
        first_nan_layer_index = nan_dict['overall'][i]
        # nan_dict['first_occurrence'].append([]) #TODO:
        if first_nan_layer_index:
            nan_dict['first_occurrence'].append([first_nan_layer_index[0], 'nan', hook_layer_names[first_nan_layer_index[0]]])
        else:
            nan_dict['first_occurrence'].append([])

    inf_all_layers_out = np.array(nan_inf_out)[:, :, 1]
    inf_dict["overall_out"] = [np.where(inf_all_layers_out[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_out[0]))]
    # 
    # inf_dict['overall'] = [np.unique(inf_dict['overall_in'][i] + inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] #in and out combined
    inf_dict['overall'] = [np.unique(inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] 
    inf_dict['flag'] = [x != [] for x in inf_dict['overall']]
    for i in range(len(inf_dict['overall'])):
        first_inf_layer_index = inf_dict['overall'][i]
        # inf_dict['first_occurrence'].append([]) #TODO:
        if first_inf_layer_index:
            inf_dict['first_occurrence'].append([first_inf_layer_index[0], 'inf', hook_layer_names[first_inf_layer_index[0]]])
        else:
            inf_dict['first_occurrence'].append([])

    for i in range(len(nan_dict['first_occurrence'])):
        nan_info = nan_dict['first_occurrence'][i]
        inf_info = inf_dict['first_occurrence'][i]
        if nan_info and inf_info:
            nan_inf_list = [nan_info, inf_info]
            if nan_info[0] == inf_info[0]:
                nan_dict['first_occur_compare'].append(nan_inf_list)
            else:
                nan_dict['first_occur_compare'].append(nan_inf_list[np.argmin([nan_info[0], inf_info[0]])])
        elif nan_info:
            nan_dict['first_occur_compare'].append(nan_info)
        elif inf_dict['first_occurrence'][i]:
            nan_dict['first_occur_compare'].append(inf_info)
        else:
            nan_dict['first_occur_compare'].append([])

    return corrupted_output, nan_dict, inf_dict

def run_simscore_hooks(save_penult_layer, hook_handles):

    save_penult_layer_out = save_penult_layer.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
    save_penult_layer.clear() #clear the hook lists, otherwise memory leakage
    for i in range(len(hook_handles)):
        hook_handles[i].remove()

    return save_penult_layer_out

def print_nan_inf_hist_v2(net, nan_dict_corr, inf_dict_corr):

    comb = np.array([nan_dict_corr['overall'][i] + inf_dict_corr['overall'][i] for i in range(len(nan_dict_corr['overall']))]).flatten()
    # mn, mx = min(nan_dict_corr['overall'] + inf_dict_corr['overall']), max(nan_dict_corr['overall'] + inf_dict_corr['overall']) #for old version
    if not comb.any():
        return

    mn, mx = min(comb), max(comb)

    layer_list = list(net.named_modules())
    layer_list_noseq = []
    for name, m in layer_list:
        if type(m) not in [torch.nn.Sequential, My_Bottleneck_Ranger]:
            layer_list_noseq.append([name, m])
    layer_list = layer_list_noseq

    info = []
    for u in range(mn, mx+1):

        in_info = ['No', 'No']
        if u in np.array(nan_dict_corr['overall_in']).flatten():
            in_info[0] = 'NaN'
        if u in np.array(inf_dict_corr['overall_in']).flatten():
            in_info[1] = 'Inf'
        out_info = ['No', 'No']
        if u in np.array(nan_dict_corr['overall_out']).flatten():
            out_info[0] = 'NaN'
        if u in np.array(inf_dict_corr['overall_out']).flatten():
            out_info[1] = 'Inf'

        info.append([u, layer_list[u], 'in', in_info, 'out', out_info])

    print(tabulate(info)) #, headers=['Name', 'Age']