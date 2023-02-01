import torch
import numpy as np
import json
import os
import glob
from collections import Iterable
import subprocess
import numpy as np
# ImageNet -----------------------------------------------

def show_gpu(cuda_device, msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[cuda_device])
    
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')

def prepare_imageNet_data():
    """
    Puts images in separate folders depending on classes.
    Renames images, i.e. shifts them.
    :return:
    """

    # label extraction and data preprocessing
    # imagenet_groundtruth = "./ILSVRC2012_validation_ground_truth.txt"
    imagenet_mapping = "./ILSVRC2012_mapping.txt"
    # imagenet_labels = "./labels.txt"
    json_read = "./imagenet_class_index.json"

    class_idx = json.load(open(json_read))

    label_dict = {}
    for i in range(len(list(class_idx.keys()))):
        class_mapping = class_idx[list(class_idx.keys())[i]][0]
        class_label = class_idx[list(class_idx.keys())[i]][1]
        label_dict[class_mapping] = class_label

    # json_read = "/home/qutub/PhD/git_repos/intel_gitlab_repos/example_images/imagenet/imagenet_class_index.json"
    # class_idx = json.load(open(json_read))

    with open(imagenet_mapping) as f:
        mapping = [line.strip() for line in f.readlines()]

    # prepare dataset
    VAL_CLASS_PATH = "./ILSVRC2012_validation_ground_truth.txt"
    VAL_DATA_PATH = "./data/imagenet_val/"
    VAL_ORI_DATA_PATH = "./data/imagenet_val_1000/*.JPEG"

    val_class = []
    with open(VAL_CLASS_PATH) as val_file:
        rows = val_file.readlines()
        for row in rows:
            row = int(row.strip())
            val_class.append(row)
    val_files = glob.glob(VAL_ORI_DATA_PATH, recursive=True)
    for file in val_files:
        file = file.replace("\\","/")
        seq_num = int(file.split("/")[-1].split("_")[-1].split(".")[0])
        print(seq_num)
        class_id = val_class[seq_num - 1]
        class_mapping = mapping[class_id - 1].split()[1]
        class_name = label_dict[class_mapping]

        if not os.path.isdir(VAL_DATA_PATH + class_name):
            os.mkdir(VAL_DATA_PATH + class_name)

        os.rename(file, VAL_DATA_PATH + class_name + "/" + file.split("/")[-1])





#onnx --------------------------------------------------

def save_torch_to_onnx(model, image_shape, onnx_file):
    dummy_input = torch.ones(image_shape, dtype=torch.float32)
    dummy_input = dummy_input[None]
    torch.onnx.export(model, dummy_input, onnx_file, export_params=True, verbose=True)



# Load and save bounds ------------------------------------


# def save_Bounds_minmax(activations_in, bnds_name):
#     """
#     Saves Ranger bounds
#     :param activations_in: list of format [[min, max], [min, max], ... ]
#     :param bnds_name: 'Vgg16_bounds_dog' for example
#     :return: saves to a txt file in /bounds
#     """

#     bnd_path = str('./bounds/' + bnds_name + '.txt')
#     f = open(bnd_path, "w+")
#     for u in range(len(activations_in)):
#         f.write(str(activations_in[u][0]) + " , " + str(activations_in[u][1]))
#         f.write("\n")
#     f.close()

#     print('Bounds saved as ' + bnds_name)



def save_Bounds_minmax(activations_in, bnds_name):
    """
    Saves Ranger bounds
    :param activations_in: list of format [[min, max], [min, max], ... ]
    :param bnds_name: 'Vgg16_bounds_dog' for example
    :return: saves to a txt file in /bounds
    """

    bnd_path = './bounds/' + str(bnds_name) + '.txt'
    f = open(bnd_path, "w+")
    for u in range(len(activations_in)):
        sv = ""
        for v in range(len(activations_in[u])-1):
            sv += str(activations_in[u][v]) + " , "
        sv += str(activations_in[u][len(activations_in[u])-1])
        # f.write(str(activations_in[u][0]) + " , " + str(activations_in[u][1]))
        f.write(sv)
        f.write("\n")
    f.close()

    print('Bounds saved to ' + bnds_name)


def get_savedBounds_minmax(filename):
    f = open(filename, "r")
    bounds = []
    if f.mode == 'r':
        contents = f.read().splitlines()
        bounds = [u.split(',') for u in contents]
    f.close()

    # bounds = [[float(n[0]), float(n[1])] for n in bounds] #make numeric
    bounds = [[float(n) for n in m] for m in bounds] #make numeric

    return bounds



# Load weights in presence of ranger layers -----------------------------------------------

def getdict_ranger(PATH, net):
    """
    Identifies which are the correct weight and bias layers of the modified network net, and changes the dictionary in path accordingly.
    :param PATH: path to saved dict with weights
    :param net: net with ranger
    :return: modified dict of the same form as the one in path
    """

    dict_vgg = torch.load(PATH)
    list_pa = list(net.named_parameters())
    list_weights_ranger = [list_pa[i][0] for i in range(len(list_pa))]
    # list_weights = list(dict_vgg.keys()) #for comparison
    dict_vgg_ranger = dict(zip(list_weights_ranger, list(dict_vgg.values())))

    return dict_vgg_ranger

# Other
def flatten(x):
    """
    Flatten any list to a single level.
    """
    global z
    z += 1
    print(z)
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def get_max_min_lists_in(activations_in):
    """
    Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
    :param activations_in: list of tuple of tensors
    :param activations_out: list of tensors
    :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
    :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
    """
    #Note:
    #activations_in structure: nr ranger layers, silly bracket, batch size, channel, height, width
    #activations_out structure: nr ranger layers, batch size, channel, height, width


    # a = max([max([torch.max(activations_in[i][0][n]).tolist() for n in range(len(activations_in[i][0]))]) for i in range(len(activations_in))])
    # print('incoming max', a) #debugging

    batch_nr = activations_in[0][0].size()[0]
    nr_rangers = len(activations_in)
    activations_in2 = []

    for b in range(batch_nr): #walk through a batch, i.e. through images
        ranger_list_in = []

        for r in range(nr_rangers):

            rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
            rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
            ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])

            # if rmax_perIm_in > 100:
            #     print('in getfct', rmax_perIm_in, rmax_perIm_out) #todo

        activations_in2.append(ranger_list_in)

    return np.array(activations_in2)


def get_max_min_lists(activations_in, activations_out, get_perc=False):
    """
    Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
    :param activations_in: list of tuple of tensors
    :param activations_out: list of tensors
    :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
    :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
    """
    #Note:
    #activations_in structure: nr ranger layers, silly bracket, batch size, channel, height, width
    #activations_out structure: nr ranger layers, batch size, channel, height, width

    
    batch_nr = activations_out[0].size()[0]
    nr_rangers = len(activations_out)
    activations_in2 = []
    activations_out2 = []

    for b in range(batch_nr): #walk through a batch (here usually just 1)
        ranger_list_in = []
        ranger_list_out = []

        for r in range(nr_rangers): #walk through a layer
            if activations_in is not None:
                rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
                rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
                ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])



            rmax_perIm_out = torch.max(activations_out[r][b]).tolist()
            rmin_perIm_out = torch.min(activations_out[r][b]).tolist()
            if get_perc:
                act_num = activations_out[r][b].cpu().numpy()
                p25 = np.percentile(act_num, 25)
                p50 = np.percentile(act_num, 50)
                p75 = np.percentile(act_num, 75)
                # p0 = min, p100 = max

                ranger_list_out.append([rmin_perIm_out, p25, p50, p75, rmax_perIm_out])
            else:
                ranger_list_out.append([rmin_perIm_out, rmax_perIm_out])


        if activations_in is not None:
            activations_in2.append(ranger_list_in)
        activations_out2.append(ranger_list_out)

    return np.array(activations_in2), np.array(activations_out2)


# def get_max_min_dicts(activations_in, activations_out):
#     """
#     Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
#     :param activations_in: dict
#     :param activations_out: dict
#     :return: act_in, act_out dicts of form name: layer name, value: [min, max]
#     """
#
#     activations_in2 = {}
#     for name, outputs in activations_in.items():
#         max_lists = [torch.max(torch.cat(outputs[n])).tolist() for n in range(len(outputs))]
#         min_lists = [torch.min(torch.cat(outputs[n])).tolist() for n in range(len(outputs))]
#         activations_in2[name] = [np.min(min_lists), np.max(max_lists)]
#
#     activations_out2 = {}
#     for name, outputs in activations_out.items():
#         max_lists = [torch.max(outputs[n]).tolist() for n in range(len(outputs))]
#         min_lists = [torch.min(outputs[n]).tolist() for n in range(len(outputs))]
#         activations_out2[name] = [np.min(min_lists), np.max(max_lists)]
#
#     return activations_in2, activations_out2



# LeNet -------------------------------------------------
#
# def get_savedBounds(filename):
#     f = open(filename, "r")
#     bounds = []
#     if f.mode == 'r':
#         contents = f.read().splitlines()
#         bounds = [float(u) for u in contents]
#     f.close()
#
#     return bounds
#
#
# def get_active_ranger(list_act, bounds_max):
#     """
#     Takes the activations list for one *batch sample*, finds the max and compares it with the bounds_max.
#     Gives back an array of dimension 1xranger layers indicating with 0 or 1 if for that batch sample the bounds were
#     exceeded before ranger cut them. For fault detection.
#     If listact is empty gives back an array of zeros. Gives +1 count for a whole batch.
#     :param list_act:
#     :param bounds_max:
#     :return: array of dimension 1xranger with 0 or 1 if for that batch sample ranger was active (1) or not (0) in that layer.
#     """
#
#     max_act = []
#     if list_act:
#         max_act = [max(sublist) for sublist in list_act]
#
#     max_act = max_act[:len(bounds_max)] #if output last layer is also saved this is ignored for the ranger_active
#     # print('max act', max_act)  # before ranger
#     # print('bounds', bounds_max)  # bounds
#
#     ranger_active_layers = np.zeros(len(bounds_max))
#     if max_act:
#         ranger_active_layers = (np.array(max_act) > np.array(bounds_max)).astype(float)
#     # print('ranger active', ranger_active_layers)
#
#     return ranger_active_layers


# VGG16 -------------------------------------------


#### helper_function.py old file migrated to new one
# class SaveOutput:
#     def __init__(self):
#         self.outputs = []

#     def __call__(self, module, module_in, module_out):
#         self.outputs.append(module_out)

#     def clear(self):
#         self.outputs = []


# class SaveInput:
#     def __init__(self):
#         self.inputs = []

#     def __call__(self, module, module_in, module_out):
#         self.inputs.append(module_in)
#         # print('check hook', module_in[0].device.index) #check on which device the hook is set

#     def clear(self):
#         self.inputs = []
# class SaveTraceOutput:
#     def __init__(self, trace_func ='sum'):
#         self.trace_func = trace_func
#         self.outputs = []

#     def __call__(self, module, module_in, module_out):
#         if self.trace_func == 'sum':
#             trace = torch.sum(module_out, (-1, -2))
#         if self.trace_func == 'mean':
#             trace = torch.mean(module_out, (-1, -2))
#         self.outputs.append(trace)

#     def clear(self):
#         self.outputs = []

# def set_ranger_hooks_v2(net, resil=Ranger_trivial):
#     """
#     Creates an instance of the classes Save_nan_inf that collect True, False values about whether nan, infs were found in the input and ouptut of any layer.
#     :param net: pytorch model
#     :return: save_nan_inf, class instance of Save_nan_inf
#     :return: hook_handles
#     """
 
#     save_acts = SaveInput()
#     hook_handles = [] #handles list could be used to remove hooks later, here not used
 
#     for _, m in net.named_modules():
#         if type(m) == resil:# if type(m)==nn.Conv2d: ...
#             handl = m.register_forward_hook(save_acts)
#             hook_handles.append(handl)
            
#     return save_acts, hook_handles

# def run_trace_hooks(trace_output, trace_hook_handles_out):

#     trace_outputs = trace_output.outputs
#     trace_output.clear() # clear the hook lists, otherwise memory leakage
#     for i in range(len(trace_hook_handles_out)):
#         trace_hook_handles_out[i].remove()

#     batch_size, _ = trace_outputs[0].shape
#     total_filters = 0
#     num_of_conv_layers = len(trace_outputs)
#     for output in trace_outputs:
#         total_filters += output.shape[1]
    
#     output_trace = None
#     for i in range(batch_size):
#         trace_image = []
#         for output in trace_outputs:
#             num_filters = output[i].shape[0]
#             output_image = output[i].reshape(num_filters)
#             trace_image.extend([output_image])

#         trace_image = torch.unsqueeze(torch.cat(trace_image, dim=0), 0)
#         # output_trace.append(output_trace)
#         if output_trace is None:
#             output_trace = trace_image
#         else:
#             output_trace = torch.cat((output_trace, trace_image), 0)
#     return output_trace

# def set_trace_hooks_conv2d(net, trace_func='sum'):
#     """
#     Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
#     :param net: pytorch model
#     :return: save_input, class instance of SaveInput
#     :return: save_output, class instance of SaveOutput
#     """
#     trace_output = SaveTraceOutput(trace_func=trace_func)
#     trace_hook_handles_out = []

#     for name, m in net.named_modules():
#         if type(m) == torch.nn.Conv2d:# if type(m)==nn.Conv2d: ...
#             # print('Ranger hook set')
#             handle_out = m.register_forward_hook(trace_output)
#             trace_hook_handles_out.append(handle_out)

#     return trace_output, trace_hook_handles_out

# def set_ranger_hooks(net):
#     """
#     Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
#     :param net: pytorch model
#     :return: save_input, class instance of SaveInput
#     :return: save_output, class instance of SaveOutput
#     """

#     save_input = SaveInput()
#     save_output = SaveOutput()

#     hook_handles_in = [] #handles list could be used to remove hooks later, here not used
#     hook_handles_out = []

#     for name, m in net.named_modules():
#         # print('check names', name, m)
#         if type(m) == Ranger:# if type(m)==nn.Conv2d: ...
#             # print('Ranger hook set')
#             handle_in = m.register_forward_hook(save_input)
#             handle_out = m.register_forward_hook(save_output)
#             # m.register_forward_hook(save_input)
#             # m.register_forward_hook(save_output)

#             hook_handles_in.append(handle_in)
#             hook_handles_out.append(handle_out)

#     # Note: dont remove handles here otherwise no output

#     return save_input, save_output, hook_handles_in, hook_handles_out


# def set_ranger_hooks_ReLU(net):
#     """
#     Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
#     :param net: pytorch model
#     :return: save_input, class instance of SaveInput
#     :return: save_output, class instance of SaveOutput
#     """

#     save_input = SaveInput()
#     save_output = SaveOutput()

#     hook_handles_in = [] #handles list could be used to remove hooks later, here not used
#     hook_handles_out = []

#     for name, m in net.named_modules():
#         # print(name, m, type(m))
#         if type(m) == torch.nn.ReLU or type(m) == torch.nn.LeakyReLU:# if type(m)==nn.Conv2d: ...
#             # print('Ranger hook set', type(m), name)
#             handle_in = m.register_forward_hook(save_input)
#             handle_out = m.register_forward_hook(save_output)

#             hook_handles_in.append(handle_in)
#             hook_handles_out.append(handle_out)

#     return save_input, save_output, hook_handles_in, hook_handles_out

# def set_ranger_hooks_conv2d(net):
#     """
#     Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
#     :param net: pytorch model
#     :return: save_input, class instance of SaveInput
#     :return: save_output, class instance of SaveOutput
#     """

#     save_input = SaveInput()
#     save_output = SaveOutput()

#     hook_handles_in = [] #handles list could be used to remove hooks later, here not used
#     hook_handles_out = []

#     for name, m in net.named_modules():
#         if type(m) == torch.nn.Conv2d:# if type(m)==nn.Conv2d: ...
#             # print('Ranger hook set')
#             handle_in = m.register_forward_hook(save_input)
#             handle_out = m.register_forward_hook(save_output)

#             hook_handles_in.append(handle_in)
#             hook_handles_out.append(handle_out)

#     return save_input, save_output, hook_handles_in, hook_handles_out

# def set_ranger_hooks_bn(net):
#     """
#     Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
#     :param net: pytorch model
#     :return: save_input, class instance of SaveInput
#     :return: save_output, class instance of SaveOutput
#     """

#     save_input = SaveInput()
#     save_output = SaveOutput()

#     hook_handles_in = [] #handles list could be used to remove hooks later, here not used
#     hook_handles_out = []

#     for name, m in net.named_modules():
#         if type(m) == torch.nn.BatchNorm2d:# if type(m)==nn.Conv2d: ...
#             # print('Ranger hook set')
#             handle_in = m.register_forward_hook(save_input)
#             handle_out = m.register_forward_hook(save_output)

#             hook_handles_in.append(handle_in)
#             hook_handles_out.append(handle_out)

#     return save_input, save_output, hook_handles_in, hook_handles_out

# def run_with_hooks(net, images, bnds):
#     """
#     Creates hooks for net, executes an input image, and eliminates hooks again. Avoids GPU memory leakage issues.
#     :param net: pytorch model
#     :param inputs: image batch
#     :return fi_outputs: (tensor) inference results
#     :return activated: (list of size batchsize), nr of ranger layers that got activated in one image, collected by batch
#     """

#     if bnds is None or (bnds == [None, None]).all():
#         fi_outputs = net(images)
#         return fi_outputs, []


#     save_input = SaveInput()
#     save_output = SaveOutput()

#     hook_handles_in = [] #handles list could be used to remove hooks later, here not used
#     hook_handles_out = []

#     ranger_count_check = 0
#     for name, m in net.named_modules():
#         if type(m) == Ranger:# if type(m)==nn.Conv2d: ...
#             # print('Ranger hook set')
#             ranger_count_check += 1
#             handle_in = m.register_forward_hook(save_input)
#             handle_out = m.register_forward_hook(save_output)

#             hook_handles_in.append(handle_in)
#             hook_handles_out.append(handle_out)

#     # print('check nr rangers', ranger_count_check, len(save_input.inputs))
#     fi_outputs = net(images) 
#     # print('check nr rangers2', len(save_input.inputs)) 

#     # Check activations ---------------------------------------
#     act_in, act_out = get_max_min_lists(save_input.inputs, save_output.outputs) 
#     # print('act_out vs outputs', act_out, save_output.outputs)
#     save_input.clear()  # clear the hook lists, otherwise memory leakage
#     save_output.clear()

#     for i in range(len(hook_handles_in)):
#         hook_handles_in[i].remove()
#         hook_handles_out[i].remove()

#     activated = [] #Ranger layers activated in one image batch!
#     for n in range(len(act_in)): #images
#         act_layers = 0
        
#         for ran in range(len(act_in[n])):
#             # if (act_in[n][ran, 0] < act_out[n][ran, 0]) or (act_in[n][ran, 1] > act_out[n][ran, 1]): #todo: just different or >, <?
#             if (act_in[n][ran, 0] < bnds[ran, 0]) or (act_in[n][ran, 1] > bnds[ran, 1]): #todo: just different or >, <?
#                 act_layers += 1
#                 # print('compare: image:', n, 'rlayer:', ran, act_in[n][ran], act_out[n][ran]) #debugging
#         activated.append(act_layers)
#     # --------------------------------------------------------

#     return fi_outputs, activated

# def get_max_min_lists_in(activations_in):
#     """
#     Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
#     :param activations_in: list of tuple of tensors
#     :param activations_out: list of tensors
#     :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
#     :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
#     """
#     #Note:
#     #activations_in structure: nr ranger layers, silly bracket, batch size, channel, height, width
#     #activations_out structure: nr ranger layers, batch size, channel, height, width


#     # a = max([max([torch.max(activations_in[i][0][n]).tolist() for n in range(len(activations_in[i][0]))]) for i in range(len(activations_in))])
#     # print('incoming max', a) #debugging

#     batch_nr = activations_in[0][0].size()[0]
#     nr_rangers = len(activations_in)
#     activations_in2 = []

#     for b in range(batch_nr): #walk through a batch, i.e. through images
#         ranger_list_in = []

#         for r in range(nr_rangers):

#             rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
#             rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
#             ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])

#             # if rmax_perIm_in > 100:
#             #     print('in getfct', rmax_perIm_in, rmax_perIm_out) #todo

#         activations_in2.append(ranger_list_in)

#     return np.array(activations_in2)

# # def get_max_min_lists(activations_in, activations_out):
# #     """
# #     Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
# #     :param activations_in: list of tuple of tensors
# #     :param activations_out: list of tensors
# #     :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
# #     :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
# #     """
# #     #Note:
# #     #activations_in structure: nr ranger layers, silly bracket, batch size, channel, height, width
# #     #activations_out structure: nr ranger layers, batch size, channel, height, width


# #     # a = max([max([torch.max(activations_in[i][0][n]).tolist() for n in range(len(activations_in[i][0]))]) for i in range(len(activations_in))])
# #     # print('incoming max', a) #debugging

# #     batch_nr = activations_in[0][0].size()[0]
# #     nr_rangers = len(activations_in)
# #     activations_in2 = []
# #     activations_out2 = []

# #     for b in range(batch_nr): #walk through a batch, i.e. through images
# #         ranger_list_in = []
# #         ranger_list_out = []

# #         for r in range(nr_rangers):

# #             rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
# #             rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
# #             ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])

# #             rmax_perIm_out = torch.max(activations_out[r][b]).tolist()
# #             rmin_perIm_out = torch.min(activations_out[r][b]).tolist()
# #             ranger_list_out.append([rmin_perIm_out, rmax_perIm_out])
# #             # if rmax_perIm_in > 100:
# #             #     print('in getfct', rmax_perIm_in, rmax_perIm_out) #todo

# #         activations_in2.append(ranger_list_in)
# #         activations_out2.append(ranger_list_out)

# #     return np.array(activations_in2), np.array(activations_out2)



# def get_max_min_lists(activations_in, activations_out, get_perc=False):
#     """
#     Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
#     :param activations_in: list of tuple of tensors
#     :param activations_out: list of tensors
#     :return: act_in, list of min-max activations, format [[[min, max], ...], ...] # images in batch, ranger layers, min-max per layer
#     :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer
#     """
#     #Note:
#     #activations_in structure: nr ranger layers, silly bracket, batch size, channel, height, width
#     #activations_out structure: nr ranger layers, batch size, channel, height, width

    
#     batch_nr = activations_out[0].size()[0]
#     nr_rangers = len(activations_out)
#     activations_in2 = []
#     activations_out2 = []

#     for b in range(batch_nr): #walk through a batch (here usually just 1)
#         ranger_list_in = []
#         ranger_list_out = []

#         for r in range(nr_rangers): #walk through a layer
#             if activations_in is not None:
#                 rmax_perIm_in = torch.max(activations_in[r][0][b]).tolist()
#                 rmin_perIm_in = torch.min(activations_in[r][0][b]).tolist()
#                 ranger_list_in.append([rmin_perIm_in, rmax_perIm_in])



#             rmax_perIm_out = torch.max(activations_out[r][b]).tolist()
#             rmin_perIm_out = torch.min(activations_out[r][b]).tolist()
#             if get_perc:
#                 act_num = activations_out[r][b].cpu().numpy()
#                 p25 = np.percentile(act_num, 25)
#                 p50 = np.percentile(act_num, 50)
#                 p75 = np.percentile(act_num, 75)
#                 # p0 = min, p100 = max

#                 ranger_list_out.append([rmin_perIm_out, p25, p50, p75, rmax_perIm_out])
#             else:
#                 ranger_list_out.append([rmin_perIm_out, rmax_perIm_out])


#         if activations_in is not None:
#             activations_in2.append(ranger_list_in)
#         activations_out2.append(ranger_list_out)

#     return np.array(activations_in2), np.array(activations_out2)



    
# ## Simplified hook on all layers -----------------------------------------------------------------------------
    
# class Save_nan_inf:
#     """
#     Inputs is list with dims: nr of layers, nr of images in batch, 2 for nan, inf.
#     """
#     def __init__(self):
#         # self.inputs = []
#         self.outputs = []

#     def __call__(self, module, module_in, module_out):
#         """
#         Sequential and Bottleneck containers input/output is not considered here (skipped) for clearer monitoring.
#         """
#         ## to track the inputs also for run_with_debug_hooks_v2, uncomment below three lines
#         # input_nan_flags = module_in[0].isnan() # first index because incoming tensor wrapped as tuple
#         # input_inf_flags = module_in[0].isinf()
#         # self.inputs.append([[input_nan_flags[i].sum().item() > 0, input_inf_flags[i].sum().item() > 0] for i in range(len(input_nan_flags))])
#         try:
#             output_nan_flags = module_out.isnan() # outgoing tensor not wrapped
#         except:
#             output_nan_flags = np.array([[0]])
#         try:
#             output_inf_flags = module_out.isinf() # outgoing tensor not wrapped
#         except:
#             output_inf_flags = np.array([[0]])
#         # try:
#         monitors = np.array([[output_nan_flags[i].sum().item() > 0, output_inf_flags[i].sum().item() > 0] for i in range(len(output_nan_flags))])
#         if isinstance(module_out, torch.Tensor):
#             if module_out.shape[0] > 1:
#                 monitors = np.array([monitors[:,0].sum().item()>0, monitors[:,1].sum().item()>0])
#         self.outputs.append(monitors)
#         # except:
#         #     x = 0

#     def clear(self):
#         # self.inputs = []
#         self.outputs = []

# class Save_penult_layer:
#     """
#     Inputs is list with dims: nr of layers, nr of images in batch, 2 for nan, inf.
#     """
#     def __init__(self):
#         # self.inputs = []
#         self.outputs = []

#     def __call__(self, module, module_in, module_out):
#         """
#         Sequential and Bottleneck containers input/output is not considered here (skipped) for clearer monitoring.
#         """
#         self.outputs.append(module_in[0])
        
#     def clear(self):
#         # self.inputs = []
#         self.outputs = []

# def set_simscore_hooks(net, model_name):
#     """
#     Creates an instance of the classes Save_nan_inf that collect True, False values about whether nan, infs were found in the input and ouptut of any layer.
#     :param net: pytorch model
#     :return: save_nan_inf, class instance of Save_nan_inf
#     :return: hook_handles
#     """
#     save_penult_layer = Save_penult_layer()
#     hook_handles = [] #handles list could be used to remove hooks later, here not used
#     hook_layer_names = []

#     # cnt = 0
#     penultimate_layer = None
#     if 'alexnet' in model_name:
#         # penultimate_layer = ['classifier.3', 'classifier.4', 'classifier.6', '57', '58', '61']
#         penultimate_layer = ['classifier.5', '31']
#         for layer_name, m in net.named_modules():
#             if layer_name in penultimate_layer:
#                 # cnt += 1
#                 # print(cnt, type(m))
#                 handle_in_out = m.register_forward_hook(save_penult_layer)
#                 hook_handles.append(handle_in_out)
#                 hook_layer_names.append(m.__module__)

#         return save_penult_layer, hook_handles, hook_layer_names
#     if 'vgg' in model_name:
#         # penultimate_layer = ['classifier.3', 'classifier.4', 'classifier.6', '57', '58', '61']
#         penultimate_layer = ['classifier.6', '61']
#         for layer_name, m in net.named_modules():
#             if layer_name in penultimate_layer:
#                 # cnt += 1
#                 # print(cnt, type(m))
#                 handle_in_out = m.register_forward_hook(save_penult_layer)
#                 hook_handles.append(handle_in_out)
#                 hook_layer_names.append(m.__module__)

#         return save_penult_layer, hook_handles, hook_layer_names
#     if 'resnet' in model_name:
#         # penultimate_layer = ['classifier.3', 'classifier.4', 'classifier.6', '57', '58', '61']
#         penultimate_layer = ['fc', '42']
#         for layer_name, m in net.named_modules():
#             if layer_name in penultimate_layer:
#                 # cnt += 1
#                 # print(cnt, type(m))
#                 handle_in_out = m.register_forward_hook(save_penult_layer)
#                 hook_handles.append(handle_in_out)
#                 hook_layer_names.append(m.__module__)

#         return save_penult_layer, hook_handles, hook_layer_names

# def set_nan_inf_hooks(net):
#     """
#     Creates an instance of the classes Save_nan_inf that collect True, False values about whether nan, infs were found in the input and ouptut of any layer.
#     :param net: pytorch model
#     :return: save_nan_inf, class instance of Save_nan_inf
#     :return: hook_handles
#     """
#     save_nan_inf = Save_nan_inf()
#     hook_handles = [] #handles list could be used to remove hooks later, here not used
#     hook_layer_names = []
#     # cnt = 0
#     for _, m in net.named_modules():
#         if np.all([x not in str(type(m)) for x in ['Sequential', 'ModuleList', 'torchvision.models', 'resiliency_methods', 'torchvision.ops.feature', 'My', 'models.yolo.Detect', 'models.yolo.Model', 'models.common']]):
#             handle_in_out = m.register_forward_hook(save_nan_inf)
#             hook_handles.append(handle_in_out)
#             hook_layer_names.append(m.__module__)
#             # print('hook set for layer', type(m)) #
#     return save_nan_inf, hook_handles, hook_layer_names

# def run_with_debug_hooks_v3(net, image, bnds, ranger_activity, nan_inf_activity, resil=Ranger_trivial):


#     nan_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
#         'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False}
#     inf_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
#         'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False}

#     activated = [] #protection layers activated in one image batch!

#     # Set naninf hooks
#     if nan_inf_activity:
#         save_nan_inf, hook_handles = set_nan_inf_hooks(net)

#     # set ranger hooks
#     if ranger_activity and (bnds is not None or (bnds != [None, None]).all()):
#         save_acts, hook_handles_act = set_ranger_hooks_v2(net, resil=resil)


#     corrupted_output = net(image)


#     # Save naninf activations
#     if nan_inf_activity:
#         # nan_inf_in, nan_inf_out = save_nan_inf.inputs, save_nan_inf.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
#         nan_inf_out = save_nan_inf.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
#         save_nan_inf.clear() #clear the hook lists, otherwise memory leakage
#         for i in range(len(hook_handles)):
#             hook_handles[i].remove()


#         # Process nan
#         nan_all_layers = np.array(nan_inf_out)[:, :, 0] #rows are layers, cols are images
#         # nan_dict["overall_in"] = [np.where(nan_all_layers_in[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_in[0]))]
#         # #
#         # # nan_all_layers_out = np.array(nan_inf_out)[:, :, 0]
#         # # nan_dict["overall_out"] = [np.where(nan_all_layers_out[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_out[0]))]
#         # #
#         # nan_dict['overall'] = [np.unique(nan_dict['overall_in'][i] + nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] #in and out combined
#         nan_dict["overall"] = [np.where(nan_all_layers[:, u] == True)[0].tolist() for u in range(len(nan_all_layers[0]))] #former in only
#         nan_dict['flag'] = [x != [] for x in nan_dict['overall']]


#         #Process inf
#         inf_all_layers = np.array(nan_inf_out)[:, :, 1]
#         # inf_dict["overall_in"] = [np.where(inf_all_layers_in[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_in[0]))]
#         # #
#         # # inf_all_layers_out = np.array(nan_inf_out)[:, :, 1]
#         # # inf_dict["overall_out"] = [np.where(inf_all_layers_out[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_out[0]))]
#         # # 
#         # inf_dict['overall'] = [np.unique(inf_dict['overall_in'][i] + inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] #in and out combined
#         inf_dict["overall"] = [np.where(inf_all_layers[:, u] == True)[0].tolist() for u in range(len(inf_all_layers[0]))]
#         inf_dict['flag'] = [x != [] for x in inf_dict['overall']]



#         # Spot first nan, inf
#         spots = True
#         nan_dict['error_cause'] = []

#         if spots:
#             # if np.array(inf_dict['overall_in']).any():
#             # inf_mins_in = [np.min(u) if u !=[] else 1000 for u in inf_dict['overall_in']] #1000 as very large number larger than nw layers
#             # nan_mins_in = [np.min(u) if u !=[] else 1000 for u in nan_dict['overall_in']]
#             inf_mins_out = [np.min(u) if u !=[] else 1000 for u in inf_dict['overall']]
#             nan_mins_out = [np.min(u) if u !=[] else 1000 for u in nan_dict['overall']]
#             # comp = [[inf_mins_in[x], nan_mins_in[x], inf_mins_out[x], nan_mins_out[x]] for x in range(len(inf_mins_in))]
#             comp = [[inf_mins_out[x], nan_mins_out[x]] for x in range(len(inf_mins_out))]
#             comp_ind = [np.where(n==np.min(n))[0].tolist() if np.min(n) < 1000 else [] for n in comp]


#             layer_list = list(net.named_modules())
#             layer_list_noseq = []
#             for name, m in layer_list:
#                 if type(m) not in [torch.nn.Sequential, My_Bottleneck_Ranger]:
#                     layer_list_noseq.append([name, m])
#             layer_list = layer_list_noseq

#             info_imgs = []
#             for i in range(len(comp_ind)): #for different images
#                 if comp_ind[i] == []:
#                     info = []
#                 else:   
#                     lay = 1000
#                     # pos = ''
#                     tp = ''
#                     # print(lay, comp, comp_ind, i)

#                     if 0 in comp_ind[i]:
#                         lay = comp[i][0]
#                         # pos = pos  + 'in'
#                         tp = tp + 'Inf'
                        
#                     if 1 in comp_ind[i]:
#                         lay = comp[i][1]
#                         # pos = pos  + 'in'
#                         tp = tp + 'Nan'

#                     # if tp == 'Nan':
#                     #     print('stop')
#                     # print(lay, comp[i[0]])
#                     info = [lay, type(layer_list[lay][1]).__name__, tp]

#                 info_imgs.append(info)
#                 # if info != []:
#                 #     print(info)

#             nan_dict['error_cause'] = info_imgs
#             # if np.array(info_imgs).any() != True:
#             # print(nan_dict['error_cause'])


#    # Save ranger activations
#     if ranger_activity and (bnds is not None or (bnds != [None, None]).all()):
#         act_in = get_max_min_lists_in(save_acts.inputs) 
#         save_acts.clear()  # clear the hook lists, otherwise memory leakage
#         for i in range(len(hook_handles_act)):
#             hook_handles_act[i].remove()

#         for n in range(len(act_in)): #images
#             act_layers = 0
#             for ran in range(len(act_in[n])):
#                 # if (act_in[n][ran, 0] < act_out[n][ran, 0]) or (act_in[n][ran, 1] > act_out[n][ran, 1]): #todo: just different or >, <?
#                 if (act_in[n][ran, 0] < bnds[ran, 0]) or (act_in[n][ran, 1] > bnds[ran, 1]): #todo: just different or >, <?
#                     act_layers += 1
#                     # print('compare: image:', n, 'rlayer:', ran, act_in[n][ran], act_out[n][ran]) #debugging
#             activated.append(act_layers)    
    

#     return corrupted_output, activated, nan_dict, inf_dict

# def run_with_debug_hooks_v2(net, image):

#     nan_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
#         'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False}
#     inf_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
#         'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False}

#     save_nan_inf, hook_handles, hook_layer_names = set_nan_inf_hooks(net)

#     corrupted_output = net(image)

#     nan_inf_out = save_nan_inf.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
    
#     save_nan_inf.clear() #clear the hook lists, otherwise memory leakage
#     for i in range(len(hook_handles)):
#         hook_handles[i].remove()


#     # # Process naninf
#     # nan_all_layers = np.array(nan_inf_out)[:, :, 0]
#     # # nan_all_layers_in = np.array(nan_inf_in)[:, :, 0] #rows are layers, cols are images
#     # # nan_dict["overall_in"] = [np.where(nan_all_layers_in[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_in[0]))]
#     # #
#     # # nan_all_layers_out = np.array(nan_inf_out)[:, :, 0]
#     # # nan_dict["overall_out"] = [np.where(nan_all_layers_out[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_out[0]))]
#     # #
#     # # nan_dict['overall'] = [np.unique(nan_dict['overall_in'][i] + nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] #in and out combined
#     # nan_dict["overall"] = [np.where(nan_all_layers[:, u] == True)[0].tolist() for u in range(len(nan_all_layers[0]))] #former in only
#     # nan_dict['flag'] = [x != [] for x in nan_dict['overall']]


#     # inf_all_layers_in = np.array(nan_inf_in)[:, :, 1]
#     # inf_dict["overall_in"] = [np.where(inf_all_layers_in[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_in[0]))]
#     # #
#     # inf_all_layers_out = np.array(nan_inf_out)[:, :, 1]
#     # inf_dict["overall_out"] = [np.where(inf_all_layers_out[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_out[0]))]
#     # # 
#     # inf_dict['overall'] = [np.unique(inf_dict['overall_in'][i] + inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] #in and out combined
#     # inf_dict['flag'] = [x != [] for x in inf_dict['overall']]

#     # Process nan
#     nan_all_layers = np.array(nan_inf_out)[:, :, 0] #rows are layers, cols are images
#     # nan_dict["overall_in"] = [np.where(nan_all_layers_in[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_in[0]))]
#     # #
#     # # nan_all_layers_out = np.array(nan_inf_out)[:, :, 0]
#     # # nan_dict["overall_out"] = [np.where(nan_all_layers_out[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_out[0]))]
#     # #
#     # nan_dict['overall'] = [np.unique(nan_dict['overall_in'][i] + nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] #in and out combined
#     nan_dict["overall"] = [np.where(nan_all_layers[:, u] == True)[0].tolist() for u in range(len(nan_all_layers[0]))] #former in only
#     nan_dict['flag'] = [x != [] for x in nan_dict['overall']]


#     #Process inf
#     inf_all_layers = np.array(nan_inf_out)[:, :, 1]
#     # inf_dict["overall_in"] = [np.where(inf_all_layers_in[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_in[0]))]
#     # #
#     # # inf_all_layers_out = np.array(nan_inf_out)[:, :, 1]
#     # # inf_dict["overall_out"] = [np.where(inf_all_layers_out[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_out[0]))]
#     # # 
#     # inf_dict['overall'] = [np.unique(inf_dict['overall_in'][i] + inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] #in and out combined
#     inf_dict["overall"] = [np.where(inf_all_layers[:, u] == True)[0].tolist() for u in range(len(inf_all_layers[0]))]
#     inf_dict['flag'] = [x != [] for x in inf_dict['overall']]


#     return corrupted_output, nan_dict, inf_dict

# def run_nan_inf_hooks(save_nan_inf, hook_handles, hook_layer_names):

#     nan_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
#         'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False, 'first_occurrence': [], 'first_occur_compare': []}
#     inf_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
#         'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False, 'first_occurrence': []}

#     # save_nan_inf, hook_handles, hook_layer_names = set_nan_inf_hooks(net)
#     # corrupted_output = net(image)len()

#     nan_inf_out = np.array(save_nan_inf.outputs) #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
    
#     save_nan_inf.clear() #clear the hook lists, otherwise memory leakage
#     for i in range(len(hook_handles)):
#         hook_handles[i].remove()

#     # Process naninf
#     # nan_all_layers_in = np.array(nan_inf_in)[:, :, 0] #rows are layers, cols are images
#     # nan_dict["overall_in"] = [np.where(nan_all_layers_in[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_in[0]))]
#     #
#     try:
#         nan_all_layers_out = np.array(nan_inf_out)[:, :, 0]
#     except:
#         nan_all_layers_out = np.array(nan_inf_out).reshape(-1,1,2)[:,:,0]

#     nan_dict["overall_out"] = [np.where(nan_all_layers_out[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_out[0]))]
#     # 
#     # nan_dict['overall'] = [np.unique(nan_dict['overall_in'][i] + nan_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] #in and out combined
#     nan_dict['overall'] = [np.unique(nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] 
#     nan_dict['flag'] = [x != [] for x in nan_dict['overall']]
#     for i in range(len(nan_dict['overall'])):
#         first_nan_layer_index = nan_dict['overall'][i]
#         if first_nan_layer_index: #TODO:
#             # print(first_nan_layer_index[0])
#             if first_nan_layer_index[0] > len(hook_layer_names):
#                 nan_dict['first_occurrence'].append([first_nan_layer_index[0], 'nan', 'layer'])
#             else:
#                 nan_dict['first_occurrence'].append([first_nan_layer_index[0], 'nan', hook_layer_names[first_nan_layer_index[0]]])
#         else:
#             nan_dict['first_occurrence'].append([])


#     # inf_all_layers_in = np.array(nan_inf_in)[:, :, 1]
#     try:
#         inf_all_layers_out = np.array(nan_inf_out)[:, :, 1]
#     except:
#         inf_all_layers_out = np.array(nan_inf_out).reshape(-1,1,2)[:,:,1]

#     inf_dict["overall_out"] = [np.where(inf_all_layers_out[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_out[0]))]
#     # 
#     # inf_dict['overall'] = [np.unique(inf_dict['overall_in'][i] + inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] #in and out combined
#     inf_dict['overall'] = [np.unique(inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] 
#     inf_dict['flag'] = [x != [] for x in inf_dict['overall']]

#     for i in range(len(inf_dict['overall'])):
#         first_inf_layer_index = inf_dict['overall'][i]
#         if first_inf_layer_index: #TODO:
#             # print(first_inf_layer_index[0], len(hook_layer_names)) #hook layer names
#             if first_inf_layer_index[0] > len(hook_layer_names):
#                 inf_dict['first_occurrence'].append([first_inf_layer_index[0], 'inf', 'layer'])
#             else:
#                 inf_dict['first_occurrence'].append([first_inf_layer_index[0], 'inf', hook_layer_names[first_inf_layer_index[0]]])
#         else:
#             inf_dict['first_occurrence'].append([])

#     for i in range(len(nan_dict['first_occurrence'])):
#         nan_info = nan_dict['first_occurrence'][i]
#         inf_info = inf_dict['first_occurrence'][i]
#         if nan_info and inf_info:
#             nan_inf_list = [nan_info, inf_info]
#             if nan_info[0] == inf_info[0]:
#                 nan_dict['first_occur_compare'].append(nan_inf_list)
#             else:
#                 nan_dict['first_occur_compare'].append(nan_inf_list[np.argmin([nan_info[0], inf_info[0]])])
#         elif nan_info:
#             nan_dict['first_occur_compare'].append(nan_info)
#         elif inf_dict['first_occurrence'][i]:
#             nan_dict['first_occur_compare'].append(inf_info)
#         else:
#             nan_dict['first_occur_compare'].append([])

#     return nan_dict, inf_dict

# ## deprecated
# def run_with_debug_hooks_simple(net, image):

#     nan_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
#         'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False, 'first_occurrence': [], 'first_occur_compare': []}
#     inf_dict = {'relu_in': [], 'relu_out': [], 'conv_in': [], 'conv_out': [], 'bn_in': [], 'bn_out': [], 'relu_in_glob': [], 'relu_out_glob': [], \
#         'conv_in_glob': [], 'conv_out_glob': [], 'bn_in_glob': [], 'bn_out_glob': [], 'overall_in': [], 'overall_out': [], 'overall': [], 'flag': False, 'first_occurrence': []}

#     save_nan_inf, hook_handles, hook_layer_names = set_nan_inf_hooks(net)

#     corrupted_output = net(image)

#     nan_inf_out = save_nan_inf.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
    
#     save_nan_inf.clear() #clear the hook lists, otherwise memory leakage
#     for i in range(len(hook_handles)):
#         hook_handles[i].remove()

#     # Process naninf
#     # nan_all_layers_in = np.array(nan_inf_in)[:, :, 0] #rows are layers, cols are images
#     # nan_dict["overall_in"] = [np.where(nan_all_layers_in[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_in[0]))]
#     #
#     nan_all_layers_out = np.array(nan_inf_out)[:, :, 0]
#     nan_dict["overall_out"] = [np.where(nan_all_layers_out[:, u] == True)[0].tolist() for u in range(len(nan_all_layers_out[0]))]
#     #
#     # nan_dict['overall'] = [np.unique(nan_dict['overall_in'][i] + nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] #in and out combined
#     nan_dict['overall'] = [np.unique(nan_dict['overall_out'][i]).tolist() for i in range(len(nan_dict['overall_out']))] 
#     nan_dict['flag'] = [x != [] for x in nan_dict['overall']]
#     for i in range(len(nan_dict['overall'])):
#         first_nan_layer_index = nan_dict['overall'][i]
#         # nan_dict['first_occurrence'].append([]) #TODO:
#         if first_nan_layer_index:
#             nan_dict['first_occurrence'].append([first_nan_layer_index[0], 'nan', hook_layer_names[first_nan_layer_index[0]]])
#         else:
#             nan_dict['first_occurrence'].append([])

#     # inf_all_layers_in = np.array(nan_inf_in)[:, :, 1]
#     # inf_dict["overall_in"] = [np.where(inf_all_layers_in[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_in[0]))]
#     #
#     inf_all_layers_out = np.array(nan_inf_out)[:, :, 1]
#     inf_dict["overall_out"] = [np.where(inf_all_layers_out[:, u] == True)[0].tolist() for u in range(len(inf_all_layers_out[0]))]
#     # 
#     # inf_dict['overall'] = [np.unique(inf_dict['overall_in'][i] + inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] #in and out combined
#     inf_dict['overall'] = [np.unique(inf_dict['overall_out'][i]).tolist() for i in range(len(inf_dict['overall_out']))] 
#     inf_dict['flag'] = [x != [] for x in inf_dict['overall']]
#     for i in range(len(inf_dict['overall'])):
#         first_inf_layer_index = inf_dict['overall'][i]
#         # inf_dict['first_occurrence'].append([]) #TODO:
#         if first_inf_layer_index:
#             inf_dict['first_occurrence'].append([first_inf_layer_index[0], 'inf', hook_layer_names[first_inf_layer_index[0]]])
#         else:
#             inf_dict['first_occurrence'].append([])

#     for i in range(len(nan_dict['first_occurrence'])):
#         nan_info = nan_dict['first_occurrence'][i]
#         inf_info = inf_dict['first_occurrence'][i]
#         if nan_info and inf_info:
#             nan_inf_list = [nan_info, inf_info]
#             if nan_info[0] == inf_info[0]:
#                 nan_dict['first_occur_compare'].append(nan_inf_list)
#             else:
#                 nan_dict['first_occur_compare'].append(nan_inf_list[np.argmin([nan_info[0], inf_info[0]])])
#         elif nan_info:
#             nan_dict['first_occur_compare'].append(nan_info)
#         elif inf_dict['first_occurrence'][i]:
#             nan_dict['first_occur_compare'].append(inf_info)
#         else:
#             nan_dict['first_occur_compare'].append([])

#     return corrupted_output, nan_dict, inf_dict

# def run_simscore_hooks(save_penult_layer, hook_handles):

#     # save_nan_inf, hook_handles, hook_layer_names = set_nan_inf_hooks(net)
#     # save_penult_layer, hook_handles, _ = set_simscore_hooks(net, model_name)

#     # corrupted_output = net(image)

#     save_penult_layer_out = save_penult_layer.outputs #format of nan_inf_in and _out is a list of length nr_network_layers and two columns with True/False for each layer depending on whether nan, infs were found or not.
    
#     save_penult_layer.clear() #clear the hook lists, otherwise memory leakage
#     for i in range(len(hook_handles)):
#         hook_handles[i].remove()

#     return save_penult_layer_out

# def print_nan_inf_hist_v2(net, nan_dict_corr, inf_dict_corr):

#     comb = np.array([nan_dict_corr['overall'][i] + inf_dict_corr['overall'][i] for i in range(len(nan_dict_corr['overall']))]).flatten()
#     # mn, mx = min(nan_dict_corr['overall'] + inf_dict_corr['overall']), max(nan_dict_corr['overall'] + inf_dict_corr['overall']) #for old version
#     if not comb.any():
#         return

#     mn, mx = min(comb), max(comb)

#     layer_list = list(net.named_modules())
#     layer_list_noseq = []
#     for name, m in layer_list:
#         if type(m) not in [torch.nn.Sequential, My_Bottleneck_Ranger]:
#             layer_list_noseq.append([name, m])
#     layer_list = layer_list_noseq

#     info = []
#     for u in range(mn, mx+1):

#         in_info = ['No', 'No']
#         if u in np.array(nan_dict_corr['overall_in']).flatten():
#             in_info[0] = 'NaN'
#         if u in np.array(inf_dict_corr['overall_in']).flatten():
#             in_info[1] = 'Inf'
#         out_info = ['No', 'No']
#         if u in np.array(nan_dict_corr['overall_out']).flatten():
#             out_info[0] = 'NaN'
#         if u in np.array(inf_dict_corr['overall_out']).flatten():
#             out_info[1] = 'Inf'

#         info.append([u, layer_list[u], 'in', in_info, 'out', out_info])

#     print(tabulate(info)) #, headers=['Name', 'Age']
