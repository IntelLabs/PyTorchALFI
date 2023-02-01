import torch
import numpy as np
from tqdm import tqdm
import torch


def update_act_minmaxlists(act_in_global_min, act_in_global_max, act_in, get_perc=False):
    """
    Update the global separate lists for min and max with the data of the current data in act_in (min, max in a single list)
    :param act_in_global_min: array of all mins (Nx0 shape)
    :param act_in_global_max: array of all maxs (Nx0 shape)
    :param act_in: array of shape BxNx2 with mins and max, B batch number, N numer of ranger layers
    :return: act_in_global_min: array of all mins (Nx0 shape)
    :return: act_in_global_max: array of all mins (Nx0 shape)
    """

    for btch in range(len(act_in)):
        act_in_btch = act_in[btch]

        if act_in_global_min is None: #first time
            act_in_global_min = np.array(act_in_btch[:, 0])
        else:
            act_in_global_min = np.vstack((act_in_global_min, act_in_btch[:, 0]))  # makes stack of rows of format Nx0
            act_in_global_min = np.min(act_in_global_min, axis=0)

        if act_in_global_max is None:
            act_in_global_max = np.array(act_in_btch)[:, 1]
        else:
            act_in_global_max = np.vstack((act_in_global_max, act_in_btch[:, 1]))  # rows are samples here
            act_in_global_max = np.max(act_in_global_max, axis=0)

        if get_perc:
            print()
            # TODO update all 5 lists

    return act_in_global_min, act_in_global_max


def extract_ranger_bounds(dataset, net, file_name, get_perc=False):
    """
    Extract ranger bounds.
    """

    act_out_global_min = None
    act_out_global_max = None

    save_output, hook_handles_out = set_ranger_hooks_ReLU(net)  # classes that save activations via hooks

    with torch.no_grad():
        print('Extracting bounds from data set' + '...')
        i = 0
        pbar = tqdm(total = len(dataset))
        while i < len(dataset):
            input_batch = dataset[i]

            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                net = net.to('cuda')

            with torch.no_grad():
                net(input_batch)

            # act_in = save_input.inputs
            act_out = save_output.outputs

            # save_input.clear() #clear the hook lists, otherwise memory leakage
            save_output.clear()

            # # Check activations (extract only min, max from the full list of activations)
            _, act_out = get_max_min_lists(None, act_out, get_perc=get_perc)

            act_out_global_min, act_out_global_max = update_act_minmaxlists(act_out_global_min, act_out_global_max, act_out)

            pbar.update(i)
            i+=1
        pbar.close()

    # below loop required to solve memory leakage
    for i in range(len(hook_handles_out)):
        hook_handles_out[i].remove()
        
    act_out_global = combine_act_minmaxlists(act_out_global_min, act_out_global_max)

    save_Bounds_minmax(act_out_global, file_name) #save outgoing (incoming?) activations, already in Bounds folder

    return act_out_global



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



def combine_act_minmaxlists(act_in_global_min, act_in_global_max):
    """
    Transforms the separate global lists for min max in one single dict.
    :param act_in_global_min: array of mins shape Nx0
    :param act_in_global_max: array of max shape Nx0
    :return: one list with [min, max] as rows
    """
    val = np.vstack((act_in_global_min, act_in_global_max))
    val = (val.T).tolist()

    return val


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


def set_ranger_hooks_ReLU(net):
    """
    Creates two instances of the classes SaveInput and SaveOutput that collect the input and ouptut to/of the Ranger layers, respectively.
    :param net: pytorch model
    :return: save_output, class instance of SaveInput
    :return: hook_handles_out, list of hook handles
    """

    save_output = SaveOutput()

    hook_handles_out = []

    for name, m in net.named_modules():
        if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]: #any type of activation function!
            # print('Ranger hook set', type(m), name)
            handle_out = m.register_forward_hook(save_output)
            hook_handles_out.append(handle_out)

    return save_output, hook_handles_out



class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []



class Range_detector:
    def __init__(self, bnd, act_list, mitigation):
        self.act_list = act_list #list of previous activations
        self.bnd = bnd #specific bound for that hook (format [min, max])
        self.mit = mitigation

    def __call__(self, module, module_in, module_out):
        module_out_clamped = torch.clamp(module_out, min=self.bnd[0], max=self.bnd[1])
        comp = torch.sub(module_out, module_out_clamped) #difference between before and after clamping
        # self.act_list.append(torch.any(comp).tolist())
        # print(torch.any(comp).tolist(), 
        self.act_list.append((torch.sum(comp) > 0).tolist())
        if self.mit:
            return module_out_clamped
        else:
            return module_out

    def clear(self):
        self.act_list = []
        self.bnd = [0, 0]



def load_bounds(filename):
    f = open(filename, "r")
    bounds = []
    if f.mode == 'r':
        contents = f.read().splitlines()
        bounds = [u.split(',') for u in contents]
    f.close()

    bounds = [[float(n[0]), float(n[1])] for n in bounds] #make numeric
    return bounds


def set_ranger_hooks(net, bnds, mitigation=False):
    """
    Sets hooks into the entire model. Hooks have individual bounds but will all write to the same list act_list.
    :param net: pytorch model
    :param bnds: list of bounds
    :param mitigation: flat whether or not Ranger mitigation should be applied.
    :return: act_list, list of saved information from Range_detector hooks.
    :return: hook_handles_out, list of handles to be cleared later
    :return: hook_list, list of hooks to be cleared later
    """

    act_list = []
    hook_handles_out = []
    hook_list = []
    cnt = 0
    for _, m in net.named_modules():
        if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid]:
            # print('check', cnt, m)
            act_hook = Range_detector(bnds[cnt], act_list, mitigation)
            handle_out = m.register_forward_hook(act_hook)
            hook_handles_out.append(handle_out)
            hook_list.append(act_hook)
            cnt += 1

    return act_list, hook_handles_out, hook_list



def run_with_range_supervision(model, input_batch, bnds, mitigation=False):
    """
    :param net: original net, pretrained
    :param input_batch: input
    :param bnds: bounds for ranger, extracted from all activation fcts
    :param mitigation: should out-of-bound values be clamped or no?
    :return: output of model inference with input_batch.
    :return: list of out-of-bound events (boolean).
    """
    
    # set the hooks
    act_list, hook_handles_out, hook_list = set_ranger_hooks(model, bnds, mitigation)

    # Prepare and run model
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')
    model.eval()
    with torch.no_grad():
        output = model(input_batch)

    # Clean up
    for i in range(len(hook_handles_out)):
        hook_handles_out[i].remove()
        hook_list[i].clear()


    return output, act_list