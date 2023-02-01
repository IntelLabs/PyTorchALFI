from netrc import netrc
import torch
from torch.autograd import Variable
import pickle
import numpy as np
# from copy import deepcopy
# import psutil
import sys
from tqdm import tqdm
# import collections
import gc
from alficore.ptfiwrap_utils.hook_functions import set_ranger_hooks, set_ranger_hooks_ReLU, get_max_min_lists
from alficore.dataloader.objdet_baseClasses.common import pad_to_square, resize
from alficore.ptfiwrap_utils.helper_functions import get_savedBounds_minmax, save_Bounds_minmax, get_max_min_lists
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr, assign_val_train
from typing import Dict, List

cuda_device = 0
if torch.cuda.is_available():
    device = torch.device(
        "cuda:{}".format(cuda_device))    
else:
    device = torch.device("cpu")
# Activation/bound extraction -----------------

def update_act_minmaxlists(act_in_global_min, act_in_global_max, act_in):
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


    return act_in_global_min, act_in_global_max


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

def preprocess_image_yolo_model(batched_inputs: List[Dict[str, torch.Tensor]]):
    """
    Normalize, pad and batch the input images.
    ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
    [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
    This is converted into a tensor batch as expected by the model
    """
    img_size = 416
    images = [x["image"]/255. for x in batched_inputs]
    # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # Pad to square resolution
    padded_imgs = [pad_to_square(img, 0)[0] for img in images]
    # Resize
    images = [resize(img, img_size) for img in padded_imgs]
    # Convert to tensor
    images = torch.stack(images).to(device)
    ## normalisde the input if neccesary
    return images



def extract_ranger_bounds(data_loader, net, file_name, dataset='imagenet'):
    """
    Extracts the ranger bounds from all the images in test_loader, for network net and saves them in bounds directory.
    Applies to all activation layers (ReLU) not the Ranger. Only saves the input bounds.
    :param data_loader: dataset loader, pytorch dataloader
    :param net: network (pretrained), pytorch model
    :param file_name: Name for bounds file, string
    :param dataset_name: Name of the dataset using which Ranger bounds are generated
    :return: list of min, max ranger inputs of the form list of [[min, max], [min, max], ...]
    :return: list of min, max ranger output of the form list of [[min, max], [min, max], ...]
    """

    act_in_global_min = None  # Lists that collect all mins and max across the image runs
    act_in_global_max = None
    act_out_global_min = None
    act_out_global_max = None
    net = net.to(device)
    net.eval()

    # memory debugging ---
    # pid = os.getpid()
    # py = psutil.Process(pid)
    # init_mem = py.memory_info()[0] / 2. ** 30
    # print(f"Torch Memory: Init memory = {init_mem:.3e}Gb")
    # ----------------------

    # save_input, save_output, hook_handles_in, hook_handles_out = set_ranger_hooks(net) #classes that save activations via hooks
    save_input, save_output, hook_handles_in, hook_handles_out = set_ranger_hooks_ReLU(net)  # classes that save activations via hooks

    with torch.no_grad():
        print('Extracting bounds from data set' + '(' + str(data_loader.__len__()) + ' batches)' + '...')
        # for j, (images, labels) in enumerate(data_loader):
        #     if j % 100 == 0:
        #         print(' Batch nr:', j)

        #     net(images) #have to call forward pass to activate the hooks, output doesnt matter here
        # data_loader = iter(data_loader)
        for i, x in enumerate(data_loader):
            if dataset == 'miovision':
                val_images = Variable(x['image']).to(device)
            elif dataset == 'imagenet':    
                val_images = x[0].to(device)
            elif dataset == 'MS CoCo' or dataset == 'coco':
                val_images = x[1].to(device)
            else:
                print('Please choose a supported dataset.')

            net(val_images.float())
            print('Batch number:', i)
            act_in = save_input.inputs
            act_out = save_output.outputs

            save_input.clear() #clear the hook lists, otherwise memory leakage
            save_output.clear()

            # # Check activations (extract only min, max from the full list of activations)
            act_in, act_out = get_max_min_lists(act_in, act_out)

            act_in_global_min, act_in_global_max = update_act_minmaxlists(act_in_global_min, act_in_global_max, act_in) #flat lists
            act_out_global_min, act_out_global_max = update_act_minmaxlists(act_out_global_min, act_out_global_max, act_out)

            # Debug memory ---------------------------------
            # memoryUse = py.memory_info()[0] / 2. ** 30
            # print(f"Torch Memory: Memory = {memoryUse:.3e}Gb")
            # print('object sizes net:', sys.getsizeof(net), net.__sizeof__())
            # print('object sizes output:', sys.getsizeof(output))
            # --------------------------------------------------
            gc.collect() # collect garbage

            # if j == 3: #break if smaller dataset should be used
            #     break

    # below loop required to solve memory leakage
    for i in range(len(hook_handles_in)):
        hook_handles_in[i].remove()
        hook_handles_out[i].remove()
        
    act_in_global = combine_act_minmaxlists(act_in_global_min, act_in_global_max)
    act_out_global = combine_act_minmaxlists(act_out_global_min, act_out_global_max)

    save_Bounds_minmax(act_out_global, file_name) #save outgoing (incoming?) activations, already in Bounds folder

    return act_in_global, act_out_global



def get_Ranger_bounds2(model, ranger_file_name, dl_attr:TEM_Dataloader_attr, get_quantiles=False, get_ftraces=False):
    # Confirmed: 72 leaky relu layers, no relu layers
    
    print('Switching to training set for bound extraction.')
    dl_attr.dl_dataset_type = 'train' #by default take bounds from training data
    dl_attr = assign_val_train(dl_attr)

    if 'coco'.lower() in dl_attr.dl_dataset_name:
        # from alficore.dataloader.coco_loader import CoCo_obj_det_dataloader
        from alficore.dataloader.coco_loader import CoCo_obj_det_native_dataloader
        print('Loading coco...')
        dataloader = CoCo_obj_det_native_dataloader(dl_attr=dl_attr, dnn_model_name = model.model._get_name().lower())
    if 'kitti'.lower() in dl_attr.dl_dataset_name.lower():
        # from alficore.dataloader.kitti_loader import Kitti2D_dataloader
        print('Loading kitti...')
        from alficore.dataloader.kitti_loader import Kitti_obj_det_native_dataloader
        dataloader = Kitti_obj_det_native_dataloader(dl_attr=dl_attr, dnn_model_name = model.model._get_name().lower())
    # if 'lyft'.lower() in dl_attr.dl_dataset_name.lower():
    #     from alficore.dataloader.lyft_loader import Lyft_dataloader
    #     print('Loading lyft...')
    #     dataloader = Lyft_dataloader(dl_attr)
    # if 'robo'.lower() in dl_attr.dl_dataset_name.lower():
    #     from alficore.dataloader.robo_loader import Robo_obj_det_dataloader
    #     print('Loading robo...')
    #     dataloader = Robo_obj_det_dataloader(dl_attr)
    # if 'fgvc'.lower() in dl_attr.dl_dataset_name.lower():
    #     from alficore.dataloader.fgvc_loader import FGVC_dataloader
    #     print('Loading fgvc...')
    #     dataloader = FGVC_dataloader(dl_attr)
    if 'imagenet'.lower() in dl_attr.dl_dataset_name.lower():
        from alficore.dataloader.imagenet_loader_rgb import Imagenet_dataloader
        print('Loading ImageNet...')
        dataloader = Imagenet_dataloader(dl_attr=dl_attr)
    if  'mnist'.lower() in dl_attr.dl_dataset_name.lower():
        print('Loading Mnist...')
        from alficore.dataloader.mnist_loader_rgb import MNIST_dataloader
        dataloader = MNIST_dataloader(dl_attr=dl_attr)


    # Extract the bounds
    BoundEx = Bound_extractor(model, dataloader)

    BoundEx.prepare_model(dl_attr.dl_device)
    BoundEx.run_info_extraction_inferences(get_quantiles, get_ftraces)
    BoundEx.postprocess_bound_data(get_quantiles, get_ftraces)
    BoundEx.save_results(ranger_file_name, get_quantiles, get_ftraces)
    BoundEx.clean_up()

    return


class Bound_extractor:
    def __init__(self, net, dataloader):
        self.qu_to_monitor = [0,10,20,30,40,50,60,70,80,90,100] #NOTE: align with hook_functions-> Range_detector_quantiles
        self.quantile_pre_summation = True #True, False #NOTE: align with hook_functions-> Range_detector_quantiles

        self.act_max_global = -np.Inf
        self.act_min_global = +np.Inf
        self.act_out_global = []
        self.qu_list_global = self.init_global_dict()
        self.net = net
        self.dataloader = dataloader
        

    def clean_up(self):
        self.act_max_global = -np.Inf
        self.act_min_global = +np.Inf
        self.act_out_global = [] #keep the regular bounds?
        self.qu_list_global = {}
        self.net = None
        self.dataloader = None


    def init_global_dict(self):
        qstr = ['q' + str(x) for x in self.qu_to_monitor]
        qu_list_global = dict(zip(qstr, [[] for n in qstr]))
        qstr_min = ['q' + str(x) + '_min' for x in self.qu_to_monitor]
        qstr_avg = ['q' + str(x) + '_avg' for x in self.qu_to_monitor]
        qstr_max = ['q' + str(x) + '_max' for x in self.qu_to_monitor]
        qu_list_global.update(dict(zip(qstr_min, [[] for n in qstr_min])))
        qu_list_global.update(dict(zip(qstr_avg, [[] for n in qstr_avg])))
        qu_list_global.update(dict(zip(qstr_max, [[] for n in qstr_max])))
        ftrace_dict = {'ftraces': [], 'ftraces_max': [], 'ftraces_min': [], 'ftraces_mu': [], 'ftraces_sigma': [], 'ftraces_fmap_disp': [], 'N_total': 0}
        qu_list_global.update(ftrace_dict)
        print('Monitoring the following properties:', list(qu_list_global.keys()))
        return qu_list_global

    def prepare_model(self, device):
        self.net = self.net.to(device)
        self.net.eval()
        self.qu_list_global['N_total'] = self.dataloader.dataset_length

    def run_info_extraction_inferences(self, get_quantiles, get_ftraces):
        with torch.no_grad():
            print('Extracting bounds from data set' + '(len=' + str(self.dataloader.dataset_length) + ')...')
            # for j, (images, labels) in enumerate(data_loader):
            #     if j % 100 == 0:
            #         print(' Batch nr:', j)
            i = 0 #image counter
            pbar = tqdm(total = self.dataloader.dataset_length)
            while self.dataloader.data_incoming:
                self.dataloader.datagen_itr()
                # print('image nr', i)

                if hasattr(self.dataloader, 'data'):
                    images = self.dataloader.data
                elif hasattr(self.dataloader, 'images'):
                    images = self.dataloader.images

                hook_list_out, hook_handles_out = set_ranger_hooks_ReLU(self.net)  # classes that save activations via hooks (all activations, for use in quantiles etc.)

                self.net(images)

                # Process hook content
                names_out = [n.lay_info[1] for n in hook_list_out]
                assert len(names_out) == len(np.unique(names_out)) 
                act_out = [n.outputs for n in hook_list_out] #if this shows multiple entries per hook then layers are reused somehow and bound extraction goes wrong

                # below loop required to solve memory leakage
                for k in range(len(hook_handles_out)):
                    hook_handles_out[k].remove()
                    hook_list_out[k].clear()


                # # Check activations (extract only min, max from the full list of activations)
                min_max_out, qu_out, ft_out, cnt_out = self.calculate_measures(act_out, get_quantiles=get_quantiles, get_ftraces=get_ftraces) #cnt_out gets overwritten but network behaves the same all the time so ok
                # print('extracted')

                # Add quantiles
                # qu_out is now of dimension order: layers, quantiles, imgs in batch
                qu_out = self.fix_unusual_layers_mod(qu_out)

                if len(qu_out)>0:
                    ind = 0
                    for n in self.qu_to_monitor:
                        self.qu_list_global['q'+str(n)].append(np.array(qu_out)[:,ind,:])
                        ind += 1
                
                # Add ftrace
                if len(ft_out)>0:
                    self.qu_list_global['ftraces'].append(ft_out) #dim: 72 x batchsize x laysize 

                # Add min, max
                self.act_max_global = np.maximum(self.act_max_global, min_max_out) #elementwise maximum 
                self.act_min_global = np.minimum(self.act_min_global, min_max_out) #elementwise minimum

                gc.collect() # collect garbage
                pbar.update(self.dataloader.curr_batch_size)
                i+=1

            pbar.close()


        lens = np.unique(cnt_out)
        if len(lens)>1 or (len(lens)==1 and lens[0]!=self.dataloader.curr_batch_size):
            print('INFO: Model appears to reuse layers that are hooked for bound extraction. We take care of this. Per inference, cnt_out=', cnt_out)

        return 
    
    def calculate_measures(self, activations_out, get_quantiles, get_ftraces):
        """
        Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
        :param activations_in: list of tuple of tensors
        :param activations_out: list of tensors
        :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer

        :qu_list_out2: has dims: nr of layers, 3 (q25, q50, q75), batch_size
        :activations_out2: has dims: nr of layers, 2
        """

        if len(activations_out) == 0: #no activations at all
            return None

        list_min_max = [] #layerwise list with mins, maxs, etc
        list_quant = []
        list_ftraces = []
        count_out = []
        

        for lay in range(len(activations_out)):
            if len(activations_out[lay]) == 0: #remove empty layer for retina net
                print('A layer returned empty monitoring output, skipping this layer for monitoring.')
                # list_min_max.append([None, None])
                continue

            mn_multi = []
            mx_multi = []
            for x in self.qu_to_monitor: #assign all q-variables (q10, q20 etc.)
                globals()[f"q{x}_multi"] = []

            ftraces_multi = []
            cnt = 0 #how much is a hook used?
            # map back multi-use to imgs
            for btch_img in range(len(activations_out[lay])): #batch size or multi-use
                
                cnt += 1
                mn = torch.min(activations_out[lay][btch_img])
                mx = torch.max(activations_out[lay][btch_img])
                mn_multi.append(mn.cpu().numpy())
                if len(mn_multi) == 0:
                    print()
                mx_multi.append(mx.cpu().numpy())

                if get_quantiles:
                    tnsr = activations_out[lay][btch_img]

                    if self.quantile_pre_summation and len(tnsr.shape) > 1:
                        dims_sum = np.arange(1,len(tnsr.shape)).tolist() #here start from one because batch is taken care of
                        tnsr = torch.sum(tnsr, dim=dims_sum)

                    tnsr = torch.flatten(tnsr) #here batches are taken care of explicitly in for loop
                    quants_all = torch.quantile(tnsr, torch.tensor(np.array(self.qu_to_monitor)/100, device=tnsr.device, dtype=tnsr.dtype)) #here batches are taken care of explicitly in for loop
                    
                    ind = 0
                    for x in self.qu_to_monitor: #assign all q-variables (q10, q20 etc.)
                        globals()[f"q{x}_multi"].append(quants_all[ind].cpu().numpy())
                        ind += 1

                if get_ftraces:
                    if len(activations_out[lay][btch_img].shape) == 1:
                        ftraces_multi.append(activations_out[lay][btch_img].tolist())
                    elif len(activations_out[lay][btch_img].shape) == 3:
                        ftraces_multi.append(torch.sum(activations_out[lay][btch_img], dim=[1,2]).tolist())
                    else:
                        print("Warning: Layer activations were found that are different from the expected dimensions of conv or fcc output.")
                        sys.exit()

            mn = np.min(mn_multi)
            mx = np.max(mx_multi)
        
            if get_quantiles:
                count_out.append(cnt)
                list_quant.append([globals()[f"q{x}_multi"] for x in self.qu_to_monitor])
            if get_ftraces:
                count_out.append(cnt)
                list_ftraces.append(ftraces_multi)

            list_min_max.append([float(mn), float(mx)])

        return np.array(list_min_max), list_quant, list_ftraces, np.array(count_out)

    def postprocess_bound_data(self, get_quantiles, get_ftraces):

        # min max lists
        self.act_out_global = np.vstack((self.act_min_global[:,0], self.act_max_global[:,1])).T.tolist() #min, max list of dim nr_layers, 2
        # save_Bounds_minmax(act_out_global, file_name) #save outgoing (incoming?) activations, already in Bounds folder
        nr_layers = len(self.act_out_global)
        
        # quantiles
        if get_quantiles:
            print('Processing quantile info...')
        
            def get_layer_list_qu(qu_list, lay):
                extr_list = []
                for btch in range(len(qu_list)):
                    extr_list.extend(qu_list[btch][lay,:].tolist())
                return np.array(extr_list)
            
            for x in self.qu_to_monitor:
                for m in range(nr_layers):
                    extr_list = get_layer_list_qu(self.qu_list_global["q"+str(x)], m)
                    self.qu_list_global["q"+str(x)+"_avg"].append([np.mean(extr_list), 1.96*np.std(extr_list)/np.sqrt(len(extr_list))])
                    self.qu_list_global["q"+str(x)+"_min"].append(np.min(extr_list))
                    self.qu_list_global["q"+str(x)+"_max"].append(np.max(extr_list))

                # # Clear raw data as it is too slow...
                self.qu_list_global["q"+str(x)] = [] #self.qu_list_global["q"+str(x)].tolist()

        
        if get_ftraces:
            print('Processing ftraces info...')

            def get_layer_list_ft(qu_list, lay):
                extr_list = []
                for btch in range(len(qu_list)):
                    extr_list.extend(qu_list[btch][lay])
                return np.array(extr_list).astype(float)


            for n in range(nr_layers):
                extr_list = get_layer_list_ft(self.qu_list_global['ftraces'], n)

                self.qu_list_global['ftraces_fmap_disp'].append([np.min(np.std(extr_list,1)), np.mean(np.std(extr_list,1)), np.max(np.std(extr_list,1)), np.std(np.std(extr_list,1))]) #first disp across fmaps, then min, max etc across samples
                self.qu_list_global['ftraces_max'].append(np.max(extr_list, 0).tolist())
                self.qu_list_global['ftraces_min'].append(np.min(extr_list, 0).tolist())
                self.qu_list_global['ftraces_mu'].append(np.mean(extr_list, 0).tolist())
                self.qu_list_global['ftraces_sigma'].append(np.std(extr_list, 0).tolist())

            # Clean up before saving (too much)
            self.qu_list_global['ftraces'] = []

        return

    def save_results(self, file_name, get_quantiles, get_ftraces):

        save_Bounds_minmax(self.act_out_global, file_name)

        if get_ftraces or get_quantiles:
            save_Bounds_quantiles(self.qu_list_global, file_name + '_quantiles_ftraces') #save both in one file

    def fix_unusual_layers_mod(self, qu_out):
        bs = self.dataloader.curr_batch_size
        unusual_layers = (np.array([len(n[0]) for n in qu_out]) != bs)
        if unusual_layers.any():
            # print('Network has unusual layers...')
            for x in range(len(unusual_layers)):
                if unusual_layers[x]:
                    qu_x = np.swapaxes(qu_out[x], 0,1)
                    mlt_factor = int(len(qu_x)/bs)
                    if mlt_factor < 1:
                        qu_x = None
                    elif mlt_factor > 1:
                        qu_x = (np.sum([qu_x[n*bs:(n+1)*bs] for n in range(mlt_factor)],0)/mlt_factor).tolist()
                        assert len(qu_x) == bs
                    qu_out[x] = np.swapaxes(qu_x,0,1)

            qu_out = [i for i in qu_out if i is not None]
        return qu_out


# def extract_ranger_bounds_auto2(data_loader, net, file_name, get_perc=False, get_ftraces=False):
#     """
#     Also works for image classification. Batch_size > 1 is supported.
#     Extracts the ranger bounds from all the images in test_loader, for network net and saves them in bounds directory.
#     Applies to all activation layers (ReLU) not the Ranger. Only saves the input bounds.
#     :param data_loader: dataset loader, pytorch dataloader
#     :param net: network (pretrained), pytorch model
#     :param file_name: Name for bounds file, string
#     :param dataset_name: Name of the dataset using which Ranger bounds are generated
#     :return: list of min, max ranger inputs of the form list of [[min, max], [min, max], ...]
#     :return: list of min, max ranger output of the form list of [[min, max], [min, max], ...]
#     """

#     act_max_global = -np.Inf
#     act_min_global = +np.Inf
#     # act_avg_global = 0.

#     qu_list_global = {'q10': [], 'q25': [], 'q50': [], 'q75':[], 'q90': [], 'q100': [], 'q10_avg': [], 'q25_avg': [], 'q50_avg': [], 'q75_avg': [], 'q90_avg': [], 'q100_avg':[],\
#         'q10_min': [], 'q25_min': [], 'q50_min': [], 'q75_min': [], 'q90_min': [], 'q100_min': [], 'q10_max': [], 'q25_max': [], 'q50_max': [], 'q75_max': [], 'q90_max': [], 'q100_max': [], 'ftraces': [], 'ftraces_max': [], 'ftraces_min': [], \
#             'ftraces_mu': [], 'ftraces_sigma': [], 'ftraces_fmap_disp': [], 'N_total': 0}


#     net = net.to(device)
#     net.eval()

#     qu_list_global['N_total'] = data_loader.dataset_length

#     with torch.no_grad():
#         print('Extracting bounds from data set' + '(len=' + str(data_loader.dataset_length) + ')...')
#         # for j, (images, labels) in enumerate(data_loader):
#         #     if j % 100 == 0:
#         #         print(' Batch nr:', j)

#         i = 0
#         pbar = tqdm(total = data_loader.dataset_length)
#         while data_loader.data_incoming:
#             data_loader.datagen_itr()

#             if hasattr(data_loader, 'data'):
#                 images = data_loader.data
#             elif hasattr(data_loader, 'images'):
#                 images = data_loader.images

#             hook_list_out, hook_handles_out = set_ranger_hooks_ReLU(net)  # classes that save activations via hooks (all activations, for use in quantiles etc.)

#             net(images)

#             # Process hook content
#             names_out = [n.lay_info[1] for n in hook_list_out]
#             assert len(names_out) == len(np.unique(names_out)) 
#             act_out = [n.outputs for n in hook_list_out] #if this shows multiple entries per hook then layers are reused somehow and bound extraction goes wrong

#             # below loop required to solve memory leakage
#             for k in range(len(hook_handles_out)):
#                 hook_handles_out[k].remove()
#                 hook_list_out[k].clear()


#             # # Check activations (extract only min, max from the full list of activations)
#             act_out, qu_out, ft_out, cnt_out = get_max_min_lists2(act_out, get_perc=get_perc, get_ftraces=get_ftraces) #cnt_out gets overwritten but network behaves the same all the time so ok

#             if qu_out != []:
                
#                 qu_list_global['q10'].append(np.array(qu_out)[:,0,:].astype(float)) #dim: 72 x batchsize
#                 qu_list_global['q25'].append(np.array(qu_out)[:,1,:].astype(float))
#                 qu_list_global['q50'].append(np.array(qu_out)[:,2,:].astype(float))
#                 qu_list_global['q75'].append(np.array(qu_out)[:,3,:].astype(float))
#                 qu_list_global['q90'].append(np.array(qu_out)[:,4,:].astype(float))
#                 qu_list_global['q100'].append(np.array(qu_out)[:,5,:].astype(float))

#                 # for n in range(len(qu_out)):
#                 #     # q10_tp = []
#                 #     # for m in range(len(qu_out[n])):
#                 #     #     q10_tp.append([float(i) for i in qu_out[n][0]])
#                 #     qu_list_global['q10'].append(np.array(qu_out)[:,0,:].astype(float))
#                 #     # qu_list_global['q10'].append([float(i) for i in qu_out[n][0]])
#                 #     qu_list_global['q25'].append([float(i) for i in qu_out[n][1]])
#                 #     qu_list_global['q50'].append([float(i) for i in qu_out[n][2]])
#                 #     qu_list_global['q75'].append([float(i) for i in qu_out[n][3]])
                
#             if ft_out != []:
#                 qu_list_global['ftraces'].append(ft_out) #dim: 72 x batchsize x laysize 


#             act_max_global = np.maximum(act_max_global, act_out) #elementwise maximum 
#             act_min_global = np.minimum(act_min_global, act_out) #elementwise minimum

#             gc.collect() # collect garbage
#             pbar.update(data_loader.curr_batch_size)
#             i+=1

#         pbar.close()


#     act_out_global = np.vstack((act_min_global[:,0], act_max_global[:,1])).T.tolist() #min, max list of dim nr_layers, 2
#     save_Bounds_minmax(act_out_global, file_name) #save outgoing (incoming?) activations, already in Bounds folder

#     lens = np.unique(cnt_out)
#     if len(lens)>1 or (len(lens)==1 and lens[0]!=data_loader.curr_batch_size):
#         print('INFO: Model appears to reuse layers that are hooked for bound extraction. We take care of this. Per inference, cnt_out=', cnt_out)

#     nr_layers = act_out.shape[0]
    

#     if get_perc:
#         print('Processing quantile info...')
       
#         def get_layer_list_qu(qu_list, lay):
#             extr_list = []
#             for btch in range(len(qu_list)):
#                 extr_list.extend(qu_list[btch][lay,:].tolist())
#             return np.array(extr_list)

        
#         for m in range(nr_layers):
#             # print('layer', m)

#             extr_list = get_layer_list_qu(qu_list_global['q10'], m)
#             qu_list_global['q10_avg'].append([np.mean(extr_list), 1.96*np.std(extr_list)/np.sqrt(len(extr_list))])
#             qu_list_global['q10_min'].append(np.min(extr_list))
#             qu_list_global['q10_max'].append(np.max(extr_list))

#             extr_list = get_layer_list_qu(qu_list_global['q25'], m)
#             qu_list_global['q25_avg'].append([np.mean(extr_list), 1.96*np.std(extr_list)/np.sqrt(len(extr_list))])
#             qu_list_global['q25_min'].append(np.min(extr_list))
#             qu_list_global['q25_max'].append(np.max(extr_list))

#             extr_list = get_layer_list_qu(qu_list_global['q50'], m)
#             qu_list_global['q50_avg'].append([np.mean(extr_list), 1.96*np.std(extr_list)/np.sqrt(len(extr_list))])
#             qu_list_global['q50_min'].append(np.min(extr_list))
#             qu_list_global['q50_max'].append(np.max(extr_list))

#             extr_list = get_layer_list_qu(qu_list_global['q75'], m)
#             qu_list_global['q75_avg'].append([np.mean(extr_list), 1.96*np.std(extr_list)/np.sqrt(len(extr_list))])
#             qu_list_global['q75_min'].append(np.min(extr_list))
#             qu_list_global['q75_max'].append(np.max(extr_list))

#             extr_list = get_layer_list_qu(qu_list_global['q90'], m)
#             qu_list_global['q90_avg'].append([np.mean(extr_list), 1.96*np.std(extr_list)/np.sqrt(len(extr_list))])
#             qu_list_global['q90_min'].append(np.min(extr_list))
#             qu_list_global['q90_max'].append(np.max(extr_list))

#             extr_list = get_layer_list_qu(qu_list_global['q100'], m)
#             qu_list_global['q100_avg'].append([np.mean(extr_list), 1.96*np.std(extr_list)/np.sqrt(len(extr_list))])
#             qu_list_global['q100_min'].append(np.min(extr_list))
#             qu_list_global['q100_max'].append(np.max(extr_list))



#             # qu_list_global['q25_avg'].append([np.mean(qu_list_global['q25'][m]), 1.96*np.std(qu_list_global['q25'][m])/np.sqrt(len(qu_list_global['q25'][m]))])
#             # qu_list_global['q50_avg'].append([np.mean(qu_list_global['q50'][m]), 1.96*np.std(qu_list_global['q50'][m])/np.sqrt(len(qu_list_global['q50'][m]))])
#             # qu_list_global['q75_avg'].append([np.mean(qu_list_global['q75'][m]), 1.96*np.std(qu_list_global['q75'][m])/np.sqrt(len(qu_list_global['q75'][m]))])

#             # qu_list_global['q10_min'].append(np.min(qu_list_global['q10'][m]))
#             # qu_list_global['q25_min'].append(np.min(qu_list_global['q25'][m]))
#             # qu_list_global['q50_min'].append(np.min(qu_list_global['q50'][m]))
#             # qu_list_global['q75_min'].append(np.min(qu_list_global['q75'][m]))

#             # qu_list_global['q10_max'].append(np.max(qu_list_global['q10'][m]))
#             # qu_list_global['q25_max'].append(np.max(qu_list_global['q25'][m]))
#             # qu_list_global['q50_max'].append(np.max(qu_list_global['q50'][m]))
#             # qu_list_global['q75_max'].append(np.max(qu_list_global['q75'][m]))


#         # Clear raw data as it is too slow...
#         qu_list_global['q10'] = []
#         qu_list_global['q25'] = []
#         qu_list_global['q50'] = []
#         qu_list_global['q75'] = []
#         qu_list_global['q90'] = []
#         qu_list_global['q100'] = []


#     if get_ftraces:
#         print('Processing ftraces info...')


#         def get_layer_list_ft(qu_list, lay):
#             extr_list = []
#             for btch in range(len(qu_list)):
#                 extr_list.extend(qu_list[btch][lay])
#             return np.array(extr_list).astype(float)


#         for n in range(nr_layers):
#             extr_list = get_layer_list_ft(qu_list_global['ftraces'], n)

#             qu_list_global['ftraces_fmap_disp'].append([np.min(np.std(extr_list,1)), np.mean(np.std(extr_list,1)), np.max(np.std(extr_list,1)), np.std(np.std(extr_list,1))]) #first disp across fmaps, then min, max etc across samples
#             qu_list_global['ftraces_max'].append(np.max(extr_list, 0).tolist())
#             qu_list_global['ftraces_min'].append(np.min(extr_list, 0).tolist())
#             qu_list_global['ftraces_mu'].append(np.mean(extr_list, 0).tolist())
#             qu_list_global['ftraces_sigma'].append(np.std(extr_list, 0).tolist())

#         # Clean up
#         qu_list_global['ftraces'] = []
        

#     if get_ftraces or get_perc:
#         print('Saving...')
#         import time
#         tic = time.time()
#         save_Bounds_quantiles(qu_list_global, file_name + '_quantiles_ftraces') #both in one file
#         toc = time.time()
#         print('time', toc - tic)

#     return act_out_global


def save_Bounds_quantiles(qu_dict, file_name):
    import json
    bnd_path = './bounds/' + str(file_name) + '.json'
    print('Saving...')
    with open(bnd_path, 'w') as outfile:
        json.dump(qu_dict, outfile)
    print('Saved quantile bounds to', bnd_path)


def rearrange_matrix_blocks(qu_list_q, nr_layers):

    try:
        harr = np.array(qu_list_q)[0*nr_layers:1*nr_layers,:]
        for x in range(1, int((np.array(qu_list_q).shape[0])/nr_layers)):
            harr = np.hstack((harr, np.array(qu_list_q)[x*nr_layers:(x+1)*nr_layers,:]))
        return harr.tolist()
    except: #for reused layers
        print('Quantile blocks are not symmetrical, reused layers?')
        harr = []
        for x in range(int(np.array(qu_list_q).shape[0])):
            if len(harr)-1 >= x % nr_layers:
                harr[x % nr_layers].extend(np.array(qu_list_q)[x,:].tolist())
            else:
                harr.append(np.array(qu_list_q)[x,:].tolist())
        return harr


# def get_max_min_lists2(activations_out, get_perc, get_ftraces):
#     """
#     Transforms the act_in, act_out dictionaries to simpler forms with only min, max per layer. Note act_in, act_out have slightly different forms.
#     :param activations_in: list of tuple of tensors
#     :param activations_out: list of tensors
#     :return: act_out are lists of form [[min, max], [min, max] ... ] for each ranger layer

#     :qu_list_out2: has dims: nr of layers, 3 (q25, q50, q75), batch_size
#     :activations_out2: has dims: nr of layers, 2
#     """

#     if len(activations_out) == 0:
#         return None

#     list_min_max = [] #layerwise list with mins, maxs, etc
#     list_quant = []
#     list_ftraces = []
#     count_out = []
    
#     for lay in range(len(activations_out)):
#         mn_multi = []
#         mx_multi = []
#         q10_multi = []
#         q25_multi = []
#         q50_multi = []
#         q75_multi = []
#         q90_multi = []
#         q100_multi = []
#         ftraces_multi = []
#         cnt = 0 #how much is a hook used?
#         for ux in range(len(activations_out[lay])): #batch size or multi-use
            
#             cnt += 1
#             mn = torch.min(activations_out[lay][ux])
#             mx = torch.max(activations_out[lay][ux])
#             mn_multi.append(mn)
#             mx_multi.append(mx)

#             if get_perc:
#                 tnsr = activations_out[lay][ux]

#                 tnsr = torch.flatten(tnsr)
#                 # torch.quantile(tnsr, #)

#                 q10, q25, q50, q75, q90, q100 = torch.quantile(tnsr, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 1.00], device=tnsr.device))
#                 # torch.quantile(tnsr, torch.tensor([0.10, 0.25, 0.50, 0.75], device=tnsr.device))
#                 # q10 = torch.quantile(tnsr, 0.10) #make one #TODO TODO
#                 # q25 = torch.quantile(tnsr, 0.25)
#                 # q50 = torch.quantile(tnsr, 0.50)
#                 # q75 = torch.quantile(tnsr, 0.75)

#                 q10_multi.append(q10)
#                 q25_multi.append(q25)
#                 q50_multi.append(q50)
#                 q75_multi.append(q75)
#                 q90_multi.append(q90)
#                 q100_multi.append(q100)

#             if get_ftraces:
#                 if len(activations_out[lay][ux].shape) == 1:
#                     ftraces_multi.append(activations_out[lay][ux].tolist())
#                 elif len(activations_out[lay][ux].shape) == 3:
#                     ftraces_multi.append(torch.sum(activations_out[lay][ux], dim=[1,2]).tolist())
#                 else:
#                     print("Warning: Layer activations were found that are different from the expected dimensions of conv or fcc output.")
#                     sys.exit()


        
#         mn = np.min(mn_multi)
#         mx = np.max(mx_multi)
    
#         if get_perc:
#             count_out.append(cnt)
#             list_quant.append([q10_multi, q25_multi, q50_multi, q75_multi, q90_multi, q100_multi])
#         if get_ftraces:
#             count_out.append(cnt)
#             list_ftraces.append(ftraces_multi)

#         list_min_max.append([float(mn), float(mx)])

#     return np.array(list_min_max), list_quant, list_ftraces, np.array(count_out)


def get_ranger_bounds_quantiles(model, dl_attr:TEM_Dataloader_attr, ranger_file_name, gen_ranger_bounds, get_percentiles, get_ftraces):
    """
    Generates (saves/overwrites to file name and reloads) or loads ranger bounds (from given file name).
    Can generate quantile thresholds as well if get_percentiles is True.
    bnds has the following structure: [[min, q25, q50, q75, max], [...]] per layer
    bnds_min_max has the structure: [[min, max], [...]] per layer
    bnds_qu has the structure: [[q25, q50, q75], [...]] per layer
    """

    ranger_file_name_full = './bounds/' + ranger_file_name + '.txt'

    if gen_ranger_bounds:
        print('New bounds to be saved as:', ranger_file_name_full)
        get_Ranger_bounds2(model, ranger_file_name, dl_attr=dl_attr, get_quantiles=get_percentiles, get_ftraces=get_ftraces)
        print('Bounds have been extracted, finishing program.')
        sys.exit()
    else:
        print('Bounds loaded:', ranger_file_name)
    
    bnds = get_savedBounds_minmax(ranger_file_name_full)
    # bnds_min_max = [[x[0], x[-1]] for x in bnds]
    # if len(bnds[0])>2:
    #     bnds_qu = [x[1:-1] for x in bnds]
    # else:
    #     bnds_qu = None

    return bnds




# Accuracy-related evaluation ------------------------------------


def evaluateAccuracy(net, test_loader, classes_gt, classes_pred, top_nr, ranger_activity):
    """
    Automatic accuracy evaluation and getting active ranger layers.
    :param net: pytorch model (with bounds)
    :param test_loader: dataset loader
    :param classes_gt: classes for ground truth
    :param classes_pred: classes for predictions (can be different from above)
    :param top_nr: top n accuracy is evaluated (e.g. 1,3,5)
    :param ranger_activity: True (active ranger layers are evaluated), False (or not).
    :return: correct (scalar, nr of correct samples)
    :return: total (scalar, nr of total images checked)
    :return: Ranger activity: Gives a list of length N (number of images), each entry in the list is number of activated ranger layers for that image.
        List is empty if flag was false.
    """

    print('Evaluate accuracy')
    # Initialize
    correct = 0
    total = 0
    batch_size = test_loader.batch_size
    ranger_act_list = []

    if ranger_activity:
        save_input, save_output = set_ranger_hooks(net) #classes that save activations via hooks

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i % 100 == 0:
                print(' Batch nr:', i)

            gt_labels = [classes_gt[n] for n in labels.tolist()]  # original labels

            output = net(images)  # have to call forward pass to activate the hooks

            # Check activations
            if ranger_activity:
                act_in, act_out = get_max_min_lists(save_input.inputs, save_output.outputs)
                save_input.clear() #clear the hook lists, otherwise memory leakage
                save_output.clear()
                activated = 0
                for n in range(len(act_in)):
                    if (act_in[n,0] < act_out[n,0]) or (act_in[n,1] > act_out[n,1]):
                        activated += 1
                        # print('in', act_in, 'out', act_out, 'active ranger')
                ranger_act_list.append(activated) #nr of active ranger layers per image


            predictions = torch.nn.functional.softmax(output, dim=1) #normalized tensors

            top_val, top_ind = torch.topk(predictions, top_nr, dim=1) #it is by default sorted
            predicted = []
            for n in range(batch_size):
                predicted.append([classes_pred[top_ind[n, m]] for m in range(top_nr)])
            # print('top val', top_val, 'top ind', top_ind)
            # print('check', predicted)
            # print('gt', gt_labels)

            total += len(gt_labels)
            correct += np.sum([gt_labels[z] in predicted[z] for z in range(batch_size)])

            # if i == 10:
            #     break


    macc = (100. * float(correct) / float(total))
    print('Accuracy of the network on the ' + str(i) + ' test images: ' + str(macc))
    return correct, total, ranger_act_list



def save_accuracy_values(list, file_name):
    """
    Saves Ranger bounds
    :param: list of format [a, b, c] to be saved
    :param file_name: name of file as string
    :return: saves to a txt file in /bounds
    """
    if list == [] or list is None:
        bnd_path = str('./result_files/' + file_name + '.txt')
        f = open(bnd_path, "w+")
        f.close()
    else:
        bnd_path = str('./result_files/' + file_name + '.txt')
        f = open(bnd_path, "w+")
        for u in range(len(list)-1):
            f.write(str(list[u]) + ", ")
        f.write(str(list[len(list)-1])) #last without comma
        f.close()

    print('Results saved as ' + file_name)


def save_metadata(a, b, c, d, file_name):
    """
    Saves 3 metadata parameters.
    :param: parameters (scalar)
    :param file_name: name of file as string
    :return: saves to a txt file in /bounds
    """

    bnd_path = str('./result_files/' + file_name + '.txt')
    f = open(bnd_path, "w+")
    f.write(str(a) + ", " + str(b) + ", " + str(c) + ", " + str(d))
    f.close()

    print('Metadata saved as ' + file_name)


def save_to_dict(dicti, file_name):
    """
    Saves a dictionary to pickle.
    :param: parameters (scalar)
    :param file_name: name of file as string
    :return: saves to a txt file in /bounds
    """

    f = open(str('./result_files/' + file_name + ".pkl"),"wb")
    pickle.dump(dicti,f)
    f.close()
    print('Dictionary saved as ' + file_name)






