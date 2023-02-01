import os
import sys
import torch
import argparse
import yaml
from copy import deepcopy
import numpy as np
from typing import Dict, List
import datetime
import pickle

# Add wrapper:
# sys.path.append("..")
sys.path.append("/home/fgeissle/fgeissle_ranger")
from alficore.wrapper.test_error_models_imgclass import TestErrorModels_ImgClass
from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, pad_to_square, resize
from alficore.dataloader.objdet_baseClasses.boxes import Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.ptfiwrap_utils.build_native_model import build_native_model
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr, assign_val_train
# from alficore.dataloader.objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog
from train_detection_model_LR3 import load_json_indiv

# # Add mmdet:
# sys.path.append("/home/fgeissle/mmdetection")
# from mmdet.apis import init_detector
# from mmdet.apis.inference import inference_detector, show_result_pyplot
import torchvision
import warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
from LeNet5.LeNet5 import LeNet_orig
import matplotlib.pyplot as plt
from object_detection.det_quant_test_img_auto1 import build_objdet_native_model_img_cls, add_img_corr
from object_detection.det_quant_test_auto2 import set_up_mmdet_model
from alficore.evaluation.eval_img_classification import extract_data, extract_predictions, extract_predictions_prob
from object_detection.quantiles_extract_features_img_plot3 import get_quant_ftrace
from util.visualization import imshow_labels, plot_fmaps, plot_fmaps_objdet
import random
from alficore.evaluation.sdc_plots.obj_det_analysis_simple import obj_det_analysis
from alficore.dataloader.objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog


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


def read_fault_file(file):
    file = open(file, 'rb')
    return pickle.load(file)

def map_preds_to_prob_vector(classes, orig, orig_prob):
    orig_pred_probs = np.zeros(len(classes))
    for x in range(len(classes)):
        x_loc = np.where(np.array(orig)==x)[0]
        if len(x_loc) == 0:
            x_prob = 0.
        else:
            x_prob = np.array(orig_prob)[x_loc]
        orig_pred_probs[x] = x_prob
    return orig_pred_probs

def flatten_list(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]

def plot_v_line(q, qname, **kwargs):
    vshift = kwargs.get('vup', 0.)
    ax = kwargs.get('ax', None)
    col = kwargs.get('col', 'b')
    if ax is not None:
        ax.axvline(x=q, color=col, lw=1, alpha=0.5, label="_nolegend_")
    else:
        plt.axvline(x=q, color=col, lw=1, alpha=0.5, label="_nolegend_")
    # plt.text(q, -0.03 +vshift, qname + '=' + str(round(q,2)), horizontalalignment='center', color='black', fontsize=14) #fontweight='bold'

def plot_h_line(q0, q10, perc_diff, **kwargs):
    # plt.axvline(x=q0, color='b')
    # print(perc_diff/(q10-q0), q0, q10)
    # plt.axhline(y=perc_diff/(q10-q0), xmin=q0, xmax=q10, color='g')
    ax = kwargs.get('ax', None)
    col = kwargs.get('col', 'g')
    if ax is not None:
        ax.plot([q0, q10], [perc_diff/(q10-q0), perc_diff/(q10-q0)], color=col)
    else:
        plt.plot([q0, q10], [perc_diff/(q10-q0), perc_diff/(q10-q0)], color=col)


def subplot_full_acts(ftraces_dict, im, lay, typ, figsz, axs, binsz = 50, no=1, label_list = []):
    dats = ftraces_dict[typ][im][lay]
    # img_lay = flatten_list(flatten_list(dats)) # dims: images, layers
    img_lay = torch.sum(torch.tensor(dats), dim=(1,2)).tolist()
    q0, q10, q20, q30, q40, q50, q60, q70, q80, q90, q100 = np.quantile(img_lay, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.figure(figsize=(9, 5)) #figsize=(13, 10)
    # axs[lay].title('Layer: ' + str(lay*10) + ", N: " + str(len(img_lay))+ ', SDC: ' + str(is_sdc)+ ', Flt layer: ' + str(flt_lay))
    if no==1:
        col = 'b'
        aph = 1.0
    elif no==2:
        col = 'orange'
        aph = 0.5
    plot_v_line(q0, 'q0', vup=-0.03, ax=axs[lay], col=col)
    plot_v_line(q10, 'q10', vup=-0.03, ax=axs[lay], col=col)
    plot_v_line(q20, 'q20', vup=-0.03, ax=axs[lay], col=col)
    plot_v_line(q30, 'q30', vup=-0.03, ax=axs[lay], col=col)
    plot_v_line(q40, 'q40', vup=-0.03, ax=axs[lay], col=col)
    plot_v_line(q50, 'q50', vup=-0.03, ax=axs[lay], col=col)
    plot_v_line(q60, 'q60', vup=-0.03, ax=axs[lay], col=col)
    plot_v_line(q70, 'q70', vup=-0.03, ax=axs[lay], col=col)
    plot_v_line(q80, 'q80', vup=-0.03,ax=axs[lay], col=col)
    plot_v_line(q90, 'q90', vup=-0.03,ax=axs[lay], col=col)
    plot_v_line(q100, 'q100', vup=-0.03,ax=axs[lay], col=col)
    # plot_h_line(q0, q10, 0.1, ax=axs[lay])
    # plot_h_line(q10, q20, 0.1, ax=axs[lay])
    # plot_h_line(q20, q30, 0.1, ax=axs[lay])
    # plot_h_line(q30, q40, 0.1, ax=axs[lay])
    # plot_h_line(q40, q50, 0.1, ax=axs[lay])
    # plot_h_line(q50, q60, 0.1, ax=axs[lay])
    # plot_h_line(q60, q70, 0.1, ax=axs[lay])
    # plot_h_line(q70, q80, 0.1, ax=axs[lay])
    # plot_h_line(q80, q90, 0.1, ax=axs[lay])
    # plot_h_line(q90, q100, 0.1, ax=axs[lay])
    #
    # axs[lay].hist(flatten_list(flatten_list(dats)), bins=50, density=True) #raw data
    axs[lay].hist(img_lay, bins=binsz, alpha=aph, density=True) #fmap sums
    # axs[lay].ticklabel_format(useOffset=True) #makes xticks scientific
    # axs[lay].ticklabel_format(style='sci') #makes yticks scientific
    axs[lay].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if np.max(img_lay) > 1e10 or np.min(img_lay) < -1e10:
        print('rescaling to log axis because of large nubmers:', np.min(img_lay), np.max(img_lay))
        axs[lay].set_xscale('symlog')
        axs[lay].axes.set_xticks(axs[lay].axes.get_xticks()[::2])
    if lay == len(axs)-1:
        axs[lay].set_xlabel('Activation sum magnitude', fontsize=14)
    
    axs[lay].set_ylabel('Density (' + label_list[lay] + ')', fontsize=14)
    txt = 'q0=' + round_and_sci(q0)  + '\n' + 'q10=' + round_and_sci(q10)  + '\n' + 'q20=' + round_and_sci(q20)  + '\n' + 'q30=' + round_and_sci(q30) + '\n' + 'q40=' + round_and_sci(q40)  + '\n' \
        + 'q50=' + round_and_sci(q50)  + '\n' + 'q60=' + round_and_sci(q60)  + '\n' + 'q70=' + round_and_sci(q70)  + '\n' + 'q80=' + round_and_sci(q80)  + '\n' \
            + 'q90=' + round_and_sci(q90)  + '\n' + 'q100=' + round_and_sci(q100)

    
    if no== 1:
        txt = 'orig:'+ '\n' + txt
        txt_x_pos = 1.06
    elif no==2:
        txt = 'corr:'+ '\n' + txt
        txt_x_pos = 1.28
    axs[lay].text(txt_x_pos, 0.0 - 0.001*lay*figsz[1]/len(axs), txt, 
         fontsize=10, fontfamily='Georgia', color='k', transform=axs[lay].transAxes) #, ha='left', va='bottom'

def round_and_sci(x):
    if abs(x) > 1e10:
        y = '{:0.2e}'.format(x)
    else: 
        y = round(x,2)
    return str(y)


def main(argv):
    """
    Workflow: 
    1. Run this file for automated experiments, additionally for range extraction. Results are in nwstore or local.
    2. Run quantiles_extract_features_plot3.py for quantile extraction and save it to nwstore.
    3. Run train_detection_model_LR3.py or ..._DT2.py or ..._fcc1.py to train anomaly detector.
    """

    # Define dataset and model specifications
    ####################################################################
    dataset_name = 'kitti' #'coco', kitti, 'robo', ppp, lyft
    model_name = 'yolov3' #'yolov3', 'retina_net', 'ssd'

    batch_size = 1 #batchsize for neurons
    num_faults = 1 #faults
    num_runs = 1 #number of runs #500
    sample_size = 1 #nr of images (sequential)
    dataset_type = 'val' #'train', 'val'

    save_to_nw = False #Save directly to nw store?

    quant_monitoring = True
    ftrace_monitoring = True

    ####################################################################

    # Set device ---------------------------------------------------------
    cuda_device = 1
    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

    # Storing location ---------------------------------------------------------
    yml_file = 'default_test_auto.yml'
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
    model = set_up_mmdet_model(model_name, dataset_name, device)

    # Adjust image scale in scenario ---------------------------------------------
    # model.model.cfg.test_pipeline[1]['transforms'][0]['keep_ratio'] = False #test added
    class_dict = model.get_class_dict()
    if 'kitti' in dataset_name.lower():
        #kitti has homogeneous image sizes, need to adjust FI to size detected in build_model -> call -> input
        (ptfw, ptfh) = (1224, 370) 
    elif 'coco' in dataset_name.lower():
        (ptfw, ptfh) = model.get_img_scale()
    set_state(yml_file_path, 'ptf_H', ptfh)
    set_state(yml_file_path, 'ptf_W', ptfw)
    # Note: for kitti: put break point in call_size to find new size and overwrite in scenario.yaml, keep keep_ratio = True in config file 

    set_state(yml_file_path, 'rnd_layer', 0)

    ###########################################
    flt_type = "Adjust_contrast" #"hwfault" #"Gaussian_noise" , Gaussian_blur, Adjust_contrast #for that put in yaml neurons and per_image
    hw_spec = ["weights", [1,3]] #weights, neurons; bit_pos
    noise_magn = 0.1
    random.seed(10) #make noise the same
    ###########################################

    if "hw" in flt_type:
        noise_magn = 0
        set_state(yml_file_path, 'rnd_bit_range', hw_spec[1])
        if hw_spec[0] == "neurons":
            set_state(yml_file_path, 'rnd_mode', "neurons")
            set_state(yml_file_path, 'inj_policy', "per_image")
        elif hw_spec[0] == "weights":
            set_state(yml_file_path, 'rnd_mode', "weights")
            set_state(yml_file_path, 'inj_policy', "per_batch")
    else: #for input corruptions
        set_state(yml_file_path, 'rnd_mode', "neurons")
        set_state(yml_file_path, 'inj_policy', "per_image")

    # # # reuse fault file
    # flt_file = '/home/fgeissle/fgeissle_ranger/result_files/yolov3_1_trials/neurons_injs/per_image/objDet_20221123-180044_1_faults_12_bits/kitti/yolov3_test_random_sbf_neurons_inj_corr_1_1_1bs_kitti_updated_rs_fault_locs.bin'
    # copy_yml_scenario = False

    # # get new fault file
    flt_file = None
    copy_yml_scenario = False

    keep_searching = True
    fault_file = None #explicit fault
    while keep_searching:
        img_ErrorModel = TestErrorModels_ObjDet(model=model, model_name=model_name, resil_name='ranger_trivial', dl_attr=dl_attr, num_faults=num_faults,\
            config_location=yml_file, fault_file=flt_file, copy_yml_scenario=copy_yml_scenario, inf_nan_monitoring=True, ranger_bounds=[], ranger_detector=False, disable_FI=False, quant_monitoring=quant_monitoring, ftrace_monitoring = ftrace_monitoring, exp_type=flt_type, corr_magn=noise_magn) #, fault_file=ff, copy_yml_scenario=True)
        img_ErrorModel.test_rand_ObjDet_SBFs_inj() 

        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)

        # Retrieve result, image, quants  ---------------------------------------------------------
        img = img_ErrorModel.dataloader.datagen_iter.next()[0]['image']

        if "hw" not in flt_type:
            img_corr = add_img_corr(deepcopy(img).unsqueeze(0), flt_type, noise_magn)[0]
        else:
            img_corr = deepcopy(img)
        classes = img_ErrorModel.dataloader.metadata.thing_classes

        # quant, ftraces
        folder = img_ErrorModel.outputdir + "/" + dataset_name
        quant_dict, ftraces_dict = get_quant_ftrace(folder)
        
        # is due check:
        is_due = np.isnan(quant_dict['corr']).any() or np.isinf(quant_dict['corr']).any()
        print('DUE', is_due)
        if is_due:
            continue

        # print('check quant_dict: corr = orig?', (np.array(quant_dict['corr']) == np.array(quant_dict['resil'])).all())
        # print('check ftraces: corr = orig?', (np.array(ftraces_dict['corr']) == np.array(ftraces_dict['resil'])).all())

        # gnd and results
        save_as_name = obj_det_analysis(folder, folder_num=0, typ='no_resil')
        backup_dict =  load_json_indiv(save_as_name.replace("images", "backup"))
        result_dict = load_json_indiv(save_as_name)

        gnd, orig, corr = backup_dict['gt'][0][0], backup_dict['orig'][0][0], backup_dict['corr'][0][0]

        assert result_dict['nr_img_all'][0] == 1, 'More than one image selected?'
        is_due = result_dict['nr_img_due'][0] > 0
        is_sdc = result_dict['nr_img_sdc'][0] > 0


        if is_sdc and not is_due:
            keep_searching = False
            fault_file = flatten_list(result_dict['flts_sdc'])
            print(fault_file)

        # keep_searching = False #TODO fix later

    
    # print('GT', gnd, 'Orig', orig, 'Corr', corr)
    # --- [original fault location 7 elements] => Meaning for NEURON injection: --- #
    # 1. batchnumber (used in: conv2d,conv3d)
    # 2. layer (everywhere)
    # 3. channel (used in: conv2d,conv3d)
    # 4. depth (used in: conv3d)
    # 5. height (everywhere)
    # 6. width (everywhere)
    # 7. value = bit pos (everywhere)
    # --- [original fault location 7 elements] => Meaning for WEIGHT injection: --- #
    # 1. layer (everywhere)
    # 2. Kth filter (everywhere)
    # 3. channel(used in: conv2d, conv3d)
    # 4. depth (used in: conv3d)
    # 5. height (everywhere)
    # 6. width (everywhere)
    # 7. value = bit pos (everywhere)
    # bit flip monitor direction
    # orig value
    # new value
    
    
    ## Demonstrate fault visually in fmaps -------------------------------
    N = 0 #img nr in batch 1 affected more. #only to plot

    layer_sizes = []
    layer = []
    for n in range(len(ftraces_dict['corr'][N])):
        layer_sizes.append(np.array(ftraces_dict['corr'][N][n]).shape) #(batch_size,) + 
        layer.append(n)

    act_list_orig = ftraces_dict['resil'][N]
    act_list_corr = ftraces_dict['corr'][N]

    ylimmax = np.max([np.max(np.array(n)) for n in act_list_corr])

    # Customize -------------------------------
    ################################################################
    bnd_path =  '/home/fgeissle/fgeissle_ranger/bounds/yolo_kitti_v3_presum.txt'
    bnds = get_savedBounds_minmax(bnd_path)

    # # reshape fcc layers: #TODO: manual: fcc layer with 120 nodes gets repshaped to a single fmap with 10x12 dimensions for visibility
    # layer_sizes[2]=(1,10,12)
    # layer_sizes[3]=(1,7,12)
    # print('reshaped fcc layers:', layer_sizes)


    # Rearrange fmaps into 2d grids for better visuality ------------
    nr_fmaps = [x[0] for x in layer_sizes]
    nr_fmaps_rearr = [[4,8], [8,8]] #TODO: manual: 6 kernels will be plotted in a 2d grid of 2x3 for visibility
    print('reshape this:', nr_fmaps, 'to this:', nr_fmaps_rearr)

    max_val = 1000 #TODO visual upper limit
    layer = layer[:2] # TODO: if you want to select specific layers

    ################################################################

    # Plot fmaps ---------------------------------------------------
    act_list_corr = [torch.clamp(torch.tensor(n), -max_val, max_val).tolist() for n in act_list_corr]
    ylimmax = float(torch.clamp(torch.tensor(ylimmax), 0, max_val))

    if "hw" in flt_type:
        extra = hw_spec[0] + "_fault, bit:" + str(int(fault_file[6]))
    else:
        extra = flt_type + ":" + str(noise_magn) #'With fault'
    ttl = " "*25 + 'No fault' + " "*130 + extra
    figsz = (30, 4* (len(layer) + int(1) + int(1)))
    # figsz = None
    y_text = 'Output' + " "*25 + 'Act2' + " "*25 + 'Act1' + " "*26 + 'Input'
    plot_fmaps_objdet(layer, act_list_orig, act_list_corr, layer_sizes, nr_fmaps_rearr, bnds, ylimmax, orig, corr, img, img_corr, add_input_plot=True, add_output_plot=True, title=ttl, classes = classes, figsz=figsz, y_text=y_text)

    save_name = "test_plot.png"
    # plt.savefig("test_plot.pdf", bbox_inches = 'tight', quality=100, pad_inches = 0.1, format='pdf')
    # plt.savefig(save_name, bbox_inches = 'tight', pad_inches = 0.1, format='png')
    plt.savefig(save_name, bbox_inches = 'tight',  pad_inches = 0.1, dpi=150, format='png')
    print('Saved as', save_name)


    # Plot quantile sums --------------------------------------------
    fig, axs = plt.subplots(len(layer))
    fig.suptitle(str('Quantile shifts (') + extra +')', fontsize=16)
    figsz = (10, 22)
    fig.tight_layout()
    label_list=["Act1", "Act2"]
    for n in range(len(layer)):
        nr_bins = np.min([layer_sizes[n][0], 100])
        # print(layer_sizes[n])
        subplot_full_acts(ftraces_dict, N, n, 'resil', figsz, axs, binsz = nr_bins, no=1,label_list=label_list)
        subplot_full_acts(ftraces_dict, N, n, 'corr', figsz, axs, binsz = nr_bins, no=2,label_list=label_list)
        axs[n].legend(['original', 'corrupted'])
    sv_path = '/home/fgeissle/fgeissle_ranger/test_plot_qu_orig_corr.png'
    plt.savefig(sv_path, dpi=150, bbox_inches='tight')
    print('saved', sv_path)


## Accuracy
# top_nr = 1 # top N results are compared
# ranger_activity = True #flag whether the number of active ranger layers should be measured
# correct, total, act_ranger_layers = evaluateAccuracy(net, test_loader, classes, classes, top_nr, ranger_activity)
# print('map', correct, total)
# print('ranger active', sum(act_ranger_layers))
# Map:
# top 1: 99.2 %


if __name__ == "__main__":
    main(sys.argv)