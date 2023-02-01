# import pickle
# bin_file = '/home/qutub/PhD/git_repos/intel_gitlab_repos/ranger/logs/fault_rates_210315_150139.bin'
# bin_data = open(bin_file, 'rb')
# runset = pickle.load(bin_data)

import os, json
import numpy as np
import pandas as pd
# from dataloader import imagenet_Dataloader, imagenet_idx2label
import collections
import matplotlib as mpl
mpl.use('Agg')
# import seaborn as sns
# sns.set()
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
import  multiprocessing as mp
from operator import itemgetter
from operator import add, sub
from alficore.dataloader.imagenet_loader import imagenet_idx2label
# from dataloader import imagenet_Dataloader, imagenet_idx2label

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
# plt.style.use('seaborn')
sns.set_theme()

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

# extract_info()
idx2label, class_idx = imagenet_idx2label()
def extract_info(err_inj_data, fault_loc=None):
    _err_inj_fi_loc = [_err_inj_data['model_label_indx']['bit_flip_pos'] for _err_inj_data in err_inj_data['model_outputs']]
    return _err_inj_fi_loc

# extract_model_output()
def extract_model_output(weig_err_data):
    df = pd.DataFrame(columns = ['fp_indx', 'gnd_label', 'class_labels', 'orig_model','corr_model', 'resil_model', 'resil_wo_fi_model', 'layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit'])
    for i in range(len(weig_err_data)):
        df_i = pd.DataFrame(columns = ['gnd_label', 'orig_model','corr_model', 'resil_model', 'resil_wo_fi_model', 'layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit'])
        try:
            gnd_label = np.array([_err_inj_data['model_label_indx']['gnd_label and fp'][0] for _err_inj_data in weig_err_data[i][0]['model_outputs']])
            fp_indx = np.array([_err_inj_data['model_label_indx']['gnd_label and fp'][1] for _err_inj_data in weig_err_data[i][0]['model_outputs']])
        except:
            gnd_label = np.array([_err_inj_data['model_label_indx']['gnd_label'][0] for _err_inj_data in weig_err_data[i]['model_outputs']])
            fp_indx = None

        class_labels = [idx2label[_gnd_label] for _gnd_label in gnd_label]
        orig_model = np.array([_err_inj_data['model_label_indx']['orig_model'][0][0] for _err_inj_data in weig_err_data[i][0]['model_outputs']])

        corr_model = np.array([_err_inj_data['model_label_indx']['corrupted_model'][0][0] for _err_inj_data in weig_err_data[i][0]['model_outputs']])

        try:
            resil_model = np.array([_err_inj_data['model_label_indx']['resil_model'][0][0] for _err_inj_data in weig_err_data[i][0]['model_outputs']])
            resil_wo_fi_model = np.array([_err_inj_data['model_label_indx']['resil_wo_fi_model'][0][0] for _err_inj_data in weig_err_data[i][0]['model_outputs']])
        except:
            resil_model = None
            resil_wo_fi_model = None

        bit_flip_pos = np.array([_err_inj_data['model_label_indx']['bit_flip_pos'] for _err_inj_data in weig_err_data[i][0]['model_outputs']])

        df_i['gnd_label'] = gnd_label
        df_i['class_labels'] = class_labels
        df_i['fp_indx'] = fp_indx
        df_i['orig_model'] = orig_model
        df_i['corr_model'] = corr_model
        df_i['resil_model'] = resil_model
        df_i['resil_wo_fi_model'] = resil_wo_fi_model
        df_i['layer'], df_i['channel_in'], df_i['channel_out'], df_i['3D channel'], df_i['height'], df_i['width'], df_i['bit'] =  np.transpose(bit_flip_pos)
        df = df.append(df_i, ignore_index = True)
    
    return df

# read_json()
def read_json(type, i, file):
    print('reading file {} - {}'.format(i, file))
    with open(os.path.join(os.getcwd(), file), 'r') as f:
        loaded_err_inj_data = json.load(f)
        # weight_err_inj_data[i]  = loaded_err_inj_data
    print('succefully loaded file {} - {}'.format(i, file))
    return loaded_err_inj_data

# extract_jsons()
def extract_jsons(**kwargs):
    """
    reads json files with predifined parameters
    """
    num_trails = kwargs.get("num_trails", 200)
    fp_snippet = kwargs.get("fp_snippet", 0)
    batchsize = kwargs.get("batchsize", [250, 250])
    fault_inj_plots = kwargs.get("fault_inj_plots", [False, True])
    fi_images = kwargs.get("fi_images", [1])
    neuron_inj_data_files = []
    weight_inj_data_files = []

    batchsize = {'neurons': batchsize[0], 'weights': batchsize[1]}
    fault_inj_plots = {'neurons': fault_inj_plots[0], 'weights': fault_inj_plots[1]}
    fi_image  = fi_images
    # fi_image = [1]
    # fi_images = [1]
    for inj in fault_inj_plots:
        if inj == 'neurons':
            for _fi_image in fi_images:
                neuron_inj_data_files.append('../all_layer_injections/{}_{}_trials_{}/json_files/{}_injs/{}_test_random_sbf_{}_inj_{}_{}_{}_results.json'.format(MODEL_NAME, num_trails, fp_snippet, inj, MODEL_NAME, inj, _fi_image, num_trails, batchsize[inj]))
                # neuron_inj_data_files.append('../pytorchFI_results/resnet50/all_layer_injections/{}_{}_trials_0/json_files/{}_injs/{}_test_random_sbf_{}_inj_{}_{}_results.json'.format(MODEL_NAME, num_trails, inj, MODEL_NAME, inj, _fi_image, num_trails))
        if inj == 'weights':    
            for _fi_image in fi_images:
                weight_inj_data_files.append('../pytorchFI_results/resnet50/ImageNet_dataset/all_layer_injections/1_sbf/ranger_4000_epochs/{}_{}_trials_{}/json_files/{}_injs/{}_test_random_sbf_{}_inj_{}_{}_{}_results.json'.format(MODEL_NAME, num_trails, fp_snippet, inj, MODEL_NAME, inj, _fi_image, num_trails, batchsize[inj]))
                # weight_inj_data_files.append('../pytorchFI_results/resnet50/CNN_injections/{}_{}_trials_0/json_files/{}_injs/{}_test_random_sbf_{}_inj_{}_{}_results.json'.format(MODEL_NAME, num_trails, inj, MODEL_NAME, inj, _fi_image, num_trails))

    neuron_err_inj_data = [None for i in range(len(neuron_inj_data_files))]
    weight_err_inj_data = [None for i in range(len(weight_inj_data_files))]

    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        if fault_inj_plots['neurons']:
            neuron_err_inj_data = pool.starmap(read_json, [('neurons', i, file) for i, file in enumerate(neuron_inj_data_files)])
        if fault_inj_plots['weights']:
            weight_err_inj_data = pool.starmap(read_json, [('weights', i, file) for i, file in enumerate(weight_inj_data_files)])
    pool.close()
    pool.join()
    return (neuron_err_inj_data, weight_err_inj_data)

# pred_dist()
"""
generates distribution of classification results 
"""
def pred_dist(err_inj_data, dataset_size='1k'):
    dataset_chunk = {'1k':1000, '50k':50000}

    pdf_err = []
    for ii in range(len(err_inj_data)):
        orig_bin = err_inj_data[ii][0] == err_inj_data[ii][1]
        corr_bin = err_inj_data[ii][1] == err_inj_data[ii][2]
        resil_bin = err_inj_data[ii][1] == err_inj_data[ii][3]
        orig_bin = [orig_bin[i:i + dataset_chunk[dataset_size]] for i in range(0, len(orig_bin), dataset_chunk[dataset_size])]
        corr_bin = [corr_bin[i:i + dataset_chunk[dataset_size]] for i in range(0, len(corr_bin), dataset_chunk[dataset_size])]
        resil_bin = [resil_bin[i:i + dataset_chunk[dataset_size]] for i in range(0, len(resil_bin), dataset_chunk[dataset_size])]
        orig_dist_acc = [np.mean(_orig_bin) for _orig_bin in orig_bin]
        corr_dist_acc = [np.mean(_corr_bin) for _corr_bin in corr_bin]
        resil_dist_acc = [np.mean(_resil_bin) for _resil_bin in resil_bin]
        pdf_err.append([orig_dist_acc, corr_dist_acc, resil_dist_acc])
    return pdf_err

def sdc_dist(err_df, resil_info, dataset_size='1k', filter=True):
    dataset_chunk = {'1k':1000, '50k':50000}
    # dataset_chunk[dataset_size]
    if resil_info:
        df = pd.DataFrame(columns = ['index', 'orig_sdc', 'corr_sdc', 'resil_sdc', 'resil_wo_fi_sdc', 'corr_due', 'resil_due', 'layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit'])
    elif resil_info==False:
        df = pd.DataFrame(columns = ['index', 'orig_sdc', 'corr_sdc', 'orig_due', 'corr_due', 'layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit'])

    for i in range(0, len(err_df), dataset_chunk[dataset_size]):
        ii = int(i/1000)
        _err_df = err_df.iloc[i:i + dataset_chunk[dataset_size]]
        if filter:
            _err_df = _err_df[_err_df['orig_bool'] == True]


        orig_sdc = 1 - np.mean(_err_df['orig_bool'])
        # corr_sdc = 1 - np.mean(_err_df['corr_bool'])
        corr_sdc = 1 - np.mean(_err_df[_err_df['nan_or_inf_flag_corr_model'] == False]['corr_bool'])
        corr_due = len(_err_df[_err_df['nan_or_inf_flag_corr_model'] == True])/len(_err_df)
        if resil_info:
            # resil_sdc = 1 - np.mean(_err_df['resil_bool'])
            resil_sdc = 1 - np.mean(_err_df[_err_df['nan_or_inf_flag_resil_model'] == False]['resil_bool'])
            resil_due = len(_err_df[_err_df['nan_or_inf_flag_resil_model'] == True])/len(_err_df)
            resil_wo_fi_sdc = 1 - np.mean(_err_df['resil_wo_fi_bool'])
        fault_location = _err_df[['layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit']]
        fault_location = fault_location.drop_duplicates()
        if len(fault_location) == 1:
            layer, channel_in, channel_out, channel, height, width, bit = fault_location.iloc[0]
        else:
            layer, channel_in, channel_out, channel, height, width, bit = [None]*len(fault_location.columns)
        if resil_info:
            df_i = pd.DataFrame([[ii, orig_sdc, corr_sdc, resil_sdc, resil_wo_fi_sdc, corr_due, resil_due, layer, channel_in, channel_out, channel, height, width, bit]], columns = ['index', 'orig_sdc', 'corr_sdc', 'resil_sdc', 'resil_wo_fi_sdc', 'corr_due', 'resil_due', 'layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit'])
        elif resil_info==False:
            df_i = pd.DataFrame([[ii, orig_sdc, corr_sdc, corr_due, layer, channel_in, channel_out, channel, height, width, bit]], columns = ['index', 'orig_sdc', 'corr_sdc', 'corr_due', 'layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit'])

        df = df.append(df_i, ignore_index = True)
    return df

def class_sdc_dist(err_df, resil_info):
    if resil_info:
        df = pd.DataFrame(columns = ['gnd_label', 'class', 'orig_acc', 'corr_acc', 'resil_acc', 'resil_wo_fi_acc', 'orig_sdc', 'corr_sdc', 'resil_sdc', 'corr_due', 'resil_due', 'resil_wo_fi_sdc'])
    elif resil_info==False:
        df = pd.DataFrame(columns = ['gnd_label', 'class', 'orig_acc', 'corr_acc', 'orig_sdc', 'corr_sdc', 'resil_sdc', 'resil_wo_fi_sdc', 'corr_due'])

    for ii, gnd_label in enumerate(err_df['gnd_label'].unique()):
        _err_df = err_df[err_df['gnd_label'] == gnd_label]
        orig_acc = np.mean(_err_df['orig_bool'])
        corr_acc = np.mean(_err_df[_err_df['nan_or_inf_flag_corr_model'] == False]['corr_bool'])
        corr_due = len(_err_df[_err_df['nan_or_inf_flag_corr_model'] == True])/len(_err_df)
        _err_df = _err_df[_err_df['orig_bool'] == True]
        orig_sdc = 1 - np.mean(_err_df['orig_bool'])
        corr_sdc = 1 - np.mean(_err_df[_err_df['nan_or_inf_flag_corr_model'] == False]['corr_bool'])

        if resil_info:
            resil_acc =np.mean(_err_df[_err_df['nan_or_inf_flag_resil_model'] == False]['resil_bool'])
            resil_due = len(_err_df[_err_df['nan_or_inf_flag_resil_model'] == True])/len(_err_df)
            resil_wo_fi_acc = np.mean(_err_df['resil_wo_fi_bool'])
            resil_sdc = 1 - np.mean(_err_df[_err_df['nan_or_inf_flag_resil_model'] == False]['resil_bool'])
            resil_wo_fi_sdc = 1 - np.mean(_err_df['resil_wo_fi_bool'])

        fault_location = _err_df[['layer', 'channel_in', 'channel_out', '3D channel', 'height', 'width', 'bit']]
        fault_location = fault_location.drop_duplicates()
        if len(fault_location) == 1:
            layer, channel_in, channel_out, channel, height, width, bit = fault_location.iloc[0]
        else:
            layer, channel_in, channel_out, channel, height, width, bit = [None]*len(fault_location.columns)
        if resil_info:
            df_i = pd.DataFrame([[gnd_label, idx2label[gnd_label], orig_acc, corr_acc, resil_acc, resil_wo_fi_acc, orig_sdc, corr_sdc, resil_sdc, resil_wo_fi_sdc, corr_due, resil_due]], columns = ['gnd_label', 'class', 'orig_acc', 'corr_acc', 'resil_acc', 'resil_wo_fi_acc', 'orig_sdc', 'corr_sdc', 'resil_sdc', 'resil_wo_fi_sdc', 'corr_due', 'resil_due'])
        elif resil_info==False:
            df_i = pd.DataFrame([[gnd_label, idx2label[gnd_label], orig_acc, corr_acc, orig_sdc, corr_sdc, corr_due]], columns = ['gnd_label', 'class', 'orig_acc', 'corr_acc', 'orig_sdc', 'corr_sdc', 'corr_due'])
        df = df.append(df_i, ignore_index = True)
    return df

def preproc_df(df, resil_info):
    df['orig output index - top5'] = df['orig output index - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    df['orig output prob - top5'] = df['orig output prob - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))

    df['corr output index - top5'] = df['corr output index - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    df['corr output prob - top5'] = df['corr output prob - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))

    df['orig_bool'] = df['gnd_label'] == df['orig output index - top5'].apply(lambda x: x[0])
    df['corr_bool'] = df['gnd_label'] == df['corr output index - top5'].apply(lambda x: x[0])

    if resil_info:
        df['resil output index - top5'] = df['resil output index - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
        df['resil output prob - top5'] = df['resil output prob - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))

        df['resil_wo_fi output index - top5'] = df['resil_wo_fi output index - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
        df['resil_wo_fi output prob - top5'] = df['resil_wo_fi output prob - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))
        df['resil_bool'] = df['gnd_label'] == df['resil output index - top5'].apply(lambda x: x[0])
        df['resil_wo_fi_bool'] = df['gnd_label'] == df['resil_wo_fi output index - top5'].apply(lambda x: x[0])

    return df

def gen_sdc_dist(df, resil_info):
    sdc_df= sdc_dist(err_df=df, resil_info=resil_info, dataset_size='1k')
    class_df = class_sdc_dist(err_df=df, resil_info=resil_info)
    return (sdc_df, class_df)  


def combine_model_df(fi_N, resil_info, **model_model_fi):
    if resil_info == False:
        model_data = pd.read_csv(model_model_fi['Ranger'])
        model_data = preproc_df(df=model_data, resil_info=resil_info)

        ## Analyse class level SDC
        sdc_data_frames, class_data_frames = gen_sdc_dist(df=model_data, resil_info=resil_info)

        model_data.rename(columns={'corr output index - top5':'ranger output index', 'corr_wo_fi output index - top5':'ranger_wo_fi output index', 'corr output prob - top5':'ranger output prob', 'corr_wo_fi output prob - top5':'ranger_wo_fi output prob'}, inplace=True)

        class_data_frames.rename(columns={'resil_sdc':'Ranger_sdc', 'resil_wo_fi_sdc':'ranger_wo_fi_sdc', 'resil_acc':'ranger_acc', 'resil_wo_fi_acc':'ranger_wo_fi_acc'}, inplace=True)

        sdc_data_frames.rename(columns={'resil_sdc':'Ranger_sdc', 'resil_wo_fi_sdc':'ranger_wo_fi_sdc', 'resil_due':'Ranger_due'}, inplace=True)
        model_data = model_data.add_suffix('_{}'.format(fi_N))
        sdc_data_frames = sdc_data_frames.add_suffix('_{}'.format(fi_N))
        class_data_frames = class_data_frames.add_suffix('_{}'.format(fi_N))
        return (model_data, sdc_data_frames, class_data_frames)

    elif resil_info == True:
        if model_model_fi['Ranger'] is not None:
            model_data = pd.read_csv(model_model_fi['Ranger']) 
            model_data = preproc_df(df=model_data, resil_info=resil_info)

            ## Analyse class level SDC
            sdc_data_frames, class_data_frames = gen_sdc_dist(df=model_data, resil_info=resil_info)

            model_data.rename(columns={'resil output index - top5':'ranger output index', 'resil_wo_fi output index - top5':'ranger_wo_fi output index', 'resil output prob - top5':'ranger output prob', 'resil_wo_fi output prob - top5':'ranger_wo_fi output prob'}, inplace=True)

            class_data_frames.rename(columns={'resil_sdc':'Ranger_sdc', 'resil_wo_fi_sdc':'ranger_wo_fi_sdc', 'resil_acc':'ranger_acc', 'resil_wo_fi_acc':'ranger_wo_fi_acc'}, inplace=True)

            sdc_data_frames.rename(columns={'resil_sdc':'Ranger_sdc', 'resil_wo_fi_sdc':'ranger_wo_fi_sdc', 'resil_due':'Ranger_due'}, inplace=True)

        if model_model_fi['Clip'] is not None:
            model_clip_data = pd.read_csv(model_model_fi['Clip']) 
            model_clip_data = preproc_df(df=model_clip_data, resil_info=resil_info)

            ## Analyse class level SDC
            sdc_model_clip_data, class_model_clip_data = gen_sdc_dist(df=model_clip_data, resil_info=resil_info)

            model_clip_data.rename(columns={'resil output index - top5':'clip output index', 'resil_wo_fi output index - top5':'clip_wo_fi output index', 'resil output prob - top5':'clip output prob', 'resil_wo_fi output prob - top5':'clip_wo_fi output prob',}, inplace=True)

            class_model_clip_data.rename(columns={'resil_sdc':'Clip_sdc', 'resil_wo_fi_sdc':'clip_wo_fi_sdc', 'resil_acc':'clip_acc', 'resil_wo_fi_acc':'clip_wo_fi_acc'}, inplace=True)

            sdc_model_clip_data.rename(columns={'resil_sdc':'Clip_sdc', 'resil_wo_fi_sdc':'clip_wo_fi_sdc', 'resil_due':'Clip_due'}, inplace=True)

            if model_model_fi['Ranger'] is not None:
                cols = class_model_clip_data.columns.difference(class_data_frames.columns)
                class_data_frames = class_data_frames.join(class_model_clip_data[cols])

                cols = sdc_model_clip_data.columns.difference(sdc_data_frames.columns)
                sdc_data_frames = sdc_data_frames.join(sdc_model_clip_data[cols])

                cols = model_clip_data.columns.difference(model_data.columns)
                model_data = model_data.join(model_clip_data[cols])
            else:
                class_data_frames = class_model_clip_data
                sdc_data_frames = sdc_model_clip_data
                model_data = model_clip_data

        if model_model_fi['BackFlip'] is not None:
            model_BackFlip_data = pd.read_csv(model_model_fi['BackFlip']) 
            model_BackFlip_data = preproc_df(model_BackFlip_data, resil_info=resil_info)

            ## Analyse class level SDC
            sdc_model_BackFlip_data, class_model_BackFlip_data = gen_sdc_dist(model_BackFlip_data, resil_info=resil_info)
            
            model_BackFlip_data.rename(columns={'resil output index - top5':'BackFlip output index', 'resil_wo_fi output index - top5':'BackFlip_wo_fi output index', 'resil output prob - top5':'BackFlip output prob', 'resil_wo_fi output prob - top5':'BackFlip_wo_fi output prob',}, inplace=True)

            class_model_BackFlip_data.rename(columns={'resil_sdc':'BackFlip_sdc', 'resil_wo_fi_sdc':'BackFlip_wo_fi_sdc', 'resil_acc':'BackFlip_acc', 'resil_wo_fi_acc':'BackFlip_wo_fi_acc'}, inplace=True)

            sdc_model_BackFlip_data.rename(columns={'resil_sdc':'BackFlip_sdc', 'resil_wo_fi_sdc':'BackFlip_wo_fi_sdc', 'resil_due':'BackFlip_due'}, inplace=True)

            if ((model_model_fi['Ranger'] is not None) | (model_model_fi['Clip'] is not None)):
                cols = class_model_BackFlip_data.columns.difference(class_data_frames.columns)
                class_data_frames = class_data_frames.join(class_model_BackFlip_data[cols])

                cols = sdc_model_BackFlip_data.columns.difference(sdc_data_frames.columns)
                sdc_data_frames = sdc_data_frames.join(sdc_model_BackFlip_data[cols])

                cols = model_BackFlip_data.columns.difference(model_data.columns)
                model_data = model_data.join(model_BackFlip_data[cols])
            else:
                class_data_frames = class_model_BackFlip_data
                sdc_data_frames = sdc_model_BackFlip_data
                model_data = model_BackFlip_data

        if model_model_fi['FmapAvg'] is not None:
            model_FmapAvg_data = pd.read_csv(model_model_fi['FmapAvg']) 
            model_FmapAvg_data = preproc_df(model_FmapAvg_data, resil_info=resil_info)

            ## Analyse class level SDC
            sdc_model_FmapAvg_data, class_model_FmapAvg_data = gen_sdc_dist(model_FmapAvg_data, resil_info=resil_info)

            model_FmapAvg_data.rename(columns={'resil output index - top5':'FmapAvg output index', 'resil_wo_fi output index - top5':'FmapAvg_wo_fi output index', 'resil output prob - top5':'FmapAvg output prob', 'resil_wo_fi output prob - top5':'FmapAvg_wo_fi output prob',}, inplace=True)


            class_model_FmapAvg_data.rename(columns={'resil_sdc':'FmapAvg_sdc', 'resil_wo_fi_sdc':'FmapAvg_wo_fi_sdc', 'resil_acc':'FmapAvg_acc', 'resil_wo_fi_acc':'FmapAvg_wo_fi_acc'}, inplace=True)

            sdc_model_FmapAvg_data.rename(columns={'resil_sdc':'FmapAvg_sdc', 'resil_wo_fi_sdc':'FmapAvg_wo_fi_sdc', 'resil_due':'FmapAvg_due'}, inplace=True)

            if ((model_model_fi['Ranger'] is not None) | (model_model_fi['Clip'] is not None) | (model_model_fi['BackFlip'] is not None)):
                cols = class_model_FmapAvg_data.columns.difference(class_data_frames.columns)
                class_data_frames = class_data_frames.join(class_model_FmapAvg_data[cols])

                cols = sdc_model_FmapAvg_data.columns.difference(sdc_data_frames.columns)
                sdc_data_frames = sdc_data_frames.join(sdc_model_FmapAvg_data[cols])

                cols = model_FmapAvg_data.columns.difference(model_data.columns)
                model_data = model_data.join(model_FmapAvg_data[cols])
            else:
                class_data_frames = class_model_FmapAvg_data
                sdc_data_frames = sdc_model_FmapAvg_data
                model_data = model_FmapAvg_data

        if model_model_fi['FmapRescale'] is not None:
            model_FmapRescale_data = pd.read_csv(model_model_fi['FmapRescale']) 
            model_FmapRescale_data = preproc_df(model_FmapRescale_data, resil_info=resil_info)

            ## Analyse class level SDC
            sdc_model_FmapRescale_data, class_model_FmapRescale_data = gen_sdc_dist(model_FmapRescale_data, resil_info=resil_info)

            model_FmapRescale_data.rename(columns={'resil output index - top5':'FmapRescale output index', 'resil_wo_fi output index - top5':'FmapRescale_wo_fi output index', 'resil output prob - top5':'FmapRescale output prob', 'resil_wo_fi output prob - top5':'FmapRescale_wo_fi output prob',}, inplace=True)


            class_model_FmapRescale_data.rename(columns={'resil_sdc':'FmapRescale_sdc', 'resil_wo_fi_sdc':'FmapRescale_wo_fi_sdc', 'resil_acc':'FmapRescale_acc', 'resil_wo_fi_acc':'FmapRescale_wo_fi_acc'}, inplace=True)

            sdc_model_FmapRescale_data.rename(columns={'resil_sdc':'FmapRescale_sdc', 'resil_wo_fi_sdc':'FmapRescale_wo_fi_sdc', 'resil_due':'FmapRescale_due'}, inplace=True)

            cols = class_model_FmapRescale_data.columns.difference(class_data_frames.columns)
            class_data_frames = class_data_frames.join(class_model_FmapRescale_data[cols])

            cols = sdc_model_FmapRescale_data.columns.difference(sdc_data_frames.columns)
            sdc_data_frames = sdc_data_frames.join(sdc_model_FmapRescale_data[cols])
            
            cols = model_FmapRescale_data.columns.difference(model_data.columns)
            model_data = model_data.join(model_FmapRescale_data[cols])

        model_data = model_data.add_suffix('_{}'.format(fi_N))
        sdc_data_frames = sdc_data_frames.add_suffix('_{}'.format(fi_N))
        class_data_frames = class_data_frames.add_suffix('_{}'.format(fi_N))
        return (model_data, sdc_data_frames, class_data_frames)