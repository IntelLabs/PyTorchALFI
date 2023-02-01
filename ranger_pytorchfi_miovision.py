import os, sys, glob
import pandas as pd
import numpy as np
import pickle
from util.helper_functions import get_savedBounds_minmax
from util.evaluate import extract_ranger_bounds
from util.ranger_automation import get_Ranger_protection
from alficore.wrapper.ptfiwrap import TestErrorModels
import alficore.dataloader.miovision.miovision_config_parser as parser
import alficore.parser.config_parser as fault_parser
from alficore.dataloader.miovision.miovision_dataloader import miovision_data_loader as data_loader
from miovision_train_eval import resnet50_net, resnet50_load_checkpoint, vgg16_net, vgg16_load_checkpoint
from resiliency_methods.Ranger import Ranger, Ranger_Clip #, Ranger_FmapRescale, Ranger_BackFlip, Ranger_FmapAvg
from alficore.dataloader.miovision.miovision_dataloader import get_label_for_testing, get_label_for_training, safety_critical_confusion
from miovision_results.utils.class_sensitivity_analysis import plot_class_resil_comparison, \
            plot_class_sdc_rate_layer, plot_class_sdc_rate_layer_average, \
            plot_sample_mean_distribution, plot_class_resil_comparison_safety_confusion, plot_class_error_distribution, render_mpl_table
             #, plot_class_sdc_fault_per_inference  
from miovision_results.utils.ranger_miovision_analysis import plot_sdc_ranger_variants, plot_sdc_safety_criticality_imposed_ranger_variants
import scipy.stats as stat
from resiliency_methods.Ranger import Ranger_trivial

import warnings
warnings.filterwarnings('ignore')


def gen_model_ranger_bounds(model, parser, ranger_file_name):
    """Generate ranger data according to https://arxiv.org/pdf/2003.13874.pdf

    Args:
        model : torchvision model to generate ranger bounds for
        parser : Miovision config parser
        ranger_file_name : filename without extension to save the ranger bounds into
    """    
    parser.val_correct = False
    train_loader, _, _ = data_loader(parser=parser)
    if not parser.generate_ranger_bounds_classes:        
        net_for_bounds = model
        extract_ranger_bounds(train_loader, net_for_bounds, ranger_file_name, parser.dataset_name) # gets also saved automatically
    else:
        for class_name in train_loader.keys():
            net_for_bounds = model
            class_dataloader = train_loader[class_name]
            ranger_file_class_name = ranger_file_name + '_' + class_name
            ranger_file_class_name = os.path.join('miovision_classes', ranger_file_class_name)
            extract_ranger_bounds(class_dataloader, net_for_bounds, ranger_file_class_name, parser.dataset_name)
    sys.exit()


def inject_fault_evaluate_ranger(ranger_file_name, model, mio_config_parser):
    """Injects faults using alficore 

    Args:
        ranger_file_name : filename without extension to save the ranger bounds into
        model : torchvision model to generate ranger bounds for
        mio_config_parser : Miovision config parser
    """    
    cuda_device = 0
    # Load bounds: ------------------------
    bnds = get_savedBounds_minmax('./bounds/' + ranger_file_name + '.txt')
    # resil = Ranger
    resils = [Ranger, Ranger_Clip]
    # resils = [Ranger_Clip, Ranger_BackFlip, Ranger_FmapAvg, Ranger_FmapRescale]
    # Change network architecture to add Ranger
    net_for_protection = model
    # model_ranger, _ = get_Ranger_protection(net_for_protection, bnds, resil=resil) 
    net_for_protection_ext, _ = get_Ranger_protection(net_for_protection, bnds, resil=Ranger_trivial) #TODO: added for layer order
    mio_config_parser.val_correct = True
    save_fault_file_dir = 'result_files/mio/msb_runs/'
    ########## Thesis experiments ##############
    fault_files = ['logs/fault_rates_vgg16_weights_lwmsb_500.bin', 'logs/fault_rates_vgg16_weights_lwmsb_600.bin']
    # num_faults = [1, 10]
    num_runs = [500, 600]    
    for i in range(len(num_runs)):
        for resil in resils:
            model_ranger, _ = get_Ranger_protection(net_for_protection, bnds, resil=resil) 
            resnet_errorModel = TestErrorModels(
                model=net_for_protection_ext, resil_model=model_ranger, model_name=mio_config_parser.model_name, cuda_device=cuda_device,
                dataset=mio_config_parser.dataset_name, store_json=True, num_faults=None, resil_method=resil.__name__, dataset_parser=mio_config_parser
                # , config_location='default_weights.yml'
                )
    # resnet_errorModel.test_random_single_bit_flip_inj(save_fault_file_dir=save_fault_file_dir)
            resnet_errorModel.test_random_single_bit_flip_inj(fault_file=fault_files[i], num_runs=num_runs[i], save_fault_file_dir=save_fault_file_dir)

        # resnet_errorModel.test_random_single_bit_flip_inj(num_runs=num_runs[i])
    #         # resnet_errorModel.test_random_single_bit_flip_inj(num_faults=num_faults[i], fault_file=fault_files[i])

    # save_fault_file_dir = 'result_files/mio/neurons/'
    # if mio_config_parser.fault_type == 'neurons_injs':
    #     for resil in resils:
    #         model_ranger, _ = get_Ranger_protection(net_for_protection, bnds, resil=resil)
    #         for class_id in get_label_for_testing.keys():
    #             mio_config_parser.neuron_inj_class_id = class_id
    #             resnet_errorModel = TestErrorModels(
    #                 model=net_for_protection_ext, resil_model=model_ranger, model_name=mio_config_parser.model_name, cuda_device=cuda_device,
    #                 dataset=mio_config_parser.dataset_name, store_json=True, num_faults=None, resil_method=resil.__name__, dataset_parser=mio_config_parser)
    #             resnet_errorModel.test_random_single_bit_flip_inj(save_fault_file_dir=save_fault_file_dir)
    sys.exit()


def fault_analysis(fault_config_parser, mio_config_parser):
    """Method to accumulate all fault csv file paths as given in the config file and either perform
    analysis on the whole dataset or classwise

    Args:
        fault_config_parser (ConfigParser): Config of default.yml
        mio_config_parser (ConfigParser): Config of miovision.yml
    """
    fault_root_folders = [mio_config_parser.model_name] # respective model folders under save_fault_file_dir path in default.yml 
    fault_trials = [] # name of fault run folders executed and saved so far e.g. resnet50_100_trials
    fault_folder_paths = []  # list to store location of fault csv files 
    fault_file_type_folder_name = 'csv_files/'
    fault_file_folders = list(os.walk(fault_config_parser.save_fault_file_dir))
    number_fault_files = 0
    fault_file_paths = []

    for folder_index in range(len(fault_file_folders)):
        root_folder = fault_file_folders[folder_index][0]
        for fault_folder in fault_root_folders:
            fault_folder_path = fault_config_parser.save_fault_file_dir + fault_folder
            if root_folder == fault_folder_path:
                number_fault_files += len(fault_file_folders[folder_index][1])
                fault_trials.extend(fault_file_folders[folder_index][1])
                for i in range(len(fault_file_folders[folder_index][1])):
                    fault_folder_paths.append(fault_folder_path)

    for i in range(number_fault_files):
        file_path = fault_folder_paths[i] + '/' + fault_trials[i] + '/' + fault_file_type_folder_name + mio_config_parser.fault_type + '/'
        fault_file_paths.append(file_path)

    ranger_comparison_csv_paths = []

    for file_path in fault_file_paths:
        csv_file_paths = glob.glob(file_path+'*.csv')
        csv_file_paths = [k for k in csv_file_paths if 'imagefp' not in k]
        csv_file_paths_filtered = []
        if mio_config_parser.faults_per_inference is not None:
            for file_path in csv_file_paths:
                if int(file_path.split('_')[-5]) in mio_config_parser.faults_per_inference and int(file_path.split('_')[-4]) in mio_config_parser.fault_epochs:
                    csv_file_paths_filtered.append(file_path)

            csv_file_paths = csv_file_paths_filtered

        if csv_file_paths:
            ranger_comparison_csv_paths.append(csv_file_paths)

    # resil_method_comparison(ranger_comparison_csv_paths, mio_config_parser)
    class_analysis_miovision(ranger_comparison_csv_paths, mio_config_parser)


def calculate_safety_criticality(df):
    """Method to fill a fault dataframe with safety-critical flags

    Args:
        df (DataFrame): Fault file dataframe

    Returns:
        [DataFrame]: DataFrame with the safety-critical flags
    """
    df['safety_critical_bool'] = np.where((df['corr_bool'] == True), True, False)
    update_indexes = df[df['safety_critical_bool'] == True].index
    original_prediction = list(df[df['safety_critical_bool'] == True]['gnd_label'])
    corrupted_predictions_top5 = list(df[df['safety_critical_bool'] == True]['corr output index - top5'])
    safety_criticial_results = safety_critical_confusion(original_prediction, corrupted_predictions_top5)
    if len(safety_criticial_results) > 0:
        df.loc[update_indexes, ['safety_critical_bool']] = safety_criticial_results

    df['safety_critical_resil_bool'] = np.where((df['safety_critical_bool'] == True), True, False)
    update_indexes = df[df['safety_critical_resil_bool'] == True].index
    df.loc[update_indexes, 'safety_critical_resil_bool'] = df.loc[update_indexes, 'resil_bool']

    return df


def class_analysis_miovision(file_paths, parser):
    """This method handles all class based activites using miovision dataset

    Args:
        file_paths (list): List of fault csv files to analyse
        parser (ConfigParser): miovision_config.yml parser file entries
    """
    resiliency_methods = ['Ranger', 'Clipper'] # change here to include any other resil
    dfs = {key:{inner_key:[] for inner_key in parser.faults_per_inference} for key in resiliency_methods} 
    sdc_classwise = pd.DataFrame(columns=['Miovision_class', 'Faults_per_inference', 'SDC_rate_per_epoch', 'Safety_critical_SDC_rate_per_epoch',
                                          'Resil_method', 'Resnet50_conv_layer', 'Confidence_score', 'Epochs_debug', 'channel'])
    due_classwise = pd.DataFrame(columns=['Miovision_class', 'Faults_per_inference', 'DUE_rate_per_epoch', 'Resil_method'])
    save_file, resil_location = initialize_params(parser, analysis_type='classwise')
    
    corrupted_model_sdc_flag = False # flag to fill the SDC details of corrupted model only once
    corrupted_model_due_flag = False # flag to fill the DUE details of corrupted model only once
    
    if not os.path.isfile(save_file):
        read_fault_files(file_paths, resil_location, resiliency_methods, dfs)
        # loops for each ranger variant resil method
        for resil_method in dfs.keys():
            fault_results_df = dfs[resil_method]
            # loops for each number of faults per inference
            for faults in fault_results_df.keys():       
                if fault_results_df[faults]:     
                # if fault_results_df[faults] and faults == 1:
                    # loops for multiple epochs fault file for a resil method and particular faults per inference
                    for fault_df in fault_results_df[faults]:
                        ## below three dictionaries are to store SDC rate, safety-critical SDC rate, Confusion score and DUE rate on a class-level
                        classes_epoch_sdc_rates = {key:{inner_key:[] for inner_key in get_label_for_training.keys()} for key in parser.faults_per_inference}
                        classes_epoch_sc_sdc_rates = {key:{inner_key:[] for inner_key in get_label_for_training.keys()} for key in parser.faults_per_inference}                          
                        classes_epoch_due_rates = {key:{inner_key:[] for inner_key in get_label_for_training.keys()} for key in parser.faults_per_inference}                               
                        classes_epoch_corr_conf_score = {key:{inner_key:[] for inner_key in get_label_for_training.keys()} for key in parser.faults_per_inference} 
                        ## below three dictionaries are to store resil SDC rate, resil safety-critical SDC rate, resil Confusion score and resil DUE rate on a class-level
                        classes_epoch_resil_sdc_rates = {key:{inner_key:[] for inner_key in get_label_for_training.keys()} for key in parser.faults_per_inference}
                        classes_epoch_resil_sc_sdc_rates = {key:{inner_key:[] for inner_key in get_label_for_training.keys()} for key in parser.faults_per_inference}
                        classes_epoch_resil_conf_score = {key:{inner_key:[] for inner_key in get_label_for_training.keys()} for key in parser.faults_per_inference}
                        classes_epoch_resil_due_rates = {key:{inner_key:[] for inner_key in get_label_for_training.keys()} for key in parser.faults_per_inference}                 

                        number_of_rows = len(fault_df.index) 
                        total_images = parser.num_classes * parser.number_of_images_each_class
                        number_of_fault_epochs = int(number_of_rows / total_images)                         
                        class_sdc_rate_dict = classes_epoch_sdc_rates[faults]
                        class_sc_sdc_rate_dict = classes_epoch_sc_sdc_rates[faults]
                        class_due_rate_dict = classes_epoch_due_rates[faults]
                        class_resil_sdc_dict = classes_epoch_resil_sdc_rates[faults]
                        class_resil_sc_sdc_dict = classes_epoch_resil_sc_sdc_rates[faults]
                        class_resil_due_rate_dict = classes_epoch_resil_due_rates[faults]
                        class_corr_conf_score = classes_epoch_corr_conf_score[faults]
                        class_resil_conf_score = classes_epoch_resil_conf_score[faults]
                        fault_layers = {key:[] for key in get_label_for_testing.keys()}
                        channels = {key:[] for key in get_label_for_testing.keys()}
                        print('Evaluating results for -- ', 'Faults per inference:', faults, ' epochs:', str(number_of_fault_epochs), 'resil_method:', resil_method)

                        if parser.fault_type == 'weights_injs':
                            # loops through each epoch corresponding to each fault location
                            for i in range(number_of_fault_epochs):
                                fault_df_epoch = fault_df.iloc[i * total_images : (i+1) * total_images, :]
                                class_groups = fault_df_epoch.groupby(by=['gnd_label'])
                                for class_id in class_groups.groups:
                                    fault_layers[class_id].append(list(fault_df_epoch['layer'])[0])
                                    channels[class_id].append(list(fault_df_epoch['channel_in'])[0])        
                                    class_df_epoch = class_groups.get_group(class_id)
                                    evaluate_class_metrics(class_df_epoch, class_id, class_due_rate_dict, class_sdc_rate_dict, class_sc_sdc_rate_dict, 
                                                        class_corr_conf_score, class_resil_sdc_dict, class_resil_sc_sdc_dict, class_resil_due_rate_dict, class_resil_conf_score)    
                            total_datapoints = number_of_fault_epochs 
                        elif parser.fault_type == 'neurons_injs':   
                            number_of_faults_class = int(number_of_rows/ parser.num_classes)
                            number_of_batch_epochs = int(number_of_faults_class / parser.batch_size)
                            for i in range(parser.num_classes):
                                class_df = fault_df.iloc[i * number_of_faults_class : (i+1) * number_of_faults_class, :]
                                class_id = np.unique(class_df['gnd_label'].values).tolist()[0]
                                for j in range(number_of_batch_epochs):
                                    class_df_batch_epoch = class_df.iloc[j * parser.batch_size : (j+1) * parser.batch_size, :]
                                    evaluate_class_metrics(class_df_batch_epoch, class_id, class_due_rate_dict, class_sdc_rate_dict, class_sc_sdc_rate_dict, 
                                                        class_corr_conf_score, class_resil_sdc_dict, class_resil_sc_sdc_dict, class_resil_due_rate_dict, class_resil_conf_score)     
                            total_datapoints = number_of_batch_epochs                      
                        
                        faults_per_inference = [faults] * total_datapoints  
                        # loops across all miovision classes to fill up the SDC data along with safety-critical confusions
                        # for the epochs corresponding to a fault file
                        for i, class_name in enumerate(class_sdc_rate_dict):
                            class_name_list = [class_name] * total_datapoints
                            if not corrupted_model_sdc_flag:
                                resil_column = ['No_protection'] * total_datapoints
                                df = {'Miovision_class': class_name_list, 'Faults_per_inference': faults_per_inference, 
                                    'SDC_rate_per_epoch': class_sdc_rate_dict[class_name], 
                                    'Safety_critical_SDC_rate_per_epoch': class_sc_sdc_rate_dict[class_name],
                                    'Resil_method': resil_column, 
                                    # 'Resnet50_conv_layer': fault_layers[get_label_for_training[class_name]],
                                    # 'channel': channels[get_label_for_training[class_name]],
                                    'Confidence_score': class_corr_conf_score[class_name]
                                    # ,'Epochs_debug': [number_of_fault_epochs] * total_datapoints
                                    }
                                sdc_classwise = sdc_classwise.append(pd.DataFrame(df))

                            ##################################################
                            ## Below code is for checking resilient methods ##
                            ##################################################
                            resil_column = [resil_method] * total_datapoints
                            df = {'Miovision_class': class_name_list, 'Faults_per_inference':faults_per_inference, 
                                    'SDC_rate_per_epoch': class_resil_sdc_dict[class_name], 
                                    'Safety_critical_SDC_rate_per_epoch': class_resil_sc_sdc_dict[class_name],
                                    'Resil_method': resil_column, 
                                    # 'Resnet50_conv_layer': fault_layers[get_label_for_training[class_name]],
                                    # 'channel': channels[get_label_for_training[class_name]],
                                    'Confidence_score': class_resil_conf_score[class_name]
                                    # ,'Epochs_debug': [number_of_fault_epochs] * total_datapoints
                                    }
                            sdc_classwise = sdc_classwise.append(pd.DataFrame(df))   

                        # loops across all miovision classes to fill up the DUE data for the epochs corresponding to a fault file                        
                        for i, class_name in enumerate(class_due_rate_dict):
                            class_name_list = [class_name] * total_datapoints
                            if not corrupted_model_due_flag:
                                resil_column = ['No_protection'] * total_datapoints
                                df = {'Miovision_class': class_name_list, 'Faults_per_inference': faults_per_inference, 
                                    'DUE_rate_per_epoch': class_due_rate_dict[class_name], 'Resil_method': resil_column}
                                due_classwise = due_classwise.append(pd.DataFrame(df))

                            ##################################################
                            ## Below code is for checking resilient methods ##
                            ##################################################
                            resil_column = [resil_method] * total_datapoints
                            df = {'Miovision_class': class_name_list, 'Faults_per_inference':faults_per_inference, 
                                    'DUE_rate_per_epoch': class_resil_due_rate_dict[class_name], 'Resil_method': resil_column}
                            due_classwise = due_classwise.append(pd.DataFrame(df))                                                         

            corrupted_model_sdc_flag = True
            corrupted_model_due_flag = True

        results_classwise = [sdc_classwise, due_classwise]
        with open(save_file, 'wb') as f:
            pickle.dump(results_classwise, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    else:
        with open(save_file, 'rb') as f:
            results_classwise = pickle.load(f)
            sdc_classwise = results_classwise[0]
            due_classwise = results_classwise[1]
            f.close()   

    sdc_classwise['confusion_score'] = sdc_classwise['SDC_rate_per_epoch'] * sdc_classwise['Confidence_score']
    # sdc_rate_class_conv_layer_analysis(sdc_classwise, parser, filtered_out_nan_epochs)
    total_epochs = len(sdc_classwise.index)/(parser.num_classes * 3) # 3 stands for corrupt sdc without protection, ranger and clipper
    
    # plot_class_resil_comparison(sdc_classwise, parser, epochs=total_epochs)
    # plot_class_resil_comparison(sdc_classwise, parser, epochs=total_epochs, attribute='Safety_critical_SDC_rate_per_epoch')
    plot_class_resil_comparison(sdc_classwise, parser, epochs=total_epochs, attribute='confusion_score')

    plot_class_resil_comparison(due_classwise, parser, epochs=total_epochs, attribute='DUE_rate_per_epoch')
    
    sdc_classwise_sc_infused = sdc_classwise.copy()
    plot_class_resil_comparison_safety_confusion(sdc_classwise_sc_infused, parser, epochs=total_epochs)

    sdc_classwise = sdc_classwise[sdc_classwise['Resil_method'] == 'No_protection'] 
    # sdc_classwise = sdc_classwise[sdc_classwise['SDC_rate_per_epoch'] > 0.2]  
    # sdc_classwise = sdc_classwise[sdc_classwise['SDC_rate_per_epoch'] < 0.8]  
    plot_class_error_distribution(sdc_classwise, parser, epochs=total_epochs)
    plot_class_error_distribution(sdc_classwise, parser, epochs=total_epochs, attribute='confusion_score')
    test_central_limit_theorem(sdc_classwise, parser)
    test_central_limit_theorem(sdc_classwise, parser, attribute='confusion_score')
    
    # plot_class_sdc_fault_per_inference(sdc_classwise, epochs=sum(parser.fault_epochs), bit_range=parser.bit_range)
    # plot_class_sdc_fault_per_inference(sdc_classwise, epochs=sum(parser.fault_epochs), attribute='Safety_critical_SDC_rate_per_epoch', bit_range=parser.bit_range)


def evaluate_class_metrics(class_df_epoch, class_id, class_due_rate_dict, class_sdc_rate_dict, class_sc_sdc_rate_dict, class_corr_conf_score, class_resil_sdc_dict, 
                                                class_resil_sc_sdc_dict, class_resil_due_rate_dict, class_resil_conf_score):
    """This method is meant to evaluate all fault metrics like SDC and DUE for each class

    Args:
        class_df_epoch (DataFrame): Input fault dataframe to evaluate the metrics for
        class_due_rate_dict (Dictionary): Dictionary to store DUE rate classwise
        class_sdc_rate_dict (Dictionary): Dictionary to store SDC rate classwise
        class_sc_sdc_rate_dict (Dictionary): Dictionary to store Safety-critical SDC rate classwise
        class_corr_conf_score (Dictionary): Dictionary to store Confusion score classwise
        class_resil_sdc_dict (Dictionary): Dictionary to store SDC rate of resiliency methods classwise
        class_resil_sc_sdc_dict (Dictionary): Dictionary to store Safety-critical SDC rate of resiliency methods classwise
        class_resil_due_rate_dict (Dictionary): Dictionary to store DUE rate of resiliency methods classwise
        class_resil_conf_score (Dictionary): Dictionary to store Confusion score of resiliency methods classwise
    """
    class_df_epoch['orig output index - top5'] = class_df_epoch['orig output index - top5'].\
                        apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    class_df_epoch['orig_bool'] = class_df_epoch['gnd_label'] == class_df_epoch['orig output index - top5'].apply(lambda x: x[0])
    class_df_epoch = class_df_epoch[class_df_epoch['orig_bool'] == True]
    ## Corrupt model DUE rate calculation and storage
    class_due_rate_epoch = len(class_df_epoch[class_df_epoch['nan_or_inf_flag_corr_model'] == True].index)/len(class_df_epoch.index)                                
    class_due_rate_dict[get_label_for_testing[class_id]].append(class_due_rate_epoch)
    class_df_epoch_filtered = class_df_epoch[class_df_epoch['nan_or_inf_flag_corr_model'] == False]

    if len(class_df_epoch_filtered.index) > 0:
        class_df_epoch_filtered = preproc_df(class_df_epoch_filtered) # to evaluate all necessary SDC related metrics
        ## Corrupt model SDC Rate calculation
        class_sdc_rate_epoch = len(class_df_epoch_filtered[class_df_epoch_filtered['corr_bool'] == True].index)/len(class_df_epoch.index)
        ## Corrupt model Safety-critical confusion calculation
        class_sc_sdc_epoch = len(class_df_epoch_filtered[class_df_epoch_filtered['safety_critical_bool'] == True].index)/len(class_df_epoch.index) 
        ## Corrupt model Confusion score calculation
        class_corr_conf_score_epoch = class_df_epoch_filtered['Confidence_probability_error'].mean()
        ## Resil model SDC rate calculation
        class_resil_sdc_epoch = len(class_df_epoch_filtered[class_df_epoch_filtered['resil_bool'] == True].index)/len(class_df_epoch.index)
        ## Resil model Safety-critical confusion calculation
        class_resil_sc_sdc_epoch = len(class_df_epoch_filtered[class_df_epoch_filtered['safety_critical_resil_bool'] == True].index)/len(class_df_epoch.index)
        ## Resil model Confusion score calculation
        class_resil_conf_score_epoch = class_df_epoch_filtered['Confidence_probability_resiliency_error'].mean()
    else:
        class_sdc_rate_epoch = 0.0
        class_sc_sdc_epoch = 0.0
        class_corr_conf_score_epoch = 0.0
        class_resil_sdc_epoch = 0.0
        class_resil_sc_sdc_epoch = 0.0
        class_resil_conf_score_epoch = 0.0

    ## Corrupt model SDC Rate storage
    class_sdc_rate_dict[get_label_for_testing[class_id]].append(class_sdc_rate_epoch)
    ## Corrupt model Safety-critical confusion storage                                                                  
    class_sc_sdc_rate_dict[get_label_for_testing[class_id]].append(class_sc_sdc_epoch)
    ## Corrupt model Confusion score storage
    class_corr_conf_score[get_label_for_testing[class_id]].append(class_corr_conf_score_epoch)
    ## Resil model SDC rate storage                           
    class_resil_sdc_dict[get_label_for_testing[class_id]].append(class_resil_sdc_epoch)
    ## Resil model Safety-critical and storage        
    class_resil_sc_sdc_dict[get_label_for_testing[class_id]].append(class_resil_sc_sdc_epoch)
    ## Resil model Confusion score storage        
    class_resil_conf_score[get_label_for_testing[class_id]].append(class_resil_conf_score_epoch)

    ## Resil model DUE rate calculation and storage     
    class_df_epoch_nan_inf = class_df_epoch[class_df_epoch['nan_or_inf_flag_corr_model'] == True]       
    class_df_epoch_due_filtered = class_df_epoch_nan_inf[class_df_epoch_nan_inf['nan_or_inf_flag_resil_model'] == False]
    class_df_epoch_due_filtered['resil output index - top5'] = class_df_epoch_due_filtered['resil output index - top5'].\
                                    apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    class_df_epoch_due_filtered['resil_bool'] = class_df_epoch_due_filtered['gnd_label'] != class_df_epoch_due_filtered['resil output index - top5'].apply(lambda x: x[0])
    class_resil_due_epoch = len(class_df_epoch_due_filtered[class_df_epoch_due_filtered['resil_bool'] == True].index)/len(class_df_epoch.index)
    class_resil_due_rate_dict[get_label_for_testing[class_id]].append(class_resil_due_epoch)                               


def read_fault_files(file_paths, resil_location, resiliency_methods, dfs):
    """Method to read fault csv files

    Args:
        file_paths (List): List of fault csv files to read
        resil_location (Integer): Location after split of resil name in entire fault file name
        resiliency_methods (List): List of resiliency method names under evaluation
        dfs (Dictionary): Dictionary of Dataframes to store fault informations after reading fault csv files
    """
    for csv_paths in file_paths:
        resil_method = csv_paths[0].split('/')[resil_location].split('_')[2]
        if resil_method not in ['Clip', 'FmapRescale', 'BackFlip', 'FmapAvg']:
            resil_method = 'Ranger'
        
        if resil_method == 'Clip':
            resil_method = 'Clipper'
        elif resil_method == 'BackFlip':
            resil_method = 'Backflip'

        if resil_method in resiliency_methods: 
            resil_dict = dfs[resil_method]
            for csv_path in csv_paths:
                faults_per_inference = int(csv_path.split('_')[-5])
                df = pd.read_csv(csv_path, index_col=None, header=0)
                print(csv_path + ' !Read success!')
                resil_dict[faults_per_inference].append(df)


def test_central_limit_theorem(df, parser, attribute='SDC_rate_per_epoch'):
    """Apply central limit theorem concepts to generate sampling distribution of sample mean classwise

    Args:
        df (DataFrame): Input Dataframe containing all class informations
        parser (ConfigParser): Configs of miovision.yml
        attribute (str, optional): [Attribute on which to apply CLT]. Defaults to 'SDC_rate_per_epoch'.
    """
    np.random.seed(parser.random_seed)
    class_sample_mean_dist_stats = pd.DataFrame(columns=['Miovision_class', 'Mean', 'Standard_error', 'Confidence_interval'])
    classwise_groups = df.groupby(by=['Miovision_class'])
    sample_mean_attribute = 'sample_mean_' + attribute        
    # loops through each class
    for class_name in classwise_groups.groups:
        class_df = classwise_groups.get_group(class_name)
        class_sample_means = []
        class_epoch_datapoints = list(class_df[attribute])
        for _ in range(30000): # 10000 # need to vary this value to check the plausibiltiy of sensitivity order
            sample_mean = np.random.choice(class_epoch_datapoints, 30).mean() # need to vary this value to check the plausibiltiy of sensitivity order
            class_sample_means.append(sample_mean)
        class_sample_means_df = pd.DataFrame(columns=[sample_mean_attribute])
        df = {sample_mean_attribute: class_sample_means}
        class_sample_means_df = class_sample_means_df.append(pd.DataFrame(df))
        class_sample_means_mean, class_sample_means_standard_error = mean_confidence_interval(class_sample_means_df[sample_mean_attribute])
        class_stats = {'Miovision_class': class_name, 'Mean': class_sample_means_mean, 'Standard_error': class_sample_means_standard_error,
                        'Confidence_interval': [[round(class_sample_means_mean - class_sample_means_standard_error, 4), 
                                round(class_sample_means_mean + class_sample_means_standard_error, 4)]]}
        class_sample_mean_dist_stats = class_sample_mean_dist_stats.append(pd.DataFrame(class_stats))
        plot_sample_mean_distribution(class_sample_means_df, class_name, sample_mean_attribute, parser)
    print('------------', sample_mean_attribute, '------------------')
    print(class_sample_mean_dist_stats)
    render_mpl_table(parser, sample_mean_attribute, class_sample_mean_dist_stats, header_columns=0, col_width=5.0)


def sdc_rate_class_conv_layer_analysis(sdc_classwise, parser, epochs):
    sdc_average_layerwise = get_average_layer_class_sdc(sdc_classwise, parser)
    plot_class_sdc_rate_layer_average(sdc_average_layerwise, epochs=epochs, bit_range=parser.bit_range)
    plot_class_sdc_rate_layer(sdc_classwise, epochs=epochs, bit_range=parser.bit_range)


def get_average_layer_class_sdc(sdc_classwise, parser):
    sdc_average_layerwise = pd.DataFrame(columns=['Miovision_class', 'Mean_SDC_rate_per_epoch', 'Resnet50_conv_layer'])
    layer_groups = sdc_classwise.groupby(by=['Resnet50_conv_layer'])
    # loops through each resnet50 layer dataframe
    for conv_layer in layer_groups.groups:
        layer_df = layer_groups.get_group(conv_layer)
        layer_fault_epochs = len(layer_df.index) / parser.num_classes
        print('Conv layer: ', conv_layer, 'number of fault epochs: ', int(layer_fault_epochs))
        classwise_groups = layer_df.groupby(by=['Miovision_class'])
        # loops through each class
        for class_name in classwise_groups.groups:
            class_layer_df = classwise_groups.get_group(class_name)
            df = {'Miovision_class': [class_name], 'Mean_SDC_rate_per_epoch': [class_layer_df['SDC_rate_per_epoch'].mean()],
                  'Resnet50_conv_layer': [conv_layer]}
            sdc_average_layerwise = sdc_average_layerwise.append(pd.DataFrame(df))
    
    return sdc_average_layerwise


def confidence_probability_check(df):
    """Method to infuse the confusion score into input dataframe

    Args:
        df (DataFrame): Input fault dataframe

    Returns:
        [DataFrame]: Fault dataframe with the calculated confusion scores
    """

    df['Confidence_probability_resiliency_error'] = np.where((df['resil_bool'] == True), 1, 0)
    update_indexes = df[df['Confidence_probability_resiliency_error'] == 1].index
    df_resil_corrupted = df[df['resil_bool'] == True]
    confidence_resil_probability_errors = prediction_probability_analysis(df_resil_corrupted, resil=True)
    df.loc[update_indexes, 'Confidence_probability_resiliency_error'] = confidence_resil_probability_errors

    # df['Confidence_probability_safety_critical_resiliency_error'] = np.where((df['safety_critical_resil_bool'] == True), 1, 0)
    # update_indexes = df[df['Confidence_probability_safety_critical_resiliency_error'] == 1].index
    # df_resil_sc_corrupted = df[df['safety_critical_resil_bool'] == True]
    # confidence_resil_sc_probability_errors = prediction_probability_analysis(df_resil_sc_corrupted, resil=True)
    # df.loc[update_indexes, 'Confidence_probability_safety_critical_resiliency_error'] = confidence_resil_sc_probability_errors

    df['Confidence_probability_error'] = np.where((df['corr_bool'] == True), 1, 0)
    update_indexes = df[df['Confidence_probability_error'] == 1].index
    df_corrupted = df[df['corr_bool'] == True]
    confidence_probability_errors = prediction_probability_analysis(df_corrupted)
    df.loc[update_indexes, 'Confidence_probability_error'] = confidence_probability_errors

    # df['Confidence_probability_safety_critical_error'] = np.where((df['safety_critical_bool'] == True), 1, 0)
    # update_indexes = df[df['Confidence_probability_safety_critical_error'] == 1].index
    # df_sc_corrupted = df[df['safety_critical_bool'] == True]
    # confidence_sc_probability_errors = prediction_probability_analysis(df_sc_corrupted)
    # df.loc[update_indexes, 'Confidence_probability_safety_critical_error'] = confidence_sc_probability_errors
    
    return df

def prediction_probability_analysis(df, resil=False):
    """Method to calculate the confusion score

    Args:
        df (DataFrame): Input fault dataframe to calculate confusion score for
        resil (bool, optional): [Flag to differentiate between whether to generate confusion score for corrupted model or resil model]. Defaults to False.

    Returns:
        [List]: List of calculated confusion scores
    """
    original_prediction = list(df['gnd_label'])
    original_prediction_probability_top5 = list(df['orig output prob - top5'])
    if not resil:
        target_prediction_top5 = list(df['corr output index - top5'])
        target_prediction_probability_top5 = list(df['corr output prob - top5'])
    else:
        target_prediction_top5 = list(df['resil output index - top5'])
        target_prediction_probability_top5 = list(df['resil output prob - top5'])

    confidence_probability_errors = []
    for i in range(len(original_prediction)):
        score = 0.0
        check_flag = False
        orig_pred = original_prediction[i]
        target_prediction = target_prediction_top5[i]
        target_prediction_probability = target_prediction_probability_top5[i]
        for j in range(4):
            if orig_pred == target_prediction[j+1]:
                corr_conf_prob = target_prediction_probability[j+1]
                score = abs(original_prediction_probability_top5[i][0] - corr_conf_prob) * (j+1) 
                check_flag = True

        if not check_flag:
            remaining_confidence = 100 - sum(target_prediction_probability)
            score = abs(original_prediction_probability_top5[i][0] - remaining_confidence) * 6 
        confidence_probability_errors.append(score)

    return confidence_probability_errors


def preproc_df(df):
    """Method to preprocess the fault dataframe

    Args:
        df (DataFrame): Input fault dataframe

    Returns:
        [DataFrame]: Preprocessed dataframe 
    """
    # df['orig output index - top5'] = df['orig output index - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    df['orig output prob - top5'] = df['orig output prob - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))
 
    df['corr output index - top5'] = df['corr output index - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    df['corr_output_top1'] = df['corr output index - top5'].apply(lambda x: x[0])
    df['corr output prob - top5'] = df['corr output prob - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))
 
    df['resil output index - top5'] = df['resil output index - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    df['resil output prob - top5'] = df['resil output prob - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))

    df['resil_wo_fi output index - top5'] = df['resil_wo_fi output index - top5'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    df['corr_bool'] = df['gnd_label'] != df['corr output index - top5'].apply(lambda x: x[0])
    df['resil_bool'] = df['gnd_label'] != df['resil output index - top5'].apply(lambda x: x[0])
    df['resil_wo_fi_bool'] = df['gnd_label'] != df['resil_wo_fi output index - top5'].apply(lambda x: x[0])
    
    df = calculate_safety_criticality(df)
    df = confidence_probability_check(df)

    # if faults_per_inference == 1:
    #     df['fault_runset'] = df[df.columns[10:17]].values.tolist()
    # else:
    #     df['layer'] = df['layer'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', '').replace(',', ''), dtype=int, sep=' '))
    #     df['channel_in'] = df['channel_in'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', '').replace(',', ''), dtype=int, sep=' '))
    #     df['channel_out'] = df['channel_out'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', '').replace(',', ''), dtype=int, sep=' '))
    #     df['3D channel'] = df['3D channel'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', '').replace(',', ''), dtype=int, sep=' '))
    #     df['height'] = df['height'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', '').replace(',', ''), dtype=int, sep=' '))
    #     df['width'] = df['width'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', '').replace(',', ''), dtype=int, sep=' '))
    #     df['bit'] = df['bit'].apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', '').replace(',', ''), dtype=int, sep=' '))
    #     df['fault_runset'] = list(df[df.columns[10:17]].to_numpy())
    return df


def resil_method_comparison(ranger_comparison_csv_paths, parser):
    """This method is to compare various ranger variants for mulitple number of faults per inference 
       (experiments for EDCC paper)

    Args:
        ranger_comparison_csv_paths ([list]): List of csv fault files with 'Ranger' in its name
        parser : Miovision config parser
    """    
    resiliency_methods = ['Ranger', 'Clipper', 'FmapRescale', 'Backflip', 'FmapAvg'] 
    dfs = {key:{inner_key:[] for inner_key in parser.faults_per_inference} for key in resiliency_methods}

    sdc_ranger_variants = pd.DataFrame(columns=['Resil_method', 'Faults_per_inference', 'SDC_rate_per_epoch', 'Safety_critical_SDC_rate_per_epoch'])
    due_ranger_variants = pd.DataFrame(columns=['Resil_method', 'Faults_per_inference', 'DUE_rate_per_epoch']) 
    safety_confusion_history = pd.DataFrame(columns=['Resil_method', 'Faults_per_inference', 'Original_class', 'Predicted_Class', 'Safety_confusion', 'Count'])                               

    save_file, resil_location = initialize_params(parser)
    corrupted_model_sdc_flag = False # flag to fill the SDC details of corrupted model only once
    corrupted_model_due_flag = False # flag to fill the DUE details of corrupted model only once

    if not os.path.isfile(save_file):
        read_fault_files(ranger_comparison_csv_paths, resil_location, resiliency_methods, dfs)

        # loops for each ranger variant resil method
        for resil_method in dfs.keys():
            fault_results_df = dfs[resil_method]
            # loops for each number of faults per inference
            for faults in fault_results_df.keys():
                safety_confusion_df = pd.DataFrame()
                if fault_results_df[faults]:
                    # loops for multiple epochs fault file for a resil method and particular faults per inference
                    for fault_df in fault_results_df[faults]:
                        # unique_faults = np.unique(fault_df_epoch[fault_df_epoch.columns[10:17:]].values, axis=0)
                        number_of_rows = len(fault_df.index) 
                        total_images = parser.num_classes * parser.number_of_images_each_class
                        number_of_epochs = int(number_of_rows / total_images)
                        faults_per_inference = [faults] * number_of_epochs   
                        corrupted_sdc_rate, corrupted_sdc_sc_rate, resil_sdc_rate, resil_sc_sdc_rate, corrupted_due_rate, resil_due_rate = ([] for i in range(6))                                                     
                        for i in range(number_of_epochs):
                            fault_df_epoch = fault_df.iloc[i * total_images : (i+1) * total_images, :]

                            fault_df_epoch_nan_inf = fault_df_epoch[fault_df_epoch['nan_or_inf_flag_corr_model'] == True]
                            fault_df_epoch_filtered = fault_df_epoch[fault_df_epoch['nan_or_inf_flag_corr_model'] == False]

                            if len(fault_df_epoch_filtered.index) > 0:
                                fault_df_epoch_filtered = preproc_df(fault_df_epoch_filtered) # to evaluate all necessary SDC related metrics
                                sdc_rate_wo_resil_epoch = len(fault_df_epoch_filtered[fault_df_epoch_filtered['corr_bool'] == True].index) * 100 / len(fault_df_epoch.index)
                                sc_sdc_rate_wo_resil_epoch = len(fault_df_epoch_filtered[fault_df_epoch_filtered['safety_critical_bool'] == True].index) * 100 / len(fault_df_epoch.index)
                            else:
                                sdc_rate_wo_resil_epoch = 0.0
                                sc_sdc_rate_wo_resil_epoch = 0.0
                            
                            due_rate_wo_resil_epoch = len(fault_df_epoch_nan_inf.index) * 100 / len(fault_df_epoch.index)
                            
                            corrupted_sdc_rate.append(sdc_rate_wo_resil_epoch)
                            corrupted_sdc_sc_rate.append(sc_sdc_rate_wo_resil_epoch)
                            corrupted_due_rate.append(due_rate_wo_resil_epoch)
                            
                            if len(fault_df_epoch_filtered.index) > 0:
                                sdc_rate_with_resil_epoch = len(fault_df_epoch_filtered[fault_df_epoch_filtered['resil_bool'] == True].index) * 100 / len(fault_df_epoch.index)
                                sc_sdc_rate_with_resil_epoch = len(fault_df_epoch_filtered[fault_df_epoch_filtered['safety_critical_resil_bool'] == True].index) * 100 / len(fault_df_epoch.index)
                            else:
                                sdc_rate_with_resil_epoch = 0.0
                                sc_sdc_rate_with_resil_epoch = 0.0

                            fault_df_epoch_nan_inf['resil output index - top5'] = fault_df_epoch_nan_inf['resil output index - top5'].\
                                        apply(lambda x: np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
                            fault_df_epoch_nan_inf['resil_bool'] = fault_df_epoch_nan_inf['gnd_label'] != fault_df_epoch_nan_inf['resil output index - top5'].\
                                        apply(lambda x: x[0])
                            
                            due_rate_with_resil_epoch = len(fault_df_epoch_nan_inf[fault_df_epoch_nan_inf['resil_bool'] == True].index) * 100 / len(fault_df_epoch.index)

                            resil_sdc_rate.append(sdc_rate_with_resil_epoch)
                            resil_sc_sdc_rate.append(sc_sdc_rate_with_resil_epoch)
                            resil_due_rate.append(due_rate_with_resil_epoch)

                            if len(fault_df_epoch_filtered.index) > 0:
                                safety_confusion_df = safety_confusion_df.append(fault_df_epoch_filtered[fault_df_epoch_filtered['corr_bool'] == True]) 

                        if not corrupted_model_sdc_flag:
                            resil_column = ['No_protection'] * number_of_epochs   
                            sdc_data_wo_resil = {'Resil_method': resil_column, 
                                                'Faults_per_inference': faults_per_inference, 'SDC_rate_per_epoch': corrupted_sdc_rate,
                                                'Safety_critical_SDC_rate_per_epoch': corrupted_sdc_sc_rate}                
                            sdc_ranger_variants = sdc_ranger_variants.append(pd.DataFrame(sdc_data_wo_resil))                          
                                                                        
                        resil_column = [resil_method] * number_of_epochs
                        sdc_data_with_resil = {'Resil_method': resil_column, 
                                            'Faults_per_inference': faults_per_inference, 'SDC_rate_per_epoch': resil_sdc_rate,
                                            'Safety_critical_SDC_rate_per_epoch': resil_sc_sdc_rate}               
                        sdc_ranger_variants = sdc_ranger_variants.append(pd.DataFrame(sdc_data_with_resil))
                        print('Faults per inference:', faults, ' epochs:' + str(number_of_epochs) + ' SDC rate without resil:', sum(corrupted_sdc_rate)/len(fault_df_epoch.index))
                        print('Faults per inference:', faults, ' epochs:' + str(number_of_epochs) + ' SDC rate with resil ', resil_method, ':', 
                                                                    sum(resil_sdc_rate)/len(fault_df_epoch.index))

                        if not corrupted_model_due_flag:
                            resil_column = ['No_protection'] * number_of_epochs   
                            due_data_wo_resil = {'Resil_method': resil_column, 
                                                'Faults_per_inference': faults_per_inference, 'DUE_rate_per_epoch': corrupted_due_rate}                
                            due_ranger_variants = due_ranger_variants.append(pd.DataFrame(due_data_wo_resil))                            
                                                                        
                        resil_column = [resil_method] * number_of_epochs
                        due_data_with_resil = {'Resil_method': resil_column, 
                                            'Faults_per_inference': faults_per_inference, 'DUE_rate_per_epoch': resil_due_rate}               
                        due_ranger_variants = due_ranger_variants.append(pd.DataFrame(due_data_with_resil))
                
                safety_confusion_history = update_safety_confusion_details(safety_confusion_history, safety_confusion_df, 
                                                                            faults, resil_method, corrupted_model_sdc_flag)
            
            corrupted_model_sdc_flag = True
            corrupted_model_due_flag = True

        with open(save_file, 'wb') as f:
            results = [sdc_ranger_variants, due_ranger_variants, safety_confusion_history]
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    else:
        with open(save_file, 'rb') as f:
            results = pickle.load(f)
            sdc_ranger_variants = results[0]
            due_ranger_variants = results[1]
            safety_confusion_history = results[2]
            f.close()    

    calculate_statistics_sdc_epochs(sdc_ranger_variants, parser, safety_confusion_history)
    
    plot_sdc_safety_criticality_imposed_ranger_variants(sdc_ranger_variants, bit_range=parser.bit_range, fault_type=parser.fault_type)
    plot_sdc_ranger_variants(sdc_ranger_variants, epochs=sum(parser.fault_epochs), bit_range=parser.bit_range, fault_type=parser.fault_type)
    plot_sdc_ranger_variants(sdc_ranger_variants, epochs=sum(parser.fault_epochs), attribute='Safety_critical_SDC_rate_per_epoch', bit_range=parser.bit_range
                            ,fault_type=parser.fault_type) 
    plot_sdc_ranger_variants(due_ranger_variants, epochs=sum(parser.fault_epochs), attribute='DUE_rate_per_epoch', bit_range=parser.bit_range
                            ,fault_type=parser.fault_type)


def update_safety_confusion_details(safety_confusion_history, update_df, num_faults, resil, corrupt_model_check_flag):
    """Method to find the safety confusion transitions for the corrupted images

    Args:
        safety_confusion_history (DataFrame): Safety confusion dataframe
        update_df (DataFrame): Update dataframe to be merge into Safet confusion dataframe
        num_faults (int): Faults per inference (num_faults as in default.yml)
        resil (str): name of resiliency method
        corrupt_model_check_flag (bool): flag to confirm whether all 'no_protection' cases have been considered already or not

    Returns:
        [DataFrame]: updated safety confusion dataframe
    """
    if not corrupt_model_check_flag:
        safety_groups = update_df.groupby(by=['safety_critical_bool', 'gnd_label', 'corr_output_top1'])
        for confusion_group in safety_groups.groups:
            info = {'Resil_method': ['No_protection'], 'Faults_per_inference': [num_faults], 'Original_class': [get_label_for_testing[confusion_group[1]]], \
                    'Predicted_Class': [get_label_for_testing[confusion_group[2]]], 'Safety_confusion': [confusion_group[0]], 
                    'Count': [len(safety_groups.get_group(confusion_group).index)]}
            safety_confusion_history = safety_confusion_history.append(pd.DataFrame(info))

    resil_safety_groups = update_df.groupby(by=['safety_critical_resil_bool', 'gnd_label', 'corr_output_top1'])
    for confusion_group in resil_safety_groups.groups:
        info = {'Resil_method': [resil], 'Faults_per_inference': [num_faults], 'Original_class': [get_label_for_testing[confusion_group[1]]],
                'Predicted_Class': [get_label_for_testing[confusion_group[2]]], 'Safety_confusion': [confusion_group[0]], 
                'Count': [len(resil_safety_groups.get_group(confusion_group).index)]}
        safety_confusion_history = safety_confusion_history.append(pd.DataFrame(info))

    return safety_confusion_history


def mean_confidence_interval(data, confidence=0.95):
    """Method to calculate mean and error margin of data sample

    Args:
        data (List): Sample data to evaluate mean and error margin for
        confidence (float, optional): [Confidence percentage]. Defaults to 0.95.

    Returns:
        [type]: [description]
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stat.sem(a)
    h = se * stat.t.ppf((1 + confidence) / 2., n-1)
    return round(m, 4), round(h, 4)


def calculate_statistics_sdc_epochs(df, parser, safety_confusion_history):
    """method to calculate statistics of fault informations and save them into output files

    Args:
        df (DataFrame): input dataframe
        parser (ConfigParser): Config of miovision.yml
        safety_confusion_history (DataFrame): Safety confusion dataframe
    """
    statistics_df = pd.DataFrame(columns=['Attribute', 'Faults_per_inference', 'Resil_method', 'Mean', 'Error_margin'])
    fault_per_inference_groups = df.groupby(by=['Faults_per_inference'])
    for fault in fault_per_inference_groups.groups:
        print('Faults per inference: ', fault)
        fault_group_df = fault_per_inference_groups.get_group(fault)
        fault_resil_groups = fault_group_df.groupby(by=['Resil_method'])
        for fault_resil_group in fault_resil_groups.groups:
            fault_resil_group_df = fault_resil_groups.get_group(fault_resil_group)
            sdc_mean, sdc_error_margin = mean_confidence_interval(list(fault_resil_group_df['SDC_rate_per_epoch']))
            sc_sdc_mean, sc_sdc_error_margin = mean_confidence_interval(list(fault_resil_group_df['Safety_critical_SDC_rate_per_epoch']))
            data = {'Attribute': ['SDC_rate_per_epoch', 'Safety_critical_SDC_rate_per_epoch'], 
                    'Faults_per_inference': [fault, fault], 'Resil_method': [fault_resil_group] * 2,
                    'Mean': [sdc_mean, sc_sdc_mean], 'Error_margin': [sdc_error_margin, sc_sdc_error_margin]}
            statistics_df = statistics_df.append(pd.DataFrame(data))
    
    file_location = 'miovision_results/resnet50_mio/experiments/ranger_variants_sdc/' + parser.fault_type + '/' + parser.bit_range + '/sdc_statistics_' + \
                       parser.fault_type + '.csv'
    safety_history_file_location = 'miovision_results/resnet50_mio/experiments/ranger_variants_sdc/' + parser.fault_type + '/' + parser.bit_range + \
                    '/safety_confusion_statistics_' + parser.fault_type + '.csv'
    statistics_df.to_csv(file_location, index=False)
    safety_confusion_history.to_csv(safety_history_file_location, index=False)


def initialize_params(parser, analysis_type='resil'):
    """Method to initialize parameters based on resiliency analysis on whole data or classwise

    Args:
        parser (ConfigParser): Config of miovision.yml
        analysis_type (str, optional): ['Resil' for whole dataset & 'classwise' for class analysis]. Defaults to 'resil'.

    Returns:
        [type]: [description]
    """
    if analysis_type == 'resil':
        save_file = 'miovision_results/' + parser.model_name + '_mio/archive_files/sdc_ranger_variants_' + parser.fault_type + '.pkl'
    elif analysis_type == 'classwise':
        if not parser.train_uniform_class_dist:
            save_file = 'miovision_results/' + parser.model_name + '_mio/archive_files/sdc_due_class_' + parser.fault_type + '_' + parser.bit_range + '_' + \
                        str(parser.faults_per_inference[0]) + '_fault_' + parser.model_name + '.pkl'
        else:
            save_file = 'miovision_results/' + parser.model_name + '_mio/archive_files/sdc_due_class_' + parser.fault_type + '_1_fault_' + \
                            parser.model_name + '_uniform_class.pkl'

    resil_location = 4

    return save_file, resil_location


def main():
    mio_config_location = 'alficore/dataloader/miovision/miovision_config.yml' 
    scenario = parser.load_scenario(conf_location=mio_config_location)
    mio_config_parser = parser.ConfigParser(**scenario)

    fault_config_location = 'scenarios/default.yml' 
    fault_scenarios = parser.load_scenario(conf_location=fault_config_location)
    fault_config_parser = fault_parser.ConfigParser(**fault_scenarios)

    if mio_config_parser.model_name == 'resnet50':
        model = resnet50_net(mio_config_parser.num_classes)
        # model = resnet50_load_checkpoint(model)
        if not mio_config_parser.train_uniform_class_dist:
            model = resnet50_load_checkpoint(model)
        else:
            model = resnet50_load_checkpoint(model, uniform_class_dict=True)
    elif mio_config_parser.model_name == 'vgg16':
        model = vgg16_net(mio_config_parser.num_classes)
        model = vgg16_load_checkpoint(model)
        
    ranger_file_name = mio_config_parser.ranger_file_name

    # # Get/Extract Ranger bounds:
    # # If they dont exist extract them here:    
    if mio_config_parser.generate_ranger_bounds:
        gen_model_ranger_bounds(model, mio_config_parser, ranger_file_name)

    # for fault injection
    if mio_config_parser.inject_fault:
        inject_fault_evaluate_ranger(ranger_file_name, model, mio_config_parser)

    fault_analysis(fault_config_parser, mio_config_parser)


if __name__ == "__main__":
    main()


