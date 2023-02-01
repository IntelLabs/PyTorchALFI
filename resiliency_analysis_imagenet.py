import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def preproc_df_imagenet(df, resil):
    df['orig output index - top5'] = df['orig output index - top5'].apply(lambda x: 
                np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    df['orig output prob - top5'] = df['orig output prob - top5'].apply(lambda x: 
                np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))
 
    df['corr output index - top5'] = df['corr output index - top5'].apply(lambda x: 
                np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
    df['corr output prob - top5'] = df['corr output prob - top5'].apply(lambda x: 
                np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))
    
    df['orig_bool'] = df['gnd_label'] == df['orig output index - top5'].apply(lambda x: x[0])
    df['corr_bool'] = df['gnd_label'] != df['corr output index - top5'].apply(lambda x: x[0])
 
    if resil:
        df['resil output index - top5'] = df['resil output index - top5'].apply(lambda x: 
                    np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
        df['resil output prob - top5'] = df['resil output prob - top5'].apply(lambda x: 
                    np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=float, sep=' '))

        # df['layer'] = df['layer'].apply(lambda x: 
        #             np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', '').replace(',', ''), dtype=int, sep=' '))
        df['resil_wo_fi output index - top5'] = df['resil_wo_fi output index - top5'].apply(lambda x: 
                    np.fromstring(x.replace('[','').replace(']','').replace('array(', '').replace(')', ''), dtype=int, sep=' '))
        df['resil_bool'] = df['gnd_label'] != df['resil output index - top5'].apply(lambda x: x[0])
        df['resil_wo_fi_bool'] = df['gnd_label'] != df['resil_wo_fi output index - top5'].apply(lambda x: x[0])

    return df


def analyse_fault_file(csv_file_path, num_of_images=1000, resil=False, nan_inf_check=False):
    fault_df = pd.read_csv(csv_file_path, index_col=None, header=0)
    print(csv_file_path + ' !Read success!')
    number_of_rows = len(fault_df.index)
    number_of_epochs = int(number_of_rows / num_of_images)
    original_accuracy = []
    corrupted_sdc_rate = []
    resil_sdc_rate = []
    corrupted_due_rate = []
    resil_due_rate = []
    # loop through the fault epochs
    for i in range(number_of_epochs):
        fault_df_epoch = fault_df.iloc[i * num_of_images : (i+1) * num_of_images, :]
        fault_df_epoch = preproc_df_imagenet(fault_df_epoch, resil)
        orig_model_correctly_classified_df = fault_df_epoch[fault_df_epoch['orig_bool'] == True]
        original_model_accuracy = len(orig_model_correctly_classified_df.index) / num_of_images
        original_accuracy.append(original_model_accuracy)
        if nan_inf_check:
            # calculates the Detected uncorrectable error (DUE) rate for corrupted model
            due_rate_wo_resil_epoch = len(orig_model_correctly_classified_df[orig_model_correctly_classified_df['nan_or_inf_flag_corr_model'] == True].index)/len(orig_model_correctly_classified_df.index)            
            corrupted_due_rate.append(due_rate_wo_resil_epoch)
            # calculates the Silent Data Corruption (SDC) rate for corrupted model
            wo_nan_inf_wo_resil_epoch = orig_model_correctly_classified_df[orig_model_correctly_classified_df['nan_or_inf_flag_corr_model'] == False]
            sdc_rate_wo_resil_epoch = len(wo_nan_inf_wo_resil_epoch[wo_nan_inf_wo_resil_epoch['corr_bool'] == True].index)/len(orig_model_correctly_classified_df.index)
        else:
            sdc_rate_wo_resil_epoch = len(orig_model_correctly_classified_df[orig_model_correctly_classified_df['corr_bool'] == True].index)/len(orig_model_correctly_classified_df.index)
        corrupted_sdc_rate.append(sdc_rate_wo_resil_epoch)
        if resil:
            if nan_inf_check:
                # calculates the Detected uncorrectable error (DUE) rate for resilient model
                due_rate_with_resil_epoch = len(orig_model_correctly_classified_df[orig_model_correctly_classified_df['nan_or_inf_flag_resil_model'] == True].index)/len(orig_model_correctly_classified_df.index)
                resil_due_rate.append(due_rate_with_resil_epoch)
                # calculates the Silent Data Corruption (SDC) rate for resilient model
                wo_nan_inf_with_resil_epoch = orig_model_correctly_classified_df[orig_model_correctly_classified_df['nan_or_inf_flag_resil_model'] == False]
                sdc_rate_with_resil_epoch = len(wo_nan_inf_with_resil_epoch[wo_nan_inf_with_resil_epoch['resil_bool'] == True].index)/len(orig_model_correctly_classified_df.index)
            else:
                sdc_rate_with_resil_epoch = len(orig_model_correctly_classified_df[orig_model_correctly_classified_df['resil_bool'] == True].index)/len(orig_model_correctly_classified_df.index)
            resil_sdc_rate.append(sdc_rate_with_resil_epoch)
        

    print('Epochs:', len(original_accuracy))
    print('Original accuracy:', np.mean(original_accuracy))
    print('Mean SDC Rate Corrupted model:', np.mean(corrupted_sdc_rate))
    if nan_inf_check:
        print('Mean DUE Rate Corrupted model:', np.mean(corrupted_due_rate))
    if resil:
        print('Mean SDC Rate Resil model:', np.mean(resil_sdc_rate))
        if nan_inf_check:
            print('Mean DUE Rate Resil model:', np.mean(resil_due_rate))


def main():
    # fault_file_path = 'result_files/mio/all_exp/vgg16_bn/vgg16_bn_Ranger_500_trials/csv_files/weights_injs/' + \
    #                         'vgg16_bn_test_random_sbf_weights_inj_10_500_50_miovision_results.csv'
    
    fault_file_path = '/home/fgeissle/ranger_repo/ranger/result_files/VPU_test/resnet50/resnet50_ranger_500_trials/csv_files/weights_injs/resnet50_test_random_sbf_weights_inj_1_500_1_imagenet_results.csv'
    analyse_fault_file(fault_file_path, num_of_images=1000)


if __name__ == "__main__":
    main()