import pickle
import os
from collections import Iterable
import numpy as np
import torch
import matplotlib.pyplot as plt

""" Format:
    # Save results ---------------------------------------------------
    result_dict = dict()
    result_dict["orig_correct_pred_top1"] = glob_correct_pred_runs_top1 # List structure: epoch -> batches -> images. 
    result_dict["fi_correct_pred_top1"] = glob_fi_correct_pred_runs_top1 # format as above ...(for all 4 here)
    result_dict["orig_correct_pred_top3"] = glob_correct_pred_runs_top3
    result_dict["fi_correct_pred_top3"] = glob_fi_correct_pred_runs_top3
    result_dict["orig_correct_pred_top5"] = glob_correct_pred_runs_top5
    result_dict["fi_correct_pred_top5"] = glob_fi_correct_pred_runs_top5
    result_dict["orig_ranger_act"] = glob_ranger_act_runs
    result_dict["fi_ranger_act"] = glob_fi_ranger_act_runs
    result_dict['class_pred'] = glob_class_pred
    result_dict['fi_class_pred'] = glob_fi_class_pred
    result_dict['fi_time_list'] = glob_time_list
    result_dict['rmse_list'] = glob_rmse_list
    
    result_dict["runset"] = runset #7xfaults
    if stop_ds_at is not None:
        ds_len = np.min([testloader.__len__(), stop_ds_at])*test_batch_size
    else:
        ds_len = testloader.__len__()*test_batch_size
    result_dict["nr_images"] = ds_len 
    result_dict["batch_size"] = test_batch_size
    result_dict["fault_rate"] = wrapper.max_fault_rate
    result_dict["fixed_nr_faults"] = wrapper.max_faults_per_image
    result_dict["Num_epochs"] = wrapper.num_runs
    result_dict["layer_types"] = wrapper.parser.layer_types
    result_dict["injection_target"] = wrapper.parser.rnd_mode
    result_dict["protection"] = "Ranger" 
    result_dict["fault_type"] = wrapper.parser.rnd_value_type 
    result_dict["nr_bits"] = wrapper.parser.rnd_value_bits 

    # Added in the process of analysis
    result_dict["acc_unfiltered"] = [mean_epochs, std_epochs] 
    result_dict["fi_acc_unfiltered"] = [fi_mean_epochs, fi_std_epochs]
    result_dict["acc_filtered"] = [mean_epochs, std_epochs] #add to dict
    result_dict["fi_acc_filtered"] = [fi_mean_epochs, fi_std_epochs]
    result_dict["sdc"] = [sdc_epochs, sdc_std_epochs] #add to dict
    result_dict["fi_sdc"] = [fi_sdc_epochs, fi_sdc_std_epochs]
    result_dict["ranger_act_nr"] = [np.sum(i) for i in orig_ranger_flattrunc]  # add to dict
    result_dict["fi_ranger_act_nr"] = [np.sum(i) for i in fi_ranger_flattrunc]
    
    result_dict["inf_time_perImage"] #in seconds
    result_dict["rmse"]
    
    # result_dict["ranger_cov"] = [ranger_epochs, ranger_std_epochs]  # add to dict
    # result_dict["fi_ranger_cov"] = [fi_ranger_epochs, fi_ranger_std_epochs]
    
    
    
    
    # Runset means:
     # meaning of rows is batches, layer, location, value
        # batches
        # modes = {"batch": parser.rnd_batch, "layer": parser.rnd_layer, "location": parser.rnd_location,
        #         "value": parser.rnd_value, "value_type": parser.rnd_value_type}
        idx = 0
        layernum = 0
        # runset is a numpy matrix with one fault per column
        # there are always 7 lines form which some are ignored for specific layer types
        # The meaning is different for neuron injection and weight injection. Neurons use batch size while weights don't
        # but instead have an additional dimension K

        # --- Meaning for NEURON injection: --- #
        # 1. batchnumber (used in: conv2d,conv3d)
        # 2. layer (everywhere)
        # 3. channel (used in: conv2d,conv3d)
        # 4. depth (used in: conv3d)
        # 5. height (everywhere)
        # 6. width (everywhere)
        # 7. value/bit position (everywhere)

        # --- Meaning for WEIGHT injection: --- #
        # 1. layer (everywhere)
        # 2. Kth filter (everywhere)
        # 3. channel(used in: conv2d, conv3d)
        # 4. depth (used in: conv3d)
        # 5. height (everywhere)
        # 6. width (everywhere)
        # 7. value/bit position (everywhere)
"""



def get_y_error_theo(results_dict, zeta):
    y_error_theo = [get_theor_error(results_dict[n]["Num_epochs"], zeta) for n in range(len(results_dict))]
    # print(y_error_theo)
    y_error_theo = [0.] + y_error_theo #first is for without faults
    return y_error_theo

def get_theor_error(nn, zeta):
    err = zeta*np.sqrt(0.25/nn)
    return err


def list_files_in_dir(dir):
    """
    Get all file names within a directory, with subfolder.
    :param dir:
    :return: list of file names
    """
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


def list_subdir_in_dir(dir):
    """
    Get all subfolder names within a folder.
    :param dir:
    :return: list of dir names
    """
    r = []
    for root, dirs, files in os.walk(dir):
        for name in dirs:
            r.append(os.path.join(root, name))
    return r


def list_direct_subdir_in_dir(dir):
    """
    Get all direct subfolder/file names within a folder.
    :param dir:
    :return: list of dir names
    """
    r = []
    for drs in os.listdir(dir):
        r.append(os.path.join(dir, drs))
    return r


def flatten(x):
    """
    Flatten any list to a single level.
    """
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def combine_images_sameEpochs(r):
    """
    :param: Adds images but keeps the same nr of epochs (one epoch that was split in two gets reunited).
    :return: returns combined dictionary of same format as first read-in dict
    """
    my_dict_final = {}  # Create an empty dictionary
    for file in r:
        if not my_dict_final: #if empty at first
            with open(file, 'rb') as f:
                my_dict_final.update(pickle.load(f))  # Update dict if it was empty
            # print('added:', file)
        else:
            with open(file, 'rb') as f:
                new_dict = pickle.load(f)
                if my_dict_final["fault_rate"] == new_dict["fault_rate"] and my_dict_final["injection_target"] == new_dict["injection_target"] \
                        and my_dict_final["nr_bits"] == new_dict["nr_bits"] and my_dict_final["fault_type"] == new_dict["fault_type"] \
                        and my_dict_final["protection"] == new_dict["protection"] and my_dict_final["layer_types"] == new_dict["layer_types"] \
                        and my_dict_final["fixed_nr_faults"] == new_dict["fixed_nr_faults"] \
                        and my_dict_final["Num_epochs"] == new_dict["Num_epochs"]: #batchsize may vary

                    my_dict_final["nr_images"] = my_dict_final["nr_images"] + new_dict["nr_images"]

                    # Combine ranger act
                    for ep in range(len(my_dict_final["orig_ranger_act"])): #for each epoch combine batches
                        my_dict_final["orig_ranger_act"][ep] = my_dict_final["orig_ranger_act"][ep] + new_dict["orig_ranger_act"][ep]
                    for ep in range(len(my_dict_final["fi_ranger_act"])):
                        my_dict_final["fi_ranger_act"][ep] = my_dict_final["fi_ranger_act"][ep] + new_dict["fi_ranger_act"][ep]

                    # Combine class pred
                    for ep in range(len(my_dict_final["class_pred"])):  # for each epoch combine batches
                        my_dict_final["class_pred"][ep] = my_dict_final["class_pred"][ep] + \
                                                               new_dict["class_pred"][ep]
                    for ep in range(len(my_dict_final["fi_class_pred"])):
                        my_dict_final["fi_class_pred"][ep] = my_dict_final["fi_class_pred"][ep] + \
                                                             new_dict["fi_class_pred"][ep]

                    # Combine time lst
                    for ep in range(len(my_dict_final["fi_time_list"])):  # for each epoch combine batches
                        my_dict_final["fi_time_list"][ep] = my_dict_final["fi_time_list"][ep] + new_dict["fi_time_list"][ep]

                    # Combine rmse lst
                    for ep in range(len(my_dict_final["rmse_list"])):  # for each epoch combine batches
                        my_dict_final["rmse_list"][ep] = my_dict_final["rmse_list"][ep] + new_dict["rmse_list"][ep]


                    # Combine predictions
                    for ep in range(len(my_dict_final["orig_correct_pred_top1"])): #for each epoch combine batches
                        my_dict_final["orig_correct_pred_top1"][ep] = my_dict_final["orig_correct_pred_top1"][ep] + new_dict["orig_correct_pred_top1"][ep]
                    for ep in range(len(my_dict_final["fi_correct_pred_top1"])): #for each epoch combine batches
                        my_dict_final["fi_correct_pred_top1"][ep] = my_dict_final["fi_correct_pred_top1"][ep] + new_dict["fi_correct_pred_top1"][ep]

                    for ep in range(len(my_dict_final["orig_correct_pred_top3"])):  # for each epoch combine batches
                        my_dict_final["orig_correct_pred_top3"][ep] = my_dict_final["orig_correct_pred_top3"][ep] + new_dict["orig_correct_pred_top3"][ep]
                    for ep in range(len(my_dict_final["fi_correct_pred_top3"])):  # for each epoch combine batches
                        my_dict_final["fi_correct_pred_top3"][ep] = my_dict_final["fi_correct_pred_top3"][ep] + new_dict["fi_correct_pred_top3"][ep]

                    for ep in range(len(my_dict_final["orig_correct_pred_top5"])):  # for each epoch combine batches
                        my_dict_final["orig_correct_pred_top5"][ep] = my_dict_final["orig_correct_pred_top5"][ep] + new_dict["orig_correct_pred_top5"][ep]
                    for ep in range(len(my_dict_final["fi_correct_pred_top5"])):  # for each epoch combine batches
                        my_dict_final["fi_correct_pred_top5"][ep] = my_dict_final["fi_correct_pred_top5"][ep] + new_dict["fi_correct_pred_top5"][ep]

                    # my_dict_final["runset"] = my_dict_final["runset"] + new_dict["runset"] #same runset was used!
                    # print('added:', file)
                else:
                    print('WARNING: folders not combined because of mismatch:')
                    print(my_dict_final["fault_rate"], new_dict["fault_rate"])
                    print(my_dict_final["injection_target"], new_dict["injection_target"])
                    print(my_dict_final["nr_bits"], new_dict["nr_bits"])
                    print(my_dict_final["fault_type"], new_dict["fault_type"])
                    print(my_dict_final["protection"], new_dict["protection"])
                    print(my_dict_final["layer_types"], new_dict["layer_types"])
                    print(my_dict_final["fixed_nr_faults"], new_dict["fixed_nr_faults"])
                    print(my_dict_final["Num_epochs"], new_dict["Num_epochs"])  # batchsize may vary
                    print('dictionaries dont match, skip file.')

    return my_dict_final


def add_epochs_sameImages(my_dict_final, new_dict):
    """
    Takes equal experiments across the same images and adds the epoch. Puts together multiple runs.
    :param: mydict_final, global dict to be updated
    :param: local dict that will be the update
    :return: returns combined dictionary of same format as first read-in dict
    """

    if not my_dict_final: #if empty at first
        # with open(file, 'rb') as f:
        my_dict_final.update(new_dict)  # Update dict if it was empty
        # print('added first')
    else:
        if my_dict_final["fault_rate"] == new_dict["fault_rate"] and my_dict_final["injection_target"] == new_dict["injection_target"] \
                and my_dict_final["fault_type"] == new_dict["fault_type"] \
                and my_dict_final["protection"] == new_dict["protection"] and my_dict_final["layer_types"] == new_dict["layer_types"] \
                and my_dict_final["fixed_nr_faults"] == new_dict["fixed_nr_faults"] \
                and my_dict_final["nr_images"] == new_dict["nr_images"]: #batchsize may vary
            # and my_dict_final["nr_bits"] == new_dict["nr_bits"]\

            my_dict_final["Num_epochs"] = my_dict_final["Num_epochs"] + new_dict["Num_epochs"]

            my_dict_final["orig_ranger_act"]= my_dict_final["orig_ranger_act"] + new_dict["orig_ranger_act"]
            my_dict_final["fi_ranger_act"] = my_dict_final["fi_ranger_act"]+ new_dict["fi_ranger_act"]

            my_dict_final["orig_correct_pred_top1"] = my_dict_final["orig_correct_pred_top1"] + new_dict["orig_correct_pred_top1"]
            my_dict_final["fi_correct_pred_top1"] = my_dict_final["fi_correct_pred_top1"] + new_dict["fi_correct_pred_top1"]
            my_dict_final["orig_correct_pred_top3"] = my_dict_final["orig_correct_pred_top3"] + new_dict["orig_correct_pred_top3"]
            my_dict_final["fi_correct_pred_top3"] = my_dict_final["fi_correct_pred_top3"] + new_dict["fi_correct_pred_top3"]
            my_dict_final["orig_correct_pred_top5"] = my_dict_final["orig_correct_pred_top5"] + new_dict["orig_correct_pred_top5"]
            my_dict_final["fi_correct_pred_top5"] = my_dict_final["fi_correct_pred_top5"] + new_dict["fi_correct_pred_top5"]

            if not (new_dict["runset"] == [] or my_dict_final["runset"] == []):
                my_dict_final["runset"] = np.hstack((my_dict_final["runset"], new_dict["runset"])) #same runset was used! (or empty if too large)

            # print('added')
        else:
            print('WARNING2: folders not combined because of mismatch:')
            print(my_dict_final["fault_rate"], new_dict["fault_rate"])
            print(my_dict_final["injection_target"], new_dict["injection_target"])
            print(my_dict_final["nr_bits"], new_dict["nr_bits"])
            print(my_dict_final["fault_type"], new_dict["fault_type"])
            print(my_dict_final["protection"], new_dict["protection"])
            print(my_dict_final["layer_types"], new_dict["layer_types"])
            print(my_dict_final["fixed_nr_faults"], new_dict["fixed_nr_faults"])
            print(my_dict_final["Num_epochs"], new_dict["Num_epochs"])  # batchsize may vary
            print('dictionaries dont match, skip file.')

    return my_dict_final


def filter_by_true(orig_pred_flat, fi_pred_flat):
    """
    Takes the second argument and filters out all rows that were not true in the first argument.
    :param orig_pred_flat: original model predictions. All rows are the same!
    :param fi_pred_flat: fault predictions.
    :return: above lists but with only those columns that were true in original list.
    """

    # print('correct images in original:', np.sum(orig_pred_flat[0]), 'from', len(orig_pred_flat[0]))  # 748 got classified correctly

    indices_true = [i for i, e in enumerate(orig_pred_flat[0]) if e]
    orig_pred_flat_filtered = orig_pred_flat[:, indices_true]
    fi_pred_flat_filtered = fi_pred_flat[:, indices_true]  # filtered to those who were correct in the original model without faults

    return orig_pred_flat_filtered, fi_pred_flat_filtered


def get_acc_std_N(orig_pred_flat):
    """
    :param orig_pred_flat: matrix with rows epochs, columns images
    :return: acc, std, N_epochs
    """
    if len(np.array(orig_pred_flat).shape) > 1: #if more than one dim
        acc_epochs = np.mean(orig_pred_flat, 1)
    else:
        acc_epochs = orig_pred_flat
    # acc_epochs = np.mean(orig_pred_flat, 1)
    mean_epochs = np.mean(acc_epochs)
    std_epochs = np.std(acc_epochs)
    return mean_epochs, std_epochs, len(orig_pred_flat)


def get_sdc_std_N(orig_pred_flat):
    """
    :param orig_pred_flat: matrix with rows epochs, columns images
    :return: sdc rate, std, N_epochs
    """
    if len(np.array(orig_pred_flat).shape) > 1:  # if more than one dim
        acc_epochs = 1 - np.mean(orig_pred_flat, 1)
    else:
        acc_epochs = 1 - np.array(orig_pred_flat)
    # acc_epochs = 1 - np.mean(orig_pred_flat, 1)
    mean_epochs = np.mean(acc_epochs)
    std_epochs = np.std(acc_epochs)
    return mean_epochs, std_epochs, len(orig_pred_flat)



def analyse_folder(folder):
    """
    Folder structure below "folder": dir like ep100a/pickle_files.
    New dict topics acc_unfiltered, fi_acc_unfiltered, acc_filtered, fi_acc_filtered, sdc, fi_sdc are created.
    :param folder: folder to be analyzed with subfolders, e.g. w1 etc
    :return: dict_all with all data combined and evaluated
    """


    # Combine multiple runs if conditions are the same (fault rate, inj_target, layer_types, protection, fault_type, nr_bits, fault_rate...)
    dict_all = {}  # dict that collect all data in folder
    sub_dirs = list_subdir_in_dir(folder) #get all subdirs of folder
    if not sub_dirs:
        print('No subfolders found in this directory:', folder)
        return

    for subd in sub_dirs:
        files_to_load = list_files_in_dir(subd)
        # print('check', files_to_load)
        if files_to_load:
            dict_united = combine_images_sameEpochs(files_to_load) #bring together files in one subdict
            add_epochs_sameImages(dict_all, dict_united) #add result to dict_all
        else:
            print('no files in this subfolder:', subd)
            continue

    # Classification predictions --------------------------------------

    # orig_ranger = dict_all["orig_ranger_act"]
    # orig_ranger_flat = np.array([flatten(i) for i in orig_ranger]) #matrix with row runs, col image class
    # fi_ranger = dict_all["fi_ranger_act"]
    # fi_ranger_flat = np.array([flatten(i) for i in fi_ranger]) #matrix with row runs, col image class

    def extract_predictions(dict_all, topic, fi_topic):

        orig_pred = dict_all[topic]
        orig_pred_flat = np.array([flatten(i) for i in orig_pred])  # matrix with row runs, col image class
        fi_pred = dict_all[fi_topic]
        fi_pred_flat = np.array([flatten(i) for i in fi_pred])  # matrix with row runs, col image class

        mean_epochs, std_epochs, N_epochs = get_acc_std_N(orig_pred_flat)
        fi_mean_epochs, fi_std_epochs, _ = get_acc_std_N(fi_pred_flat)

        newtopic = "acc_unfiltered" + topic[-5:]
        fi_newtopic = "fi_acc_unfiltered" + fi_topic[-5:]
        dict_all[newtopic] = [mean_epochs, std_epochs]  # add to dict
        dict_all[fi_newtopic] = [fi_mean_epochs, fi_std_epochs]

        orig_pred_flat_filt, fi_pred_flat_filt = filter_by_true(orig_pred_flat, fi_pred_flat)

        # Check accuracy filtered:
        mean_epochs, std_epochs, _ = get_acc_std_N(orig_pred_flat_filt)
        fi_mean_epochs, fi_std_epochs, _ = get_acc_std_N(fi_pred_flat_filt)
        # print('filtered')
        # print('acc', 'mean', mean_epochs, 'std', std_epochs)
        # print('fi_acc', 'mean', fi_mean_epochs, 'std', fi_std_epochs)

        newtopic = "acc_filtered" + topic[-5:]
        fi_newtopic = "fi_acc_filtered" + fi_topic[-5:]
        dict_all[newtopic] = [mean_epochs, std_epochs]  # add to dict
        dict_all[fi_newtopic] = [fi_mean_epochs, fi_std_epochs]

        # Get SDC rates:
        sdc_epochs, sdc_std_epochs, _ = get_sdc_std_N(orig_pred_flat_filt)
        fi_sdc_epochs, fi_sdc_std_epochs, _ = get_sdc_std_N(fi_pred_flat_filt)
        # print('sdc', 'mean', sdc_epochs, 'std', sdc_std_epochs)
        # print('fi_sdc', 'mean', fi_sdc_epochs, 'std', fi_sdc_std_epochs)

        newtopic = "sdc" + topic[-5:]
        fi_newtopic = "fi_sdc" + fi_topic[-5:]
        dict_all[newtopic] = [sdc_epochs, sdc_std_epochs]  # add to dict
        dict_all[fi_newtopic] = [fi_sdc_epochs, fi_sdc_std_epochs]

    extract_predictions(dict_all, "orig_correct_pred_top1", "fi_correct_pred_top1")
    extract_predictions(dict_all, "orig_correct_pred_top3", "fi_correct_pred_top3")
    extract_predictions(dict_all, "orig_correct_pred_top5", "fi_correct_pred_top5")

    # Times
    times = dict_all["fi_time_list"]
    times_flat = np.array([flatten(i) for i in times])  # matrix with row runs, col image class
    mean_epochs, std_epochs, _ = get_acc_std_N(times_flat)
    btch_nr = dict_all["batch_size"]
    newtopic = "inf_time_perImage"
    dict_all[newtopic] = [mean_epochs/btch_nr, std_epochs/btch_nr]  # add to dict

    # RMSE
    errors = dict_all["rmse_list"]
    errors_flat = np.array([flatten(i) for i in errors])  # matrix with row runs, col image class
    mean_epochs, std_epochs, _ = get_acc_std_N(errors_flat)
    newtopic = "rmse"
    dict_all[newtopic] = [mean_epochs, std_epochs]  # add to dict

    return dict_all


def get_selection(folder_all, selection):
    """
    :param folder_all: Name of scenario folder, e.g. 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/scenario1_weights_top1/'
    :param selection: Name of selection folder, e.g. 'Ranger'
    :return: dict with all sorted data.
    """
    results_sel = []
    folders_sel = list_direct_subdir_in_dir(folder_all + selection)
    for fl in folders_sel:
        dict_alln = analyse_folder(fl)

        if dict_alln:
            results_sel.append(dict_alln)

    # print(results_sel)
    results_sel = sort_by_fr(results_sel) #order by fault rate just in case

    return results_sel


def sort_by_fr(results):
    """
    Brings list of dicts in another order such that they are sorted by increasing fault rate.
    :param results: dict
    :return: dict list sorted
    """

    fr_list = [results[i]["fixed_nr_faults"] for i in range(len(results))]
    # print(fr_list)
    results_2 = [x for y, x in sorted(zip(fr_list, results))]

    return results_2




def get_plotdata_bytopic(results, x_topic, y_topic, zeta):
    x_data = [i[x_topic] for i in results]
    num_ep = [i["Num_epochs"] for i in results]

    y_data = [i[y_topic][0] for i in results]
    y_error = [i[y_topic][1] for i in results]
    # Adjust error:
    y_error = np.array(y_error) * zeta / np.sqrt(np.array(num_ep)) #error measure here

    return x_data, y_data, y_error


def get_fault_nofault_combined(results_ranger, x_topic, y_topic, zeta):
    x_data0, y_data0, y_error0 = get_plotdata_bytopic(results_ranger, x_topic, y_topic[0], zeta)
    x_data, y_data, y_error = get_plotdata_bytopic(results_ranger, x_topic, y_topic[1], zeta)
    x_data = flatten([0, x_data])
    y_data = flatten([y_data0[0], y_data])
    y_error = flatten([y_error0[0], y_error])
    return x_data, y_data, y_error



def get_ranger_act_images(dict_used, y_topic):
    """
    :param dict_used: one dictionary of results
    :param y_topic: name of topic (ranger_act)
    :return: m_act: average portion of activated images (wrt to whole dataset) by 1 or n fixed faults.
    :return: std_act std error of m_act.
    """

    act_ranger = dict_used[y_topic] #original list: runs, batches, images
    act_ranger_flat = np.array([flatten(i) for i in act_ranger])  # eliminate batches: matrix with row runs, col image class
    act_ranger_flat_maxone = np.array([np.sign(k) for k in act_ranger_flat]) #truncate all acts greater one to one
    portion_images_act = np.array([np.sum(i) for i in act_ranger_flat_maxone])/np.shape(act_ranger_flat_maxone)[1] #fraction of images in the dataset (1000) that gets activated by the given faults
    # print(act_ranger_flat_maxone)
    # print(portion_images_act) #portion of activated images for that fault
    # nr_images_act = np.sum(act_ranger_flat_maxone) #nr of images with at least one ranger activation in all epochs with the given faultrate

    m_act = np.mean(portion_images_act) #mean of portion of images activated by 1 or given n faults
    std_act = np.std(portion_images_act)
    return m_act, std_act



def plotDistribution(dict_type, pic_folder, save_all, title_name, plot_topic, x_topic):
    """
    Plots the content of plot_topic as a histogram against the topic x_topic.
    """

    fig, axs = plt.subplots(ncols=len(dict_type), sharey=True)
    fig.suptitle(title_name, fontsize=16)
    axs = flatten([axs])

    for nrax in range(len(axs)):
        fi_pred = dict_type[nrax][plot_topic]
        fi_pred_flat = np.array([flatten(i) for i in fi_pred])  # matrix with row runs, col image class
        acc_epochs = np.mean(fi_pred_flat, 1)

        nr_faults = dict_type[nrax][x_topic]
        axs[nrax].hist(acc_epochs)

        axs[nrax].set_title('Faults: ' + str(nr_faults))
        axs[nrax].set_xlabel('accuracy')
        axs[nrax].set_ylabel('frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Save
    if save_all:
        pic_name = pic_folder + "acc_dist_" + title_name + ".png"
        fig.savefig(pic_name, dpi=150)



def get_ranger_coverage(dict_used, x_topic, y_topic, zeta):
    x_data = [i[x_topic] for i in dict_used]
    y_data = []
    y_error = []
    for j in range(len(dict_used)):
        m_act, std_act = get_ranger_act_images(dict_used[j], y_topic[1])
        y_data.append(m_act)
        y_error.append(std_act * zeta / np.sqrt(dict_used[j]["Num_epochs"]))

    # Add zero faults:
    m_act0, std_act0 = get_ranger_act_images(dict_used[0], y_topic[
        0])  # just from first example in dict_used, they are all the same (no fault)
    x_data = flatten([0, x_data])
    y_data = flatten([m_act0, y_data])
    y_error = flatten([std_act0 * zeta / np.sqrt(dict_used[0]["Num_epochs"]), y_error])

    return x_data, y_data, y_error



def plot_accuracy(results_ranger, x_topic, y_topic, zeta, theo_errors, axs, leg, name_str):
    # simplified plotting of accuracy values

    if results_ranger is not None:
        x_data, y_data, y_error = get_fault_nofault_combined(results_ranger, x_topic, y_topic, zeta)

        if theo_errors:  # add theoretical error
            y_error_theo = get_y_error_theo(results_ranger, zeta)  # theoretical worst-case error
            axs.errorbar(x_data, y_data, y_error_theo, capsize=5)  # ls = ''
            # eb[-1][0].set_linestyle('--')
        else:
            axs.errorbar(x_data, y_data, y_error, capsize=5)

        leg.append(name_str)
        print('acc ' + name_str, x_data, y_data)




def analyse_miscl(fi_ranger_act, fi_prediction, ranger_act, prediction):
    """
    :param fi_ranger_act: ranger act to be analysed (e.g. with faults)
    :param fi_prediction: predictions to be analysed (e.g. with faults)
    :param ranger_act:
    :param prediction:
    :return: lists of all epochs: oob, oob and misclassified, in bounds, in bounds and miscl,
    version that are filtered by only those images that are correctly classified in the control set (e.g. without faults)
    """

    # Clarify dimensions:
    eps = len(ranger_act)
    btchs = len(ranger_act[0])
    ims = len(ranger_act[0][0])
    nr_imgs = btchs*ims #1000 images here sum

    # Initialize
    oob_eps = []
    oob_mcl_eps = []
    ib_eps = []
    ib_mcl_eps = []
    #
    oob_eps_filt = [] #filtered, e.g. only those were original pred was correct
    oob_mcl_eps_filt = []
    ib_eps_filt = []
    ib_mcl_eps_filt = []


    for epoch in range(eps):

        oob_count = 0
        ib_count = 0
        oob_mcl_count = 0
        ib_mcl_count = 0
        #
        oob_count_filt = 0
        ib_count_filt = 0
        oob_mcl_count_filt = 0
        ib_mcl_count_filt = 0


        ep = ranger_act[epoch]
        pred = prediction[epoch]
        fi_ep = fi_ranger_act[epoch]
        fi_pred = fi_prediction[epoch]
        nr_imgs_filt = np.sum(pred)

        for btch in range(btchs):
            for im in range(ims):

                act_fi = fi_ep[btch][im] #0 or 1 if activated
                # print(act_fi)
                pr_fi = fi_pred[btch][im] #true/false if prediction is correct or not
                # print(pr_fi)
                act_orig = ep[btch][im]  # 0 or 1 if activated, for control
                pr_orig = pred[btch][im]  # true/false if prediction is correct or not, for control

                if act_fi > 0: #activated with fault
                    oob_count += 1
                    if not pr_fi:
                        oob_mcl_count +=1
                else:
                    ib_count += 1
                    if not pr_fi:
                        ib_mcl_count += 1

                # if original control pic (without faults) is correctly classified, add as well here
                if pr_orig:
                    if act_fi > 1:  # activated with fault
                        oob_count_filt += 1
                        if not pr_fi:
                            oob_mcl_count_filt += 1
                    else:
                        ib_count_filt += 1
                        if not pr_fi:
                            ib_mcl_count_filt += 1


        oob_eps.append(oob_count/nr_imgs)
        oob_mcl_eps.append(oob_mcl_count/nr_imgs)
        ib_eps.append(ib_count/nr_imgs)
        ib_mcl_eps.append(ib_mcl_count/nr_imgs)
        #
        oob_eps_filt.append(oob_count_filt / nr_imgs_filt)
        oob_mcl_eps_filt.append(oob_mcl_count_filt / nr_imgs_filt)
        ib_eps_filt.append(ib_count_filt / nr_imgs_filt)
        ib_mcl_eps_filt.append(ib_mcl_count_filt / nr_imgs_filt)

        # print('oob', oob_count, oob_count_filt)
        # print(nr_imgs_filt, nr_imgs, btchs, ims)

    return oob_eps, oob_mcl_eps, ib_eps, ib_mcl_eps, oob_eps_filt, oob_mcl_eps_filt, ib_eps_filt, ib_mcl_eps_filt



def get_cond_p(oob_eps, oob_mcl_eps, zeta):
    """Calculate mean and errors of the oob lists etc.
    return: probability oob, probability oob and mcl, conditional prob mcl given that oob
    """

    m_oob, std, n = get_acc_std_N(oob_eps)
    err_oob = std*zeta/np.sqrt(n)
    # print('chance of oob', m, err)

    m_oob_mcl, std, n = get_acc_std_N(oob_mcl_eps)
    err_oob_mcl = std*zeta/np.sqrt(n)
    # print('chance of oob and mcl', m, err)

    if m_oob > 0.:
        m_mcl_given_oob = m_oob_mcl/m_oob
        err_oob_given_mcl = np.sqrt((1/m_oob)**2 * err_oob_mcl**2 + (m_oob_mcl/m_oob**2)**2 * err_oob**2)
    else:
        m_mcl_given_oob = 0.
        err_oob_given_mcl = 0.
    # print(m_mcl_given_oob, err_oob_given_mcl)

    # return m_mcl_given_oob, err_oob_given_mcl
    return m_oob, m_oob_mcl, m_mcl_given_oob








## Currently not used:
# def get_ranger_counts(dict_all_topic):
#     """
#     Checks how many images and ranger layers got activated in total.
#     :param dict_all_topic: topic with ranger data
#     :return: total number of activated ranger layers
#     :return: total number of images with any ranger activation
#     :return: largest number of ranger activations found per image
#     """
#     orig_ranger_act_nonzeroflat = [i for i in flatten(dict_all_topic) if i > 0]
#     tot_act_ranger = np.sum(orig_ranger_act_nonzeroflat)
#     im_act_ranger = len(orig_ranger_act_nonzeroflat)
#     max_per_im = np.max(orig_ranger_act_nonzeroflat)
#
#     return tot_act_ranger, im_act_ranger, max_per_im






