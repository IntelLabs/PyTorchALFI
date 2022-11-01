# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import logging.config
import os
import pickle
import sys
from enum import Enum
import random
import numpy as np
import torch
import yaml
import collections
from torchsummary import summary
from pytorchfi.pytorchfi.errormodels import \
    single_bit_flip_func as fault_injector
from alficore.ptfiwrap_utils.pyfihelpers import get_number_of_neurons, \
    get_number_of_weights, random_batch_element, random_layer_element, \
    random_neuron_location, random_weight_location, random_value, \
    random_layer_weighted
from ..parser.config_parser import ConfigParser


logging.config.fileConfig(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../fi.conf')))
log = logging.getLogger()


class rnd_types(Enum):
    """Enumeration to avoid mistakes when using fault injection type (weight or neuron)
    as function parameters.

    Args:
        Enum (enum.Enum): inherits from Enum
    """    
    neurons = 1
    weights = 2


# class batchIter: 
#     def __init__(self, batchsize):
#         self.batchsize = batchsize
#         self.batches = np.arange(self.batchsize)
#         self.index = 0

#     def getNext(self):
#         ret = self.batches[self.index]
#         self.index = (self.index + 1) % self.batchsize
#         return ret


class ptfiwrap:
    """main wrapper class that controls the generation of faults and the initialization of 
    faulty models.
    """    
    def __init__(self, model, **kwargs):
        """Parses config file and initializes model to be corrupted.

        Args:
            model (pytorch model, optional): Trained model to be corrupted.
            
        """        
        self.runset = np.array([])
        self.value_type = ""
        self.rnd_value_type = ""
        self.CURRENT_FAULTS = {}
        self.CURRENT_NUM_FAULTS = -1
        self.input_num = kwargs.get("input_num", 1)
        #self.cuda_device = kwargs.get("cuda_device", 0)
        self.device = kwargs.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.config_location = kwargs.get("config_location", "default.yml")
        self.scenario_data = kwargs.get("scenario_data", None)
        self.create_runset = kwargs.get("create_runset", True)
        # scenario file needs to be read in any case
        # because it also contains information on pytorchfi parameters
        # and data loader
        if self.config_location:
            self.config_location = os.path.join('scenarios', self.config_location)
        else:
            self.config_location = "scenarios/default.yml"  # hardcode for now
        if not self.scenario_data:
            scenario = self.__load_scenario(self.config_location)
        else:
            scenario = self.scenario_data
        self.parser = ConfigParser(**scenario)

        self.fileDir = os.path.dirname(sys.argv[0])
        # check validity of entries in default.yml
        valid, reason = self.parser.isvalid(self.fileDir)
        if not valid:
            log.error("Scenario config not valid, exiting!")
            log.error("Reason:")
            for r in reason:
                log.error(r)
            sys.exit()

        
        self.net = model.to(self.device)
        if self.parser.print:
            self.print_model_details(self.net)


    def __post_init(self):
        self.dataset_size = self.parser.dataset_size
        self.num_runs = self.parser.num_runs  # number of epochs runs
        self.max_faults_per_image = self.parser.max_faults_per_image
        self.max_fault_rate = self.parser.max_fault_rate
        self.batch_size = self.parser.ptf_batch_size
        self.__create_pytorchfi()
        self.num_faults = self.__get_numfaults()

    def __prepare_runtime_faultgeneration(self):
        self.__create_pytorchfi()
        self.__prepare_runmodes()

    def __create_runset(self):

        # Option 1: actual FI values are read from a file generated
        # in a previous run
        if not self.parser.read_from_file == "-1":
            try:
                f = open(self.parser.read_from_file, 'rb')
                self.runset = pickle.load(f)
                self.runset_updated = self.__adjust_rs_inj_policy()
                f.close()
            except OSError as e:
                log.error("Unable to read from file {}: {}".format(
                    self.parser.read_from_file, e))
                sys.exit()

        # actual FI values are generated from scenario file
        ##############################################
        #            Scenario parsing                #
        ##############################################
        # if random value generation is configured, create a static set
        # of random values according to the number of runs
        else:
            # initiate PytorchFi model
            self.__prepare_runmodes()
            if self.parser.rnd_mode == "neurons":
                self.runset = self.__fill_values(
                    rnd_types.neurons, self.num_faults, self.modes,
                    self.pytorchfi)
            else:
                self.runset = self.__fill_values(
                    rnd_types.weights, self.num_faults, self.modes,
                    self.pytorchfi)
            self.runset_updated = self.__adjust_rs_inj_policy()
            # runset is saved to experiment folder by test_error_models
            #####################
            #    End parsing    #
            #####################

    def __adjust_rs_inj_policy(self):
        runset = self.runset
        if self.parser.rnd_mode == 'neurons':
            if self.parser.inj_policy == "per_epoch":
                _runset = np.tile(runset[:, 0:self.max_faults_per_image],(1, self.dataset_size))
                for i in range(1, self.num_runs):
                    tile_runset = np.tile(runset[:, i*self.max_faults_per_image:i*self.max_faults_per_image+self.max_faults_per_image],(1, self.dataset_size))
                    _runset = np.hstack([_runset, tile_runset])
                return _runset
            if self.parser.inj_policy == 'per_image':
                # return runset[:,:self.num_runs*self.dataloader.dataset_length*self.num_faults-1]
                return runset
            elif self.parser.inj_policy == 'per_batch':
                batches = int(np.ceil(self.num_faults*self.dataset_size/self.batch_size))
                _runset = np.tile(runset[:, 0:self.max_faults_per_image], (1, self.batch_size))
                for i in range(1, batches):
                    tile_runset = np.tile(runset[:, i*self.max_faults_per_image:i*self.max_faults_per_image+self.max_faults_per_image],(1, self.batch_size))
                    _runset = np.hstack([_runset, tile_runset])
                _runset = _runset[:,:self.num_runs*self.dataset_size]
                return _runset
        elif self.parser.rnd_mode == 'weights':
            if self.parser.inj_policy == 'per_batch':
                batches = int(np.ceil(self.num_runs*self.dataset_size/self.batch_size))
                _runset = np.tile(runset[:, 0:self.max_faults_per_image], (1, self.batch_size))
                curr_dataset_size = self.batch_size
                batch_size        = self.batch_size
                for i in range(1, batches):
                    if curr_dataset_size + batch_size  > self.dataset_size:
                        batch_size = self.dataset_size - (curr_dataset_size)
                        batch_size = batch_size if batch_size>0 else self.batch_size
                        curr_dataset_size = 0
                    else:
                        batch_size = self.batch_size
                        curr_dataset_size = curr_dataset_size + batch_size
                    tile_runset = np.tile(runset[:, i*self.max_faults_per_image:i*self.max_faults_per_image+self.max_faults_per_image],(1, batch_size))
                    _runset = np.hstack([_runset, tile_runset])
                return _runset
            # if self.parser.inj_policy == "per_epoch":
            elif self.parser.inj_policy == 'per_epoch':
                _runset = np.tile(runset[:, 0:self.max_faults_per_image], (1, self.dataset_size))
                for i in range(1, self.num_runs):
                    tile_runset = np.tile(runset[:, i*self.max_faults_per_image:i*self.max_faults_per_image+self.max_faults_per_image], (1, self.dataset_size))
                    _runset = np.hstack([_runset, tile_runset])
                return _runset
            # return runset

    def __prepare_runmodes(self):
        if self.parser.rnd_value_type == "number":
            modes = {"layer": self.parser.rnd_layer,
                     "location": self.parser.rnd_location,
                     "value": self.parser.rnd_value, "value_type":
                         self.parser.rnd_value_type,
                     "value_min": self.parser.rnd_value_min,
                     "value_max": self.parser.rnd_value_max}
        else:  # prepare for bitflip
            modes = {"layer": self.parser.rnd_layer, 
                     "location": self.parser.rnd_location, "value": self.parser.rnd_value,
                     "value_type": self.parser.rnd_value_type,
                     "value_bits": self.parser.rnd_value_bits,
                     "bit_range": self.parser.rnd_bit_range}

        if self.parser.rnd_mode == "neurons":
            modes["batch"] = self.parser.rnd_batch
            modes["batchsize"] = self.parser.ptf_batch_size
            # fill values for batches
        self.modes = modes

    def __create_pytorchfi(self):
        self.run_type = self.parser.run_type  # single or accumulated
        layer_types = self.parser.layer_types
        if "conv3d" in self.parser.layer_types:
            self.CONV3D = True
        else:
            self.CONV3D = False
        self.value_type = self.parser.rnd_mode # weights or neurons
        self.rnd_value_type = self.parser.rnd_value_type # bitflip

        # using the class single_bit_flip_func from pytorchfi errormodels.py
        # in order to access the bit flip functionality
        ptfiwrap_obj = self
        self.pytorchfi = fault_injector(
                model=self.net, model_attr=self.parser, ptfiwrap=ptfiwrap_obj, layer_types=layer_types, input_num=self.input_num)

    def __get_numfaults(self):
    # def __get_numfaults(self, pytorchfi, num_runs, max_faults_per_image,
    #                     dataset_size, max_fault_rate, rnd_mode):
        pytorchfi=self.pytorchfi
        num_runs=self.num_runs
        max_faults_per_image=self.max_faults_per_image
        dataset_size=self.dataset_size
        max_fault_rate=self.max_fault_rate
        rnd_mode=self.parser.rnd_mode
        inj_policy = self.parser.inj_policy
        batches = self.parser.ptf_batch_size
        num_faults = 0
        dim = 0
        print()
        if max_faults_per_image >= 1 and max_fault_rate == -1.0:
        ##TODO: adapt the following code using injection policy "per_batch"
            if rnd_mode == "neurons":
                if inj_policy == "per_image":
                    num_faults = num_runs * max_faults_per_image \
                                * dataset_size
                elif inj_policy == "per_epoch":
                    num_faults = num_runs * max_faults_per_image
                elif inj_policy == "per_batch":
                    num_faults = int(np.ceil(num_runs * max_faults_per_image \
                                * dataset_size/batches))
            elif rnd_mode == "weights":
                if inj_policy == "per_image":
                    num_faults = num_runs * max_faults_per_image * dataset_size
                elif inj_policy == "per_epoch":
                    num_faults = num_runs * max_faults_per_image
                elif inj_policy == "per_batch":
                    num_faults = int(np.ceil(num_runs * max_faults_per_image \
                                * dataset_size/batches))
        elif max_fault_rate >= 0 and max_faults_per_image == -1:
            ## TODO: Not sure why ,ax_fault_rate is >=0 because max_fault_rate >=1 already exists
            if rnd_mode == "neurons":
                dim = get_number_of_neurons(pytorchfi)
                num_faults = num_runs * max_fault_rate * dim \
                             * dataset_size / batches
            elif rnd_mode == "weights":
                dim = get_number_of_weights(pytorchfi)
                num_faults = num_runs * max_fault_rate * dim * dataset_size
        else:
            log.error("setting of max_faults_per_image and max_fault_rate is \
                inconsistent. Please only set one.\n"
                      "exiting!")
            sys.exit()
        if num_faults < 1:
            num_faults = 1
        return int(num_faults)

    def __get_data_loader(self):
        return self.loader

    def __load_scenario(self, conf_location):
        """
        Load content of scenario file (yaml)
        :param conf_location: relative path to scenario configuration,
        default is scenarios/default.yml
        :return: dict from yaml file
        """
        try:
            fileDir = os.path.dirname(sys.argv[0])
            # fileDir = os.path.dirname(os.path.realpath(__file__))
            # Obtain the current working directory
            config_file = os.path.join(fileDir, conf_location)
            document = open(config_file)
            scenario = yaml.safe_load(document)
            return scenario

        except OSError:
            try: #try in parent folder
                from pathlib import Path
                fileDir = Path().resolve()
                config_file = os.path.join(fileDir, conf_location)
                document = open(config_file)
                scenario = yaml.safe_load(document)
                return scenario
            except:
                log.error("Unable to open file {}, exiting!".format(conf_location))
                sys.exit()
        except yaml.YAMLError:
            log.error("Error reading yaml file {}".format(conf_location))
            sys.exit()

    def get_single_fault(self, fi_type, value_type, pfi_model, **kwargs):
        """[generates single fault with position and value depending in
        fi_type and value_type]

        :param fi_type: [sets either neurons or weights fault injection]
        :type fi_type: [Enum rnd_types]
        :param value_type: [type of value to be injected,
        either 'number' or 'bitflip']
        :type value_type: [String]
        :param pfi_model: [pytorchfi faulty model object]
        :type pfi_model: [pytorchfi.pytorchfi.errormodels.single_bit_flip_func]
        :return: [Dictionary with parameters for fault injection depending
        on fi_type and value_type.
        Keys are: batch, layer, rnd_type{neurons|weights}, k, c, d, h, w,
        value_type{number|bitflip}, value]
        :rtype: [dict]
        """
        fix_batch = kwargs.get("batch_num", -1)
        fix_layer = kwargs.get("layer_num", -1)
        fix_channel = kwargs.get("channel_num", -1)
        fix_value = kwargs.get("value_num", -1)
        min_value = kwargs.get("value_min", -1)
        max_value = kwargs.get("value_max", -1)
        value_bits = kwargs.get("value_bits", -1)
        bit_range = kwargs.get("bit_range", [])
        fault = {}
        if fix_layer > -1:
            fault["layer"] = fix_layer
        else:
            if self.parser.rnd_layer_weighted:
                if self.parser.rnd_mode == "weights":
                    fault["layer"] = random_layer_weighted(pfi_model, "weights")
                elif self.parser.rnd_mode == "neurons":
                    fault["layer"] = random_layer_weighted(pfi_model, "neurons")
            else:
                fault["layer"] = random_layer_element(pfi_model)
        if fi_type == rnd_types.neurons:
            fault["rnd_type"] = "neurons"
            if fix_batch > -1:
                fault["batch"] = fix_batch
            else:
                fault["batch"] = random_batch_element(pfi_model)
            (c, d, h, w) = random_neuron_location(pfi_model, fault["layer"])
            if fix_channel > -1:
                max_channels = pfi_model.get_fmaps_num(fault["layer"])
                if fix_channel < max_channels:
                    c = fix_channel
                else:
                    log.error("requested channel larger than max channels for "
                              "layer, using random value")

        if fi_type == rnd_types.weights:
            fault["rnd_type"] = "weights"
            index = random_weight_location(pfi_model, fault["layer"])
            c = -1
            k = -1
            d = -1
            h = index[-2]
            w = index[-1]
            if len(index) >= 5:  # conv2d or conv3d layer
                k = index[1]
                if fix_channel > -1:
                    c = fix_channel
                else:
                    c = index[2]
                if len(index) == 6:
                    d = index[3]
            fault["k"] = k  # only relevant for weights
        fault["c"] = c
        fault["d"] = d
        fault["h"] = h
        fault["w"] = w

        if value_type == "number":
            fault["value_type"] = value_type
            if fix_value > -1:
                fault["value"] = fix_value
            elif min_value > -1 and max_value > -1:
                fault["value"] = random_value(min_value, max_value)
            else:
                log.error("No value or value range given for number!")
                return
        elif value_type in ["bitflip", "stuckat_0", "stuckat_1"] :
            fault["value_type"] = value_type
            if bit_range:
                if len(bit_range) == 1:
                    fault["value"] = bit_range[0]
                elif len(bit_range) == 2:
                    fault["value"] = random.randint(bit_range[0], bit_range[1])
                else:  # other values make no sense here
                    pass
            elif value_bits > -1:
                fault["value"] = random.randint(0, value_bits - 1)
            else:
                log.error("No max bits given for {}!".format(value_type))
                return
        ## TODO: Activate bitflip_bounds feature
        # elif value_type in ["bitflip_bounds", "bitflip_weighted"]:
        else:
            fault["value"] = -1
        return fault

    def __tile_batch(self, batchsize, total):
        sequence = np.arange(batchsize)
        reps = -(-total // batchsize)
        tiles = np.tile(sequence, reps)
        res = np.zeros(total, dtype=int)
        res[:] = tiles[:total]
        return res

    def __fill_values(self, fi_type, runs, modes, pfi_model):
        """
        Generated the set of faults to be injected base on the random section
        of the scenario definition file.
        :param fi_type: content of rnd_mode in scenario definition file
        {neurons|weights}
        :param runs: number of FI runs
        :param modes: a portion from the scenario file relevant for
        random content
        :param pfi_model: fault inject model from PytorchFi
        :return: runset, a numpy.array representation of the intended faults
        to be injected
        """
        # meaning of rows is batches, layer, location, value
        layernum = 0
        # runset is a numpy matrix with one fault per column
        # there are always 7 lines form which some are ignored for specific
        # layer types
        # The meaning is different for neuron injection and weight injection.
        # Neurons use batch size while weights don't
        # but instead have an additional dimension K

        # --- Meaning for NEURON injection: --- #
        # 1. batchnumber (used in: conv2d,conv3d)
        # 2. layer (everywhere)
        # 3. channel (used in: conv2d,conv3d)
        # 4. depth (used in: conv3d)
        # 5. height (everywhere)
        # 6. width (everywhere)
        # 7. value (everywhere)

        # --- Meaning for WEIGHT injection: --- #
        # 1. layer (everywhere)
        # 2. Kth filter (everywhere)
        # 3. channel(used in: conv2d, conv3d)
        # 4. depth (used in: conv3d)
        # 5. height (everywhere)
        # 6. width (everywhere)
        # 7. value (everywhere)

        runset = np.full((7, runs), -1)  # values that are not explicitely set
        # remain -1 and are later ignored during injection

        ####################################################
        #   Fetch first fault for first column of runset   #
        ####################################################

        kwargs = {}
        batch_mode = ""
        layer_mode = modes["layer"]
        location_mode = modes["location"]
        value_mode = modes["value"]
        if fi_type == rnd_types.neurons:
            batch_mode = modes["batch"]
            # the only difference is that for neurons we can specify batches
            # and for weights not
            batchsize = modes["batchsize"]
            if batch_mode == "each":
                kwargs["batch_num"] = 0
            elif batch_mode == "same":
                batchnum = random_batch_element(pfi_model)
                kwargs["batch_num"] = batchnum

        value_type = modes["value_type"]
        if value_type == "number":
            kwargs["value_min"] = modes["value_min"]
            kwargs["value_max"] = modes["value_max"]
        else:
            kwargs["value_bits"] = modes["value_bits"]
            kwargs["bit_range"] = modes["bit_range"]

        # layers

        if self.__isnumber(layer_mode):  # fix a layer through config file
            try:
                layernum = int(layer_mode)
                max_layers = pfi_model.get_total_conv()
                if (layernum + 1) > max_layers:
                    layernum = max_layers - 1
            except Exception as e:
                layernum = 0
                log.debug(e)
            kwargs["layer_num"] = layernum

        # get first column of runset
        fault = self.get_single_fault(fi_type, modes["value_type"],
                                      pfi_model, **kwargs)

        # fault Keys are: batch, layer, rnd_type{neurons|weights}, k, c, d, h,
        # w, value_type{number|bitflip}, value]
        if fi_type == rnd_types.neurons:
            runset[0, 0] = fault["batch"]
            runset[1, 0] = fault["layer"]
        else:
            runset[0, 0] = fault["layer"]
            runset[1, 0] = fault["k"]
        runset[2, 0] = fault["c"]
        runset[3, 0] = fault["d"]
        runset[4, 0] = fault["h"]
        runset[5, 0] = fault["w"]
        runset[6, 0] = fault["value"]

        # keep first columnt of fault
        first_fault = fault

        ####################################################
        #   fill rest of runset based on modes for         #
        #   batches, layers, location and values           #
        #   (each, same, change)                           #
        ####################################################

        if batch_mode == "same" and layer_mode == "same" and \
                location_mode == "same" and value_mode == "same":
            runset = np.tile(runset[:, 0], runs)
            return runset

        # batch_mode=each iterates through the batch numbers but keeps
        # the rest of the fault identical
        if fi_type == rnd_types.neurons and \
                batch_mode == "each" and batchsize > 1:
            batch_fault = fault
            for i in np.arange(runs // batchsize):
                for j in np.arange(batchsize):
                    ind = i * batchsize + j
                    runset[0, ind] = j
                    runset[1, ind] = batch_fault["layer"]
                    runset[2, ind] = batch_fault["c"]
                    runset[3, ind] = batch_fault["d"]
                    runset[4, ind] = batch_fault["h"]
                    runset[5, ind] = batch_fault["w"]
                    runset[6, ind] = batch_fault["value"]
                if layer_mode == "same":
                    kwargs["layer_num"] = first_fault["layer"]
                batch_fault = self.get_single_fault(
                    fi_type, modes["value_type"], pfi_model, **kwargs)
                if location_mode == "same":
                    batch_fault["d"] = first_fault["d"]
                    batch_fault["h"] = first_fault["h"]
                    batch_fault["w"] = first_fault["w"]
                if value_mode == "same":
                    batch_fault["value"] = first_fault["value"]

        else:
            for i in np.arange(1, runs):
                if layer_mode == "same":
                    kwargs["layer_num"] = first_fault["layer"]
                fault = self.get_single_fault(
                    fi_type, modes["value_type"], pfi_model, **kwargs)
                if location_mode == "same":
                    fault["d"] = first_fault["d"]
                    fault["h"] = first_fault["h"]
                    fault["w"] = first_fault["w"]
                if value_mode == "same":
                    fault["value"] = first_fault["value"]
                if fi_type == rnd_types.neurons:
                    runset[0, i] = fault["batch"]
                    runset[1, i] = fault["layer"]
                else:
                    runset[0, i] = fault["layer"]
                    runset[1, i] = fault["k"]
                runset[2, i] = fault["c"]
                runset[3, i] = fault["d"]
                runset[4, i] = fault["h"]
                runset[5, i] = fault["w"]
                runset[6, i] = fault["value"]
            
            if fi_type == rnd_types.neurons and \
                batch_mode == "change" and batchsize > 1:
                # tile batches according to batchsize
                # to equally distribute faults over batch
                runset[0] = self.__tile_batch(batchsize, runs)

        # Debug distribution of weighted layers
        if fi_type == rnd_types.neurons:
            layer_weights = self.pytorchfi.get_layer_neuron_weights()
            freq = collections.Counter(np.sort(runset[1]))
            # print("printing Neuron based layer weights")
        else:
            layer_weights = self.pytorchfi.get_layer_weight_weights()
            freq = collections.Counter(np.sort(runset[0]))
            # print("printing Weight based layer weights")
        # print("layer emphasis: {}".format(list(layer_weights)))
        # print("Map of layer number to actual number of faults")
        # for (key, value) in freq.items():
        #     print(key, " -> ", value)
        return runset


    def __tile_epoch(self, epoch, batchsize, total):
        sequence = np.arange(batchsize)
        reps = -(-total // batchsize)
        tiles = np.tile(sequence, reps)
        res = np.zeros(total, dtype=int)
        res[:] = tiles[:total]
        return res

    def __prepare_dumpfile(self, basedir, subdir, timestamp, basename, ext):
        """
        Creates full path string for file to store the FI values as binary
        :param basedir: Absolute root directory
        :param subdir: Relative directory under basedir
        :param timestamp: String representing a timestamp
        :param basename: Core portion of file name to which the timestring and
        an extension is attached.
        :param ext: String for extension
        :return: Absolute complete directory + file name to store FI values
        """
        dir = os.path.join(basedir, subdir)
        file = basename + "_" + timestamp + "." + ext
        try:
            if not os.path.exists(dir):
                os.mkdir(dir)
            return os.path.join(dir, file)
        except OSError:
            return False

    def __isnumber(self, value):
        try:
            int(value)
            return True
        except Exception as e1:
            log.debug(e1)
            try:
                float(value)
                return True
            except Exception as e2:
                log.debug(e2)
                return False

    def print_model_details(self, model):
        """
        print details about a loaded trained pytorch model
        :param model: The model where faults are to be injected
        """
        # Print model's structure
        print("Model's structure:")
        if self.parser.ptf_D != -1:
            input_size = (
                self.parser.ptf_C, self.parser.ptf_D, self.parser.ptf_H,
                self.parser.ptf_W)
        else:
            input_size = (
                self.parser.ptf_C, self.parser.ptf_H, self.parser.ptf_W)
        summary(model, input_size=input_size,
                batch_size=self.parser.ptf_batch_size)
        # print(model)

    def get_value_type(self):
        return self.value_type

    def get_runs(self):
        return self.num_runs

    def get_faults(self):
        return self.num_faults

    def get_runset(self):
        return self.runset

    def get_scenario(self):
        return self.parser.__dict__

    def set_scenario(self, scenario, create_runset=None):
        if create_runset is None:
            create_runset = self.create_runset
        self.parser = ConfigParser(**scenario)
        # valid, reason = self.parser.isvalid(os.path.dirname(sys.argv[0])) # needed when overriding wrapper with num_runs
        valid, reason = self.parser.isvalid()
        if not valid:
            log.error("Scenario config not valid, exiting!")
            log.error("Reason:")
            for r in reason:
                log.error(r)
            sys.exit()
        self.__post_init()
        if create_runset:
            self.__create_runset()

    def get_fimodel_iter(self, num_faults_per_image=None,
                         fault_rate=None, resume_pointer=None):
        """start an iteration through the runset of prepared faults

        :param num_faults_per_image: how many faults each image should receive,
        defaults to None
        :type num_faults_per_image: int, optional
        :param fault_rate: fault rate related to either size of weights or
        number of neurons depending on test type, defaults to None
        :type fault_rate: float, optional
        :yield: yield can be used to receive the next number of faults
        depending on batch size
        :rtype: pytorchfi model with faults applied
        """
        rs = self.runset_updated if self.value_type == "neurons" else self.runset
        self.num_faults_forall_images = num_faults_per_image
        self.fault_rate = fault_rate
        self.CURRENT_FAULTS = {}
        n, batchsize = self.__get_num_faults_for_run()
        if n == 1:
            self.CURRENT_NUM_FAULTS = 1
            if not resume_pointer:
                resume_pointer = 0
            for i in range(resume_pointer, self.num_faults):
                c = int(rs[2, i])
                d = int(rs[3, i])
                h = int(rs[4, i])
                w = int(rs[5, i])
                value = rs[6, i]

                if self.value_type == "neurons":
                    # here we have batchsize=1
                    # set batch = 0
                    #batch = int(rs[0, i])
                    batch = 0
                    conv_num = int(rs[1, i])
                    inj_model = self.__get_inj_model(
                        self.value_type, c, d, h, w, value, conv_num=conv_num,
                        batch=batch)
                    self.__set_current_faults(
                        self.value_type, c, d, h, w, value, conv_num=conv_num,
                        batch=batch)
                else:
                    conv_num = int(rs[0, i])
                    k = int(rs[1, i])
                    inj_model = self.__get_inj_model(
                        self.value_type, c, d, h, w, value, conv_num=conv_num,
                        k=k)
                    self.__set_current_faults(
                        self.value_type, c, d, h, w, value, conv_num=conv_num,
                        k=k)
                    self.CURRENT_FAULTS["k"] = k
                    self.CURRENT_FAULTS["batch"] = -1
                self.CURRENT_FAULTS["vtype"] = self.value_type
                yield inj_model
        else:
            loop = True
            while loop:
                # print('in the loop...')
                if not self.fault_rate and self.num_faults_forall_images > 1:
                    faults_per_run = self.num_faults_forall_images
                else:
                    x = np.random.binomial(n=n, p=self.fault_rate, size=1)
                    faults_per_run = x[0]
                self.CURRENT_NUM_FAULTS = faults_per_run
                if faults_per_run == 0:
                    log.warning("no faults are injected."
                                " Is fault_rate too low?"
                                " Returning original model.")
                    yield self.pytorchfi.get_original_model()
                else:
                    # log.info("Number of faults for this batch group: {}".
                    # format(faults_per_run))
                    # log.info("Number of batches: {} and faults per image: {}"
                    # .format(batchsize, int(faults_per_run/batchsize)))
                    # log.info("#############################################\n")
                    i = self.pytorchfi.ptfi_batch_pointer
                    c = [int(x) for x in rs[2, i:i + faults_per_run]]
                    d = [int(x) for x in rs[3, i:i + faults_per_run]]
                    h = [int(x) for x in rs[4, i:i + faults_per_run]]
                    w = [int(x) for x in rs[5, i:i + faults_per_run]]
                    value = rs[6, i:i + faults_per_run]
                    if self.value_type == "neurons":
                        if batchsize > 1:  # spread faults equally over batches
                            batch = self.__tile_batch(self.pytorchfi.
                                                      _RUNTIME_BATCH_SIZE,
                                                      faults_per_run)
                            # write back modified batch indices to runset
                            rs[0, i:i + faults_per_run] = batch[:rs[0, i:i + faults_per_run].shape[0]]
                        else:
                            batch = np.zeros(faults_per_run, dtype=int)
                            # write back modified batch indices to runset
                            rs[0, i:i + faults_per_run] = batch
                        conv_num = [int(x) for x in rs[1, i:i + faults_per_run]]
                        inj_model = self.__get_inj_model(
                            self.value_type, c, d,
                            h, w, value, conv_num=conv_num, batch=batch)
                        self.__set_current_faults(
                            self.value_type, c, d, h, w,
                            value, conv_num=conv_num, batch=batch)
                    else:
                        conv_num = [int(x) for x in rs[0, i:i + faults_per_run]]
                        k = [int(x) for x in rs[1, i:i + faults_per_run]]
                        inj_model = self.__get_inj_model(
                            self.value_type, c, d, h, w, value,
                            conv_num=conv_num, k=k)
                        self.__set_current_faults(
                            self.value_type, c, d, h, w, value,
                            conv_num=conv_num, k=k)
                    # i += faults_per_run
                    # log.info("applied faults: \n batch %s \n layer %s \n c %s \n h %s \n v %s \n value %s"
                    #     % (
                    #         self.CURRENT_FAULTS["batch"],
                    #         self.CURRENT_FAULTS["conv_num"],
                    #         self.CURRENT_FAULTS["c"],
                    #         self.CURRENT_FAULTS["h"],
                    #         self.CURRENT_FAULTS["w"],
                    #         self.CURRENT_FAULTS["value"],
                    #     )
                    # )
                    yield inj_model

    def __set_current_faults(
            self, vtype, c, d, h, w, value, conv_num=None, batch=None, k=None):
        if batch is not None:
            self.CURRENT_FAULTS["batch"] = batch
        elif k is not None:
            self.CURRENT_FAULTS["k"] = k
        if conv_num is not None:
            self.CURRENT_FAULTS["conv_num"] = conv_num
        self.CURRENT_FAULTS["vtype"] = vtype
        self.CURRENT_FAULTS["c"] = c
        self.CURRENT_FAULTS["d"] = d
        self.CURRENT_FAULTS["h"] = h
        self.CURRENT_FAULTS["w"] = w
        self.CURRENT_FAULTS["value"] = value

    def get_fault_for_image(self, i):
        """given a certain batchsize return the faults applied to image i
        in the current batch
        This function expects self.CURRENT_FAULTS to be set which
        contains the faults
        for the whole batch.

        :param i: position of image in batch
        :type i: int
        """
        # only for neuron fault injection are there different faults per batch item
        if self.CURRENT_FAULTS["vtype"] == "neurons" and self.batch_size > 1:
            fault_image = self.CURRENT_NUM_FAULTS // self.batch_size
            c = self.CURRENT_FAULTS["c"][i * fault_image:i * fault_image + fault_image]
            d = self.CURRENT_FAULTS["d"][i * fault_image:i * fault_image + fault_image]
            h = self.CURRENT_FAULTS["h"][i * fault_image:i * fault_image + fault_image]
            w = self.CURRENT_FAULTS["w"][i * fault_image:i * fault_image + fault_image]
            conv = self.CURRENT_FAULTS["conv_num"][
                   i * fault_image:i * fault_image + fault_image]
                #    [i * fault_image:i * fault_image + fault_image]
            value = self.CURRENT_FAULTS["value"][
                    i * fault_image:i * fault_image + fault_image]
            batch = self.CURRENT_FAULTS["batch"][
                    i * fault_image:i * fault_image + fault_image]
            return [batch, conv, c, d, h, w, value]
        elif self.CURRENT_FAULTS["vtype"] == "neurons" and self.batch_size == 1:
            c = self.CURRENT_FAULTS["c"]
            d = self.CURRENT_FAULTS["d"]
            h = self.CURRENT_FAULTS["h"]
            w = self.CURRENT_FAULTS["w"]
            conv = self.CURRENT_FAULTS["conv_num"]
            value = self.CURRENT_FAULTS["value"]
            batch = self.CURRENT_FAULTS["batch"]
            return [batch, conv, c, d, h, w, value]
        else:
            c = self.CURRENT_FAULTS["c"]
            d = self.CURRENT_FAULTS["d"]
            h = self.CURRENT_FAULTS["h"]
            w = self.CURRENT_FAULTS["w"]
            conv = self.CURRENT_FAULTS["conv_num"]
            value = self.CURRENT_FAULTS["value"]
            k = self.CURRENT_FAULTS["k"]
            return [conv, k, c, d, h, w, value]

    def __get_num_faults_for_run(self):
        batchsize = self.batch_size
        # rnd_value_type = self.rnd_value_type
        if (not self.num_faults_forall_images) and (not self.fault_rate):
            # print("both num_faults_per_image and fault rate are unset!")
            if not self.parser.max_fault_rate == -1:
                # print("using max_fault_rate: {}".format(
                #     self.parser.max_fault_rate))
                self.fault_rate = self.parser.max_fault_rate
            else:
                # print("using max_faults_per_image: {}".format(
                #     self.parser.max_faults_per_image))
                self.num_faults_forall_images = self.parser.max_faults_per_image

        if self.fault_rate and self.num_faults_forall_images:
            self.fault_rate = None
        pytorchfi = self.pytorchfi
        if self.fault_rate and self.parser.max_fault_rate >= 0:
            if self.parser.rnd_mode == "weights":
                n = get_number_of_weights(pytorchfi)
            else:
                self.fault_rate = self.fault_rate * batchsize
                n = get_number_of_neurons(pytorchfi)

        elif self.num_faults_forall_images and self.max_faults_per_image > 0:
            if self.parser.rnd_mode == "neurons":
                self.num_faults_forall_images = self.num_faults_forall_images * batchsize
            n = self.num_faults_forall_images

        if n:
            n = int(n)
        else:
            log.error("inconsistent setting of faults in config file and call"
                      " to function 'get_fimodel_iter'."
                      " Shutting down!")
            sys.exit()
        return n, batchsize

    def __get_inj_model(
            self, vtype, c, d, h, w, value, conv_num, batch=None, k=None):
        rnd_value_type = self.rnd_value_type
        pytorchfi = self.pytorchfi
        if vtype == "neurons":
            if rnd_value_type == "number":
                inj_model = pytorchfi.declare_neuron_fi(
                    batch=batch, conv_num=conv_num,
                    c=c, clip=d, h=h, w=w, value=value)
            else:
                # pytorchfi.bit_loc = list(value)
                # print('stuck here?', value)
                # if isinstance(value, list) or isinstance(value, np.ndarray):
                #    pytorchfi.bit_loc = list(value)
                # else:
                pytorchfi.set_bit_loc(value)
                inj_model = pytorchfi.declare_neuron_fi(
                    batch=batch,
                    conv_num=conv_num,
                    c=c,
                    clip=d,
                    h=h,
                    w=w,
                    function=pytorchfi.single_bit_flip_signed_across_batch)
        else:
            if rnd_value_type == "number":
                inj_model = pytorchfi.declare_weight_fi(
                    conv_num=conv_num, k=k, c=c, clip=d, h=h, w=w, value=value)
            else:
                # print('stuck here?', value)
                pytorchfi.set_bit_loc(value)
                inj_model = pytorchfi.declare_weight_fi(
                    conv_num=conv_num, k=k, c=c, clip=d, h=h, w=w,
                    function=pytorchfi.single_bit_flip_weights
                )
        return inj_model
