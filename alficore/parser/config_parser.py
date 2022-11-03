# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import os
import logging

log = logging.getLogger()


class ConfigParser:
    def __init__(self, **entries):
        self.print = False
        self.fi_logfile = "-1"
        self.read_from_file = "-1"
        self.num_runs = -1
        self.max_faults_per_image = -1
        self.max_fault_rate = -1.0
        self.run_type = "-1"
        self.value_type = "-1"
        self.layer_types = []
        self.rnd_mode = "-1"
        self.rnd_batch = "-1"
        self.rnd_layer = "-1"
        self.rnd_layer_weighted = False
        self.rnd_location = "-1"
        self.rnd_value = "-1"
        self.rnd_value_type = "-1"
        self.rnd_value_min = -1.0
        self.rnd_value_max = -1.0
        self.rnd_value_bits = -1
        self.rnd_bit_range = []
        self.ds_location = "-1"
        self.ds_batch_size = -1
        self.ds_loader_class = "-1"
        self.ptf_H = -1
        self.ptf_W = -1
        self.ptf_D = -1
        self.ptf_C = -1
        self.ptf_batch_size = -1
        self.ds_test_size = -1
        self.save_fault_file_dir = None
        self.inj_policy = None
        self.__dict__.update(entries)
        self.run_type_opts = ["single", "accumulate"]
        self.value_type_opts = ["static", "random"]
        self.rnd_mode_opts = ["neurons", "weights"]
        self.rnd_batch_opts = ["each", "same", "change"]
        # valid for layer, location, value
        self.rnd_feature_opts = ["same", "change"]
        ## TODO: Activate bitflip_bounds feature
	    # self.layer_boundsfile = "-1"
        self.rnd_value_type_opts = ["number", "bitflip", "bitflip_bounds", "bitflip_weighted", "stuckat_0", "stuckat_1"]
        self.layer_types_opts = ["conv2d", "fcc", "conv3d"]
        self.inj_policy_opts = ["per_image", "per_epoch", "per_batch"]
        self.reason = []
        self.valid = True

    def isvalid(self, filedir=None):
        if (self.print is not True) and (
                self.print is not False):  # not boolean
            self.reason.append("Wrong value for print: {}".format(self.print))
            self.valid = False
        if self.fi_logfile == "-1":
            log.info("fi_logfile not set, ignoring")
        if self.read_from_file == "-1":
            log.info("read_from_file not set, ignoring")
        elif not os.path.isabs(self.read_from_file):
            file_location = os.path.join(filedir, self.read_from_file)
            if not os.path.exists(file_location):
                self.reason.append("Wrong value for read_from_file {}".format(
                    self.read_from_file))
                self.valid = False
        if self.num_runs == -1:
            log.warning("num_runs not set, setting to 1")
            self.num_runs = 1
        if self.max_faults_per_image == -1:
            if self.max_fault_rate == -1.0:
                log.warning("max_faults_per_image not set, setting to 1")
                log.warning("max_fault_rate not set, ignoring")
                self.max_faults_per_image = 1
            else:
                log.warning("max_faults_per_image not set, ignoring")
                if not self.isnumber(self.max_fault_rate):
                    self.reason.append("max_fault_rate is not a number")
                    self.valid = False

        elif not (self.max_faults_per_image == -1) \
                and self.max_fault_rate == -1:
            if not self.isnumber(self.max_faults_per_image):
                self.reason.append("max_faults_per_image is not a number")
                self.valid = False
        elif not (self.max_faults_per_image == -1) and not (
                self.max_fault_rate == -1):
            self.reason.append(
                "max_faults_per_image and max_fault_rate are both set,"
                " please chose only one.")
            self.valid = False
        if self.run_type == "-1":
            log.warning("run_type not set, setting to single")
            self.run_type = "single"
        elif self.run_type not in self.run_type_opts:
            self.reason.append(
                "Wrong value for run_type: {}".format(self.run_type))
            self.valid = False
        if not self.layer_types:
            log.warning("layer_types not set, setting to conv2d")
            self.layer_types = ("conv2d")
        else:
            for lt in self.layer_types:
                if lt not in self.layer_types_opts:
                    self.reason.append(
                        "Wrong value for layer_types: {}".format(lt))
                    self.valid = False
            self.layer_types = tuple(self.layer_types)
        if self.rnd_mode == "-1":
            log.warning("rnd_mode not set, setting to 'neurons'")
            self.rnd_mode = "neurons"
        if self.rnd_mode == 'weights':
            if self.inj_policy not in ['per_epoch', 'per_batch']:
                self.reason.append("Wrong value for inj_policy selected for the fault injections in weights; weight fault injection only supports 'per_epoch' and 'per_batch'\
                : {}".format(self.sourcedev))
                self.valid = False
        elif self.rnd_mode not in self.rnd_mode_opts:
            self.reason.append(
                "Wrong value for rnd_mode: {}".format(self.rnd_mode))
            self.valid = False
        if self.rnd_batch == "-1":
            self.reason.append("rnd_batch not set.")
            self.valid = False
        elif self.rnd_batch not in self.rnd_batch_opts:
            self.reason.append(
                "Wrong value for rnd_batch: {}".format(self.rnd_batch))
            self.valid = False
        if self.rnd_layer == "-1":
            self.reason.append("rnd_batch not set.")
            self.valid = False
        elif (self.rnd_layer not in self.rnd_feature_opts) and (
                not self.isnumber(self.rnd_layer)):
            self.reason.append(
                "Wrong value for rnd_layer: {}".format(self.rnd_layer))
            self.valid = False
        if self.rnd_location == "-1":
            self.reason.append("rnd_location not set.")
            self.valid = False
        elif self.rnd_location not in self.rnd_feature_opts:
            self.reason.append(
                "Wrong value for rnd_location: {}".format(
                    self.rnd_location))
            self.valid = False
        # layer == change and location == same does not make sense
        # because layer dimensions differ
        elif self.rnd_location == "same" and self.rnd_layer == "change":
            self.rnd_location = "change"
            log.warn("layer == change and location == same does not make"
                        " sense because layer dimensions differ!")
            log.warn("chaning location to 'change'")
        if self.rnd_value == "-1":
            self.reason.append("rnd_value not set.")
            self.valid = False
        elif self.rnd_value not in self.rnd_feature_opts:
            self.reason.append(
                "Wrong value for rnd_value: {}".format(self.rnd_value))
            self.valid = False
        if self.rnd_value_type == "-1":
            self.reason.append("rnd_value_type not set.")
            self.valid = False
        elif self.rnd_value_type not in self.rnd_value_type_opts:
            self.reason.append(
                "Wrong value for rnd_value_type: {}".format(
                    self.rnd_value_type))
            self.valid = False
        elif self.rnd_value_type == "number":
            if self.rnd_value_min == -1.0:
                self.reason.append("rnd_value_min not set.")
                self.valid = False
            elif not self.isnumber(self.rnd_value_min):
                self.reason.append(
                    "value for rnd_value_min is not a number: {}".format(
                        self.rnd_value_min))
                self.valid = False
            if self.rnd_value_max == -1.0:
                self.reason.append("rnd_value_max not set.")
                self.valid = False
            elif not self.isnumber(self.rnd_value_max):
                self.reason.append(
                    "value for rnd_value_max is not a number: {}".format(
                        self.rnd_value_max))
                self.valid = False
        elif self.rnd_value_type == "bitflip":
            if self.rnd_value_bits == -1:
                self.reason.append("rnd_value_bits not set.")
                self.valid = False
            elif not self.isnumber(self.rnd_value_bits):
                self.reason.append(
                    "value for rnd_value_bits is not a number: {}".format(
                        self.rnd_value_bits))
                self.valid = False
            if self.rnd_bit_range:
                if len(self.rnd_bit_range) > 2:
                    self.reason.append(
                        "rnd_bit_range expects a maximum of 2 values, "
                        "{} given.".format(len(self.rnd_bit_range)))
                    self.valid = False
                elif len(self.rnd_bit_range) == 2:
                    if self.rnd_bit_range[0] > self.rnd_bit_range[1]:
                        # swap elements
                        self.rnd_bit_range[:] = self.rnd_bit_range[::-1]
                    if max(self.rnd_bit_range) > self.rnd_value_bits - 1:
                        self.reason.append("rnd_bit_range contains too "
                                            "large values"
                                            ",should not be larger than "
                                            "rnd_value_bits")
                        self.valid = False
                else:  # one element only
                    if max(self.rnd_bit_range) > self.rnd_value_bits - 1:
                        self.reason.append(
                            "rnd_bit_range is too large"
                            ",should not be larger than rnd_value_bits")
                        self.valid = False
        elif self.rnd_value_type == "stuckat_0" or self.rnd_value_type == "stuckat_1":
            if self.inj_policy != "per_epoch":
                self.reason.append("Permanent faults: stuck at 0/1 is only compatible with injectin policy-per_epoch; current injection policy: {}".format(self.inj_policy))
                self.valid = False
        ## TODO: Activate bitflip_bounds feature
        # elif self.rnd_value_type == "bitflip_bounds":
        #     if self.layer_boundsfile == "-1":
        #         self.reason.append(
        #             "value for rnd_value_type is {} and the corresponding bounds file is not given: {}"
        #             .format(self.rnd_value_type))
        #         self.valid = False

        if self.ptf_H == -1:
            self.reason.append("ptf_H not set.")
            self.valid = False
        elif not self.isnumber(self.ptf_H):
            self.reason.append(
                "value for ptf_H is not a number: {}"
                .format(self.ptf_H))
            self.valid = False
        if self.ptf_W == -1:
            self.reason.append("ptf_W not set.")
            self.valid = False
        elif not self.isnumber(self.ptf_W):
            self.reason.append(
                "value for ptf_W is not a number: {}"
                .format(self.ptf_W))
            self.valid = False
        if self.ptf_batch_size == -1:
            log.warning("ptf_batch_size not set, setting to 1.")
            self.ptf_batch_size = 1
        elif not self.isnumber(self.ptf_batch_size):
            self.reason.append(
                "value for ptf_batch_size is not a number: {}"
                .format(self.ptf_batch_size))
            self.valid = False
        if self.ptf_C == -1:
            log.warning("ptf_C not set. Setting to 3")
            self.ptf_C = 3
        if "3dconv" in self.layer_types:
            if self.ptf_D == -1:
                self.reason.append("ptf_D not set.")
                self.valid = False
            elif not self.isnumber(self.ptf_D):
                self.reason.append(
                    "value for ptf_D is not a number: {}"
                    .format(self.ptf_D))
                self.valid = False
        ## TODO: Activate bitflip_bounds feature
        # if self.rnd_value_type == "bitflip_bounds":
        #     if self.layer_boundsfile == "-1":
        #         self.reason.append(
        #             "value for rnd_value_type is {} and the corresponding bounds file is not given: {}"
        #             .format(self.rnd_value_type))
        #         self.valid = False
        if self.inj_policy is None:
                self.reason.append("mandatory value for injection policy not set")
                self.valid = False
        elif self.inj_policy not in self.inj_policy_opts:
            self.reason.append(
                "Wrong value for inj_policy: {}".format(self.inj_policy_opts))
        return self.valid, self.reason

    def isnumber(self, value):
        try:
            int(value)
            return True
        except Exception as e:
            log.debug(e)
            try:
                float(value)
                return True
            except Exception as e1:
                log.debug(e1)
                return False
