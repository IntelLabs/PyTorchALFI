# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT


import argparse
import os
import sys
import logging.config
import json
import numpy as np
import pickle

logging.config.fileConfig(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../fi.conf')))
log = logging.getLogger()


class ResTrans():

    def __init__(self):
        self.batchsize = 1
        args = self.parse_arguments()
        self.input_file = args.inputfile
        self.fault_type = args.faulttype
        self.faults_per_image = args.faultsperimage
        self.output_file = args.outputfile
        self.fileDir = os.path.dirname(sys.argv[0])
        if self.fault_type == "neuron":
            if not args.batchsize:
                print("Please give batchsize if fault_type = neuron! Exiting!")
                sys.exit()
            else:
                self.batchsize = args.batchsize
        if not self.file_exists(self.input_file):
            print("File {} does not exist, exiting!".format(self.input_file))
            sys.exit()
        self.output_file = self.prepare_outputfile(self.output_file)
        self.err_inj_data = self.read_json(self.input_file)
        self.runset = self.extract_faults()
        try:
            f = open(self.output_file, 'wb')
            pickle.dump(self.runset, f)
            print("Output written to {}!".format(self.output_file))
        except Exception as e:
            print("error writing output file: {}".format(e))

    def parse_arguments(self):
        parser = argparse.ArgumentParser(
            description='Extract fault positions from results json')
        parser.add_argument(
            'inputfile', metavar='I', help='json file with'
            ' complete test results')
        parser.add_argument(
            'faulttype', metavar='F', choices=['weight', 'neuron'],
            help='set fault type as weight or neuron')
        parser.add_argument(
            'faultsperimage', metavar='P', type=int,
            help='how many faults per image does the key bit_flip_pos contain')
        parser.add_argument(
            'outputfile', metavar='O', 
            help="binary output file readable from alficore")
        parser.add_argument(
            '--batchsize', metavar='b', type=int,
            help='Batchsize used during test')
        return parser.parse_args()

    def file_exists(self, file):
        if not os.path.isabs(file):
            file_location = os.path.join(self.filedir, file)
        else:
            file_location = file
        if not os.path.exists(file_location):
            return False
        else:
            return True

    def prepare_outputfile(self, file):
        if not os.path.isabs(file):
            file_location = os.path.join(self.filedir, file)
        else:
            file_location = file
        dir = os.path.dirname(file_location)
        try:
            if not os.path.exists(dir):
                os.mkdir(dir)
            return file
        except OSError as e:
            print("Error creating directory for outputfile {}: {}"
                  .format(dir, e))

    def read_json(self, file):
        print('reading file {}'.format(file))
        with open(file, 'r') as f:
            loaded_err_inj_data = json.load(f)
            # weight_err_inj_data[i]  = loaded_err_inj_data
        print('succefully loaded file {}'.format(file))
        return loaded_err_inj_data

    def extract_faults(self):
        """[summary]
        """
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
        outputs = self.err_inj_data["model_outputs"]
        cols = self.faults_per_image
        output_size = len(outputs) * cols
        runset = np.full((7, output_size), -1)

        # tile the whole runset with batch numbers
        if self.fault_type == "neuron":
            batchnums = self.__tile_batch(
                    self.batchsize, output_size)
            runset[0, :] = batchnums

        idx = 0
        for output in outputs:
            bit_flip_pos = np.array(output["model_label_indx"]["bit_flip_pos"])
            if self.fault_type == "weight":
                if len(bit_flip_pos) == 6:
                    insert = np.full((7, cols), -1)
                    insert[0:3, :] = bit_flip_pos[0:3, :]
                    insert[4:, :] = bit_flip_pos[3:, :]
                elif len(bit_flip_pos) == 7:
                    insert = bit_flip_pos
                runset[:, idx:idx+cols] = insert.reshape(-1, cols)
            else:
                if len(bit_flip_pos) == 5:
                    insert = np.full((6, cols), -1)
                    insert[0:2, :] = bit_flip_pos[0:2, :]
                    insert[3:, :] = bit_flip_pos[2:, :]
                elif len(bit_flip_pos) == 6:
                    insert = bit_flip_pos
                runset[1:, idx:idx+cols] = insert.reshape(-1, cols)
            idx = idx + cols

        return runset

    def __tile_batch(self, batchsize, total):
        sequence = np.arange(batchsize)
        reps = -(-total//batchsize)
        tiles = np.tile(sequence, reps)
        res = np.zeros(total, dtype=int)
        res[:] = tiles[:total]
        return res


if __name__ == "__main__":
    rt = ResTrans()
