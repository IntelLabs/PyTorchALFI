"""
pytorchfi.errormodels provides different error models out-of-the-box for use.
"""

import random
import logging
import torch
from ..pytorchfi import core
import struct
from ..pytorchfi.util import CircularBuffer, Map_Dict, get_savedBounds_minmax
import numpy as np
import torch.nn as nn
from math import fabs

"""
helper functions
"""


def random_batch_element(pfi_model):
    return random.randint(0, pfi_model.get_total_batches() - 1)


def random_neuron_location(pfi_model, conv=-1):
    if conv == -1:
        conv = random.randint(0, pfi_model.get_total_conv() - 1)

    c = pfi_model.get_fmaps_num(conv)
    if not c == -1:
        c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
    d = pfi_model.get_fmaps_D(conv)
    if not d == -1:
        d = random.randint(0, d - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

    return (c, d, h, w)


def random_weight_location(pfi_model, conv=-1):
    loc = list()

    if conv == -1:
        corrupt_layer = random.randint(0, pfi_model.get_total_conv() - 1)
    else:
        corrupt_layer = conv
    loc.append(corrupt_layer)

    curr_layer = 0
    layer_dim = 0
    tmp_param_size = []
    for module in pfi_model.get_original_model().modules():
        # for name, param in pfi_model.get_original_model().named_parameters():
        # print(type(module))
        # if "conv" in name and "weight" in name:
        # todo verify position of k
        # print("Type {}".format(type(module)))
        if __verify_layer(pfi_model, module):
            for name, param in module.named_parameters():
                tmp_param_size.append(list(param.size()))
                if curr_layer == corrupt_layer and "weight" in name:
                    # logging.info("Layer for injection: {}".format(name))
                    layer_dim = len(param.size()) + 1
                    for dim in param.size():
                        loc.append(random.randint(0, dim - 1))
            # print (curr_layer)
            curr_layer += 1
        # print ("loop {}".format(looptest))
        # looptest += 1
    """ print("Relevant param size of original model")
    for tmp in tmp_param_size:
        print("{}".format(tmp)) """

    assert curr_layer == pfi_model.get_total_conv()
    assert len(loc) == layer_dim

    return tuple(loc)


def get_number_of_weights(pfi_model):
    size = 0
    for module in pfi_model.get_original_model().modules():
        if __verify_layer(pfi_model, module):
            size += sum(
                param.numel() for name, param in module.named_parameters()
                if "weight" in name)
    return size


def get_number_of_neurons(pfi_model):
    return pfi_model.get_neuron_num()


def __verify_layer(pfi_model, module):
    ret = False
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) \
            or isinstance(module, nn.Linear):
        if pfi_model.LAYER_TYPE_CONV2D and isinstance(module, nn.Conv2d):
            ret = True
        elif pfi_model.LAYER_TYPE_CONV3D and isinstance(module, nn.Conv3d):
            ret = True
        elif pfi_model.LAYER_TYPE_FCC and isinstance(module, nn.Linear):
            ret = True
    return ret


def random_multi_weight_location(pfi_model, num_faults=1):
    # TODO: add support for 3DConv (parameter d)
    random.seed(random.randint(1, 100000))
    conv_max = pfi_model.get_total_conv()
    conv_prob = int(random.randint(1, 99)*conv_max/100)
    conv_prob = 1 if conv_prob == 0 else conv_prob
    # sample distribution randomly - gamma, beta
    random.seed(random.randint(1, 100000))
    sampled_conv = random.sample(range(0, conv_max - 1), conv_prob)
    sampled_conv.sort()
    np.random.seed(random.randint(1, 100000))
    weighted_dist = np.random.random(len(sampled_conv))

    weighted_dist = weighted_dist*num_faults
    weighted_dist = np.around(weighted_dist).astype(int)
    excess_faults = sum(weighted_dist) - num_faults
    while excess_faults != 0:
        if len(sampled_conv) == 1:
            rand_indx = 0
        else:
            random.seed(random.randint(1, 100000))
            rand_indx = random.randint(0, len(sampled_conv) - 1)
            non_zero_indx = [i for i in weighted_dist.nonzero()[0]]
            if len(non_zero_indx) == 0:
                rand_indx = 0
            else:
                random.seed(random.randint(1, 100000))
                rand_indx = random.choice(non_zero_indx)

        random.seed(random.randint(1, 100000))
        rand_weight_faults = max(random.randint(
            0, weighted_dist[rand_indx]), 1)
        weighted_dist[rand_indx] = \
            weighted_dist[rand_indx] - min(rand_weight_faults, excess_faults)
        excess_faults = sum(weighted_dist) - num_faults
    sampled_conv = [sampled_conv[i] for i in weighted_dist.nonzero()[0]]
    weighted_dist = [weighted_dist[i] for i in weighted_dist.nonzero()[0]]
    assert sum(weighted_dist) == num_faults
    conv = []
    k = []
    c = []
    h = []
    w = []
    for i in range(len(sampled_conv)):
        fault_injs = weighted_dist[i]
        _loc = list()
        corrupt_layer = sampled_conv[i]
        _loc.append(corrupt_layer)
        curr_layer = 0
        for name, param in pfi_model.get_original_model().named_parameters():
            if (("conv" in name) or
                    ("downsample.0" in name)) and "weight" in name:
                if curr_layer == corrupt_layer:
                    _conv = [corrupt_layer for _ in range(fault_injs)]
                    random.seed(random.randint(1, 100000))
                    _k = random.choices(range(0, param.size()[0] - 1),
                                        k=fault_injs)
                    random.seed(random.randint(1, 100000))
                    _c = random.choices(
                        range(0, param.size()[1] - 1), k=fault_injs)
                    try:
                        random.seed(random.randint(1, 100000))
                        _h = random.choices(
                            range(0, param.size()[2] - 1), k=fault_injs)
                    except Exception as e:
                        logging.debug(e)
                        random.seed(random.randint(1, 100000))
                        _h = random.randint(0, param.size()[2] - 1)
                        _h = [_h for _ in range(fault_injs)]
                    try:
                        random.seed(random.randint(1, 100000))
                        _w = random.choices(
                            range(0, param.size()[3] - 1), k=fault_injs)
                    except Exception as e:
                        logging.debug(e)
                        random.seed(random.randint(1, 100000))
                        _w = random.randint(0, param.size()[3] - 1)
                        _w = [_w for _ in range(fault_injs)]
                    assert len(_conv + _k + _c + _h + _w) % fault_injs == 0
                    conv.extend(_conv)
                    k.extend(_k)
                    c.extend(_c)
                    h.extend(_h)
                    w.extend(_w)
                curr_layer += 1
        assert curr_layer == pfi_model.get_total_conv()
    assert len(conv + k + c + h + w) % num_faults == 0
    return (conv, k, c, h, w)


def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


"""
Neuron Perturbation Models
"""


# @ssq
def random_multi_neuron_location(pfi_model, num_faults=1):
    """
    Returns the randomly chosen neurons
    """

    random.seed(random.randint(1, 100000))
    conv_max = pfi_model.get_total_conv()
    conv_prob = int(random.randint(1, 99)*conv_max/100)
    conv_prob = 1 if conv_prob == 0 else conv_prob
    random.seed(random.randint(1, 100000))
    sampled_conv = random.sample(range(0, conv_max - 1), conv_prob)
    sampled_conv.sort()
    random.seed(random.randint(1, 100000))
    np.random.seed(random.randint(1, 100000))
    weighted_dist = np.random.random(len(sampled_conv))
    weighted_dist = weighted_dist*num_faults
    weighted_dist = np.around(weighted_dist).astype(int)
    excess_faults = sum(weighted_dist) - num_faults
    while excess_faults != 0:
        if len(sampled_conv) == 1:
            rand_indx = 0
        else:
            random.seed(random.randint(1, 100000))
            rand_indx = random.randint(0, len(sampled_conv) - 1)
            non_zero_indx = [i for i in weighted_dist.nonzero()[0]]
            if len(non_zero_indx) == 0:
                rand_indx = 0
            else:
                random.seed(random.randint(1, 100000))
                rand_indx = random.choice(non_zero_indx)
        # try:
        random.seed(random.randint(1, 100000))
        rand_weight_faults = max(
            random.randint(0, weighted_dist[rand_indx]), 1)
        # except:
        #     rand_weight_faults = 0
        weighted_dist[rand_indx] = weighted_dist[rand_indx] - min(
            rand_weight_faults, excess_faults)
        excess_faults = sum(weighted_dist) - num_faults
    sampled_conv = [sampled_conv[i] for i in weighted_dist.nonzero()[0]]
    weighted_dist = [weighted_dist[i] for i in weighted_dist.nonzero()[0]]
    conv = []
    c = []
    h = []
    w = []
    i = 0
    while i < len(sampled_conv):
        _conv = sampled_conv[i]
        fault_injs = weighted_dist[i]
        c_max = pfi_model.get_fmaps_num(_conv)
        h_max = pfi_model.get_fmaps_H(_conv)
        w_max = pfi_model.get_fmaps_W(_conv)
        random.seed(random.randint(1, 100000))
        channels_perc = int(random.randint(1, 100)*c_max/100)
        channels_perc = 1 if channels_perc == 0 else channels_perc
        channels_perc = c_max-2 if channels_perc >= c_max-1 else channels_perc
        random.seed(random.randint(1, 100000))
        _c = random.sample(range(1, c_max - 1), channels_perc)
        random.seed(random.randint(1, 100000))
        _c = [random.choice(_c) for _ in range(fault_injs)]
        random.seed(random.randint(1, 100000))
        _h = random.choices(range(1, h_max - 1), k=fault_injs)
        random.seed(random.randint(1, 100000))
        _w = random.choices(range(1, w_max - 1), k=fault_injs)
        random.seed(random.randint(1, 100000))
        _conv = [_conv for _ in range(fault_injs)]
        try:
            assert len(_conv + _c + _h + _w) % fault_injs == 0
        except Exception as e:
            logging.debug(e)
            continue
        i = i + 1
        conv.extend(_conv)
        c.extend(_c)
        h.extend(_h)
        w.extend(_w)
    assert len(conv + c + h + w) % num_faults == 0
    return (conv, c, h, w)


# single random neuron error in single batch element
def random_neuron_inj(pfi_model, min_val=-1, max_val=1):
    b = random_batch_element(pfi_model)
    (conv, C, D, H, W) = random_neuron_location(pfi_model)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_neuron_fi(
        batch=b, conv_num=conv, c=C, clip=D, h=H, w=W, value=err_val
    )


# single random neuron error in each batch element.
def random_neuron_inj_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, conv_num, c_rand, d_rand, h_rand, w_rand, value = (
        [] for i in range(7))

    if not randLoc:
        (conv, C, D, H, W) = random_neuron_location(pfi_model)
    if not randVal:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (conv, C, D, H, W) = random_neuron_location(pfi_model)
        if randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        conv_num.append(conv)
        c_rand.append(C)
        d_rand.append(D)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, clip=d_rand, h=h_rand,
        w=w_rand, value=value
    )


# one random neuron error per layer in single batch element
def random_inj_per_layer(pfi_model, min_val=-1, max_val=1):
    batch, conv_num, c_rand, d_rand, h_rand, w_rand, value = (
        [] for i in range(7))

    b = random_batch_element(pfi_model)
    for i in range(pfi_model.get_total_conv()):
        (conv, C, D, H, W) = random_neuron_location(pfi_model, conv=i)
        batch.append(b)
        conv_num.append(conv)
        c_rand.append(C)
        d_rand.append(D)
        h_rand.append(H)
        w_rand.append(W)
        value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, clip=d_rand, h=h_rand,
        w=w_rand, value=value
    )


# one random neuron error per layer in each batch element
def random_inj_per_layer_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, conv_num, c_rand, d_rand, h_rand, w_rand, value = (
        [] for i in range(7))

    for i in range(pfi_model.get_total_conv()):
        if not randLoc:
            (conv, C, D, H, W) = random_neuron_location(pfi_model, conv=i)
        if not randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi_model.get_total_batches()):
            if randLoc:
                (conv, C, D, H, W) = random_neuron_location(pfi_model, conv=i)
            if randVal:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            conv_num.append(conv)
            c_rand.append(C)
            d_rand.append(D)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, clip=d_rand, h=h_rand,
        w=w_rand, value=value
    )


class single_bit_flip_func(core.fault_injection):
    def __init__(self, model, model_attr, ptfiwrap, **kwargs):
        super().__init__(model, h=model_attr.ptf_H, w=model_attr.ptf_W, batch_size=model_attr.ptf_batch_size, \
            c=model_attr.ptf_C, clip=model_attr.ptf_D, **kwargs)
        logging.basicConfig(
            format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")
        logging.getLogger().setLevel('INFO')
        self.bits = kwargs.get("bits", 8)
        self.rnd_value_type = model_attr.rnd_value_type
        self.ptfiwrap = ptfiwrap
        self.bit_loc = None
        self.ptfi_batch_pointer = -1
        self.ptfi_batch_pointer_curr = -1
        self.total_captured_faults = self.compute_runset_length()
        self.LayerRanges = []
        self.bit_flips_direc = np.array([None]*self.total_captured_faults)
        self.bit_flips_monitor = np.array([None]*self.total_captured_faults)
        self.value_monitor = np.array([[None]*self.total_captured_faults,[None]*self.total_captured_faults])
        ## TODO: Activate bitflip_bounds feature
        # if model_attr.rnd_value_type == "bitflip_bounds":
        #     self.bounds = get_savedBounds_minmax(model_attr.layer_boundsfile)

    @classmethod
    def from_kwargs(cls, model, c=1, h=32, w=32, clip=-1, batch_size=1, rnd_value_type='bitflip', **kwargs):
        model_attr = Map_Dict({"ptf_C": c, "ptf_H":h, "ptf_W":w, "ptf_D":clip, "ptf_batch_size":batch_size, "rnd_value_type":rnd_value_type})
        return cls(model, model_attr, **kwargs)

    def compute_runset_length(self):
        if self.ptfiwrap.value_type == 'neurons':
            runset_length = self.ptfiwrap.num_runs*self.ptfiwrap.dataset_size*self.ptfiwrap.max_faults_per_image
        elif self.ptfiwrap.value_type == 'weights':
            if self.ptfiwrap.parser.inj_policy == "per_epoch":
                runset_length = self.ptfiwrap.num_runs*self.ptfiwrap.max_faults_per_image
            if self.ptfiwrap.parser.inj_policy == 'per_batch':
                runset_length = int(np.ceil(self.ptfiwrap.num_runs*self.ptfiwrap.dataset_size*self.ptfiwrap.max_faults_per_image/self.ptfiwrap.batch_size))
        return runset_length

    def set_conv_max(self, data):
        self.LayerRanges = data

    def reset_conv_max(self, data):
        self.LayerRanges = []

    def get_conv_max(self, layer):
        return self.LayerRanges[layer]

    def set_bit_loc(self, value):
        if isinstance(value, np.ndarray):
            size = len(value)
            self.bit_loc = CircularBuffer(max_size=size)
            for item in value:
                self.bit_loc.enqueue(item)
        else:
            self.bit_loc = value

    def _twos_comp_shifted(self, val, nbits):
        if val < 0:
            val = (1 << nbits) + val
        else:
            val = self._twos_comp(val, nbits)
        return val

    def _twos_comp(self, val, bits):
        # compute the 2's complement of int value val
        if (val & (1 << (bits - 1))) != 0:
            # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)  # compute negative value
        return val  # return positive value as is

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logging.info("orig value:", orig_value)
        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logging.info("quantum:", quantum)
        logging.info("twos_comple:", twos_comple)
        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info("bits:", bits)
        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        assert len(bits) == total_bits
        logging.info("sign extend bits", bits)
        # flip a bit
        # use MSB -> LSB indexing
        assert bit_pos < total_bits
        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info("bits", bits_str_new)
        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")
        # convert to quantum
        assert bits_str_new.isdigit()
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logging.info("out", out)

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info("new_value", new_value)

        return torch.tensor(new_value, dtype=save_type)

    def __single_bit_flip(self, orig_value, bit_pos):
        array_pointer = self.ptfi_batch_pointer + self.ptfi_batch_pointer_curr
        save_type = orig_value.dtype
        float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0')
                                for c in struct.pack('!f', orig_value.item()))
        assert (len(float_to_bin) - 1 >= bit_pos),\
            "Bit position {} too large for size of value: {}"\
            .format(bit_pos, len(float_to_bin))
        # float_to_bin[bit_pos] = 1 - float_to_bin[bit_pos] # 1 to 0 or 0 to 1
        # logging.info("original value: {}".format(orig_value))
        # logging.info("original bitmap: {}".format(float_to_bin))
        self.bit_flips_monitor[array_pointer] = bit_pos
        if float_to_bin[bit_pos] == "1":
            new_float = float_to_bin[:bit_pos] + '0' + float_to_bin[bit_pos+1:]
            self.bit_flips_direc[array_pointer] = 0
        else:
            new_float = float_to_bin[:bit_pos] + '1' + float_to_bin[bit_pos+1:]
            self.bit_flips_direc[array_pointer] = 1
        # logging.info("changed bitmap: {}".format(new_float))
        f = int(new_float, 2)
        bin_to_float = struct.unpack('f', struct.pack('I', f))[0]
        corr_value = torch.tensor(bin_to_float, dtype=save_type)
        self.value_monitor[0, array_pointer] = orig_value.item()
        self.value_monitor[1, array_pointer] = corr_value.item()        
        return corr_value

    def single_bit_flip(self, orig_value, bit_pos):
        array_pointer = self.ptfi_batch_pointer_curr
        save_type = orig_value.dtype
        float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0')
                                for c in struct.pack('!f', orig_value.item()))
        assert (len(float_to_bin) - 1 >= bit_pos),\
            "Bit position {} too large for size of value: {}"\
            .format(bit_pos, len(float_to_bin))
        # self.bit_flips_monitor = np.append(self.bit_flips_monitor, bit_pos)
        self.bit_flips_monitor[array_pointer] = bit_pos
        if float_to_bin[bit_pos] == "1":
            new_float = float_to_bin[:bit_pos] + '0' + float_to_bin[bit_pos+1:]
            # self.bit_flips_direc = np.append(self.bit_flips_direc, 0)
            self.bit_flips_direc[array_pointer] = 0
        else:
            new_float = float_to_bin[:bit_pos] + '1' + float_to_bin[bit_pos+1:]
            # self.bit_flips_direc = np.append(self.bit_flips_direc, 1)
            self.bit_flips_direc[array_pointer] = 1
        # logging.info("changed bitmap: {}".format(new_float))
        f = int(new_float, 2)
        bin_to_float = struct.unpack('f', struct.pack('I', f))[0]
        corr_value = torch.tensor(bin_to_float, dtype=save_type)
        # self.value_monitor = np.append(self.value_monitor, [[orig_value.item()],[corr_value.item()]], 1)
        self.value_monitor[0, array_pointer] = orig_value.item()
        self.value_monitor[1, array_pointer] = corr_value.item()
        return corr_value

    def single_bit_flip_bounds(self, orig_value, bounds:list=[-1, 1]):
        """
        Value flipped bit remains within the bounds given. 
        All bits are searched and the one with higher weightage is chosen.
        Assuming it is IEEE 32 bit format
        """
        array_pointer = self.ptfi_batch_pointer + self.ptfi_batch_pointer_curr
        save_type = orig_value.dtype
        ## 
        bounds = [min(orig_value, bounds[0]), max(orig_value, bounds[1])]
        bit_pos_ = np.array([]).astype(np.uint)
        weighted_bits = np.array([])
        for i in range(32):
            flipped_val = self.__single_bit_flip(orig_value, i).item()
            if flipped_val >= bounds[0] and flipped_val <= bounds[1]:
                weighted_bits = np.append(weighted_bits, fabs(flipped_val - orig_value.item()))
                bit_pos_ = np.append(bit_pos_, i)
        weighted_bits = weighted_bits/weighted_bits.sum()
        bit_pos =int(random.choices(population=bit_pos_, k=1, weights=weighted_bits)[0])
        float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0')
                                for c in struct.pack('!f', orig_value))
        assert (len(float_to_bin) - 1 >= bit_pos),\
            "Bit position {} too large for size of value: {}"\
            .format(bit_pos, len(float_to_bin))
        self.bit_flips_monitor = np.append(self.bit_flips_monitor, bit_pos)
        if float_to_bin[bit_pos] == "1":
            new_float = float_to_bin[:bit_pos] + '0' + float_to_bin[bit_pos+1:]
            self.bit_flips_direc = np.append(self.bit_flips_direc, 0)
        else:
            new_float = float_to_bin[:bit_pos] + '1' + float_to_bin[bit_pos+1:]
            self.bit_flips_direc = np.append(self.bit_flips_direc, 1)
        # logging.info("changed bitmap: {}".format(new_float))
        f = int(new_float, 2)
        bin_to_float = struct.unpack('f', struct.pack('I', f))[0]
        return torch.tensor(bin_to_float, dtype=save_type)

    def single_bit_flip_stuckat(self, orig_value, bit_pos, stuckat:int):
        """
        Value flipped bit remains within the bounds given. 
        All bits are searched and the one with higher weightage is chosen.
        Assuming it is IEEE 32 bit format
        """
        save_type = orig_value.dtype
        float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0')
                                for c in struct.pack('!f', orig_value.item()))
        assert (len(float_to_bin) - 1 >= bit_pos),\
            "Bit position {} too large for size of value: {}"\
            .format(bit_pos, len(float_to_bin))
        # float_to_bin[bit_pos] = 1 - float_to_bin[bit_pos] # 1 to 0 or 0 to 1
        # logging.info("original value: {}".format(orig_value))
        # logging.info("original bitmap: {}".format(float_to_bin))
        self.bit_flips_monitor = np.append(self.bit_flips_monitor, bit_pos)
        new_float = float_to_bin[:bit_pos] + str(stuckat) + float_to_bin[bit_pos+1:]
        self.bit_flips_direc = np.append(self.bit_flips_direc, int(stuckat))
        # logging.info("changed bitmap: {}".format(new_float))
        f = int(new_float, 2)
        bin_to_float = struct.unpack('f', struct.pack('I', f))[0]
        corr_value = torch.tensor(bin_to_float, dtype=save_type)
        self.value_monitor = np.append(self.value_monitor, [[orig_value.item()],[corr_value.item()]], 1)
        return corr_value

    def single_bit_flip_weighted(self, orig_value):
        """
        All bits are searched and the one with higher weightage is chosen.
        Assuming it is IEEE 32 bit format
        """
        save_type = orig_value.dtype
        bit_pos_ = np.array([]).astype(np.uint)
        weighted_bits = np.array([])
        for i in range(32):
            flipped_val = self.__single_bit_flip(orig_value, i).item()
            weighted_bits = np.append(weighted_bits, fabs(flipped_val - orig_value.item()))
            bit_pos_ = np.append(bit_pos_, i)
        weighted_bits = weighted_bits/weighted_bits.sum()
        bit_pos =int(random.choices(population=bit_pos_, k=1, weights=weighted_bits)[0])
        float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0')
                                for c in struct.pack('!f', orig_value))
        assert (len(float_to_bin) - 1 >= bit_pos),\
            "Bit position {} too large for size of value: {}"\
            .format(bit_pos, len(float_to_bin))
        self.bit_flips_monitor = np.append(self.bit_flips_monitor, bit_pos)
        if float_to_bin[bit_pos] == "1":
            new_float = float_to_bin[:bit_pos] + '0' + float_to_bin[bit_pos+1:]
            self.bit_flips_direc = np.append(self.bit_flips_direc, 0)
        else:
            new_float = float_to_bin[:bit_pos] + '1' + float_to_bin[bit_pos+1:]
            self.bit_flips_direc = np.append(self.bit_flips_direc, 1)
        # logging.info("changed bitmap: {}".format(new_float))
        f = int(new_float, 2)
        bin_to_float = struct.unpack('f', struct.pack('I', f))[0]
        return torch.tensor(bin_to_float, dtype=save_type)

    def single_bit_flip_weights(self, orig_value):
        # range_max = self.get_conv_max(self.get_curr_conv())
        # logging.info("curr_conv: {}".format(self.get_curr_conv()))
        # logging.info("range_max", range_max)
        self.ptfi_batch_pointer_curr = self.ptfi_batch_pointer
        prev_value = torch.tensor(orig_value)
        if type(self.bit_loc) == CircularBuffer:
            bit_loc = self.bit_loc.front()
        else:
            bit_loc = self.bit_loc
        if bit_loc == -1:
            rand_bit = random.randint(0, self.bits - 1)
        else:
            rand_bit = bit_loc

        # logging.info("rand_bit: {}".format(rand_bit))
        # new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)
        new_value = self.single_bit_flip(prev_value, rand_bit)

        return new_value

    def single_bit_flip_signed_across_batch(self, module, input, output):
            corrupt_conv_set = self.get_corrupt_conv()
            # range_max = self.get_conv_max(self.get_curr_conv())
            # logging.info("curr_conv: {}".format(self.get_curr_conv()))
            # logging.info("range_max", range_max)
            # print("in bitflip: module id {}".format(id(module)))
            # print(self.get_curr_conv())
            if len(module.new_id) > 1:
                curr_id = module.new_id.pop(0)
            else:
                curr_id = module.new_id[0]

            if type(corrupt_conv_set) == list:
                inj_list = list(
                    filter(
                        lambda x: corrupt_conv_set[x] == curr_id,
                        range(len(corrupt_conv_set)),
                    )
                )
                for i in inj_list:
                    self.assert_inj_bounds(index=i)

                    # print(list(output.size()))
                    # print("batch {} C {} H {} W {}".format(self.CORRUPT_BATCH[i],
                    # self.CORRUPT_C[i],self.CORRUPT_H[i],self.CORRUPT_W[i]))
                    real_batch = list(output.size())[0]
                    if self.CORRUPT_C[i] >= 0:
                        if self.CORRUPT_BATCH[i] >= real_batch:
                            """
                            TODO: few models like faster-rcnn (detectron2 need this)
                            In few object detection models, the tensor shape in intermediate layers
                            gets expanded from [B, C_i, H_i, W_i] to [B * N, C_k, H_k, W_k]
                            N = number of object proposals/bounding box proposals (dependent on model's architecture)
                            self.CORRUPT_BATCH[i] = real_batch - 1
                            """
                            return
                        if self.CORRUPT_CLIP[i] >= 0:  # 3dconv layer
                            # print(list(output.size()))
                            # print("{}".format(self.CORRUPT_BATCH))
                            prev_value = output[self.CORRUPT_BATCH[i]][
                                self.CORRUPT_C[i]][self.CORRUPT_CLIP[i]][
                                    self.CORRUPT_H[i]][
                                self.CORRUPT_W[i]
                            ]
                        else:  # 2DConv layer
                            prev_value = output[self.CORRUPT_BATCH[i]][
                                self.CORRUPT_C[i]][self.CORRUPT_H[i]][
                                self.CORRUPT_W[i]
                            ]
                    else:
                        if self.CORRUPT_H[i] >= real_batch:
                            #self.CORRUPT_BATCH[i] = real_batch - 1
                            return
                        prev_value = output[self.CORRUPT_H[i]][self.CORRUPT_W[i]]

                    self.ptfi_batch_pointer_curr = self.ptfi_batch_pointer + i
                    if isinstance(self.bit_loc, CircularBuffer):
                        bit_loc = self.bit_loc.buffer[i]
                    else:
                        bit_loc = self.bit_loc
                    if bit_loc == -1:
                        rand_bit = random.randint(0, self.bits - 1)
                    else:
                        rand_bit = bit_loc

                    # logging.info("rand_bit: {}".format(rand_bit))
                    # new_value = self._flip_bit_signed(prev_value,
                    # range_max, rand_bit)
                    if self.rnd_value_type == "bitflip" or self.rnd_value_type == "stuckat_1":
                        new_value = self.single_bit_flip(prev_value, rand_bit)
                    ## TODO: Activate bitflip_bounds feature
                    # elif self.rnd_value_type == "bitflip_bounds":
                    #     new_value = self.single_bit_flip_bounds(prev_value, bounds=self.bounds[curr_id])
                    elif self.rnd_value_type == "bitflip_weighted":
                        new_value = self.single_bit_flip_weighted(prev_value)
                    # logging.info("new value: {}\n".format(new_value))
                    # TODO support 3d conv
                    if self.CORRUPT_C[i] >= 0:
                        if self.CORRUPT_CLIP[i] >= 0:  # 3dconv layer
                            output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                                self.CORRUPT_CLIP[i]][self.CORRUPT_H[i]][
                                self.CORRUPT_W[i]
                            ] = new_value
                        else:
                            output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                                self.CORRUPT_H[i]][
                                self.CORRUPT_W[i]
                            ] = new_value
                    else:
                        output[self.CORRUPT_H[i]][self.CORRUPT_W[i]] = new_value
            else:
                self.assert_inj_bounds()
                corrupt_name = self.OUTPUT_LOOKUP[corrupt_conv_set]
                # logging.debug("module.new_name {} corrupt_name {}".
                # format(module.new_name,corrupt_name))
                prev_value = torch.tensor(0)
                if curr_id == corrupt_conv_set:
                    if self.CORRUPT_C >= 0:
                        if self.CORRUPT_CLIP >= 0:  # 3dconv layer
                            # print(list(output.size()))
                            # print("{}".format(self.CORRUPT_BATCH))
                            real_batch = list(output.size())[0]
                            if self.CORRUPT_BATCH >= real_batch:
                                self.CORRUPT_BATCH = real_batch - 1
                            prev_value = output[self.CORRUPT_BATCH][
                                self.CORRUPT_C][self.CORRUPT_CLIP][self.CORRUPT_H][
                                self.CORRUPT_W
                            ]
                        else:  # 2DConv layer
                            real_batch = list(output.size())[0]
                            if self.CORRUPT_BATCH >= real_batch:
                                # self.CORRUPT_BATCH = real_batch - 1
                                """
                                @debug 
                                In few object detection models, the tensor shape in intermediate layers
                                gets expanded from [B, C_i, H_i, W_i] to [B * N, C_k, H_k, W_k]
                                N = number of object proposals/bounding box proposals (dependent on model's architecture)
                                """
                                self.CORRUPT_BATCH = real_batch - 1
                            prev_value = output[self.CORRUPT_BATCH][
                                self.CORRUPT_C][
                                self.CORRUPT_H][
                                self.CORRUPT_W
                            ]
                    else:
                        prev_value = output[self.CORRUPT_H][self.CORRUPT_W]

                    if type(self.bit_loc) == CircularBuffer:
                        bit_loc = self.bit_loc.front()
                    else:
                        bit_loc = self.bit_loc
                    if bit_loc == -1:
                        rand_bit = random.randint(0, self.bits - 1)
                    else:
                        rand_bit = bit_loc
                    self.ptfi_batch_pointer_curr = self.ptfi_batch_pointer
                    # logging.info("rand_bit: {}".format(rand_bit))
                    # new_value = self._flip_bit_signed(prev_value,
                    # range_max, rand_bit)
                    new_value = self.single_bit_flip(prev_value, rand_bit)
                    # logging.info("new value: {}\n".format(new_value))
                    # TODO support 3d conv
                    if self.CORRUPT_C >= 0:
                        if self.CORRUPT_CLIP >= 0:  # 3dconv layer
                            output[self.CORRUPT_BATCH][self.CORRUPT_C][
                                self.CORRUPT_CLIP][self.CORRUPT_H][
                                self.CORRUPT_W
                            ] = new_value
                        else:
                            output[self.CORRUPT_BATCH][self.CORRUPT_C][
                                self.CORRUPT_H][
                                self.CORRUPT_W
                            ] = new_value
                    else:
                        output[self.CORRUPT_H][self.CORRUPT_W] = new_value

def random_neuron_single_bit_inj_batched(
        pfi_model, layer_ranges, randLoc=True):
    pfi_model.set_conv_max(layer_ranges)
    batch, conv_num, c_rand, d_rand, h_rand, w_rand = ([] for i in range(6))
    if not randLoc:
        (conv, C, D, H, W) = random_neuron_location(pfi_model)
    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (conv, C, D, H, W) = random_neuron_location(pfi_model)
        batch.append(i)
        conv_num.append(conv)
        c_rand.append(C)
        d_rand.append(D)
        h_rand.append(H)
        w_rand.append(W)
    return pfi_model.declare_neuron_fi(
        batch=batch,
        conv_num=conv_num,
        c=c_rand,
        clip=d_rand,
        h=h_rand,
        w=w_rand,
        function=pfi_model.single_bit_flip_signed_across_batch,
    )

def random_neuron_single_bit_inj(pfi_model, layer_ranges):
    pfi_model.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi_model)
    (conv, C, D, H, W) = random_neuron_location(pfi_model)
    # print(batch, ",", conv, ",", C, ",", H, ",", W, ",", pfi_model.bit_loc)
    return pfi_model.declare_neuron_fi(
        batch=batch,
        conv_num=conv,
        c=C,
        clip=D,
        h=H,
        w=W,
        function=pfi_model.single_bit_flip_signed_across_batch,
    )

def random_neuron_single_bit_inj_layer(pfi_model, layer_no):

    batch = random_batch_element(pfi_model)
    conv = layer_no
    if conv == -1:
        conv = random.randint(0, pfi_model.get_total_conv() - 1)
    c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
    d = random.randint(0, pfi_model.get_fmaps_D(conv) - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        conv_num=conv,
        c=c,
        clip=d,
        h=h,
        w=w,
        function=pfi_model.single_bit_flip_signed_across_batch,
    )


"""
Weight Perturbation Models
"""


def random_weight_inj(
        pfi_model, corrupt_conv=-1, min_val=-1, max_val=1, faulty_val=None):
    index = random_weight_location(pfi_model, corrupt_conv)
    c_in = -1
    k = -1
    d = -1
    kH = index[-2]
    kW = index[-1]
    if len(index) >= 5:  # conv2d or conv3d layer
        k = index[1]
        c_in = index[2]
        if len(index) == 6:
            d = index[3]
    if faulty_val is None:
        faulty_val = random_value(min_val=min_val, max_val=max_val)

    # @ssq changed the return type
    orig_value = pfi_model.declare_weight_fi(
        conv_num=corrupt_conv, k=k, c=c_in,
        clip=d, h=kH, w=kW, value=faulty_val
    )
    return orig_value, faulty_val


def random_multi_single_bit_flip_weight_inj(pfi_model, num_faults=1, bits=8):
    conv, k, c_in, d, kH, kW, rand_bits = ([] for i in range(7))
    # for b in range(pfi_model.get_total_batches()):
    (conv, k, c_in, kH, kW) = random_multi_weight_location(
        pfi_model, num_faults=num_faults)
    rand_bits = random.choices([i for i in range(bits)], k=num_faults)
    assert len(conv + k + c_in + kH + kW + rand_bits) % num_faults == 0

    # @ssq changed the return type
    pfi_model.set_rand_bit_pos(bit_pos=rand_bits)
    pfi_model.declare_weight_fi(
        conv_num=conv, k=k, c=c_in, h=kH, w=kW, function=single_bit_flip
    )
    bit_flip_pos = [conv, k, c_in, kH, kW, rand_bits]
    return bit_flip_pos


def zeroFunc_rand_weight(pfi_model):
    (conv, k, c_in, kH, kW) = random_weight_location(pfi_model)
    return pfi_model.declare_weight_fi(
        function=_zero_rand_weight, conv_num=conv, k=k, c=c_in, h=kH, w=kW
    )


def _zero_rand_weight(data, location):
    newData = data[location] * 0
    return newData


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
            model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at {} and {}'.format(key_item_1[0],
                      key_item_2[0]))
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
        return True
    else:
        return False
