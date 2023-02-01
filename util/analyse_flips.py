import torch
import struct
import random
import matplotlib.pyplot as plt
import numpy as np
from math import isnan, log


def single_bit_flip(orig_value, bit_pos):
    save_type = orig_value.dtype
    float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', orig_value.item()))
    assert (len(float_to_bin) - 1 >= bit_pos), "Bit position {} too large for size of value: {}" \
        .format(bit_pos, len(float_to_bin))
    # print("original value: {}".format(orig_value))
    # print("original bitmap: {}".format(float_to_bin))
    if float_to_bin[bit_pos] == "1":
        new_float = float_to_bin[:bit_pos] + '0' + float_to_bin[bit_pos + 1:]
    else:
        new_float = float_to_bin[:bit_pos] + '1' + float_to_bin[bit_pos + 1:]
    # print("changed bitmap: {}".format(new_float))
    f = int(new_float, 2) #converts to binary with base 2
    bin_to_float = struct.unpack('f', struct.pack('I', f))[0]
    return torch.tensor(bin_to_float, dtype=save_type)


def plot_fct(x_set, flipped_set, bit_pos, title):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(title)
    axs[0].hist(x_set)
    axs[0].set_xlabel('starting values')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(flipped_set)
    axs[1].set_xlabel('flipped values')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(bit_pos, bins = np.arange(32+1)-0.5)
    # axs[2].set_xticks(np.array(range(32)))
    axs[2].set_xlabel('flipped bit')
    axs[2].set_ylabel('Frequency')


##
nr_samples = 10000
bnd_orig = 100.

# # This defines the range of the input values before flipping
range_low = 0. #0.
range_high = bnd_orig #100. #2.


x_set = [random.uniform(range_low, range_high) for n in range(nr_samples)]

bit_pos = [random.randint(0, 31) for n in range(nr_samples)]

# Filter by large values
flipped_set = [float(single_bit_flip(torch.tensor(x_set[n]), bit_pos[n])) for n in range(nr_samples)]

# Remove nans
non_sing_list = [False if isnan(x) else True for x in flipped_set] #or:  or x==np.inf. Will only happen if you flip exactly 1.0
flipped_set_filt = np.array(flipped_set)[non_sing_list]
bit_pos_filt = np.array(bit_pos)[non_sing_list]
x_set_filt = np.array(x_set)[non_sing_list]


# # Set specific (e.g. large) value range of the flipped output

# Note: values 1->2 never get out of bounds, see nan inf rules
# bnd_test_low = bnd_orig*2**64 # only bit 1 flipped -> comes from small numbers 0->1 #set to 0!
# bnd_test_up = np.inf #
# bnd_test_low = bnd_orig*2**32 # only bit 2 flipped -> 2->100, homogeneous #set to 2!
# bnd_test_up = bnd_orig*2**64 #
# bnd_test_low = bnd_orig*2**16 # only bit 3 flipped -> 2->100
# bnd_test_up = bnd_orig*2**32 #
# bnd_test_low = bnd_orig*2**8 # only bit 4 flipped -> 2->100
# bnd_test_up = bnd_orig*2**16 #
# bnd_test_low = bnd_orig*2**4 # only bit 5 flipped -> comes from 6->100
# bnd_test_up = bnd_orig*2**8 #
# bnd_test_low = bnd_orig*2**2 # bit 5-6 flipped -> comes from values 2->32 (more on the 32 end)
# bnd_test_up = bnd_orig*2**4 #
# bnd_test_low = bnd_orig*2**1 # bit 6-7 flipped -> comes from values 12->100 (more on the 100 end) #set to 100
# bnd_test_up = bnd_orig*2**2 #
# bnd_test_low = bnd_orig*2**0 # bit 6-12 flipped -> comes from values 6->100 (more on the 100 end)
# bnd_test_up = bnd_orig*2**1 #

# Three domains works for now:
# bnd_test_low = bnd_orig*2**64 # b
# bnd_test_up = bnd_orig*2**128 #

bnd_test_low = bnd_orig*2**1 #
bnd_test_up = bnd_orig*2**64 #
#
# bnd_test_low = bnd_orig*2**0 #
# bnd_test_up = bnd_orig*2**1 #


mask_ge = np.ma.masked_greater_equal(flipped_set_filt, bnd_test_low, copy=True).mask
mask_le = np.ma.masked_less_equal(flipped_set_filt, bnd_test_up, copy=True).mask
mask_in_bounds = np.logical_and(mask_ge, mask_le)


flipped_beyond_bound = flipped_set_filt[mask_in_bounds]
bit_beyond_bound = bit_pos_filt[mask_in_bounds]
x_set_beyond_bound = x_set_filt[mask_in_bounds]


# Show which bits have been flipped
if len(flipped_beyond_bound) == 0:
    print('No large values found')
else:
    print('before flip: min',  "{:e}".format(min(x_set_beyond_bound)), 'max', "{:e}".format(max(x_set_beyond_bound)))
    print('after flip: min', "{:e}".format(min(flipped_beyond_bound)), 'max', "{:e}".format(max(flipped_beyond_bound)))
    print('flipped bits: min', min(bit_beyond_bound), 'max', max(bit_beyond_bound))

    # print('comp 64', "{:e}".format(bnd_orig * 2 ** 64))
    # print('comp 32', "{:e}".format(bnd_orig * 2 ** 32))
    # print('comp 16', "{:e}".format(bnd_orig * 2 ** 16))



## Plot
plt.close('all')

# All
plot_fct(x_set_filt, flipped_set_filt, bit_pos_filt, 'All values')

# Only large results
plot_fct(x_set_beyond_bound, flipped_beyond_bound, bit_beyond_bound, 'Large values')




##
# # log(100.,2)
# a = 1./2.**64
# print(a)
# b = single_bit_flip(torch.tensor(a), 1)
# print(b)
# print('comp', "{:e}".format(100*2**32))
# print('comp', "{:e}".format(100*2**64))


##

# Observations: --------------------
# -Only values between 0-1 can get very large (> bnd*2**64) because only they have b1=0 before flip. flip bit 1
# -1.0 exactly becomes inf under flip of bit 1
# - >1-2 can never get larger than they already are (2), i.e. if bnd > 2 they never get large
# values 2-100 can be maximally bnd_test*2**64 (flip bits 2-12)
# values from bnd to 2*2**64 come from pretty much all values (?)


# 99% percentile?