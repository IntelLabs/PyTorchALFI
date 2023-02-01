from util.helper_functions import flatten
import matplotlib.pyplot as plt
import torch
import torchvision
import struct
from math import isnan, log
from miovision_train_eval import resnet50_net, resnet50_load_checkpoint, vgg16_net, vgg16_load_checkpoint
import pickle, os
import numpy as np
from alficore.models.yolov3.darknet import Darknet

## Get weights
# PATH = '../state/vgg16-397923af.pth' #load weights
# net = VGG16_ranger(num_classes=1000, bounds=None)
# net.load_state_dict(getdict_ranger(PATH, net)) #load the pretrained weights

# # net = torchvision.models.resnet50(pretrained=True, progress=True)
# # net = resnet50_net(num_classes=11)
# # net = resnet50_load_checkpoint(net)
# net = vgg16_net(num_classes=11)
# net = vgg16_load_checkpoint(net)
# # net = torchvision.models.vgg16(pretrained=True, progress=True)
# net.eval()

# net = Darknet("alficore/models/yolov3/config/yolov3-kitti.cfg")
# net.load_darknet_weights("alficore/models/yolov3/weights/yolov3-kitti.weights")
# net = Darknet("alficore/models/yolov3/config/yolov3.cfg")
# net.load_darknet_weights("alficore/models/yolov3/weights/yolov3.weights")
# net.eval()

net = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True) #pretrained on coco2017, number of classes is 91 instead of 80 so mapping needed
net.eval()


def get_flat_list(net):

    ## Extract weight parameters
    all_params = list(net.named_parameters())

    ##
    names = [all_params[i][0] for i in range(len(all_params))]
    numbers = [all_params[i][1] for i in range(len(all_params))]

    # Keywords can be found from the names list
    # Note: For ResNet use "conv" as keyword otherwise it also gets the batchnorm layers
    # Note: for vgg use "weight" as keyword
    kw = "weight" #"conv"
    # kw = "conv"
    w_ind = np.array([1 if kw in names[n] and "bias" not in names[n] else 0 for n in range(len(names))]).astype(bool) #get layers with weights
    names_conv2d = np.array(names)[w_ind]
    numbers_conv2d = np.array(numbers)[w_ind]

    # if not os.path.isfile('resnet50_mio/archive_files/weights_check.pkl'):
    #     flat_layer_list = [flatten(numbers_conv2d[n].detach().numpy()) for n in range(len(numbers_conv2d))] #list of weights grouped by conv layer
    #     with open('resnet50_mio/archive_files/weights_check.pkl', 'wb') as f:
    #         pickle.dump(flat_layer_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    #         f.close()
    # else:
    #     with open('resnet50_mio/archive_files/weights_check.pkl', 'rb') as f:
    #         flat_layer_list = pickle.load(f)
    #         f.close()

    flat_layer_list = [flatten(numbers_conv2d[n].detach().numpy()) for n in range(len(numbers_conv2d))] #list of weights grouped by conv layer
    flat = flatten(flat_layer_list)
    # print(np.min(flat), np.max(flat))

    return flat


## Plot
# example_list = [np.random.randn(10) for n in range(16)]
# example_list = flat_layer_list
# print('nr of layers', len(example_list))

# dim1 = 10 # 4 #10 #rows
# dim2 = 5 #4 #5 #cols

# fig, axs = plt.subplots(dim1, dim2, figsize=(15,20)) #16 conv layers
# for a in range(dim1):
#     for b in range(dim2):
#         # print(a*dim2 + b)
#         if a*dim2+b >= len(example_list):
#             break
#         axs[a][b].hist(example_list[a*dim2+b])
#         axs[a][b].set_title('layer ' + str(a * dim2 + b))

# axs[0][0].set_xlabel('conv2d weight values')
# axs[0][0].set_ylabel('Frequency')

# # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.subplots_adjust(top = 0.95, bottom=0.05, hspace=1., wspace=0.6) #create more space between rows
# plt.show()



# ##
# # pic_folder = 'C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/FI experiments/pics/'
# # fig.savefig(pic_folder + 'weight_distribution.png', dpi=150)


# ## Which bits can be flipped in weights?
# flat = flatten(flat_layer_list)
# print(np.min(flat), np.max(flat))
# flat_abs = [abs(x) for x in flat]
# print(np.min(flat_abs), np.max(flat_abs))
# # 7.58*1e-13 - 1.32; bit 1 always free, bit 2 always occ (1e-19), bit 3 occ or bit 4+5 both occ


# ##
def convert_from_float32(orig_value):
    save_type = orig_value.dtype
    float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', orig_value.item()))
    return float_to_bin


# def single_bit_flip(orig_value, bit_pos):
#     save_type = orig_value.dtype
#     float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', orig_value.item()))
#     assert (len(float_to_bin) - 1 >= bit_pos), "Bit position {} too large for size of value: {}" \
#         .format(bit_pos, len(float_to_bin))
#     # print("original value: {}".format(orig_value))
#     # print("original bitmap: {}".format(float_to_bin))
#     if float_to_bin[bit_pos] == "1":
#         new_float = float_to_bin[:bit_pos] + '0' + float_to_bin[bit_pos + 1:]
#     else:
#         new_float = float_to_bin[:bit_pos] + '1' + float_to_bin[bit_pos + 1:]
#     # print("changed bitmap: {}".format(new_float))
#     f = int(new_float, 2) #converts to binary with base 2
#     bin_to_float = struct.unpack('f', struct.pack('I', f))[0]
#     return torch.tensor(bin_to_float, dtype=save_type)

# #
# flt = torch.tensor(1.0300)
# # bn = single_bit_flip(flt, 1)
# bn = convert_2_float32(flt)
# print(flt, bn)


# bit_freq = []
# for n in range(len(flat)):
#     bn = convert_2_float32(flat[n])
#     print(bn[1:9])

#     if n > 10:
#         break



# Example ------------------
# flat = get_flat_list(net)
# bit_freq = [0, 0, 0, 0, 0, 0, 0, 0, 0] #0+8 exp bits
# for n in range(len(flat)):
#     bn = convert_from_float32(flat[n])
#     for u in range(8+1):
#         # print(bn[u+1])
#         if bn[u] == "1":
#             bit_freq[u] += 1


# bit_freq = np.array(bit_freq)/len(flat)
# ##
# fig = plt.figure()
# x = np.array(range(8+1)) #+1

# plt.bar(x, bit_freq)

# ax = fig.gca()
# ax.set_xlabel('bit position')
# ax.set_ylabel('p(is 1)')

# # Plot labels in correct scientific notation
# round_to = 2
# for i, v in enumerate(bit_freq):
#     ax.text(i + - 0.25, v + .01, np.round(v, round_to))  # , color='blue', fontweight='bold')

# # plt.savefig('resnet50_mio/experiments/weights.pdf')
# # plt.savefig('plots/yolo_weights_kitti.pdf')
# # plt.savefig('plots/yolo_weights_coco.pdf')
# plt.savefig('plots/retinanet_weights_coco.pdf')
