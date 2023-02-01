from attic.VGG16_ranger import VGG16_ranger
from util.dataset_loaders import *
from util.visualization import *
from util.evaluate import *
import torchvision
from util.hook_functions import run_with_hooks
import struct

# modelsummary package taken from:
# https://github.com/graykode/modelsummary
# Url for weight download here:
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py

# todo universal visualization fmaps



##

def single_bit_flip(orig_value, bit_pos):
    save_type = orig_value.dtype
    float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', orig_value.item()))
    assert (len(float_to_bin) - 1 >= bit_pos), "Bit position {} too large for size of value: {}" \
        .format(bit_pos, len(float_to_bin))
    print("original value: {}".format(orig_value))
    print("original bitmap: {}".format(float_to_bin))
    if float_to_bin[bit_pos] == "1":
        new_float = float_to_bin[:bit_pos] + '0' + float_to_bin[bit_pos + 1:]
    else:
        new_float = float_to_bin[:bit_pos] + '1' + float_to_bin[bit_pos + 1:]
    print("changed bitmap: {}".format(new_float))
    f = int(new_float, 2)
    bin_to_float = struct.unpack('f', struct.pack('I', f))[0]
    return torch.tensor(bin_to_float, dtype=save_type)


def back_bit_flip(obs_value, bnd_orig):
    """
    :param obs_value: an observed tensor value that is *above* the bnd_origin.
    :param bnd_orig: Ranger bound for this layer
    :return: recovered tensor value that is within the bounds.
    """
    if not bnd_orig: #if bound 0, or None
        return obs_value

    if obs_value < 2. * bnd_orig:
        # flip is in the last exp bit (8) or in a fraction bit
        a_rec = torch.tensor(bnd_orig, dtype=obs_value.dtype)
    else:
        # flip is in the higher exponential bits (1-7)
        # Get exponential bits that are now 1 (thus have potentially been flipped)
        float_to_bin = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', obs_value.item()))
        msb = []
        for bt in range(1, 8):  # (from 1-7) skip sign bit 0 and forget last exp bit 8 (since its only x2),
            if float_to_bin[bt] == '1':
                msb.append(bt)
        print('exp bits that are one:', msb)

        # Associated flip factors
        flip_factors = np.array([2 ** (2 ** (8 - n)) for n in msb])
        print('flip factors', ["{:e}".format(x) for x in flip_factors])
        est_true_flip = float(obs_value) / bnd_orig #thats the *minimum* scale factor of the error
        print('min scaling by error', est_true_flip)
        backflip_pos_list = np.array(msb)[flip_factors > est_true_flip]
        if backflip_pos_list:
            # backflip_pos = backflip_pos_list[0] #choose the largest one
            a_rec = single_bit_flip(obs_value, backflip_pos_list[0])
        else:
            a_rec = torch.tensor(bnd_orig, dtype=obs_value.dtype) #if it was oob before (very unlikely)

        print('possible bit pos', backflip_pos_list, 'best match', backflip_pos_list[0])

    return a_rec




#
# orig = torch.tensor(7.9445e+27)
bnd_orig = 124.
# bitflip_pos = 7
# obs_value = single_bit_flip(torch.tensor(orig), bitflip_pos)

obs_value = torch.tensor(7.9445e+27)
print('observed', obs_value)

a_rec = back_bit_flip(obs_value, bnd_orig)
print('recovered', a_rec)
a_rec2 = single_bit_flip(obs_value, 1)
print('recovered', a_rec2)


## Load dataset:
test_batch_size = 4
(test_loader, classes_orig, classes_sorted) = load_data_imageNet_piece(test_batch_size)

##
# with open('classes_sorted.txt', 'w') as f:
#     for item in classes_sorted:
#         f.write("%s\n" % item)
# with open('classes_orig.txt', 'w') as f:
#     for item in classes_orig:
#         f.write("%s\n" % item)


## Path to weights:
PATH = './state/vgg16-397923af.pth' #load weights


## Get/Extract Ranger bounds:
# # If they dont exist extract them here:
# net_for_bounds = VGG16_ranger(num_classes=1000, bounds=None)
# net_for_bounds.load_state_dict(getdict_ranger(PATH, net_for_bounds)) #load the pretrained weights
# net_for_bounds.eval()
# name = "Vgg16_bounds_ImageNet1000" #name of saved file
# act_input, act_output = extract_ranger_bounds(test_loader, net_for_bounds, name) #gets also saved automatically
# print('check Ranger input', act_input)
# print('check Ranger output', act_output)


# If bounds exist load them:
bnds = get_savedBounds_minmax("./bounds/Vgg16_bounds_ImageNet1000_backup.txt")
print('Bounds used:', bnds)
net = VGG16_ranger(num_classes=1000, bounds=bnds)
net.load_state_dict(getdict_ranger(PATH, net)) #load the pretrained weights
net.eval()





## Accuracy
# top_nr = 5 # top N results are compared
# ranger_activity = False #flag whether the number of active ranger layers should be measured
# correct, total, act_ranger_layers = evaluateAccuracy(net, test_loader, classes_sorted, classes_orig, top_nr, ranger_activity)
# print('map', correct, total)
# print('ranger active', act_ranger_layers)
#
# # Map:
# # top 1: 72.3 %
# # top 3: 86.7 %
# # top 5: 90.9 %






## Example inference
dataiter = iter(test_loader) #get one random pic batch (four images)
images, labels = dataiter.next()
top_nr = 3

with torch.no_grad():
    # output = net(images)

    output, activated = run_with_hooks(net, images)

print('activated', activated)

predictions = torch.nn.functional.softmax(output, dim=1)  # normalized tensors
top_val, top_ind = torch.topk(predictions, top_nr, dim=1)  # it is by default sorted
predicted = []
for n in range(test_batch_size):
    predicted.append([classes_orig[top_ind[n, m]] for m in range(top_nr)])

gt_labels = [classes_sorted[n] for n in labels.tolist()]  # original labels

col_ch = 3
# imshow_labels(images, predicted, gt_labels, col_ch, test_batch_size)
# imshow_labels(img, pred_labels, fi_pred_labels, gt_labels, col_ch, batch_size)

# # plt.savefig("test_plot.pdf", bbox_inches = 'tight', pad_inches = 0.1, format='pdf')
# plt.savefig("test_plot.png", quality=100, format = 'png')


## Count parameters
# summary(net, torch.zeros(images.size()), show_input=True)
# # summary(net, torch.zeros(images.size()), show_hierarchical=True)


## Save to onnx
#
# bnds = get_savedBounds_minmax("./bounds/Vgg16_bounds_dog.txt")
# net = VGG16_ranger(num_classes=1000, bounds=bnds)
#
# PATH = './state/vgg16-397923af.pth'
# net.load_state_dict(getdict_ranger(PATH, net)) #load the pretrained weights
# net.eval()
#
# onnx_file = 'VGG16_ranger2' + ".onnx"
# save_torch_to_onnx(net, input_batch[0].shape, onnx_file)


# Compare to original torchvision vgg16
PATH = './state/vgg16-397923af.pth'
net2 = torchvision.models.vgg16(pretrained=False, progress=True)
net2.load_state_dict(torch.load(PATH))
net2.eval()
list_ch2 = list(net2.children())
# onnx_file = 'VGG16' + ".onnx"
# save_torch_to_onnx(net2, test_batch_size[0].shape, onnx_file)
# print('onnx exported')



## Inference step with fault injection
from pytorchfi import core

# create model with injected fault
channel_nr = 3 #1 for mnist
pcore = core.fault_injection(net, c=channel_nr, h=32, w=32, batch_size=4)  # automate the dimensions here

# Weight fault ----
conv_i = 0 #or 2
k = 1  # kernel
c_i = 0  # channel
h_i = 2  # height
w_i = 2  # width
inj_value_i = 100.0
fi_net = pcore.declare_weight_fi(conv_num=conv_i, k=k, c=c_i, h=h_i, w=w_i, value=inj_value_i)


with torch.no_grad():
    output, activated = run_with_hooks(fi_net, images)

print('fi_activated', activated)

predictions = torch.nn.functional.softmax(output, dim=1)  # normalized tensors
top_val, top_ind = torch.topk(predictions, top_nr, dim=1)  # it is by default sorted
predicted = []
for n in range(test_batch_size):
    predicted.append([classes_orig[top_ind[n, m]] for m in range(top_nr)])

imshow_labels(images, predicted, gt_labels, col_ch, test_batch_size)





