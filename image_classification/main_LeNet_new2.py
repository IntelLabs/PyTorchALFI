import sys
sys.path.append("/home/fgeissle/ranger_repo/ranger")
from LeNet_orig import LeNet_orig
from alficore.dataloader.mnist_loader import MNIST_dataloader
from alficore.ptfiwrap_utils.evaluate import evaluateAccuracy, extract_ranger_bounds
from alficore.ptfiwrap_utils.helper_functions import getdict_ranger, get_savedBounds_minmax
from alficore.evaluation.visualization import *
from pytorchfi.pytorchfi import core
# from util.visualization import plot_fmap_layer, customize_cmap
# from alficore.ptfiwrap_utils.evaluate import get_Ranger_protection, get_Ranger_protection_trivial
# from alficore.ptfiwrap_utils.hook_functions import run_with_hooks_actlist
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec


# Use for visualization -----------------------------------------------------------------


# LeNet code based on:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html and following pages
# https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html
# https://pypi.org/project/pytorchfi/#usage
# https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html
# https://www.kaggle.com/vincentman0403/pytorch-v0-3-1b-on-mnist-by-lenet


def get_labels(outputs, top_nr, classes):
    predictions = torch.nn.functional.softmax(outputs, dim=1)  # normalized tensors
    top_val, top_ind = torch.topk(predictions, top_nr, dim=1)  # it is by default sorted
    predicted = []
    for n in range(test_batch_size):
        predicted.append([classes[top_ind[n, m]] for m in range(top_nr)])
    return predicted


def run_FI(net, channel_nr, test_batch_size, conv_i, k, c_i, h_i, w_i, inj_value_i, images):

    pcore = core.fault_injection(net, c=channel_nr, h=32, w=32,
                                 batch_size=test_batch_size)  # automate the dimensions here.
    fi_net = pcore.declare_weight_fi(conv_num=conv_i, k=k, c=c_i, h=h_i, w=w_i, value=inj_value_i)
    # fi_net = pcore.declare_weight_fi(conv_num=(conv_i, conv_i), k=(k, k+1), c=(c_i,c_i), h=(h_i, h_i), w=(w_i, w_i), value=(inj_value_i, 20))
    fi_net.eval()
    # print('check weights', fi_net.__getattr__('0').weight[k][c_i][h_i][w_i])

    # Predict ----------------------------
    with torch.no_grad():  # turn off gradient calculation
        outputs, list_act_in, list_act_out = run_with_hooks_actlist(net, images)  # input and output of Ranger layers
        fi_outputs, fi_list_act_in, fi_list_act_out = run_with_hooks_actlist(fi_net, images)

    _, predicted = torch.max(outputs, 1)  # select highest value (~instead of softmax)
    _, fi_predicted = torch.max(fi_outputs, 1)  # select highest value

    predicted_labels = get_labels(outputs, top_nr, classes)
    fi_predicted_labels = get_labels(fi_outputs, top_nr, classes)
    print('pred no fault', predicted_labels, 'pred fault', fi_predicted_labels)

    return predicted_labels, fi_predicted_labels, outputs, fi_outputs, list_act_out, fi_list_act_out


## Load dataset:
train_batch_size = 4
test_batch_size = 4 #note: if test_batchsize > 1 and only one N=0 is plotted then can be confusing
dataset = MNIST_dataloader()
train_loader = dataset.data_loader
# (train_loader, test_loader, classes) = load_data_mnist(train_batch_size, test_batch_size)


## Get weights:
channel_nr = 1 #1 for mnist, 3 for cifar
net_orig = LeNet_orig(color_channels=channel_nr)

# Train weights:
# col_ch = 1 #greyscale for mnist, color for cifar
# net = LeNet_ranger3(color_channels=col_ch, bounds=None)
net = net_orig
from image_classification.Net_trainer2 import Net_trainer2
trainer = Net_trainer2()
lrt = 0.001 #learning rate, ~0.001 (mnist)
mmt = 0.9 #momentum ~ 0.9 (mnist)
nr_epochs = 10
weightfile_name = 'lenet5-mnist'
PATH = trainer.train(net, train_loader, weightfile_name, lrt, mmt, nr_epochs)
print('training done')
# sys.exit()

# Or just get path to pretrained weights:
PATH = './state/lenet5-mnist_backup.pth' #load weights
net_orig.load_state_dict(torch.load(PATH)) #load the pretrained weights
net_orig.eval()



## Get Ranger bounds

# # If bounds dont exist yet extract them:
# p = 0.2
# (bounds_train_loader, _) = load_Subtraindata_mnist(train_batch_size, p)
# net_for_bounds = net_orig
# # net_for_bounds.load_state_dict(getdict_ranger(PATH, net_for_bounds)) #load the pretrained weights
# # net_for_bounds.eval()
# name = "LeNet_bounds_mnist_act" #name of saved file
# act_input, act_output = extract_ranger_bounds(bounds_train_loader, net_for_bounds, name) #gets also saved automatically
# print('check Ranger input', len(act_input))
# print('check Ranger output', len(act_output))


# If bounds exist load them:
bnds = get_savedBounds_minmax("./bounds/LeNet_bounds_mnist_act.txt")
print('Bounds used:', bnds)

net, bnds_mod = get_Ranger_protection(net_orig, bnds)
bnds = bnds_mod
net.eval()

# For comparison
net_orig, _ = get_Ranger_protection_trivial(net_orig, bnds) #add trivial ranger bounds for plotting, optional



## Example inference
dataiter = iter(test_loader) #get one random pic batch (four images)
images, labels = dataiter.next()
top_nr = 3

# # Only inference without faults:
# with torch.no_grad():
#     output = net_orig(images)
#
# predicted = get_labels(output, top_nr, classes)
# # predictions = torch.nn.functional.softmax(output, dim=1)  # normalized tensors
# # top_val, top_ind = torch.topk(predictions, top_nr, dim=1)  # it is by default sorted
# # predicted = []
# # for n in range(test_batch_size):
# #     predicted.append([classes[top_ind[n, m]] for m in range(top_nr)])
#
# gt_labels = [classes[n] for n in labels.tolist()]  # original labels
#
# col_ch = channel_nr
# fig_res = imshow_labels(images, predicted, gt_labels, col_ch, test_batch_size)




## Count parameters
# summary(net, torch.zeros(images.size()), show_input=True)





## Demonstrate fault visually in fmaps -------
# FCC are reshaped here for better visualization: 120=10x12 split here, 84 = 21*4; 10=1*10

# Create model with fault (weight or neuron injection) --------------------------
# pcore = core.fault_injection(net, c=channel_nr, h=32, w=32, batch_size=test_batch_size) #automate the dimensions here.
# pcore_comp = core.fault_injection(net_orig, c=channel_nr, h=32, w=32, batch_size=test_batch_size) #automate the dimensions here.

# Weight fault
conv_i = 0 #0 = 1.conv, 1=2.conv ... 4 = fc3.
inj_value_i = 1000.0
if conv_i < 2:
    k = 1 #kernel (see in fmap), number of filters
    c_i = 0 #channel (~third dim)
    h_i = 0 #height
    w_i = 0 #width
else:
    k = -1 #kernel (see in fmap), number of filters
    c_i = -1 #channel (~third dim)
    h_i = 4 #height
    w_i = 10 #width



predicted_labels, fi_predicted_labels, outputs, fi_outputs, list_act_out, fi_list_act_out = run_FI(net, channel_nr, test_batch_size, conv_i, k, c_i, h_i, w_i, inj_value_i, images)

predicted_labels_orig, fi_predicted_labels_orig, outputs_orig, fi_outputs_orig, list_act_out_orig, fi_list_act_out_orig = run_FI(net_orig, channel_nr, test_batch_size, conv_i, k, c_i, h_i, w_i, inj_value_i, images)


# Plot fmaps ----------------------------------
# Note: plot only one column each and then for different ranger types?
# Define which layers to plot
N = 3 # select batch to plot #TODO:
layer = [0, 2, 5, 6] #layers to plot
add_output_plot = 1 #0 #should the output tensor be plotted too or not?


# Reshape linear fmaps for better visuality
# layer_sizes = [np.array(x.shape) for x in list_act_out] #automated version but reshape manually the last layers!
layer_sizes = [(test_batch_size, 6, 28, 28), (test_batch_size, 6, 14, 14), (test_batch_size, 16, 10, 10), (test_batch_size, 16, 5, 5), (test_batch_size, 20, 20), (test_batch_size, 10, 12), (test_batch_size, 4, 21)]

nr_fmaps = [layer_sizes[x][1] if len(layer_sizes[x])>3 else 1 for x in range(len(layer_sizes)) ]
nr_fmaps_rearr = [[2,3], [2,3], [4,4], [4,4], [1,1], [1,1], [1,1]]

ylimmax = inj_value_i #choose end of colorbar

fig = plot_fmaps_grid(N, list_act_out, fi_list_act_out, outputs, fi_outputs, layer, layer_sizes, nr_fmaps_rearr, add_output_plot, bnds, ylimmax)
fig_orig = plot_fmaps_grid(N, list_act_out_orig, fi_list_act_out_orig, outputs_orig, fi_outputs_orig, layer, layer_sizes, nr_fmaps_rearr, add_output_plot, bnds, ylimmax) #optional

gt_labels = [classes[n] for n in labels.tolist()]  # original labels
fig_res = imshow_labels(images, predicted_labels, fi_predicted_labels, gt_labels, channel_nr, test_batch_size)




##
# # Save pics ---
# # Note: execute savefig twice for correct format
#
# # name = 'test'
# # name = 'LeNet_vis_noRanger'
# name = 'LeNet_vis_Ranger'
# # name = 'LeNet_vis_Clipping'
# # name = 'LeNet_vis_Backflip'
# # name = 'LeNet_vis_Rescale'
# # name = 'LeNet_vis_avfmap'
# # png option 1,2 (sometimes have to execute twice to get correct pic)
# fig.savefig("C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/Dependable AI WS2/VIsualisation Ranger/" + name + ".png", quality=100, bbox_inches = 'tight', format = 'png', dpi= 300)
# fig_orig.savefig("C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/Dependable AI WS2/VIsualisation Ranger/" + name + "_comp.png", quality=100, bbox_inches = 'tight', format = 'png', dpi= 300)
# fig_res.savefig("C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/Dependable AI WS2/VIsualisation Ranger/" + name + "_pred.png", quality=100, bbox_inches = 'tight', format = 'png') #dpi here messes up text
# # plt.savefig("test_plot.pdf", bbox_inches = 'tight', pad_inches = 0.1, format='pdf')


# # pdf option
# fig.savefig("C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/Dependable AI WS2/VIsualisation Ranger/" + name + ".pdf", quality=100, bbox_inches = 'tight', format = 'pdf')
# fig_res.savefig("C:/Users/fgeissle/OneDrive - Intel Corporation/Desktop/Dependable AI WS2/VIsualisation Ranger/" + name + "_pred.pdf", quality=100, bbox_inches = 'tight', format = 'pdf')



## Accuracy
# top_nr = 1 # top N results are compared
# ranger_activity = True #flag whether the number of active ranger layers should be measured
# correct, total, act_ranger_layers = evaluateAccuracy(net, test_loader, classes, classes, top_nr, ranger_activity)
# print('map', correct, total)
# print('ranger active', sum(act_ranger_layers))

# Map:
# top 1: 99.2 %

