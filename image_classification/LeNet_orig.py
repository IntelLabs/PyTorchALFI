import torch
import torch.nn as nn
import numpy as np
# from Ranger import Ranger

# Ranger paper:
# https://arxiv.org/abs/2003.13874

# LeNet code based on:
# https://engmrk.com/lenet-5-a-classic-cnn-architecture/
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html and following pages
# https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html
# https://pypi.org/project/pytorchfi/#usage
# https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html


class LeNet_orig(nn.Module):

    def __init__(self, color_channels=3):
        super(LeNet_orig, self).__init__()

        # Config
        self.ImageSize = (32, 32)
        self.InChannels = color_channels
        # self.Nr_rangers = 7
        # self.Bounds = np.reshape([None] * (self.Nr_rangers * 2), (self.Nr_rangers, 2))
        # if bounds is not None and len(bounds) >= self.Nr_rangers and len(bounds[0]) >= 2:
        #     self.Bounds = np.array(bounds)

        # Layers
        self.convBlock1 = self.make_conv_block(self.InChannels, 6, 5, 0) #Ranger 0,1
        self.convBlock2 = self.make_conv_block(6, 16, 5, 2) #Ranger 2,3
        # self.flatten_Ranger = Ranger(self.Bounds[4]) # Ranger 4
        self.fc1 = self.make_fcc_block(16 * 5 * 5, 120, 5) # Ranger 5
        self.fc2 = self.make_fcc_block(120, 84, 6)  # Ranger 6
        self.fc3 = nn.Linear(84, 10)



    def make_conv_block(self, in_channels, out_channels, kernel_size, ranger_start_nr):
        """
        Creates one convolutional block. Contains two Ranger layers.
        :param in_channels: nr of input channels
        :param out_channels: output channels for conv layer
        :param kernel_size: for conv layer
        :param ranger_start_nr: list in bounds list that the first Ranger layer gets
        :return: Container with convolutional block.
        """
        # Note: Conv2d: default stride = 1, default padding = 0
        # Note: MaxPool2d: default stride = kernel_size, padding = 0
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)]
        layers += [nn.ReLU(inplace=True)]
        # layers += [Ranger(self.Bounds[ranger_start_nr])]
        layers += [nn.MaxPool2d(kernel_size=2)] #stride is by default = kernel size
        # layers += [Ranger(self.Bounds[ranger_start_nr + 1])]  # stride is by default = kernel size

        return nn.Sequential(*layers)


    def make_fcc_block(self, in_channels, out_channels, ranger_start_nr):
        """
        Creates one fcc block. Contains one Ranger layer.
        :param in_channels: fcc input
        :param out_channels: fcc output
        :param ranger_start_nr: list in bounds list that the first Ranger layer gets
        :return: Container with fcc block.
        """
        layers = []
        layers += [nn.Linear(in_channels, out_channels)]
        layers += [nn.ReLU(inplace=True)]
        # layers += [Ranger(self.Bounds[ranger_start_nr])]

        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = torch.flatten(x, 1)
        # x = self.flatten_Ranger(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
