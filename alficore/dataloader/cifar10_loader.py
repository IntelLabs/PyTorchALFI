"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.

[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np

from .abs_loader import Abstract_Loader
from .utils_data import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from ..ptfiwrap_utils.utils import *

class Cifar10_Loader(Abstract_Loader):
    def __init__(self):
        self.dataset_size = 0

    def get_test_loader(self, data_dir,
                        batch_size,
                        shuffle=True,
                        valid_size=0.1,
                        num_workers=1,
                        pin_memory=False,
                        std = None,
                        mean = None):
        """
        Utility function for loading and returning a multi-process
        test iterator over the CIFAR-10 dataset.

        If using CUDA, num_workers should be set to 1 and pin_memory to True.

        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - shuffle: whether to shuffle the dataset after every epoch.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
          True if using GPU.

        Returns
        -------
        - data_loader: test set iterator.
        """

        data_dir = rejoin_path(data_dir)

        normalize = transforms.Normalize(
             mean=[0.5, 0.5, 0.5],
             std=[0.5, 0.5, 0.5],
        )

        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((valid_size >= 0) and (valid_size <= 1)), error_msg


        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )
        self.dataset_size = len(dataset)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return data_loader

    def get_dataset_size(self):
        return self.dataset_size