# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import os, sys
import glob
from torchvision import models
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pytorchfi.test.unit_tests.test_neuron_errormodels import TestNeuronErrorModels as Testneuronerrormodels
from pytorchfi.test.unit_tests.test_weight_errormodels import TestWeightErrorModels as Testweighterrormodels
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
from pytorchfi.pytorchfi.core import fault_injection as pfi_core
# from pytorchfi.errormodels import single_bit_flip_func as pfi_core_func
from pytorchfi.pytorchfi.errormodels import (
    random_inj_per_layer,
    random_inj_per_layer_batched,
    random_neuron_inj,
    random_neuron_inj_batched,
    random_neuron_single_bit_inj,
    random_neuron_single_bit_inj_batched,
    random_weight_inj,
)
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset, default_loader, IMG_EXTENSIONS
from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

def Imagenet_class_distribution():
    ## under development
    json_read = "/home/qutub/PhD/data/datasets/imagenet/imagenet_class_index.json"
    class_idx = json.load(open(json_read))
    Label_dict = {}
    synset_id = {}
    for i in range(len(list(class_idx.keys()))):
        # class_mapping = class_idx[list(class_idx.keys())[i]][0]
        synset_label = class_idx[list(class_idx.keys())[i]][0]
        class_label = class_idx[list(class_idx.keys())[i]][1]
        Label_dict[class_label] = i
        synset_id[synset_label] = i
    datasplit = [len(os.listdir(os.path.join(path,dir))) for dir in dirs]
    pass

def random_class_sampling(random_classes=20):
    ## under development
    dest = "/nwstore/datasets/ImageNet/ILSVRC/random20classes_FI"
    root = "/nwstore/datasets/ImageNet/ILSVRC/sort"
    sampling = random.choices(os.listdir(root), k=random_classes)
    for sample in sampling:
        dest_dir = os.path.join(dest,sample)
        source_dir = os.path.join(root, sample)
        shutil.copytree(source_dir,dest_dir)

def imagenet_idx2label():
    json_read = "/nwstore/datasets/ImageNet/imagenet_class_index.json"
    # json_read = "/media/qutub/nwstore/datasets/ImageNet/imagenet_class_index.json"
    class_idx = json.load(open(json_read))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return idx2label, class_idx


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        if root == 'train':
            root = "/nwstore/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train"
            # root = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train"
            self.datasplit = 'train'
        elif root == 'val':
            # root = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/sort"
            root = "/nwstore/datasets/ImageNet/ILSVRC/random20classes_FI"
            # root = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/random20classes_FI"
            self.datasplit = 'val'
        else:
            raise Exception('Invalid root directory')

        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        json_read = "/nwstore/datasets/ImageNet/imagenet_class_index.json"
        # json_read = "/media/qutub/nwstore/datasets/ImageNet/imagenet_class_index.json"
        class_idx = json.load(open(json_read))
        Label_dict = {}
        synset_id = {}
        for i in range(len(list(class_idx.keys()))):
            # class_mapping = class_idx[list(class_idx.keys())[i]][0]
            synset_label = class_idx[list(class_idx.keys())[i]][0]
            class_label = class_idx[list(class_idx.keys())[i]][1]
            Label_dict[class_label] = i
            synset_id[synset_label] = i
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        # class_to_idx = {classes[i]: i for i in range(len(classes))}
        if self.datasplit == 'train':
            class_to_idx = {classes[i]: synset_id[classes[i]] for i in range(len(classes))}
        elif self.datasplit == 'val':
            class_to_idx = {classes[i]: Label_dict[classes[i]] for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
    
class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

def prep_val_imagenet():
    _, class_idx = imagenet_idx2label() # _ = idx2label
    imagenet_groundtruth = "/home/qutub/PhD/data/datasets/imagenet/ILSVRC2012_validation_ground_truth.txt"
    imagenet_mapping = "/nwstore/datasets/imagenet/ILSVRC2012_mapping.txt"
    imagenet_labels = "/home/qutub/PhD/data/datasets/imagenet/labels.txt"

    label_dict = {}
    Label_dict = {}
    for i in range(len(list(class_idx.keys()))):
        class_mapping = class_idx[list(class_idx.keys())[i]][0]
        class_label = class_idx[list(class_idx.keys())[i]][1]
        label_dict[class_mapping] = class_label
        Label_dict[class_label] = i

    with open(imagenet_mapping) as f:
        mapping = [line.strip() for line in f.readlines()]

    VAL_CLASS_PATH = "/nwstore/datasets/imagenet/ILSVRC2012_validation_ground_truth.txt"
    VAL_DATA_PATH = "/nwstore/datasets/imagenet/val/"
    VAL_ORI_DATA_PATH = "/nwstore/datasets/imagenet/images/*.JPEG"
    
    val_class = []
    with open(VAL_CLASS_PATH) as val_file:
        rows = val_file.readlines()
        for row in rows:
            row = int(row.strip())
            val_class.append(row)
    val_files = glob.glob(VAL_ORI_DATA_PATH)
    for file in val_files:
        seq_num = int(file.split("/")[-1].split("_")[-1].split(".")[0])
        class_id = val_class[seq_num - 1]
        class_mapping = mapping[class_id - 1].split()[1]
        class_name = label_dict[class_mapping]
    
        if not os.path.isdir(VAL_DATA_PATH + class_name):
            os.mkdir(VAL_DATA_PATH + class_name)
    
        os.rename(file, VAL_DATA_PATH + class_name + "/" + file.split("/")[-1])

    ## for validation dataset
    # import ntpath
    # import glob
    # import json
    # import xml.etree.ElementTree as ET
    # import shutil
    # VAL_ORI_DATA_PATH = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/Data/CLS-LOC/val/*.JPEG"
    # VAL_GROUND_TRUTH = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/Annotations/CLS-LOC/val"
    # VAL_DATA_PATH = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/sort/"
    # # VAL_DATA_PATH = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/random20classes_FI"
    # val_data_files = glob.glob(VAL_ORI_DATA_PATH)
    # json_read = "/home/qutub/PhD/data/datasets/imagenet/imagenet_class_index.json"
    # class_idx = json.load(open(json_read))
    # Label_dict = {}
    # print('works')
    # for i in range(len(list(class_idx.keys()))):
    #     class_mapping = class_idx[list(class_idx.keys())[i]][0]
    #     class_label = class_idx[list(class_idx.keys())[i]][1]
    #     Label_dict[class_mapping] = i, class_label


    # for file in val_data_files:
    #     file_basename = os.path.splitext(os.path.basename(file))[0]
    #     class_name = class_mapping[file_basename][1]

    #     if not os.path.isdir(VAL_DATA_PATH + class_name):
    #         os.mkdir(VAL_DATA_PATH + class_name)

    #     # os.rename(file, VAL_DATA_PATH + class_name + "/" + file.split("/")[-1])    
    #     shutil.copyfile(file, VAL_DATA_PATH + class_name + "/" + file.split("/")[-1])

    # import ntpath
    # import glob
    # import json
    # import xml.etree.ElementTree as ET
    # import shutil
    # import xml.etree.ElementTree as ET
    
    # import shutil
    # source = "/nwstore/sayanta/random20classes_FI"
    # destination = "/nwstore/sayanta/random20classes_FI_test"

    # os.makedirs(destination, exist_ok=True)
    # # code to move the files from sub-folder to main folder.
    # folders = os.listdir(source)
    # for folder in folders:
    #     _folder = os.path.join(source, folder)
    #     list_folder = os.listdir(_folder)
    #     for _file in list_folder:
    #         file_name = os.path.join(_folder, _file)
    #         shutil.copy(file_name, destination )
    # print("Files Moved")
    
    # def extract_synset_id(file):
    #     synset_id = []
    #     xmltree = ET.parse(file)
    #     objects = xmltree.findall("object")
    #     for object_iter in objects:
    #         _synset_id = object_iter.find("name")
    #         synset_id.append(_synset_id.text)
    #     return synset_id[0]
        
    # VAL_ORI_DATA_PATH = "/nwstore/datasets/ImageNet/ILSVRC/random20classes_club/*.JPEG"
    # VAL_GROUND_TRUTH = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/Annotations/CLS-LOC/val"
    # # VAL_DATA_PATH = "/media/qutub/nwstore/datasets/ImageNet/ILSVRC/random20classes_FI"
    # val_data_files = glob.glob(VAL_ORI_DATA_PATH)
    # json_read = "/nwstore/datasets/ImageNet/imagenet_class_index.json"
    # class_idx = json.load(open(json_read))
    # Label_dict = {}


    # annotations ="/nwstore/datasets/ImageNet/ILSVRC/Annotations/CLS-LOC/val/"

    # for i in range(len(list(class_idx.keys()))):
    #     class_mapping = class_idx[list(class_idx.keys())[i]][0]
    #     class_label = class_idx[list(class_idx.keys())[i]][1]
    #     Label_dict[class_mapping] = i, class_label

    # val_data_ann = {}
    # for file in val_data_files:
    #     file_basename = os.path.splitext(os.path.basename(file))[0]
    #     ann_file = os.path.join(annotations, file_basename + '.xml')
        
    #     synset_id = extract_synset_id(ann_file)

    #     val_data_ann[file_basename] = synset_id, Label_dict[synset_id][0], Label_dict[synset_id][1]

    # file_path = "/nwstore/datasets/ImageNet/ILSVRC/random20classes_club_labels.txt"
    # with open(file_path, 'w') as fp:
    #     json.dump(val_data_ann, fp)

    # file_path = "/nwstore/datasets/ImageNet/ILSVRC/random20classes_club_label_ids.txt"
    # val_data_ann_text = ['{} {}'.format(key, val_data_ann[key][1]) for key in val_data_ann.keys()]
    # with open(file_path, 'w') as f:
    #     for line in val_data_ann_text:
    #         f.write(line)
    #         f.write('\n')

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

    def get_datasetfp(self):
        return [sample[0] for sample in self.samples]

def imagenet_Dataloader(root='val', batch_size=1, shuffle=False, transform=transform, sampling=False, sampleN=None):

    _, class_idx = imagenet_idx2label() # _ = idx2label
    Label_dict = {}
    for i in range(len(list(class_idx.keys()))):
        # class_mapping =  class_idx[list(class_idx.keys())[i]][0]
        class_label = class_idx[list(class_idx.keys())[i]][1]
        Label_dict[class_label] = i
    dataset = ImageFolderWithPaths(root=root, transform=transform)
    datasetfp = dataset.get_datasetfp()
    if sampling:
        if sampleN is None:
            sampleN = 20
        
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), int(sampleN*len(dataset)/100), replace=False))
        datasetfp = dataset.dataset.get_datasetfp()

    imageNet_VAL_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return len(dataset), imageNet_VAL_dataloader, datasetfp

class Imagenet_dataloader():

    def __init__(self, dl_attr:TEM_Dataloader_attr):
        """
        Args:
        dataset_type (str) : Type of dataset shall be used - train, test, val... Defaults to val.
        batch_size   (uint): batch size. Defaults to 1.
        shuffle      (str) : Shuffling of the data in the dataloader. Defaults to False.
        sampleN      (uint): Percentage of dataset lenth to be sampled. Defaults to None.
        transform    (obj) : Several transforms composed together. Defaults to None.
        num_workers  (uint): Number of workers to be used to load the dataset. Defaults to 1.
        device       (uint): Cuda/Cpu device to be used to load dataset. Defaults to Cuda 0 if available else CPuU.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        # prep_val_imagenet()
        super().__init__()
        self.dl_attr         = dl_attr
        self.device          = dl_attr.dl_device if self.dl_attr.dl_device is not None else torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        self.dl_attr.dl_device  = self.device
        if isinstance(self.dl_attr.dl_sampleN, float) and self.dl_attr.dl_sampleN>1:
            self.dl_attr.dl_sampleN = int(self.dl_attr.dl_sampleN)

        ## TODO self.classes and self.class_idx is probably not the class label names and idx. Need to confirm. This may be is extracted
        ## and stored in Label_dict
        self.classes, self.class_idx = imagenet_idx2label() # _ = idx2label
        Label_dict = {}
        for i in range(len(list(self.class_idx.keys()))):
            # class_mapping =  class_idx[list(class_idx.keys())[i]][0]
            class_label = self.class_idx[list(self.class_idx.keys())[i]][1]
            Label_dict[class_label] = i
        self.dataset = ImageFolderWithPaths(root=self.dl_attr.dl_dataset_type, transform=self.dl_attr.dl_transform)
        self.datasetfp = self.dataset.get_datasetfp()

        if self.dl_attr.dl_sampleN <= 1:
            self.dl_attr.dl_sampleN = self.dl_attr.dl_sampleN*len(self.dataset)

        if self.dl_attr.dl_sampleN is not None:
            self.dataset = torch.utils.data.Subset(self.dataset, np.random.choice(len(self.dataset), self.dl_attr.dl_sampleN, replace=False))
            self.datasetfp = self.dataset.dataset.get_datasetfp()

        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.dl_attr.dl_batch_size, shuffle=self.dl_attr.dl_shuffle, num_workers=self.dl_attr.dl_num_workers)
        self.datagen_iter = iter(self.data_loader)
        self.dataset_length = len(self.dataset)
        self.datagen_iter_cnt = 0

        self.data_incoming = True
        self.images = -1
        self.labels = -1
        print("ImageNet Dataset loaded in {}; {} dataset - Length : {}".format(self.device, self.dl_attr.dl_dataset_type, len(self.dataset)))

    def datagen_itr(self):
        if self.datagen_iter_cnt < self.dataset_length:
            self.data_incoming = True
            self.images, self.labels, self.image_path = self.datagen_iter.next()
            self.image_path = [self.datasetfp.index(i) for i in self.image_path]
            self.curr_batch_size = len(self.images)
            self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
            self.images = self.images.to(self.device)
            self.labels = self.labels.to(self.device)
            if self.datagen_iter_cnt == self.dataset_length: # added to stop repeat of last batch of images
                self.data_incoming = False
        else:
            self.data_incoming = False

    def datagen_reset(self):
        self.images = -1
        self.labels = -1
        self.datagen_iter = iter(self.data_loader)
        self.data_incoming = True
        self.datagen_iter_cnt = 0
