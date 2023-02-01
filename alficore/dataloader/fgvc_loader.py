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
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pytorchfi.test.unit_tests.test_neuron_errormodels import TestNeuronErrorModels as Testneuronerrormodels
from pytorchfi.test.unit_tests.test_weight_errormodels import TestWeightErrorModels as Testweighterrormodels
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




def prep_fgvc(typ='test'):

    _, label_dct = get_class_mapping(typ) 

    VAL_DATA_PATH = "/nwstore/datasets/FGVC/datasets/FGVC-aircraft/data/images/" + typ +"/"
    if not os.path.isdir(VAL_DATA_PATH):
        os.mkdir(VAL_DATA_PATH)
    VAL_ORI_DATA_PATH = "/nwstore/datasets/FGVC/datasets/FGVC-aircraft/data/images/*.jpg"
    
    val_files = glob.glob(VAL_ORI_DATA_PATH)

    print('Reorganizing images...')
    cnt = 0
    for file in val_files:
        print(cnt)
        cnt += 1
        # seq_num = int(file.split("/")[-1].split("_")[-1].split(".")[0])
        seq_num = file[-11:-4]
        if seq_num in label_dct.keys():
            class_name = label_dct[seq_num]
            class_name = class_name.replace("/", "") 
    
            if not os.path.isdir(VAL_DATA_PATH + class_name):
                os.mkdir(VAL_DATA_PATH + class_name)
        
            os.rename(file, VAL_DATA_PATH + class_name + "/" + file.split("/")[-1])


def get_label_info(filename):
    f = open(filename, "r")
    bounds = []
    if f.mode == 'r':
        contents = f.read().splitlines()
        bounds = [u.split(',') for u in contents]
    f.close()
    return bounds


def get_class_mapping(typ="test"):
    if not (typ == "test" or typ=="trainval"):
        print("Warning: Typ not allowed, choose test or trainval")
        sys.exit()
    info = get_label_info('/nwstore/datasets/FGVC/datasets/FGVC-aircraft/data/images_variant_' + typ + '.txt')
    info_fam = get_label_info('/nwstore/datasets/FGVC/datasets/FGVC-aircraft/data/images_manufacturer_' + typ + '.txt')
    if typ == "test":
        info_scores = get_label_info('/nwstore/datasets/FGVC/datasets/FGVC-aircraft/data/test.txt')
    elif typ == "trainval":
        info_scores = get_label_info('/nwstore/datasets/FGVC/datasets/FGVC-aircraft/data/train.txt')


    scs = []
    nms = []
    img_nrs = []
    nms_nrs = []
    for i in range(len(info)):
        var = info[i][0][8:]
        nr = info[i][0][:7]
        man = info_fam[i][0][8:]
        aircraft_name = (man + " " + var, info[i][0][:7])[0]

        score = int(info_scores[i][0][12:])
        if score not in scs:
            scs.append(score)
            nms.append(aircraft_name)
        img_nrs.append(nr)
        nms_nrs.append(aircraft_name)

    scs = list(np.array(scs) -1)
    dct = {scs[i]: nms[i] for i in range(len(scs))}
    label_dct = {img_nrs[i]: nms_nrs[i] for i in range(len(img_nrs))}
    return dct, label_dct



# class FGVC_dataloader_old():
#     def __init__(self, **kwargs):
#         """
#         Args:
#         dataset_type (str) : Type of dataset shall be used - train, test, val... Defaults to val.
#         batch_size   (uint): batch size. Defaults to 1.
#         shuffle      (str) : Shuffling of the data in the dataloader. Defaults to False.
#         sampleN      (uint): Percentage of dataset lenth to be sampled. Defaults to None.
#         transform    (obj) : Several transforms composed together. Defaults to None.
#         num_workers  (uint): Number of workers to be used to load the dataset. Defaults to 1.
#         device       (uint): Cuda/Cpu device to be used to load dataset. Defaults to Cuda 0 if available else CPuU.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.
#         """

#         self.dataset_type = kwargs.get("dataset_type", 'val')
#         self.batch_size = kwargs.get("batch_size", 1)
#         self.shuffle = kwargs.get("shuffle", False)
#         self.sampleN = kwargs.get("sampleN", None)
#         self.transform = kwargs.get("transform", None)
#         self.num_workers = kwargs.get("num_workers", 1)
#         self.device = kwargs.get("device", torch.device(
#         "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"))

#         self.images = -1
#         self.labels = -1

#         # Reorganize folder structure (only once)
#         # prep_fgvc("test")
#         # prep_fgvc("trainval")
        
#         ## TODO self.classes and self.class_idx is probably not the class label names and idx. Need to confirm. This may be is extracted
#         ## and stored in Label_dict
#         # self.classes, self.class_idx = imagenet_idx2label() # _ = idx2label
#         # Label_dict = {}
#         # for i in range(len(list(self.class_idx.keys()))):
#         #     # class_mapping =  class_idx[list(class_idx.keys())[i]][0]
#         #     class_label = self.class_idx[list(self.class_idx.keys())[i]][1]
#         #     Label_dict[class_label] = i
#         # self.dataset = ImageFolderWithPaths(root=self.dataset_type, transform=transform)

#         # self.datasetfp = self.dataset.get_datasetfp()
#         # if self.sampleN is not None:
#         #     self.dataset = torch.utils.data.Subset(self.dataset, np.random.choice(len(self.dataset), int(self.sampleN*len(self.dataset)/100), replace=False))
#         #     self.datasetfp = self.dataset.dataset.get_datasetfp()


#         root = '/nwstore/datasets/FGVC/datasets/FGVC-aircraft'
#         self.data_loader, self.dataset = read_dataset(input_size=448, batch_size=self.batch_size, root=root, type=self.dataset_type)

#         self.datagen_iter = iter(self.data_loader)
#         self.dataset_length = 10000 #len(self.dataset)
#         self.data_incoming = True
#         self.datagen_iter_cnt = 0
#         print("ImageNet Dataset loaded in {}; {} dataset ".format(self.device, self.dataset_type ))

#     def datagen_itr(self):
#         if self.datagen_iter_cnt < self.dataset_length:
#             self.data_incoming = True
#             self.images, self.labels, self.image_path = self.datagen_iter.next()
#             # self.image_path = [self.datasetfp.index(i) for i in self.image_path]
#             self.curr_batch_size = len(self.images)
#             self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
#             self.images = self.images.to(self.device)
#             self.labels = self.labels.to(self.device)
#             if self.datagen_iter_cnt == self.dataset_length: # added to stop repeat of last batch of images
#                 self.data_incoming = False
#         else:
#             self.data_incoming = False

#     def datagen_reset(self):
#         self.images = -1
#         self.labels = -1
#         self.datagen_iter = iter(self.data_loader)
#         self.data_incoming = True
#         self.datagen_iter_cnt = 0



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
            root = "/nwstore/datasets/FGVC/datasets/FGVC-aircraft/data/images/trainval"
            self.datasplit = 'train'
        elif root == 'val':
            root = "/nwstore/datasets/FGVC/datasets/FGVC-aircraft/data/images/test"
            self.datasplit = 'val'
        else:
            raise Exception('Invalid root directory')

        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        # {'American_lobster': 122, 'albatross': 146, 'balance_beam': 416, 'cabbage_butterfly': 324, 'cellular_telephone': 487, 'flagpole': 557, 'gibbon': 368, 'groenendael': 224, 'langur': 374, 'malinois': 225, 'moving_van': 675, 'patas': 371, 'planetarium': 727, 'radiator': 753, ...}

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
        self.input_size = 448

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
        if self.datasplit=='val':
            typ = "test"
        elif self.datasplit=='train':
            typ = "trainval"
        class_idx, _  = get_class_mapping(typ) # _ = idx2label

        for u in list(class_idx.keys()):
            if "/" in class_idx[u]:
                # print(u)
                # break
                class_idx[u] = class_idx[u].replace("/", "")

        
        classes = list(class_idx.values())
        class_idx = {str(list(class_idx.values())[i]): list(class_idx.keys())[i] for i in range(len(list(class_idx.values())))}

        # classes
        # ['American_lobster', 'albatross', 'balance_beam', 'cabbage_butterfly', 'cellular_telephone', 'flagpole', 'gibbon', 'groenendael', 'langur', 'malinois', 'moving_van', 'patas', 'planetarium', 'radiator', ...]
        # class_to_idx
        # {'American_lobster': 122, 'albatross': 146, 'balance_beam': 416, 'cabbage_butterfly': 324, 'cellular_telephone': 487, 'flagpole': 557, 'gibbon': 368, 'groenendael': 224, 'langur': 374, 'malinois': 225, 'moving_van': 675, 'patas': 371, 'planetarium': 727, 'radiator': 753, ...}

        return classes, class_idx


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = np.array(self.loader(path))
        # img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        

        if self.datasplit=='train':
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size, self.input_size), transforms.InterpolationMode.BICUBIC)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            # img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            # n_digits = 4
            # img = (img * 10**n_digits).round() / (10**n_digits)


        return img, target


    
    # def __getitem__(self, index):

    #         if self.is_train:
    #             img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]

    #             if len(img.shape) == 2:
    #                 img = np.stack([img] * 3, 2)
    #             img = Image.fromarray(img, mode='RGB')

    #             img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
    #             # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
    #             # img = transforms.RandomCrop(self.input_size)(img)
    #             img = transforms.RandomHorizontalFlip()(img)
    #             img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

    #             img = transforms.ToTensor()(img)
    #             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    #         else:
    #             img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
    #             if len(img.shape) == 2:
    #                 img = np.stack([img] * 3, 2)
    #             img = Image.fromarray(img, mode='RGB')
    #             img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
    #             # img = transforms.CenterCrop(self.input_size)(img)
    #             img = transforms.ToTensor()(img)
    #             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    #         return img, target



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


class FGVC_dataloader():

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
        # # Reorganize folder structure (only once)
        # prep_fgvc("test")
        # prep_fgvc("trainval")

        self.dataset_type=dl_attr.dl_dataset_type
        self.batch_size=dl_attr.dl_batch_size
        self.transform=dl_attr.dl_transform
        self.sampleN=dl_attr.dl_sampleN
        self.shuffle=dl_attr.dl_shuffle
        self.num_workers = dl_attr.dl_num_workers
        self.device = dl_attr.dl_device

        self.images = -1
        self.labels = -1
        
        ## TODO self.classes and self.class_idx is probably not the class label names and idx. Need to confirm. This may be is extracted
        ## and stored in Label_dict
        if self.dataset_type=="val":
            typ = "test"
        elif self.dataset_type=="train":
            typ = "trainval"
        self.class_idx, _  = get_class_mapping(typ) # _ = idx2label

        for u in list(self.class_idx.keys()):
            if "/" in self.class_idx[u]:
                # print(u)
                # break
                self.class_idx[u] = self.class_idx[u].replace("/", "")

        self.classes = list(self.class_idx.values())

    #     self.classes
    # ['tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco', ...]
    # len(self.classes)
    # 1000
    # self.class_idx
    # {'0': ['n01440764', 'tench'], '1': ['n01443537', 'goldfish'], '2': ['n01484850', 'great_white_shark'], '3': ['n01491361', 'tiger_shark'], '4': ['n01494475', 'hammerhead'], '5': ['n01496331', 'electric_ray'], '6': ['n01498041', 'stingray'], '7': ['n01514668', 'cock'], '8': ['n01514859', 'hen'], '9': ['n01518878', 'ostrich'], '10': ['n01530575', 'brambling'], '11': ['n01531178', 'goldfinch'], '12': ['n01532829', 'house_finch'], '13': ['n01534433', 'junco'], ...}
    # len(self.class_idx)
        # Label_dict = {}
        # for i in range(len(list(self.class_idx.keys()))):
        #     # class_mapping =  class_idx[list(class_idx.keys())[i]][0]
        #     class_label = self.class_idx[list(self.class_idx.keys())[i]][1]
        #     Label_dict[class_label] = i


        self.dataset = ImageFolderWithPaths(root=self.dataset_type, transform=self.transform)
        self.datasetfp = self.dataset.get_datasetfp()
        if self.sampleN is not None:
            if dl_attr.dl_sampleN <= 1:
                self.sampleN = int(self.sampleN*len(self.dataset))

            self.dataset = torch.utils.data.Subset(self.dataset, np.random.choice(len(self.dataset), self.sampleN, replace=False))
            self.datasetfp = self.dataset.dataset.get_datasetfp()

        self.data_incoming = True
        self.images = -1
        self.labels = -1
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        self.datagen_iter = iter(self.data_loader)
        self.dataset_length = len(self.dataset)
        self.datagen_iter_cnt = 0
        print("FGVC Dataset loaded in {}; {} dataset - Length : {}".format(self.device, self.dataset_type, len(self.dataset)))

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



# import torch
# import os
# import imageio

# class FGVC_aircraft():
#     def __init__(self, input_size, root, is_train=True, data_len=None):
#         self.input_size = input_size
#         self.root = root
#         self.is_train = is_train
#         train_img_path = os.path.join(self.root, 'data', 'images')
#         test_img_path = os.path.join(self.root, 'data', 'images')
#         train_label_file = open(os.path.join(self.root, 'data', 'train.txt'))
#         test_label_file = open(os.path.join(self.root, 'data', 'test.txt'))
#         train_img_label = []
#         test_img_label = []
#         for line in train_label_file:
#             train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
#         for line in test_label_file:
#             test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
#         self.train_img_label = train_img_label[:data_len]
#         self.test_img_label = test_img_label[:data_len]

#     def __getitem__(self, index):
#         if self.is_train:
#             img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
#             if len(img.shape) == 2:
#                 img = np.stack([img] * 3, 2)
#             img = Image.fromarray(img, mode='RGB')

#             img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
#             # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
#             # img = transforms.RandomCrop(self.input_size)(img)
#             img = transforms.RandomHorizontalFlip()(img)
#             img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

#             img = transforms.ToTensor()(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

#         else:
#             img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
#             if len(img.shape) == 2:
#                 img = np.stack([img] * 3, 2)
#             img = Image.fromarray(img, mode='RGB')
#             img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
#             # img = transforms.CenterCrop(self.input_size)(img)
#             img = transforms.ToTensor()(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

#         return img, target

#     def __len__(self):
#         if self.is_train:
#             return len(self.train_img_label)
#         else:
#             return len(self.test_img_label)


# def read_dataset(input_size, batch_size, root, type):
    
#     if type == 'train':
#         # print('Loading Aircraft trainset')
#         trainset = FGVC_aircraft(input_size=input_size, root=root, is_train=True)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                                   shuffle=True, num_workers=8, drop_last=False)
#         return trainset, trainloader

#     elif type == 'val':                                         
#         # print('Loading Aircraft testset')
#         testset = FGVC_aircraft(input_size=input_size, root=root, is_train=False)
#         testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                                  shuffle=False, num_workers=8, drop_last=False)
#         return testset, testloader
#     else:
#         print('Please choose supported dataset')
#         os._exit()

#         # dataset = datasets.FGVCAircraft(root=data_dir, train=False, shuffle=False, batch_size=batch_size, transform=transform, sampling=False, download=True)
#         # self.dataset_size = len(dataset)
