# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

"""
Abstract class to define necessary functions for each data loader

"""
from abc import ABC, abstractmethod
import os
import random
import torch
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
from .objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog
from .objdet_baseClasses.common import DatasetFromList, MapDataset, DatasetMapper, trivial_batch_collator

class Abstract_Loader(ABC):
    def __init__(self, dl_attr:TEM_Dataloader_attr, dnn_model_name:str="yolov3_ultra"):
        """
        Args:
        dl_attr
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
        self.dl_attr         = dl_attr
        self.dataset_name  = self.dl_attr.dl_dataset_name
        self.dnn_model_name  = dnn_model_name
        self.shuffle=self.dl_attr.dl_shuffle
        self.device          = dl_attr.dl_device if self.dl_attr.dl_device is not None else torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        self.dl_attr.dl_device  = self.device

        if isinstance(self.dl_attr.dl_sampleN, float) and self.dl_attr.dl_sampleN>1:
            self.dl_attr.dl_sampleN = int(self.dl_attr.dl_sampleN)
        
        self.register_instances(dataset_name=self.dl_attr.dl_dataset_name, metadata={}, json_file=self.dl_attr.dl_gt_json,\
                image_root=self.dl_attr.dl_img_root, dl_attr=self.dl_attr)
   

    def init_dataloader(self):
        '''
        This function needs to be called from child class
        Overwrite for specific data loader 
        '''
        self.dataset_dict = DatasetCatalog.get(self.dl_attr.dl_dataset_name)
        self.metadata = MetadataCatalog.get(self.dl_attr.dl_dataset_name)
        self.dataset = DatasetFromList(self.dataset_dict, copy=True)
        self.dataset = MapDataset(self.dataset, DatasetMapper())
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.dl_attr.dl_num_workers,
            batch_size=self.dl_attr.dl_batch_size,
            collate_fn=trivial_batch_collator,
        )
        self.data_incoming = True
        self.data = -1
        self.curr_batch_size = 0
        self.datagen_iter = iter(self.data_loader)
        self.dataset_length = len(self.dataset)
        self.datagen_iter_cnt = 0
        print("{} Dataset loaded in {}; {} dataset - Length : {}".format(self.dl_attr.dl_dataset_name, self.dl_attr.dl_device, self.dl_attr.dl_dataset_type, len(self.dataset)))

    @abstractmethod
    def load_json(self, json_file, image_root, dl_attr:TEM_Dataloader_attr, dataset_name=None, extra_annotation_keys=None):
        """
        Load a json file with COCO's instances annotation format.
        Currently supports instance detection, instance segmentation,
        and person keypoints annotations.

        Args:
            json_file (str): full path to the json file in COCO instances annotation format.
            image_root (str or path-like): the directory where the images in this json file exists.
            dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
                When provided, this function will also do the following:

                * Put "thing_classes" into the metadata associated with this dataset.
                * Map the category ids into a contiguous range (needed by standard dataset format),
                and add "thing_dataset_id_to_contiguous_id" to the metadata associated
                with this dataset.

                This option should usually be provided, unless users need to load
                the original json content and apply more processing manually.
            extra_annotation_keys (list[str]): list of per-annotation keys that should also be
                loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
                "category_id", "segmentation"). The values for these keys will be returned as-is.
                For example, the densepose annotations are loaded in this way.

        Returns:
            list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
            `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
            If `dataset_name` is None, the returned `category_ids` may be
            incontiguous and may not conform to the Detectron2 standard format.

        Notes:
            1. This function does not read the image files.
            The results do not have the "image" field.
        """
        raise NotImplementedError
    

    def split_data(self, dataset_len:int, sampleN=0.1, random_sample:bool=False):
        orig_split = list(range(dataset_len))
        if random_sample:
            train_split = random.sample(orig_split, int(len(orig_split) * sampleN))
        else:
            train_split = orig_split[0:int(len(orig_split) * sampleN)]
        val_split = list(set(orig_split) - set(train_split))
        return train_split, val_split

    def register_instances(self, json_file, image_root, dataset_name, metadata={}, dl_attr:TEM_Dataloader_attr=None):
        """
        Register a dataset in COCO's json annotation format for
        instance detection, instance segmentation and keypoint detection.
        (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
        `instances*.json` and `person_keypoints*.json` in the dataset).

        This is an example of how to register a new dataset.
        You can do something similar to this function, to register new datasets.

        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            metadata (dict): extra metadata associated with this dataset.  You can
                leave it as an empty dict.
            json_file (str): path to the json instance annotation file.
            image_root (str or path-like): directory which contains all the images.
        ex: register_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
        """
        assert isinstance(dataset_name, str), dataset_name
        assert isinstance(json_file, (str, os.PathLike)), json_file
        assert isinstance(image_root, (str, os.PathLike)), image_root
        # 1. register a function which returns dicts
        DatasetCatalog.register(dataset_name, lambda: self.load_json(json_file=json_file, image_root=image_root, dl_attr=dl_attr, dataset_name=dataset_name))

        # 2. Optionally, add metadata about this dataset,
        # since they might be useful in evaluation, visualization or logging
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
        )

    def datagen_itr(self):
        if self.data_incoming == True:
            self.data = self.datagen_iter.next()
            self.curr_batch_size = len(self.data)
            self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
            # self.images = self.images.to(self.device)
        else:
            raise ValueError("{} dataloader reached its limit, monitor the bool data_incoming  flag\
                to reset the data iterator for epochs to continue".format(self.dl_attr.dl_dataset_name))
        if self.datagen_iter_cnt >= self.dataset_length:
            self.data_incoming = False

    def datagen_reset(self):
        self.data = -1
        self.datagen_iter = iter(self.data_loader)
        self.data_incoming = True
        self.datagen_iter_cnt = 0

