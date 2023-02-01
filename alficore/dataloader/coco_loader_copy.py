# from alficore.models.yolov3
import os
import contextlib
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from enum import Enum
import io
import logging
from alficore.models.yolov3.utils import parse_data_config, load_classes
import torch.nn.functional as functional
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from .objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog
from .objdet_baseClasses.boxes import BoxMode
from .objdet_baseClasses.common import DatasetFromList, MapDataset, DatasetMapper, trivial_batch_collator
from PIL import Image
import torch
import random
from alficore.ptfiwrap_utils.augmentations import horisontal_flip, pad_to_square, resize
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
from random import shuffle

"""
This file contains functions to parse COCO-format annotations into dicts in "pytorchfi wrapper format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_coco_json", "load_sem_seg", "convert_to_coco_json", "register_coco_instances"]

class CoCo_Dataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        
        for i in range(len(self.img_files)):
            self.img_files[i] = os.path.join(os.path.dirname(list_path), self.img_files[i].rstrip()[1:])
        self.len_img_files = len(self.img_files)

        ## TODO: What is the purpose of the function below
        for i in range(self.len_img_files):
            if i < len(self.img_files):
                self.path_img = self.img_files[i].rstrip()
                if os.path.exists(self.path_img):
                    continue
                else:
                    del self.img_files[i]
                    i=i-1

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        for i in range(self.len_img_files):
            if i < len(self.label_files):
                self.path_label = self.label_files[i].rstrip()
                if os.path.exists(self.path_label):
                    continue
                else:
                    del self.label_files[i]
                    del self.img_files[i]
                    i=i-1

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        else:
            return None

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

def coco_loader(root='val', img_size=416, batch_size=1, shuffle:bool=False, sampling=False, sampleN=None, augment=False, multiscale=False, num_workers=1):
    # TODO: Add dataset path
    data_config = parse_data_config('./alficore/models/yolov3/config/coco.data')
    if root == "train":
        path = data_config["train"]
    elif root == "val":
        path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    coco_dataset = CoCo_Dataset(path, img_size=img_size, augment=augment, multiscale=multiscale)

    if sampling:
        if sampleN is None:
            sampleN = 20
        dataset = torch.utils.data.Subset(coco_dataset, np.random.choice(len(coco_dataset), int(sampleN*len(coco_dataset)/100), replace=False))
        CoCo_dataset = dataset
    else:
        CoCo_dataset = coco_dataset
    dataloader = torch.utils.data.DataLoader(CoCo_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=coco_dataset.collate_fn)
    return dataloader, class_names

class CoCo_dataloader():

    def __init__(self, **kwargs):  
        """
        Args:
        dataset_type (str) : Type of dataset shall be used - train, test, val... Defaults to val.
        batch_size   (uint): batch size. Defaults to 1.
        shuffle      (str) : Shuffling of the data in the dataloader. Defaults to False.
        sampleN      (uint): Percentage of dataset lenth to be sampled. Defaults to None.
        transform  (obj) : Several transforms composed together. Defaults to predefined transform.
        num_workers  (uint): Number of workers to be used to load the dataset. Defaults to 1.
        device       (uint): Cuda/Cpu device to be used to load dataset. Defaults to Cuda 0 if available else CPU.
        multiscale   (bool): boolean value to enable multiscaling of images. Defaults to False
        img_size     (uint): Image size. Defaults to 416
        augment      (bool): data augmentation bool variable. Defaults to False
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """        

        self.dataset_type = kwargs.get("dataset_type", 'val')
        self.batch_size = kwargs.get("batch_size", 1)
        self.shuffle = kwargs.get("shuffle", False)
        self.sampleN = kwargs.get("sampleN", None)
        self.transform = kwargs.get("transform", transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
        self.num_workers = kwargs.get("num_workers", 1)
        self.device = kwargs.get("device", torch.device(
        "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"))
        self.multiscale = kwargs.get("multiscale", False)
        self.img_size = kwargs.get("img_size", 416)
        self.augment = kwargs.get("augment", False)
        
        self.images = -1
        self.labels = -1

        data_config = parse_data_config('./alficore/models/yolov3/config/coco.data')
        if self.dataset_type == "train":
            path = data_config["train"]
        elif self.dataset_type == "val":
            path = data_config["valid"]
        self.classes = load_classes(data_config["names"])

        self._dataset = CoCo_Dataset(path, img_size=self.img_size, augment=self.augment, multiscale=self.multiscale)
        self.dataset = self._dataset

        if self.sampleN is not None:
            self.dataset = torch.utils.data.Subset(self._dataset, np.random.choice(len(self._dataset), int(self.sampleN*len(self._dataset)/100), replace=False))
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self._dataset.collate_fn)

        self.datasetfp = None
        self._image_path = -1
        self.datagen_iter = iter(self.data_loader)
        self.dataset_length = len(self.dataset)
        self.datagen_iter_cnt = 0
        print("CoCo Dataset loaded in {}; {} dataset - Length : {}".format(self.device, self.dataset_type, len(self.dataset)))

    def datagen_itr(self):
        if self.datagen_iter_cnt < self.dataset_length:
            self.data_incoming = True
            self.image_path, self.images, self.labels  = self.datagen_iter.next()
            self.image_path = [self._image_path for i in range(len(self.images))]
            self.curr_batch_size = len(self.images)
            self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
            self.images = self.images.to(self.device)
            self.labels = self.labels.to(self.device)
        else:
            self.data_incoming = False

    def datagen_reset(self):
        self.images = -1
        self.labels = -1        
        self.datagen_iter = iter(self.data_loader)
        self.data_incoming = True
        self.datagen_iter_cnt = 0

## object detection dataloader

def split_data(dataset_len:int, sampleN=0.1, random_sample:bool=False):
    orig_split = list(range(dataset_len))
    if random_sample:
        train_split = random.sample(orig_split, int(len(orig_split) * sampleN))
    else:
        train_split = orig_split[0:int(len(orig_split) * sampleN)]
    val_split = list(set(orig_split) - set(train_split))
    return train_split, val_split
    
def load_coco_json(json_file, image_root, dl_attr:TEM_Dataloader_attr, dataset_name=None, extra_annotation_keys=None):
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
    from pycocotools.coco import COCO

    timer = Timer()
    # json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dl_attr.dl_dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dl_attr.dl_dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    if dl_attr.dl_sampleN:
        if dl_attr.dl_sampleN <= 1:
            sampleN = dl_attr.dl_sampleN
        else:
            sampleN = dl_attr.dl_sampleN/len(img_ids)
        val_split, _ = split_data(dataset_len=len(img_ids), sampleN=sampleN, random_sample=dl_attr.dl_random_sample)
        img_ids = [img_ids[i] for i in val_split]
    if dl_attr.dl_shuffle:
        shuffle(img_ids)
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts

def register_coco_instances(dataset_name, metadata={}, json_file='/nwstore/datasets/COCO/coco2014/annotations/instances_val2014.json',\
    image_root='/nwstore/datasets/COCO/coco2014/val2014', dl_attr:TEM_Dataloader_attr=None):
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
    ex: register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
    """
    assert isinstance(dataset_name, str), dataset_name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(dataset_name, lambda: load_coco_json(json_file=json_file, image_root=image_root, dl_attr=dl_attr, dataset_name=dataset_name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

class CoCo_obj_det_dataloader:

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
        super().__init__()
        self.dl_attr         = dl_attr
        self.dnn_model_name  = dnn_model_name
        self.device          = dl_attr.dl_device if self.dl_attr.dl_device is not None else torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        self.dl_attr.dl_device  = self.device
        
        if isinstance(self.dl_attr.dl_sampleN, float) and self.dl_attr.dl_sampleN>1:
            self.dl_attr.dl_sampleN = int(self.dl_attr.dl_sampleN)
        self.release = 2014 if "2014" in self.dl_attr.dl_dataset_name else 2017
        dataset_name  = 'coco{}/{}'.format(str(self.release), self.dl_attr.dl_dataset_type)
        if self.dl_attr.dl_dataset_type == 'val':
            register_coco_instances(dataset_name=dataset_name, metadata={}, json_file='/nwstore/datasets/COCO/coco{}/annotations/instances_val{}.json'.format(str(self.release), str(self.release)),\
                image_root='/nwstore/datasets/COCO/coco{}/val{}'.format(str(self.release), str(self.release)), dl_attr=self.dl_attr)
        else:
            register_coco_instances(dataset_name=dataset_name, metadata={}, json_file='/nwstore/datasets/COCO/coco{}/annotations/instances_train{}.json'.format(str(self.release), str(self.release)),\
                image_root='/nwstore/datasets/COCO/coco{}/train{}'.format(str(self.release), str(self.release)), dl_attr=self.dl_attr)
    
        self.dataset_dict = DatasetCatalog.get('coco{}/{}'.format(str(self.release), self.dl_attr.dl_dataset_type))
        self.metadata = MetadataCatalog.get('coco{}/{}'.format(str(self.release),  self.dl_attr.dl_dataset_type))
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
        print("CoCo Obj Det Dataset loaded in {}; {} dataset - Length : {}".format(self.dl_attr.dl_device, self.dl_attr.dl_dataset_type, len(self.dataset)))

    def datagen_itr(self):
        if self.data_incoming == True:
            self.data = self.datagen_iter.next()
            self.curr_batch_size = len(self.data)
            self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
            # self.images = self.images.to(self.device)
        else:
            raise ValueError("CoCo dataloader reached its limit, monitor the bool data_incoming  flag\
                to reset the data iterator for epochs to continue")
        if self.datagen_iter_cnt >= self.dataset_length:
            self.data_incoming = False

    def datagen_reset(self):
        self.data = -1
        self.datagen_iter = iter(self.data_loader)
        self.data_incoming = True
        self.datagen_iter_cnt = 0