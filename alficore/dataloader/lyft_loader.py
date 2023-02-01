import sys
from enum import Enum
sys.path.append('/nwstore/qutub/intel_git/torchauto/src')
from torchauto.datasets.lyft import LyftDataset
from .objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog
from .objdet_baseClasses.boxes import BoxMode
from .objdet_baseClasses.common import DatasetFromList, MapDataset, DatasetMapper, trivial_batch_collator
from PIL import Image
from torchauto.generators import YoloObjectEncoder, ImageObjectEncoder
import torch
import torchauto.geometry as geometry
from torchauto.utils.iou import overlap
import os
import random
import numpy as np
import random
from random import shuffle
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr

__things_classes_kitti__ = ['Car', 'Pedestrian', 'Van', 'Truck', 'Cyclist', 'DontCare', 'Person_sitting', 'Tram', 'Misc']

__things_classes__ = ['car', 'pedestrian', 'animal', 'other_vehicle', 'bus', 'motorcycle', 'truck', 'bicycle']

class_mapping_yolov3_lyft = {'car':0, 'pedestrian':1, 'animal':2, 'other_vehicle':3, 'bus':4, 'motorcycle':5, 'truck':6, 'bicycle':7}
# class_mapping_yolov3_lyft = {'car':0, 'pedestrian':0, 'animal':0, 'other_vehicle':0, 'bus':0, 'motorcycle':0, 'truck':0, 'bicycle':0}

# [{"id": 0, "name": "car"}, {"id": 1, "name": "pedestrian"}, {"id": 2, "name": "animal"}, {"id": 3, "name": "other_vehicle"}, {"id": 4, "name": "bus"}, {"id": 5, "name": "motorcycle"}, {"id": 6, "name": "truck"}, {"id": 7, "name": "bicycle"}]
class Dataloader_Mode(Enum):
    IMAGE    = 0
    SEQUENCE = 1

def load_lyft_objects(dirname : str = '/nwstore/datasets/Lyft/perception/training/images', split : float = 0.8):
    """
    Args:
        root (str): path to the raw dataset. e.g., "/nwstore/datasets/KITTI/2d_oject/training".
        split (str): train or validation split. e.g., "train" or "val".
        split_perc (float): percentage of split. eg: 0.9
            split_perc => 0:0.9 of total dataset
            split_perc => 0.9: of total dataset
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in pytorchfiWrapper standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    ## TODO: add split part directly inside MultiModalDatasetlyft: will save much time
    lyft_det_dataset = LyftDataset(dirname)
    orig_split = list(range(len(lyft_det_dataset)))
    train_split, val_split = split_data(dataset_len=len(lyft_det_dataset), split=split)
    # train_split = list(lyft_det_dataset.db)[0:int(split*len(lyft_det_dataset.db))]
    # val_split = list(lyft_det_dataset.db)[int(split*len(lyft_det_dataset.db)):]
    train_dict = load_lyft_dict(dataset=lyft_det_dataset, split=train_split)
    val_dict = load_lyft_dict(dataset=lyft_det_dataset, split=val_split)
    return train_dict, val_dict

def split_data(dataset_len:int, split=0.1, random:bool=False):
    orig_split = list(range(dataset_len))
    if random:
        train_split = random.sample(orig_split, int(len(orig_split) * split))
    else:
        train_split = orig_split[0:int(len(orig_split) * split)]
    val_split = list(set(orig_split) - set(train_split))
    return train_split, val_split


def load_lyft_dict(dataset : LyftDataset, dl_attr:TEM_Dataloader_attr, dirname:str='', dnn_model_name:str='', image_encoder=None):
    """
    Args:
        root (str): path to the raw dataset. e.g., "/nwstore/datasets/KITTI/2d_oject/training".
        split (str): train or validation split. e.g., "train" or "val".

    Returns:
        list[dict]: a list of dicts in pytorchfiWrapper standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    ## TODO: add split part directly inside MultiModalDatasetlyft: will save much time

    dataset_dicts = []
    sample_channels = False
    image_encoder = ImageObjectEncoder(dataset.class_mapping, encoding='two_corner_rect')
    # frame = 'IMAGE_{}'.format(image_channel)
    check_overlap = True
    min_size = (8,10)
    images_path = os.path.join(os.path.dirname(dirname), 'images')
    total_images = sum([len(dataset.scenes[scene]['indices']) for scene in range(len(dataset.scenes))]) #22679 - total ; #158757 - total samples #len([name for name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, name))]) ## total images in Lyft dataset is 158757
    # total_images = dataset.scenes[len(dataset.scenes)-1]['indices'][-1]
    sampleN         = dl_attr.dl_sampleN
    shuffle         = dl_attr.dl_shuffle
    random_sample   = dl_attr.dl_random_sample
    scenes          = dl_attr.dl_scenes   
    sensor_channels = dl_attr.dl_sensor_channels

    sampleN         = int(sampleN*total_images) if sampleN<= 1 else sampleN
    sample_channels = True if sensor_channels[-1] == "random" else sample_channels
    dataloader_mode = Dataloader_Mode(0) if dl_attr.dl_mode == "image" else Dataloader_Mode(1)

    if dataloader_mode.value == Dataloader_Mode.IMAGE.value:
        # dataset_frame_indx = dataset.scenes[0]['indices']
        if random_sample:
            dataset_frame_indx = random.sample(range(total_images), sampleN)
        else:
            dataset_frame_indx = list(range(sampleN))
    elif dataloader_mode.value == Dataloader_Mode.SEQUENCE.value:
        if scenes[-1] == -1:
            scenes = len(dataset.scenes)
        dataset_frame_indx = []
        for scene in scenes:
            dataset_frame_indx.extend(dataset.scenes[scene]['indices'])
        if sampleN:
            dataset_frame_indx = dataset_frame_indx[:sampleN]
    if shuffle:
        shuffle(dataset_frame_indx)
    image_sizes = []
    for id, idx in enumerate(dataset_frame_indx):
        record = {}
        data_frame = dataset.get_frame(idx)
        if sample_channels:
            sensor_channels = random.sample(data_frame.channels, 1)
        for image_channel in sensor_channels:
            image_size = data_frame.channel_data[image_channel].size
            image_sizes.append(image_size)
            frame = 'IMAGE_{}'.format(image_channel)
            # sample = dataset._get_sample(idx)
            # sample_data = dataset.database.get('sample_data', dataset.samples[idx]['data'][sensor_channel])

            # filename = os.path.join(os.path.dirname(dirname), sample_data['filename'])
            if not image_channel in data_frame.channels:
                continue
            record["channel_id"]  = data_frame.channels.index(image_channel)
            record["channel"]     = image_channel
            record["file_name"]   = str(data_frame.channel_paths[image_channel])
            record["image_id"]    = id
            record["height"]      = image_size[1]
            record["width"]       = image_size[0]

            # Annotation
            annotations = data_frame.annotations
            objs = []
            rects = []
            boxes = []
            instance_ids = []
            category_ids = []
            positions = []

            for annotation in annotations:
                # anno = dataset.database.get('sample_annotation', anno_token)
                rect = geometry.Rect.from_annotation(annotation, data_frame.transform_tree, target_frame = frame)
                visible = rect.in_region(0, 0, *image_size, include_cropped = True, visibility_threshold = 0.25)

                if dnn_model_name.lower() == 'yolov3_ultra':
                    if annotation.label.lower() not in class_mapping_yolov3_lyft.keys():
                        category_id = class_mapping_yolov3_kitti['misc']
                    else:
                        category_id = class_mapping_yolov3_lyft[annotation.label]
                else:
                    assert True == False, "Warning: Lyft dataloader is being by other DNN model whose category_id may not be right and needs the right assignment."
                    category_id = dataset.class_mapping[annotation.label] - 1


                if visible and rect.w >= min_size[0] and rect.h >= min_size[1]:
                    box, label = image_encoder(annotation, rect, data_frame.transform_tree, image_channel, image_size)
                    rects.append(rect)
                    boxes.append(box)
                    category_ids.append(category_id)
                    instance_ids.append(annotation.instance_id)
                    positions.append([annotation.x, annotation.y, annotation.z])
                    # labels.append(label)

            if check_overlap and len(boxes) > 1:
                scores = np.array([1.0 / (1.0 + rect.z) for rect in rects], dtype=np.float32)
                overlap_rects = np.array([rect.two_corner_rect() for rect in rects], dtype=np.float32)
                overlap_keep = overlap(overlap_rects, scores, threshold=0.70)
                overlap_indices = overlap_keep.nonzero()[0].tolist()
                boxes = [boxes[i] for i in overlap_indices]
                category_ids = [category_ids[i] for i in overlap_indices]
                instance_ids = [instance_ids[i] for i in overlap_indices]
                positions = [positions[i] for i in overlap_indices]

            if len(boxes) == 0:
                boxes.append([])
                category_ids.append([])
                instance_ids.append([])
                positions.append([])
                        # target[channel] = dict()
                        # target[channel]['boxes'] = np.array(boxes).astype(np.float32)
                        # target[channel]['labels'] = np.array(labels).astype(np.int64)
            for i, bbox in enumerate(boxes):
                obj = {
                    "bbox": boxes[i],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": category_ids[i],
                    "instance_id": instance_ids[i],
                    "position": positions[i]
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts



def register_lyft_objects(dl_attr:TEM_Dataloader_attr, dnn_model_name:str='', dirname:str = '/nwstore/datasets/Lyft/perception/training/train_data'):
    """
    
    """
    # TODO: change training and testing using dataset_type arg
    lyft_dataset = LyftDataset(dirname)

    if dnn_model_name.lower() == 'yolov3_ultra'.lower():
        thing_classes = list(class_mapping_yolov3_lyft.keys())
    else:
        thing_classes = list(lyft_dataset.class_mapping.keys())
    # lyft_dataset.class_mapping = class_mapping_yolov3_lyft


    DatasetCatalog.register(dl_attr.dl_dataset_name + '/{}'.format(dl_attr.dl_dataset_type), lambda: load_lyft_dict(dataset = lyft_dataset, dl_attr=dl_attr, dirname=dirname, dnn_model_name=dnn_model_name))
    MetadataCatalog.get(dl_attr.dl_dataset_name + '/{}'.format(dl_attr.dl_dataset_type)).set(
        thing_classes=thing_classes, dirname=dirname, sensor_channels=dl_attr.dl_scenes, dataset=dl_attr.dl_dataset_name, evaluator_type='coco')

class Lyft_dataloader:

    def __init__(self, dl_attr:TEM_Dataloader_attr, dnn_model_name:str="yolov3_ultra"):
        """
        Args:
        dl_attr
            dataset_type     (str) : Type of dataset shall be used - train, test, val... Defaults to val.
            batch_size       (uint): batch size. Defaults to 1.
            shuffle          (str) : Shuffling of the data in the dataloader. Defaults to False.
            sampleN          (uint): Percentage of dataset lenth to be sampled. Defaults to None.
            transform        (obj) : Several transforms composed together. Defaults to None.
            num_workers      (uint): Number of workers to be used to load the dataset. Defaults to 1.
            device           (uint): Cuda/Cpu device to be used to load dataset. Defaults to Cuda 0 if available else CPuU.
            scenes           (list): List of scenes to be included from range of 180; scenes = [-1] leads to include all scenes
            sensor_channels   (list): List of image channels to be considered from the range of 6 different cameras.
            dataloader_mode  (list): Mode of dataloader -> IMAGE(random images from any cam channel)/SEQUENCE(video sequences)
            dirname          (str): location of dataset files
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dl_attr         = dl_attr
        self.dnn_model_name  = dnn_model_name
        self.device          = dl_attr.dl_device if self.dl_attr.dl_device is not None else torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        if isinstance(self.dl_attr.dl_sampleN, float) and self.dl_attr.dl_sampleN>1:
            self.dl_attr.dl_sampleN = int(self.dl_attr.dl_sampleN)
            
        self.dirname = self.dl_attr.dl_dirname
        if self.dirname:
            register_lyft_objects(dirname=self.dirname, dl_attr=self.dl_attr, dnn_model_name=self.dnn_model_name)
        else:
            register_lyft_objects(dl_attr=self.dl_attr, dnn_model_name=self.dnn_model_name)
        self.dataset_dict = DatasetCatalog.get('lyft/{}'.format(self.dl_attr.dl_dataset_type))
        self.metadata = MetadataCatalog.get('lyft/{}'.format(self.dl_attr.dl_dataset_type))
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
        self.datagen_iter = iter(self.data_loader)
        self.dataset_length = len(self.dataset)
        self.curr_batch_size = 0
        self.datagen_iter_cnt = 0
        print("Lyft Dataset loaded in {}; {} dataset - Length : {}".format(self.device, self.dl_attr.dl_dataset_type, len(self.dataset)))

    def datagen_itr(self):
        if self.data_incoming == True:
            self.data = self.datagen_iter.next()
            self.curr_batch_size = len(self.data)
            self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
            # self.images = self.images.to(self.device)
        else:
            raise ValueError("Lyft dataloader reached its limit, monitor the bool data_incoming  flag\
                to reset the data iterator for further epochs to continue")
        if self.datagen_iter_cnt >= self.dataset_length:
            self.data_incoming = False

    def datagen_reset(self):
        self.data = -1
        self.datagen_iter = iter(self.data_loader)
        self.data_incoming = True
        self.datagen_iter_cnt = 0
