import sys
sys.path.append('/nwstore/qutub/intel_git/torchauto/src')
from torchauto.datasets.kitti import KittiDataset
from .objdet_baseClasses.catalog import DatasetCatalog, MetadataCatalog
from .objdet_baseClasses.boxes import BoxMode
from .objdet_baseClasses.common import DatasetFromList, MapDataset, DatasetMapper, trivial_batch_collator
from PIL import Image
import torch
import random
from random import shuffle
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr

# __things_classes__ = ['Car', 'Pedestrian', 'Van', 'Truck', 'Cyclist', 'DontCare', 'Person_sitting', 'Tram', 'Misc']
__things_classes__ = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

## old yolov3 model
# class_mapping_yolov3_kitti = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}

## yolov3 ultra
# class_mapping_yolov3_kitti = {'Car': 0, 'Pedestrian': 1, 'Van': 2, 'Truck': 3, 'Cyclist': 4, 'DontCare': 5, 'Person_sitting': 6, 'Tram': 7, 'Misc': 8}
class_mapping_yolov3_kitti = {'Car': 0,  'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5,  'Tram': 6, 'Misc': 7, 'DontCare': 8}


################################################################
# {"id": 0, "name": "Car"}, {"id": 1, "name": "Pedestrian"}, 
# {"id": 2, "name": "Van"}, {"id": 3, "name": "Truck"}, 
# {"id": 4, "name": "Cyclist"}, {"id": 5, "name": "DontCare"}, 
# {"id": 6, "name": "Person_sitting"}, {"id": 7, "name": "Tram"}, 
# {"id": 8, "name": "Misc"}]
################################################################
def load_kitti2D_objects(dirname : str = '/nwstore/datasets/KITTI/2d_oject/training', split : float = 0.8):
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
    ## TODO: add split part directly inside MultiModalDatasetKitti2D: will save much time
    kitti2d_det_dataset = KittiDataset(dirname)
    orig_split = list(range(len(kitti2d_det_dataset)))
    train_split, val_split = split_data(dataset_len=len(kitti2d_det_dataset), split=split)
    # train_split = list(kitti2d_det_dataset.db)[0:int(split*len(kitti2d_det_dataset.db))]
    # val_split = list(kitti2d_det_dataset.db)[int(split*len(kitti2d_det_dataset.db)):]
    train_dict = load_kitti2D_dict(dataset=kitti2d_det_dataset, split=train_split)
    val_dict = load_kitti2D_dict(dataset=kitti2d_det_dataset, split=val_split)
    return train_dict, val_dict

def split_data(dataset_len:int, sampleN=0.1, random_sample:bool=False):
    orig_split = list(range(dataset_len))
    if random_sample:
        train_split = random.sample(orig_split, int(len(orig_split) * sampleN))
    else:
        train_split = orig_split[0:int(len(orig_split) * sampleN)]
    val_split = list(set(orig_split) - set(train_split))
    return train_split, val_split


def load_kitti2D_dict(dataset : KittiDataset, dl_attr:TEM_Dataloader_attr, split_list, dnn_model_name:str=''):
    """
    Args:
        root (str): path to the raw dataset. e.g., "/nwstore/datasets/KITTI/2d_oject/training".
        split (str): train or validation split. e.g., "train" or "val".

    Returns:
        list[dict]: a list of dicts in pytorchfiWrapper standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    ## TODO: add split part directly inside MultiModalDatasetKitti2D: will save much time

    dataset_dicts = []
    if split_list == None:
        split_list = list(range(len(dataset.db)))

    for id, idx in enumerate(split_list):
        record = {}
        
        filename = str(dataset.db[idx]['CAM_FRONT_LEFT'])
        # height, width = cv2.imread(filename).shape[:2]
        width, height = Image.open(filename).size
        record["file_name"] = filename
        record["image_id"] = id
        record["height"] = height
        record["width"] = width
      
        annos = dataset.db[idx]['annotations']
        objs = []
        for anno in annos:
            if dnn_model_name.lower() == 'yolov3':
                if anno['name'].lower() == 'DontCare'.lower():
                    continue
                category_id = class_mapping_yolov3_kitti[anno['name']]
            else:
                category_id = dataset.class_mapping[anno['name']] - 1
            obj = {
                "bbox": anno['two_corner_rect'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "truncated": anno['truncated'],
                "visible": anno['visible'],
                "category_id": category_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_kitti2D_objects(dl_attr:TEM_Dataloader_attr, dirname:str = '/nwstore/datasets/KITTI/2d_object/training', dnn_model_name:str=''):
    """
    
    """
    kitti2d_det_dataset = KittiDataset(dirname)
    if dl_attr.dl_sampleN <= 1:
        dl_attr.dl_sampleN = dl_attr.dl_sampleN
    else:
        dl_attr.dl_sampleN = dl_attr.dl_sampleN/len(kitti2d_det_dataset)
    ## if split = 0.2, then val_split = 0.2, train_split = 0.8 which is discarded.
    val_split, train_split = split_data(dataset_len=len(kitti2d_det_dataset), sampleN=dl_attr.dl_sampleN, random_sample=dl_attr.dl_random_sample)
    if dl_attr.dl_shuffle:
        shuffle(val_split)

    if dnn_model_name.lower() == 'yolov3'.lower():
        thing_classes = list(class_mapping_yolov3_kitti.keys())
    else:
        thing_classes = list(kitti2d_det_dataset.class_mapping.keys())


    DatasetCatalog.register(dl_attr.dl_dataset_name + '/{}'.format(dl_attr.dl_dataset_type), lambda: load_kitti2D_dict(dataset = kitti2d_det_dataset, dl_attr=dl_attr, split_list=val_split, dnn_model_name=dnn_model_name))
    MetadataCatalog.get(dl_attr.dl_dataset_name + '/{}'.format(dl_attr.dl_dataset_type)).set(
        thing_classes=thing_classes, dirname=dirname, sampleN=dl_attr.dl_sampleN, year=2012, dataset=dl_attr.dl_dataset_name
    )
    MetadataCatalog.get(dl_attr.dl_dataset_name + '/{}'.format(dl_attr.dl_dataset_type)).set(
        evaluator_type='coco'
    )

class Kitti2D_dataloader:

    def __init__(self, dl_attr:TEM_Dataloader_attr, dnn_model_name:str="yolov3_ultra"):
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
        super().__init__()
        self.dl_attr         = dl_attr
        self.dnn_model_name  = dnn_model_name
        self.device          = dl_attr.dl_device if self.dl_attr.dl_device is not None else torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        self.dl_attr.dl_device  = self.device
        if isinstance(self.dl_attr.dl_sampleN, float) and self.dl_attr.dl_sampleN>1:
            self.dl_attr.dl_sampleN = int(self.dl_attr.dl_sampleN)
        
        register_kitti2D_objects(dl_attr=self.dl_attr, dnn_model_name=self.dnn_model_name)
        self.dataset_dict = DatasetCatalog.get('kitti/{}'.format(self.dl_attr.dl_dataset_type))
        self.metadata = MetadataCatalog.get('kitti/{}'.format(self.dl_attr.dl_dataset_type))
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
        print("Kitti Dataset loaded in {}; {} dataset - Length : {}".format(self.dl_attr.dl_device, self.dl_attr.dl_dataset_type, len(self.dataset)))

    def datagen_itr(self):
        if self.data_incoming == True:
            self.data = self.datagen_iter.next()
            self.curr_batch_size = len(self.data)
            self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
            # self.images = self.images.to(self.device)
        else:
            raise ValueError("Kitti dataloader reached its limit, monitor the bool data_incoming  flag\
                to reset the data iterator for epochs to continue")
        if self.datagen_iter_cnt >= self.dataset_length:
            self.data_incoming = False

    def datagen_reset(self):
        self.data = -1
        self.datagen_iter = iter(self.data_loader)
        self.data_incoming = True
        self.datagen_iter_cnt = 0