# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import sys
import os
from os.path import dirname as up
from collections import namedtuple
sys.path.append(os.getcwd())

## uncomment this for debug
# sys.path.append(os.getcwd())

from alficore.evaluation.visualization.visualization import Visualizer, ColorMode
from alficore.ptfiwrap_utils.utils import  read_json
from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
from PIL import Image, TiffImagePlugin
TiffImagePlugin.DEBUG = False
########################################################
# image = np.random.randn(416,416,3)
trails = 2
fault = 1
# inj_policy = 'per_epoch'
inj_policy = 'per_image'
# inj_policy = 'per_batch'
inj_type = 'neurons'
inj_type = 'weights'


uuid = 'objDet_20220331-125720_1_faults_[1,4]_bits' #coco
# uuid = 'objDet_20220331-143056_1_faults_[1,4]_bits' #robo
# uuid = 'objDet_20220331-151035_1_faults_[1,4]_bits' #ppp

# model_name = 'det2_fasterRCNN'
model_name = 'yolov3_ultra'
dataset = 'coco2017'
# dataset = 'lyft'
# dataset = 'kitti'
epoch = 0
batch_size = 1
######################################################################

""""
list of image ids to be visualised
img_ids = None or
img_ids = [0,1,2]
Image Id should match with the gt_json or dt_json file
"""
img_ids = [80, 100, 127, 143, 647]

epochs = list(range(trails))

gt_json_file = '../pytorchalfi/result_files/output_20221007-122944_yolo_kitti/kitti/coco_format.json'
dt_json_file = '../pytorchalfi/result_files/output_20221007-122944_yolo_kitti/kitti/orig_model/epochs/0/coco_instances_results_0_epoch.json'

gt_json_data = read_json(gt_json_file)
thing_classes = [a['name'] for a in gt_json_data['categories']]
id_map = None
if "coco" in dataset:
    from pycocotools.coco import COCO
    coco_api = COCO(gt_json_file)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    # meta.thing_classes = thing_classes
    id_map = {v: i for i, v in enumerate(cat_ids)}
    # meta.thing_dataset_id_to_contiguous_id = id_map


MetadataCatalog.get('{}/{}'.format(dataset, 'val')).set(
        thing_classes=thing_classes, dataset=dataset, thing_dataset_id_to_contiguous_id=id_map)

metadata = MetadataCatalog.get('{}/{}'.format(dataset, 'val'))


visualizer = Visualizer(metadata=metadata, instance_mode=ColorMode.IMAGE, vis_mode='offline')
Vis_Features = namedtuple('vis_features', 'only_bbox vis_tpfpfn text_info')


"""
DRAW_INSTANCE_GT_PRED_OFFLINE
##### ----- visualises all the predictions -----#####

viz_gt     = visualise ground truth
viz_dt     = visualise original detections
viz_cdt    = visualise corrupt detections
viz_rcdt   = visualise resiliency applied corrupt detections
epochs     = epochs to be considered from the experiment folder. ex: [0, 1, ..., 25] or [4, 23] or [9]
img_ids    = images to be considered from all the chosen epochs. ex: [0, 1, ..., 1000] or [87, 798] or [50]
             None means the function will visualise all the available img_ids.

vis_features ::
    a. only_bbox  : 
        Visualise detections with filled bounding boxes against black and white background (B&W image); stores images in 'vis/images_only_bbox' if set to true.
        True
        False  => Visualise detections with original image and bounding boxes; stores images in 'vis/images'.
        None   => Visualises detections using normal images and also with filled black and white bounding boxes.
    b. vis_tpfpfn :
        Visualise TP, FP and FN in selected color format + detailed info on count
        True   => Visualise TP, FP and FN in selected color code format for TP, FP and FN respectively.
        False  => default mode, which visualises images and detections normally with random colors for each bounding box
    c. text_info  :
        Prints number of TPs, FPs, FNs on top of image
        True
        False
    both only_bbox and vis_tpfpfn cannot be true at once.
"""
pool_executors = 12
vis_features = Vis_Features(False, True, True)
if vis_features.only_bbox and vis_features.vis_tpfpfn:
    sys.exit('vis_features: both only_bbox and vis_tpfpfn cannot be true at once.')
visualizer.draw_instance_gt_pred_offline(img_ids=img_ids, no_imgs=None, viz_gt=True, viz_dt=True, viz_cdt=False, viz_rdt=False, viz_rcdt=False, gt_json_file=gt_json_file,
    dt_json_file=dt_json_file, resil_name='ranger', epoch=epochs, vis_features = vis_features, pool_executors=pool_executors, batch_size=batch_size)


"""
SAVE_VIDEO
----- save the visualised predictions into a video format -----
##### 



viz_path  => path to images to be compiled for the video
             /../**/vis/images -> considers visualised detections with original image
             /../**/vis/images_only_bbox -> considers visualises detections with filled bounding boxes against a black and white background (B&W image)
vis_types => compile videos from either 
             'gt'   -> ground truth
             'dt'   -> original detections
             'cdt'  -> corrupt detections
             'rcdt' -> resiliency applied corrupt detections
epochs    =  epochs to be considered from the experiment folder. ex: [0, 1, ..., 25] or [4, 23] or [9]
uncomment line -> "visualizer.save_video(args)"
"""
# vis_types = ['dt', 'cdt']
# viz_path = os.path.join(os.path.dirname(gt_json_file), 'vis/images')
# visualizer.save_video(viz_path=viz_path, epochs=epochs, vis_types=vis_types) ## saves prediction videos


# epochs = [0]
# vis_iso_types=  [None]
# viz_path = os.path.join(os.path.dirname(gt_json_file), 'vis/images_only_bbox')
# visualizer.save_blob_video(viz_path=viz_path, epochs=epochs, vis_iso_types=vis_iso_types, only_bbox=True) ## saves isolated videos

