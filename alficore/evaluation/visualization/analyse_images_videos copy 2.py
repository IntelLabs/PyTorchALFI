import sys
import os
from os.path import dirname as up
from collections import namedtuple
# sys.path.append(sys.argv[0])
# sys.path.insert(0, up(up(up(os.getcwd()))))
## comment this for debug
sys.path.append(up(up(up(os.getcwd()))))

## uncomment this for debug
# sys.path.append(os.getcwd())

from alficore.evaluation.visualization.visualization import Visualizer, ColorMode
from alficore.ptfiwrap_utils.utils import  read_json
from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
from PIL import Image, TiffImagePlugin
TiffImagePlugin.DEBUG = False
# image = np.random.randn(416,416,3)
trails = 3 #nr of epochs here
fault = 1
# inj_policy = 'per_epoch'
inj_policy = 'per_image'
# inj_policy = 'per_batch'
inj_type = 'neurons'
# inj_type = 'weights'

# uuid objDet_20220128-112931_1_faults_[1]_bits  100 neuron
# uuid = 'objDet_20220316-152500_1_faults_[0,8]_bits'
uuid = 'objDet_20220322-001237_1_faults_[0,8]_bits'
# uuid = "objDet_20220316-121554_1_faults_[0,8]_bits" #coco
# uuid = "objDet_20220316-122047_1_faults_[0,8]_bits" ## 100 weights #robo
# model_name = 'det2_fasterRCNN'
model_name = 'yolov3_ultra'
# dataset = 'coco2017'
dataset = 'robo'
# dataset = 'kitti'
epoch = 0
batch_size = 1

# coco = CoCo_obj_det_dataloader(dataset_type='val', batch_size=4, sampleN=0.001, num_workers=4)

# metadata = MetadataCatalog.get('coco2017/{}'.format('val'))
# img_ids=[139, 285, 632, 724, 776, 785, 802, 872]
# img_ids = [0]
img_ids = None
epochs = list(range(trails))
ground_truth_file = '/home/fgeissle/ranger_repo/ranger/result_files/{}_{}_trials/{}_injs/{}/{}/{}/val/coco_format.json'
original_epoch_json_file = '/home/fgeissle/ranger_repo/ranger/result_files/{}_{}_trials/{}_injs/{}/{}/{}/val/orig_model/epochs/{}/coco_instances_results_{}_epoch.json'
# /home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_1_trials/neurons_injs/per_image/objDet_20220316-122047_1_faults_[0,8]_bits/robo/val/orig_model/epochs/0/coco_instances_results_0_epoch.json
# /home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_1_trials/neurons_injs/per_image/objDet_20220316-122047_1_faults_[0,8]_bits/robo/val/orig_model/epochs/0/coco_instances_results_0_epoch.json
# /home/fgeissle/ranger_repo/ranger/result_files/yolov3_ultra_10_trials/neurons_injs/per_image/objDet_20220316-122047_1_faults_[0,8]_bits/robo/val/coco_format.json

gt_json_file = ground_truth_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset)
dt_json_file = original_epoch_json_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset, epoch, epoch)

gt_json_data = read_json(gt_json_file)
thing_classes = [a['name'] for a in gt_json_data['categories']]
from copy import deepcopy
sv = deepcopy(thing_classes[0] )
thing_classes[0] = thing_classes[1]
thing_classes[1] = sv

MetadataCatalog.get('{}/{}'.format(dataset, 'val')).set(
        thing_classes=thing_classes, dataset=dataset)
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
        True   => Visualise detections with filled bounding boxes (B&W image); stores images in 'vis/images'.
        False  => Visualise detections with original image and colourfull bounding boxes; stores images in 'vis/images_only_bbox'.
        None   => Visualises detections using normal images and also with filled black and white bounding boxes.
    b. vis_tpfpfn :
        True   => Visualise TP, FP and FN in selected color format + detailed info on count.
        False  => default mode, which visualises images and detections normally with random colors
    c. text_info  :
        True   => Prints number of TPs, FPs, FNs
        False  => only visualises the TPs, FPs and FNs in different colors without the total numbers
    both only_bbox and vis_tpfpfn cannot be true at once.
"""
vis_features = Vis_Features(False, True, True)
if vis_features.only_bbox and vis_features.vis_tpfpfn:
    sys.exit('vis_features: both only_bbox and vis_tpfpfn cannot be true at once.')
visualizer.draw_instance_gt_pred_offline(img_ids=img_ids, no_imgs=None, viz_gt=True, viz_dt=True, viz_cdt=True, viz_rdt=True, viz_rcdt=True, gt_json_file=gt_json_file,
    dt_json_file=dt_json_file, resil_name='ranger', epoch=epochs, vis_features = vis_features)


"""
##### ----- visualises all the predictions -----#####

DRAW_INSTANCE_GT_PRED_OFFLINE

viz_path  => path to images to be compiled for the video
             /../**/vis/images -> considers visualised detections with original image
             /../**/vis/images_only_bbox -> considers visualises detections with filled bounding boxes (B&W image)
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

