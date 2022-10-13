# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

# Copyright (c) Facebook, Inc. and its affiliates.
# # intel copyright
from enum import Enum, unique
from numpy import inf, insert
from pathlib import Path
import numpy as np
import pathlib
import colorsys
import logging
import os
import sys
from os.path import dirname as up
import yaml
import math
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from numpy.lib.type_check import imag
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from alficore.evaluation.ivmod_metric import ivmod_metric
from alficore.evaluation.visualization.colormap import random_color
from alficore.ptfiwrap_utils.utils import read_faultbin_file, read_json, read_yaml
from PIL import Image
import json
from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
from ...dataloader.objdet_baseClasses.boxes import BoxMode, Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
import math
import concurrent.futures
import concurrent

from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

__all__ = ["ColorMode", "VisImage", "Visualizer"]


_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """

def _create_text_labels_catch_exception(class_names, i):
    if i <= len(class_names):
        return class_names[i]
    else:
        return class_names[-1]
    
def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [_create_text_labels_catch_exception(class_names, i) for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

class Visualizer(object):
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions,sem_seg,panoptic_seg_predictions,dataset_dict}`
    that draw composite data in some pre-defined style.

    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according
    to different heuristics, as long as the results still look visually reasonable.

    To obtain a consistent style, you can implement custom drawing functions with the
    abovementioned primitive methods instead. If you need more customized visualization
    styles, you can process the data yourself following their format documented in
    tutorials (:doc:`/tutorials/models`, :doc:`/tutorials/datasets`). This class does not
    intend to satisfy everyone's preference on drawing styles.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    # TODO implement a fast, rasterized version using OpenCV

    def __init__(self, img_rgb=None, metadata=None, scale=1.0, vis_mode='online', instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        self.cpu_device = torch.device("cpu")
        self.save_folder_path = None
        self.scale = scale
        if metadata is None:
            metadata = MetadataCatalog.get("__nonexist__")
        self.metadata = metadata

        if vis_mode == 'online':
            self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
            self.output = VisImage(self.img, scale=self.scale)
        # too small texts are useless, therefore clamp to 9
            self._default_font_size = max(
                np.sqrt(self.output.height * self.output.width) // 90, 10 // self.scale
            )
        self._instance_mode = instance_mode

    @classmethod
    def check_bbox(self, bbox):
        # bboxes = [_bbox if (isinstance(_bbox, (float, int)) and _bbox>=0) else 0 for _bbox in bbox]
        bboxes = [v if not (math.isinf(v) or math.isnan(v)) else 5 for v in bbox]
        # for _bbox in bboxes:
        #     print(_bbox)
        #     if isinstance(_bbox, (float, int)):
        #         # print(_bbox)
        #         pass
        #     else:
        #         print('I found infinity')        
        return bboxes

    def create_instance(self, img_size=416, annotations=None):
        """
        pred_boxes should be passed in raw XYWH format and internally this function converts them back to native
        format preferred by alficore functions viz XYXY  format.
        """
        pred_boxes_xywh = [annotations[i]['bbox'] for i in range(len(annotations))]
        pred_boxes_xyxy = [BoxMode.convert(box, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for box in pred_boxes_xywh]
        pred_boxes = [torch.Tensor(pred_boxes) for pred_boxes in pred_boxes_xyxy]
        try:
            pred_scores = [annotations[i]['score'] for i in range(len(annotations))]
        except:
            pred_scores = [0]*len(pred_boxes)
        if "coco" in self.metadata.name.lower() or "robo" in self.metadata.name.lower() or "ppp" in self.metadata.name.lower():
            """
            for CoCo datasets
            """
            try:
                pred_classes = [self.metadata.thing_dataset_id_to_contiguous_id[annotations[i]['category_id']] for i in range(len(annotations))]
            except:
                print("Input has the wrong class labels (0-80) instead of (1-91)?)")
                sys.exit()
        else:
            pred_classes = [annotations[i]['category_id'] for i in range(len(annotations))]
        
        if not len(pred_classes):
            pred_classes = [-1]*len(pred_boxes)
        if not len(pred_scores):
            pred_scores = [0]*len(pred_boxes)
        
        instance = Instances(416)
        instance.set("pred_boxes", Boxes(torch.stack(pred_boxes, 0)) if len(pred_boxes)>0 else  Boxes([]))
        instance.set("pred_classes", torch.Tensor(pred_classes).type(torch.ByteTensor))
        instance.set("scores", torch.Tensor(pred_scores))
        return instance

    def visualize_ground_truth(self, img_ids=None, vis_features=None):
        img_ids = self.img_ids
        vis_features = self.vis_features
        iou_thresh = vis_features.iou_thresh if hasattr(vis_features, "iou_thresh") else 0.5
        eval_mode = vis_features.eval_mode if hasattr(vis_features, "eval_mode") else "iou+class_labels"
        coco_gt = self.gt_json_data
        index_list, list_nr_images, valid_inds = find_non_annotated_images(coco_gt)
        coco_labels_grouped = group_objects_by_imageId(coco_gt['annotations'], list_nr_images)

        for img_id in img_ids:
            if vis_features.vis_tpfpfn:
                tp_fp_fn = ivmod_metric(coco_labels_grouped[img_id], coco_labels_grouped[img_id], iou_thresh=iou_thresh, eval_mode=eval_mode) #get one dict per image
                tp_gt_instance = self.create_instance(annotations=tp_fp_fn['tp'])
                fp_gt_instance = self.create_instance(annotations=tp_fp_fn['fp'])
                fn_gt_instance = self.create_instance(annotations=tp_fp_fn['fn'])
                # gt_instance = [tp_gt_instance, fp_gt_instance, fn_gt_instance]
            #Draw Ground Truth Annotations
            if vis_features.only_bbox is None or vis_features.only_bbox is True:
                gt_anno = [anno for anno in coco_gt['annotations'] if anno['image_id'] == img_id]
                gt_instance = self.create_instance(annotations=gt_anno)

            if vis_features.only_bbox is None:
                vis_types = [True, False] ## [only_bbox: True, only_bbox: False]
            else:
                vis_types = [vis_features.only_bbox]
            for vis_type in vis_types:
                save_folder_path = os.path.join(os.path.dirname(self.gt_json_file), 'vis', 'images_only_bbox' if vis_type else 'images')
                save_folder_path = os.path.join(save_folder_path, str(img_id))
                self.save_file_name = os.path.join(save_folder_path, 'test_viz_'+str(img_id)+'_gt.png')
                if os.path.exists(self.save_file_name): 
                    print('File already exists, exiting...')
                    continue
                image_file_name =  [img['file_name'] for img in coco_gt['images'] if img['id'] == img_id][0]
                img_rgb = np.array(Image.open(image_file_name))
                if vis_type:
                    img_rgb = np.zeros((img_rgb.shape))
                self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
                # self.output = VisImage(self.img)
                # self._default_font_size = max(
                #                 np.sqrt(self.output.height * self.output.width) // 90, 10 // self.scale
                #             )
                if vis_features.vis_tpfpfn and not vis_type:
                    orig_label = "TP: " + str(len(tp_fp_fn['tp'])) + ", " + "FP: " + str(len(tp_fp_fn['fp'])) + ", " + "FN: " + str(len(tp_fp_fn['fn']))
                    self.simple_visualization(inset_str=orig_label, tp_instances=tp_gt_instance, fp_instances=fp_gt_instance, fn_instances=fn_gt_instance)
                else:
                    self.output = VisImage(self.img)
                    self._default_font_size = max(
                                    np.sqrt(self.output.height * self.output.width) // 90, 10 // self.scale
                                )                    
                    vis_output_gt = self.draw_instance_predictions(predictions=gt_instance, only_bbox=vis_type)
                
                    os.makedirs(save_folder_path, exist_ok=True)
                    vis_output_gt.save(self.save_file_name)
                    print('saved gt visualisation at {}'.format(self.save_file_name))

    def visualize_detection(self, epoch=0, _type='dt'):
        if _type == 'dt':
            dt_json_file = self.dt_json_file
        elif _type == 'rdt':
            dt_json_file = self.dt_json_file.replace('orig', self.resil_name)
        elif _type == 'cdt':
            dt_json_file = self.dt_json_file.replace('orig', 'corr').replace('0/', '{}/'.format(epoch)).replace('_0_', '_{}_'.format(epoch))
        elif _type == 'rcdt':
            dt_json_file = self.dt_json_file.replace('orig', self.resil_name + '_corr').replace('0/', '{}/'.format(epoch)).replace('_0_', '_{}_'.format(epoch))
        img_ids = self.img_ids
        vis_features = self.vis_features
        dt_json_data = read_json(dt_json_file)
        iou_thresh   = vis_features.iou_thresh if hasattr(vis_features, "iou_thresh") else 0.5
        eval_mode    = vis_features.eval_mode if hasattr(vis_features, "eval_mode") else "iou+class_labels"
        coco_gt = self.gt_json_data
        coco_dt      =  dt_json_data
        index_list, list_nr_images, valid_inds = find_non_annotated_images(coco_gt)
        coco_detecs_grouped = group_objects_by_imageId(coco_dt, list_nr_images)

        index_list, list_nr_images, valid_inds = find_non_annotated_images(coco_gt)
        coco_labels_grouped = group_objects_by_imageId(coco_gt['annotations'], list_nr_images)
        injection_type, faults = injType_and_faults_(up(self.gt_json_file), batch_size=self.batch_size, _type=_type)
        for n in range(len(img_ids)):
            img_id = img_ids[n]
            if vis_features.vis_tpfpfn:
                tp_fp_fn = ivmod_metric(coco_labels_grouped[img_id], coco_detecs_grouped[img_id], iou_thresh=iou_thresh, eval_mode=eval_mode) #get one dict per image
                tp_dt_instance = self.create_instance(annotations=tp_fp_fn['tp'])
                fp_dt_instance = self.create_instance(annotations=tp_fp_fn['fp'])
                fn_dt_instance = self.create_instance(annotations=tp_fp_fn['fn'])
                if not(len(fp_dt_instance.pred_boxes)) and not(len(tp_dt_instance.pred_boxes)):
                    print('\n\n' + '\033[1m No detections in {} visualisation - {}\n\n'.format(_type, self.save_file_name))
            if vis_features.only_bbox is None or vis_features.only_bbox is True:
                dt_anno = [anno for anno in coco_dt if anno['image_id'] == img_id]
                dt_instance = self.create_instance(annotations=dt_anno)
                if not(len(dt_instance.pred_boxes)):
                    print('\n\n' + '\033[1m No detections in {} visualisation - {}\n\n'.format(_type, self.save_file_name))

            if vis_features.only_bbox is None:
                vis_types = [True, False] ## [only_bbox: True, only_bbox: False]
            else:
                vis_types = [vis_features.only_bbox]
            for vis_type in vis_types:
                save_folder_path = os.path.join(os.path.dirname(self.gt_json_file), 'vis', 'images_only_bbox' if vis_type else 'images')
                if _type == 'dt' or _type == 'rdt':
                    save_folder_path = os.path.join(save_folder_path, str(img_id))
                else:
                    save_folder_path = os.path.join(save_folder_path, str(img_id), str(epoch))
                self.save_file_name = os.path.join(save_folder_path, 'test_viz_'+str(img_id)+'_{}.png'.format(_type))
                if os.path.exists(self.save_file_name):
                    ## skipping already rendered visualisations.
                    continue
                image_file_name =  [img['file_name'] for img in self.gt_json_data['images'] if img['id'] == img_id]
                img_rgb = np.array(Image.open(image_file_name[0]))
                if vis_type:
                    img_rgb = np.zeros((img_rgb.shape))
                self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
                if vis_features.vis_tpfpfn and not vis_type:
                    if "c" in _type: ## selecting the right text for the type of fault corruption info
                        if injection_type == 'weights':
                            label = "TP: " + str(len(tp_fp_fn['tp'])) + ", " + "FP: " + str(len(tp_fp_fn['fp'])) + ", " + "FN: " + str(len(tp_fp_fn['fn'])) \
                                + ", " + "b: " + str(int(faults[6, n+epoch*(len(img_ids))])) + ", " + "l: " + str(int(faults[0, n+epoch*(len(img_ids))]))
                        elif injection_type == "neurons":
                            label = "TP: " + str(len(tp_fp_fn['tp'])) + ", " + "FP: " + str(len(tp_fp_fn['fp'])) + ", " + "FN: " + str(len(tp_fp_fn['fn'])) \
                                + ", " + "b: " + str(int(faults[6, n+epoch*(len(img_ids))])) + ", " + "l: " +  str(int(faults[1, n+epoch*(len(img_ids))]))
                    else:
                        label = "TP: " + str(len(tp_fp_fn['tp'])) + ", " + "FP: " + str(len(tp_fp_fn['fp'])) + ", " + "FN: " + str(len(tp_fp_fn['fn']))
                    self.simple_visualization(inset_str=label, tp_instances=tp_dt_instance, fp_instances=fp_dt_instance, fn_instances=fn_dt_instance)
                else:    
                    self.output = VisImage(self.img)
                    self._default_font_size = max(
                        np.sqrt(self.output.height * self.output.width) // 90, 10 // self.scale
                    )                
                    vis_output_dt = self.draw_instance_predictions(predictions=dt_instance, only_bbox=vis_type)
                    os.makedirs(save_folder_path, exist_ok=True)
                    vis_output_dt.save(self.save_file_name)
                    print('saved {} visualisation at {}'.format(_type, self.save_file_name))

    def save_video(self, viz_path, epochs, vis_types=['dt'], only_bbox=False):
        path = pathlib.PurePath(viz_path)
        path = pathlib.PurePath(path)
        path_suffix = path.name
        path_suffix = path_suffix.split("_")
        path_suffix = "_".join(path_suffix[1:]) if len(path_suffix)>1 else path_suffix[0]
        for epoch in epochs:
            for viz in vis_types:
                sample_epoch = 0
                sample_img_id = 0
                if viz in ['gt', 'dt', 'rdt']:
                    # if not vis_iso_type:
                    img_path = os.path.join(viz_path, str(sample_img_id), "test_viz_{}_{}.png".format(sample_img_id, viz))
                    output = os.path.join(os.path.dirname(viz_path), 'videos', "test_viz_{}_{}.mp4".format(viz, path_suffix))
                    # else:
                    #     img_path = os.path.join(viz_path, str(sample_img_id), vis_iso_type, "test_viz_{}_{}.png".format(sample_img_id, viz))
                    #     output = os.path.join(os.path.dirname(viz_path), 'videos', vis_iso_type, "test_viz_{}_{}.mp4".format(viz, vis_iso_type))
                elif viz in ['cdt', 'rcdt']:
                    img_path = os.path.join(viz_path, str(sample_img_id), str(epoch), "test_viz_{}_{}.png".format(sample_img_id, viz))
                    output = os.path.join(os.path.dirname(viz_path), 'videos', "{}".format(epoch), "test_viz_{}_{}.mp4".format(viz, path_suffix))
                if not os.path.exists(img_path):
                    print("img {} missing".format)
                sample_frame = cv2.imread(img_path)
                height, width, channels = sample_frame.shape
                if not os.path.exists(output):
                    os.makedirs(os.path.dirname(output), exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
                    out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # org
                    org = (50, 50)
                    # fontScale
                    fontScale = 1
                    # Blue color in BGR
                    color = (0, 255, 0)
                    
                    # Line thickness of 2 px
                    thickness = 2
                    for img_id in range(len(os.listdir(viz_path))):
                        if viz in ['gt', 'dt', 'rdt']:
                            img_path = os.path.join(viz_path, str(img_id), "test_viz_{}_{}.png".format(img_id, viz))
                        elif viz in ['cdt', 'rcdt']:
                            img_path = os.path.join(viz_path, str(img_id), str(epoch), "test_viz_{}_{}.png".format(img_id, viz))
                        if not os.path.exists(img_path):
                            print("{} is missing".format(img_path))
                            # exit()
                        frame = cv2.imread(img_path)
                        frame = cv2.putText(frame, 'frame id: {}'.format(img_id), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        out.write(frame) # Write out frame to video

                        # cv2.imshow('video',frame)
                        # if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                        #     break

                    # Release everything if job is finished
                    out.release()
                    # cv2.destroyAllWindows()

                    print("The output video is {}".format(output))
                else:
                    print("The output video {} already exists".format(output))

    def draw_instance_gt_pred_offline(self, no_imgs=5, img_ids=None, **kwargs):
        """
        img_ids: list of image_ids that are to be visualized
        no_imgs: no of images to be chosen at random and visualized
        Only one of img_ids and no_imgs need to be specified
        If no img_ids is provided as input, it selects no_imgs images at random for visualization
        If both img_ids and no_images are None, the func will plot all the images
        viz_gt: Set to True to visualize ground truth
        viz_dt: Set to True to visualize detection
        viz_cdt: Set to True to visualize corr detection
        viz_rdt: Set to True to visualize resiliency applied detection
        viz_rcdt: Set to True to visualize resiliency applied + corr detection
        """
        self.img_ids = img_ids
        self.no_imgs = no_imgs
        self.gt_json_file = kwargs.get("gt_json_file", None)
        self.dt_json_file =  kwargs.get("dt_json_file", None)
        self.viz_gt =  kwargs.get("viz_gt", True)        
        self.viz_dt =  kwargs.get("viz_dt", True)
        self.viz_cdt =  kwargs.get("viz_cdt", False)
        self.viz_rdt =  kwargs.get("viz_rdt", False)
        self.viz_rcdt =  kwargs.get("viz_rcdt", False)
        self.resil_name =  kwargs.get("resil_name", 'ranger')
        self.epoch_list =  kwargs.get("epoch", [0])
        self.vis_features  =  kwargs.get("vis_features", None)
        self.pool_executors = kwargs.get("pool_executors", 1)
        self.batch_size = kwargs.get("batch_size", 1)
        self.gt_json_data = read_json(self.gt_json_file)

        if self.img_ids is None and self.no_imgs is not None:
            self.image_ids = [img['id'] for img in self.gt_json_data['images']]
            self.img_ids = np.random.choice(self.image_ids, self.no_imgs).tolist()
        elif self.img_ids is None and self.no_imgs is None:
            self.img_ids = [img['id'] for img in self.gt_json_data['images']]

        # self.save_folder_path = os.path.join(os.path.dirname(self.gt_json_file), 'vis', 'images_only_bbox' if self.only_bbox else 'images')
        if self.viz_gt:
            self.visualize_ground_truth()
        if self.viz_dt:
            self.visualize_detection(_type='dt')
        if self.viz_rdt:
            self.visualize_detection(_type='rdt')
        
        for epoch in self.epoch_list:
            if self.viz_cdt and epoch is not None:
                self.visualize_detection(epoch=epoch, _type='cdt')
            if self.viz_rcdt and epoch is not None:
                json_file = self.dt_json_file.replace('orig', self.resil_name + '_corr').replace('0/', '{}/'.format(epoch)).replace('_0_', '_{}_'.format(epoch))
                self.visualize_detection(epoch=epoch, _type='rcdt')
    
    def draw_instance_corr_pred_offline(self, no_imgs=5, img_ids=None):
        """
        img_ids: list of image_ids that are to be visualized
        no_imgs: no of images to be chosen at random and visualized
        Only one of img_ids and no_imgs need to be specified
        If no image_ids is provided as input, it selects no_imgs images at random for visualization
        vis_gt: Set to True to visualize ground truth
        vis_dt: Set to True to visualize detection
        """
        ## gt visualisation
        # gt_obj = first visualise gt using the seperate func that you have written 
        ## dt visualisation
        # dt_obj = use draw_instance_predictions  func to visualise predictions
        # final_img = gt_obj.fig/img and dt_objfig/img side by side
        # matplot lib to plot these 2 fun
        # return final_img
        self.visualize_detection(no_imgs, img_ids)

    
    def draw_instance_predictions(self, predictions, **kwargs):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        kitti_labels = ['Car', 'Pedestrian', 'Van', 'Truck', 'Cyclist', 'DontCare', 'Person_sitting', 'Tram', 'Misc']
        ids = kwargs.get("ids", None)
        only_bbox = kwargs.get("only_bbox", False)
        text_shift = kwargs.get("text_shift", None)
        if text_shift is not None:
            self.text_shift = text_shift
        else:
            self.text_shift = (0,0)

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        class_names = self.metadata.get("thing_classes", None)
        # if self.metadata.get('dataset') is not None:
        #     if self.metadata.dataset == 'kitti': #replace for kitti since not yet updated
        #         class_names = ['Car', 'Pedestrian', 'Van', 'Truck', 'Cyclist', 'DontCare', 'Person_sitting', 'Tram', 'Misc'] #Only for kitti
        #     elif self.metadata.dataset == 'lyft':
        #         class_names = ['car', 'pedestrian', 'animal', 'other_vehicle', 'bus', 'motorcycle', 'truck', 'bicycle']
        #         # ['animal': 9, 'bicycle': 5, 'bus': 4, 'car': 1, 'emergency_vehicle': 4, 'motorcycle': 5, 'other_vehicle': 7, 'pedestrian': 7, 'truck': 4}
        
        labels = _create_text_labels(classes, scores, class_names)
        # labels = _create_text_labels(classes, scores, kitti_labels)
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            alpha = 0.5
            colors = kwargs.get("colors", None)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
            alpha = 0.3

        if ids is not None:
            labels2 = []
            for l in range(len(labels)):
                ls = labels[l].split()
                labels2.append(ls[0] + ' ' + str(ids[l]) + ' ' + ls[1])
            labels = labels2
        self.overlay_instances(
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
            only_bbox=only_bbox,
        )
        return self.output


    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]

            colors = None
            category_ids = [x["category_id"] for x in annos]
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                    for c in category_ids
                ]
            names = self.metadata.get("thing_classes", None)
            labels = _create_text_labels(
                category_ids,
                scores=None,
                class_names=names,
                is_crowd=[x.get("iscrowd", 0) for x in annos],
            )
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )
        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
        only_bbox=False
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                if not only_bbox:
                    self.draw_box(boxes[i], edge_color=color)
                if only_bbox:
                    self.draw_box(boxes[i], edge_color=None, fill=True, face_color='white', alpha=1)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    text_pos = (x0 + (x1-x0)/2, y0 + (y1-y0)/2)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                if not only_bbox:
                    self.draw_text(
                        labels[i],
                        tuple(np.array(text_pos) + np.array(self.text_shift)),
                        color=lighter_color,
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                    )


        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    def overlay_rotated_instances(self, boxes=None, labels=None, assigned_colors=None):
        """
        Args:
            boxes (ndarray): an Nx5 numpy array of
                (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image.
            labels (list[str]): the text to be displayed for each instance.
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = len(boxes)

        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        if boxes is not None:
            areas = boxes[:, 2] * boxes[:, 3]

        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs]
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            self.draw_rotated_box_with_label(
                boxes[i], edge_color=colors[i], label=labels[i] if labels is not None else None
            )

        return self.output

    """
    Primitive drawing functions:
    """

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-", fill=False, face_color=None):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=fill,
                edgecolor= edge_color,
                facecolor =face_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_rotated_box_with_label(
        self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None
    ):
        """
        Draw a rotated box with label on its top-left corner.

        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): image object with box drawn.
        """
        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            6 if area < _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3
        )

        theta = angle * np.pi / 180.0
        c = np.cos(theta)
        s = np.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
            )
            self.draw_text(label, text_pos, color=label_color, font_size=font_size, rotation=angle)

        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output

    def draw_binary_mask(
        self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5, area_threshold=0
    ):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn in the object's center of mass.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component small than this will not be shown.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes:
            # draw polygons for regular masks
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
        else:
            # TODO: Use Path/PathPatch to draw vector graphics:
            # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            has_valid_segment = True
            self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))

        if text is not None and has_valid_segment:
            # TODO sometimes drawn on wrong objects. the heuristics here can improve.
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
            largest_component_id = np.argmax(stats[1:, -1]) + 1

            # draw text on the largest component, as well as other very large components.
            for cid in range(1, _num_cc):
                if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                    # median is more stable than centroid
                    # center = centroids[largest_component_id]
                    center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                    self.draw_text(text, center, color=lighter_color)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    """
    Internal methods:
    """

    def _jitter(self, color):
        """
        Randomly modifies given color to produce a slightly different color than the color given.

        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the [0.0, 1.0] range.

        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
                color after being jittered. The values in the list are in the [0.0, 1.0] range.
        """
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _create_grayscale_image(self, mask=None):
        """
        Create a grayscale version of the original image.
        The colors in masked area, if given, will be kept.
        """
        img_bw = self.img.astype("f4").mean(axis=2)
        img_bw = np.stack([img_bw] * 3, axis=2)
        if mask is not None:
            img_bw[mask] = self.img[mask]
        return img_bw

    def _change_color_brightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes):
            if boxes.tensor.is_cuda:
                return boxes.tensor.cpu().detach().numpy()
            else:
                return boxes.tensor.detach().numpy()
        else:
            return np.asarray(boxes)

    def _convert_masks(self, masks_or_polygons):
        """
        Convert different format of masks or polygons to a tuple of masks and polygons.

        Returns:
            list[GenericMask]:
        """
        # m = masks_or_polygons
        # if isinstance(m, PolygonMasks):
        #     m = m.polygons
        # if isinstance(m, BitMasks):
        #     m = m.tensor.numpy()
        # if isinstance(m, torch.Tensor):
        #     m = m.numpy()
        # ret = []
        # for x in m:
        #     if isinstance(x, GenericMask):
        #         ret.append(x)
        #     else:
        #         ret.append(GenericMask(x, self.output.height, self.output.width))
        # return ret
        pass

    def get_output(self):
        """
        Returns:
            output (VisImage): the image output containing the visualizations added
            to the image.
        """
        return self.output

    def simple_visualization(self, **kwargs):
        # Specific class conversions for coco are included.

        ids = kwargs.get('ids', None)
        inset_str = kwargs.get('inset_str', None)
        tp_instances = kwargs.get('tp_instances', None)
        fp_instances = kwargs.get('fp_instances', None)
        fn_instances = kwargs.get('fn_instances', None)
        only_bbox = kwargs.get('only_bbox', False)

        # colors = kwargs.get('colors', None)

        from copy import deepcopy
        from alficore.dataloader.objdet_baseClasses.instances import Instances

        def copy_instance(out_instance):
            if out_instance is None:
                return None
            tp_instances_copy = Instances(out_instance.image_size)
            if isinstance(out_instance.pred_boxes, Boxes):
                tp_instances_copy.pred_boxes = deepcopy(out_instance.pred_boxes.tensor.detach().cpu())
            else:
                tp_instances_copy.pred_boxes = deepcopy(out_instance.pred_boxes)
            tp_instances_copy.pred_classes = deepcopy(out_instance.pred_classes.detach())
            if isinstance(out_instance.scores, list):
                tp_instances_copy.scores = deepcopy(out_instance.scores)
            else:
                tp_instances_copy.scores = deepcopy(out_instance.scores.detach())
            return tp_instances_copy

        tp_instances_copy = copy_instance(tp_instances)
        fp_instances_copy = copy_instance(fp_instances)
        fn_instances_copy = copy_instance(fn_instances)

        visualizer = Visualizer(self.img, self.metadata) #labels are strings
        
        if fp_instances_copy is not None and fn_instances_copy is None: #one extra list
            # Put different colors to distinguish
            colors1 = [np.array([0,1,0]) for n in range(len(tp_instances_copy.scores))]
            colors2 = [np.array([0.8,0.2,0]) for n in range(len(fp_instances_copy.scores))]
            visualizer.draw_instance_predictions(tp_instances_copy, ids=ids, colors=colors1, only_bbox=only_bbox)
            visualizer.draw_instance_predictions(fp_instances_copy, ids=ids, colors=colors2, only_bbox=only_bbox)
        elif fp_instances_copy is not None and fn_instances_copy is not None: #two extra lists
            # Put different colors to distinguish
            colors1 = [np.array([0,1,0]) for n in range(len(tp_instances_copy.scores))] #green, tp
            colors2 = [np.array([0.8,0.2,0]) for n in range(len(fp_instances_copy.scores))] #orange: fp
            colors3 = [np.array([0,0,1]) for n in range(len(fn_instances_copy.scores))] #blue: fn
            visualizer.draw_instance_predictions(tp_instances_copy, ids=ids, colors=colors1, only_bbox=only_bbox)
            visualizer.draw_instance_predictions(fp_instances_copy, ids=ids, colors=colors2, only_bbox=only_bbox)
            visualizer.draw_instance_predictions(fn_instances_copy, ids=ids, colors=colors3, only_bbox=only_bbox)
        else:
            visualizer.draw_instance_predictions(tp_instances_copy, ids=ids, only_bbox=only_bbox)


        os.makedirs(os.path.dirname(self.save_file_name), exist_ok=True)

        if inset_str is not None:
            fig, ax = visualizer.output.fig, visualizer.output.ax
            fig.text(0.5, 0.97, inset_str, horizontalalignment='center', verticalalignment='center', fontsize=14, color='black', bbox=dict(facecolor='white', edgecolor='none', pad=2.0))

        visualizer.output.save(self.save_file_name)
        print('saved gt visualisation at {}'.format(self.save_file_name))



def simple_visualization(input_dict, out_instance, output_path, dataset_name, **kwargs):
    # Specific class conversions for coco are included.

    ids = kwargs.get('ids', None)
    inset_str = kwargs.get('inset_str', None)
    extra_boxes = kwargs.get('extra_boxes', None)
    extra_boxes2 = kwargs.get('extra_boxes2', None)
    only_bbox = kwargs.get('only_bbox', False)
    # colors = kwargs.get('colors', None)

    from copy import deepcopy
    from alficore.dataloader.objdet_baseClasses.instances import Instances

    def copy_instance(out_instance):
        if out_instance is None:
            return None
        vis_copy = Instances(out_instance.image_size)
        if isinstance(out_instance.pred_boxes, Boxes):
            vis_copy.pred_boxes = deepcopy(out_instance.pred_boxes.tensor.detach().cpu())
        else:
            vis_copy.pred_boxes = deepcopy(out_instance.pred_boxes)
        vis_copy.pred_classes = deepcopy(out_instance.pred_classes.detach())
        if isinstance(out_instance.scores, list):
            vis_copy.scores = deepcopy(out_instance.scores)
        else:
            vis_copy.scores = deepcopy(out_instance.scores.detach())
        return vis_copy

    vis_copy = copy_instance(out_instance)
    extra_vis_copy = copy_instance(extra_boxes)
    extra_vis_copy2 = copy_instance(extra_boxes2)

    from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
    metadata = MetadataCatalog.get(dataset_name + '/val')

    #if dataset_name == "coco2014" or dataset_name == "coco2017" or dataset_name == 'robo' or dataset_name == 'ppp':
    #    dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
    #
    #    # Set back classes from 91 len to 80 len only to match with string labels of classes
    #    try:
    #        vis_copy.pred_classes = torch.tensor([dataset_id_to_contiguous_id[x] for x in vis_copy.pred_classes.tolist()], dtype=torch.uint8) #move to 0-80 class numbers to match with strings
    #        if extra_vis_copy is not None:
    #            extra_vis_copy.pred_classes = torch.tensor([dataset_id_to_contiguous_id[x] for x in extra_vis_copy.pred_classes.tolist()], dtype=torch.uint8) #move to 0-80 class numbers to match with strings
    #        if extra_vis_copy2 is not None:
    #            extra_vis_copy2.pred_classes = torch.tensor([dataset_id_to_contiguous_id[x] for x in extra_vis_copy2.pred_classes.tolist()], dtype=torch.uint8) #move to 0-80 class numbers to match with strings
    #    except:
    #        print("Input has the wrong class labels (0-80) instead of (1-91)?)")
    #        return

    img = Image.open(input_dict['file_name'])
    visualizer = Visualizer(np.array(img), metadata) #labels are strings

    # visualizer = Visualizer(np.array(img)) #labels are numbers
    
    if extra_vis_copy is not None and extra_vis_copy2 is None: #one extra list
        # Put different colors to distinguish
        colors1 = [np.array([0,1,0]) for n in range(len(vis_copy.scores))] #green, tp
        colors2 = [np.array([0.8,0.2,0]) for n in range(len(extra_vis_copy.scores))] #orange: fp
        visualizer.draw_instance_predictions(vis_copy, ids=ids, colors=colors1)
        # visualizer.output.save('test.png')
        visualizer.draw_instance_predictions(extra_vis_copy, ids=ids, colors=colors2, text_shift=(0,10))
    elif extra_vis_copy is not None and extra_vis_copy2 is not None: #two extra lists
        # Put different colors to distinguish
        colors1 = [np.array([0,1,0]) for n in range(len(vis_copy.scores))] #green, tp
        colors2 = [np.array([0.8,0.2,0]) for n in range(len(extra_vis_copy.scores))] #orange: fp
        colors3 = [np.array([0,0,1]) for n in range(len(extra_vis_copy2.scores))] #blue: fn
        visualizer.draw_instance_predictions(vis_copy, ids=ids, colors=colors1)
        visualizer.draw_instance_predictions(extra_vis_copy, ids=ids, colors=colors2, text_shift=(0,10))
        visualizer.draw_instance_predictions(extra_vis_copy2, ids=ids, colors=colors3, text_shift=(0,20))
    else:
        visualizer.draw_instance_predictions(vis_copy, ids=ids)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if inset_str is not None:
        fig, ax = visualizer.output.fig, visualizer.output.ax
        fig.text(0.5, 0.97, inset_str, horizontalalignment='center', verticalalignment='center', fontsize=14, color='black', bbox=dict(facecolor='white', edgecolor='none', pad=2.0))

    visualizer.output.save(output_path)
    print('saved as:', output_path)


def simple_visualization_direct_img(input_dict, out_instance, output_path, dataset_name, **kwargs):
    # Specific class conversions for coco are included.

    ids = kwargs.get('ids', None)
    inset_str = kwargs.get('inset_str', None)
    extra_boxes = kwargs.get('extra_boxes', None)
    extra_boxes2 = kwargs.get('extra_boxes2', None)
    only_bbox = kwargs.get('only_bbox', False)
    # colors = kwargs.get('colors', None)

    from copy import deepcopy
    from alficore.dataloader.objdet_baseClasses.instances import Instances

    def copy_instance(out_instance):
        if out_instance is None:
            return None
        vis_copy = Instances(out_instance.image_size)
        if isinstance(out_instance.pred_boxes, Boxes):
            vis_copy.pred_boxes = deepcopy(out_instance.pred_boxes.tensor.detach().cpu())
        else:
            vis_copy.pred_boxes = deepcopy(out_instance.pred_boxes)
        vis_copy.pred_classes = deepcopy(out_instance.pred_classes.detach())
        if isinstance(out_instance.scores, list):
            vis_copy.scores = deepcopy(out_instance.scores)
        else:
            vis_copy.scores = deepcopy(out_instance.scores.detach())
        return vis_copy

    vis_copy = copy_instance(out_instance)
    extra_vis_copy = copy_instance(extra_boxes)
    extra_vis_copy2 = copy_instance(extra_boxes2)

    from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
    metadata = MetadataCatalog.get(dataset_name + '/val')

    # if dataset_name == "coco2014" or dataset_name == "coco2017" or dataset_name == 'robo' or dataset_name == 'ppp':
    #     dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id

    #     # Set back classes from 91 len to 80 len only to match with string labels of classes
    #     try:
    #         vis_copy.pred_classes = torch.tensor([dataset_id_to_contiguous_id[x] for x in vis_copy.pred_classes.tolist()], dtype=torch.uint8) #move to 0-80 class numbers to match with strings
    #         if extra_vis_copy is not None:
    #             extra_vis_copy.pred_classes = torch.tensor([dataset_id_to_contiguous_id[x] for x in extra_vis_copy.pred_classes.tolist()], dtype=torch.uint8) #move to 0-80 class numbers to match with strings
    #         if extra_vis_copy2 is not None:
    #             extra_vis_copy2.pred_classes = torch.tensor([dataset_id_to_contiguous_id[x] for x in extra_vis_copy2.pred_classes.tolist()], dtype=torch.uint8) #move to 0-80 class numbers to match with strings
    #     except:
    #         print("Input has the wrong class labels (0-80) instead of (1-91)?)")
    #         return

    # img = Image.open(input_dict['file_name'])
    img = input_dict['image'].permute(1,2,0)
    visualizer = Visualizer(np.array(img), metadata) #labels are strings

    # visualizer = Visualizer(np.array(img)) #labels are numbers
    
    if extra_vis_copy is not None and extra_vis_copy2 is None: #one extra list
        # Put different colors to distinguish
        colors1 = [np.array([0,1,0]) for n in range(len(vis_copy.scores))] #green, tp
        colors2 = [np.array([0.8,0.2,0]) for n in range(len(extra_vis_copy.scores))] #orange: fp
        visualizer.draw_instance_predictions(vis_copy, ids=ids, colors=colors1)
        # visualizer.output.save('test.png')
        visualizer.draw_instance_predictions(extra_vis_copy, ids=ids, colors=colors2, text_shift=(0,10))
    elif extra_vis_copy is not None and extra_vis_copy2 is not None: #two extra lists
        # Put different colors to distinguish
        colors1 = [np.array([0,1,0]) for n in range(len(vis_copy.scores))] #green, tp
        colors2 = [np.array([0.8,0.2,0]) for n in range(len(extra_vis_copy.scores))] #orange: fp
        colors3 = [np.array([0,0,1]) for n in range(len(extra_vis_copy2.scores))] #blue: fn
        visualizer.draw_instance_predictions(vis_copy, ids=ids, colors=colors1)
        visualizer.draw_instance_predictions(extra_vis_copy, ids=ids, colors=colors2, text_shift=(0,10))
        visualizer.draw_instance_predictions(extra_vis_copy2, ids=ids, colors=colors3, text_shift=(0,20))
    else:
        visualizer.draw_instance_predictions(vis_copy, ids=ids)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if inset_str is not None:
        fig, ax = visualizer.output.fig, visualizer.output.ax
        fig.text(0.5, 0.97, inset_str, horizontalalignment='center', verticalalignment='center', fontsize=14, color='black', bbox=dict(facecolor='white', edgecolor='none', pad=2.0))

    visualizer.output.save(output_path)
    print('saved as:', output_path)
    
    
def simple_visualisation_bb(img, boxes, output_path, box_repr=0, labels=None):
    """
    Draw bounding boxes

    Args:
        predictions

    Returns:
        output (VisImage): image object with visualizations.
    """
    if isinstance(img, dict):
        img = Image.open(img['file_name'])
    visualizer = Visualizer(np.array(img))
    if box_repr == 0:
        boxes = [BoxMode.convert(box, BoxMode.XYXY_ABS, BoxMode.XYXY_ABS) for box in boxes]
    elif box_repr == 1:
        boxes = [BoxMode.convert(box, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for box in boxes]

    colors = None
    alpha = 0.5
    visualizer.overlay_instances(
        boxes=boxes,
        assigned_colors=colors,
        labels=labels,
        alpha=alpha,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visualizer.output.save(output_path)

def find_non_annotated_images(coco_gt):
    """
    Returns:
    - Indices of not-annotated images
    - list of IDs of valid images
    - list of indices of valid images
    """

    img_ids_gt = [n["image_id"] for n in coco_gt["annotations"]]
    img_ids_gt2 = [n["id"] for n in coco_gt["images"]]
    # print("images without annotations", set(img_ids_gt2) - set(img_ids_gt))
    unannotated_images = list(np.sort(list(set(img_ids_gt2) - set(img_ids_gt))))
    list_nr_images = list(np.sort(list(set(img_ids_gt))))

    index_list = []
    for x in unannotated_images:
        ind = np.where(np.array(img_ids_gt2) == x)[0]
        if ind:
            index_list.append(ind[0])
    
    all_inds = list((range(len(img_ids_gt2))))
    valid_inds = np.sort(list(set(index_list) ^ set(all_inds)))

    return index_list, list_nr_images, valid_inds

def group_objects_by_imageId(coco_labels, img_ids_ref):
    """
    Splits list of result dicts into sublits that belong to the same image_id.
    """

    img_ids_gt = [n["image_id"] for n in coco_labels] #possible that some ids do not show up
    # img_ids_sorted = list(np.sort(list(set(img_ids_gt))))

    coco_labels_grouped = {key: None for key in range(len(np.unique(img_ids_gt)))}

    for x in img_ids_ref:
        coco_labels_1img = []
        for u in range(len(img_ids_gt)):
            if img_ids_gt[u] == x:
                coco_labels_1img.append(coco_labels[u])
        coco_labels_grouped[x] = coco_labels_1img

    return coco_labels_grouped

def injType_and_faults_(folder_path, batch_size, _type):
    if "c" in _type:
        filelist = list( Path(folder_path).glob('**/*updated_rs_fault_locs.bin'))
        filelist = [filelist[a] for a in range(len(filelist)) if "{}bs".format(str(batch_size)) in filelist[a].name]
        if len(filelist)==2 and filelist[0]>filelist[1]:
            fault_file =  str(filelist[0])
        elif len(filelist)==2 and filelist[0]<filelist[1]:
            fault_file =  str(filelist[1])
        else:
            fault_file = filelist[0]
        faults = read_faultbin_file(fault_file)

        ## read injection type:
        yaml_file = list(Path(folder_path).glob('**/*.yml'))[0]
        data = read_yaml(yaml_file)[0]
        injection_type = data['rnd_mode']
        return injection_type, faults
    else:
        return None, None

def simple_xx_coverage(input_dict, out_instance, **kwargs):
    # Specific class conversions for coco are included.

    # ids = kwargs.get('ids', None)
    # inset_str = kwargs.get('inset_str', None)
    # fp_instances = kwargs.get('fp_instances', None)
    # fn_instances = kwargs.get('fn_instances', None)
    # only_bbox = kwargs.get('only_bbox', True)
    # colors = kwargs.get('colors', None)

    from copy import deepcopy
    from alficore.dataloader.objdet_baseClasses.instances import Instances

    def copy_instance(out_instance):
        if out_instance is None:
            return None
        tp_instances_copy = Instances(out_instance.image_size)
        if isinstance(out_instance.pred_boxes, Boxes):
            tp_instances_copy.pred_boxes = deepcopy(out_instance.pred_boxes.tensor.detach().cpu())
        else:
            tp_instances_copy.pred_boxes = deepcopy(out_instance.pred_boxes)
        tp_instances_copy.pred_classes = deepcopy(out_instance.pred_classes.detach())
        if isinstance(out_instance.scores, list):
            tp_instances_copy.scores = deepcopy(out_instance.scores)
        else:
            tp_instances_copy.scores = deepcopy(out_instance.scores.detach())
        return tp_instances_copy

    tp_instances_copy = copy_instance(out_instance)
    img = Image.open(input_dict['file_name'])
    visualizer = Visualizer(np.array(img)) #labels are strings


    visualizer.draw_instance_predictions(tp_instances_copy, only_bbox=True)
    visualizer.output.save('test.png')
    im_frame = np.array(Image.open('test.png'))[:,:,:3] #remove alpha channel
    # out_area, total_area = np.sum(im_frame>0), np.prod(im_frame.shape[:2])
    # occ_ratio = np.sum(im_frame>0)/np.prod(im_frame.shape[:2])

    return im_frame
