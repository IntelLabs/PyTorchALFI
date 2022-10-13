# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import os
import contextlib

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import logging
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from .objdet_baseClasses.catalog import MetadataCatalog
from .objdet_baseClasses.boxes import BoxMode

from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
from .abs_loader import Abstract_Loader
from random import shuffle

"""
This file contains functions to parse COCO-format annotations into dicts in "pytorchfi wrapper format".
"""

logger = logging.getLogger(__name__)


class CoCo_obj_det_native_dataloader(Abstract_Loader):

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
        
        super().__init__(dl_attr,dnn_model_name)
        # for own dataloader implementation overwrite init_dataloader() and call with self.init_dataloader()
        super().init_dataloader()
        
    
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
            val_split, _ = self.split_data(dataset_len=len(img_ids), sampleN=sampleN, random_sample=dl_attr.dl_random_sample)
            img_ids = [img_ids[i] for i in val_split]
        if dl_attr.dl_shuffle:
            shuffle(img_ids)
        imgs = coco_api.loadImgs(img_ids)
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
            # file_name, height, width and image_id are the minimum fields required
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
