# Copyright (c) Facebook, Inc. and its affiliates.
## TODO: add repo link
import json, codecs
import re
import numpy as np
import os
from tabulate import tabulate
import pycocotools.mask as mask_util

from alficore.evaluation.coco_evaluation import COCOEvaluator
from ..dataloader.objdet_baseClasses.boxes import Boxes, BoxMode, pairwise_iou
from ..dataloader.objdet_baseClasses.coco_generic import convert_to_coco_json
from ..dataloader.objdet_baseClasses.common import create_small_table
from ..dataloader.objdet_baseClasses.catalog import MetadataCatalog

class BASELINEevaluator(COCOEvaluator):
    """
    This is designed specifically for Yolo models. Features containing bounding box,
    objectness and class score information are compared to the final detections after
    NMS (none max suppression). 
    """

    def __init__(
        self,
        dataset_name,
        outputdir=None,
        sampleN = None,
        model_name = None,
        model_type = None
    ):
        super().__init__(dataset_name=dataset_name, outputdir=outputdir, sampleN=sampleN, model_type=model_type, model_name=model_name)
        
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., Object Detection models).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that containes fields like pred_boxes, scores, classes`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                #TODO replace with own json function
                prediction["instances"] = instances_to_baseline_json(instances, input["image_id"])
            if "features" in output:
                #TODO safe features to file
                features = output["features"].to(self._cpu_device)
                prediction["features"] = features_to_baseline_json(features, input["image_id"])
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None, save_pred=False, epoch=0):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
            TODO: switch to clean way of passing epoch arg
        """
        super().evaluate(img_ids, save_pred, epoch)

        predictions = self._predictions
        if "features" in predictions[0]:
            feature_results = list([x["features"] for x in self._predictions])
            if self._outputdir:
                file_name = os.path.join(self._outputdir, self.dataset_name, self.model_type, 'epochs', str(epoch), "all_features_{}_{}_epoch.json".format(self.model_type, epoch))
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                self._logger.info("Saving results to {}".format(file_name))
                with open(file_name, "w") as f:
                    json.dump(feature_results, f)
                    f.flush()
                    f.close()

def features_to_baseline_json(features, img_id):
    """
    store complete features for an image

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    features = features.detach().cpu().numpy()
    d_type = str(features.dtype)
    f = str(codecs.encode(features.tostring(), 'hex'))
    # remove b and quotes to be able to decode back later
    f = re.sub(r'^b\'','',f)
    f = re.sub(r'\'$','',f)
    result = {
        "image_id": img_id,
        "shape": features.shape,
        "type_str": d_type,
        "features": f,
    }
    # return to numpy array by command
    # np.fromstring(results["features"].tostring(), dtype=result["type_str"]).reshape(results["shape"].shape)
    return result


def instances_to_baseline_json(instances, img_id):
    """
    Add additional data on stored features that led to a particular bounding box toan "Instances" 
    object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.detach().numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    has_features = instances.has("feat_boxes")
    if has_features:
        feat_bbox = instances.feat_boxes.tensor.detach().numpy()
        feat_bbox = BoxMode.convert(feat_bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        feat_bbox = feat_bbox.tolist()
        feat_objectness = instances.feat_objectness.tolist()
        feat_cls_scores = instances.feat_cls_scores.tolist()
        feat_index = instances.feat_idx.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "bbox_mode": BoxMode.XYWH_ABS,
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        if has_features:
            result["feat_bbox"] = feat_bbox[k]
            result["feat_objectness"] = feat_objectness[k]
            result["feat_cls_scores"] = feat_cls_scores[k]
            result["feat_index"] = feat_index[k]
        results.append(result)
    return results
