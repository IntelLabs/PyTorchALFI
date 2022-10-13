import io
import os
import sys
import json
import shutil
import logging
import datetime
import numpy as np
from filelock import FileLock


from .catalog import MetadataCatalog
from .boxes import Boxes, BoxMode

logger = logging.getLogger(__name__)

def convert_to_coco_dict(dataloader, dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in pytorchalfi's standard format into COCO json format.

    Generic dataset description can be found here:
    https://pytorchfiWrapper.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in pytorchalfi's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = dataloader.dataset_dict
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if to_bbox_mode == BoxMode.XYWH_ABS:
                bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()
            else:
                area = Boxes([bbox]).area()[0].item()

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for pytorchalfi.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_coco_json(dataloader, dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in pytorchalfi's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in pytorchalfi's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    # Path.mkdir(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ## remove file lock if NFS file system is being used to store the results/json files
    with FileLock(output_file):
        if os.path.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataloader, dataset_name)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with io.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            try:
                shutil.move(tmp_file, output_file)
            except:
                logger.error(
                f"remove file lock context in func convert_to_coco_json if NFS file system is being used to store the results/json files")
                sys.exit()