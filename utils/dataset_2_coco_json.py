# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import os
import mmcv
import json

#########################################################
#modify this section to adjust to your dataset location #
#########################################################
kitti_img_path = '/nwstore/datasets/KITTI/2d_object/training/image_2'
kitti_label_path = '/nwstore/datasets/KITTI/2d_object/training/label_2'
kitti_names = '/nwstore/datasets/KITTI/2d_object/training/kitti.names'

# output file
kitti_label_json_path = '/nwstore/datasets/KITTI/2d_object/training/image_2_label_CoCo_format.json'

############################
# end modification section #
############################

# def convert_balloon_to_coco():
kitti_names_contents = kitti_names.readlines()
kitti_images = os.listdir(kitti_img_path)
kitti_labels = os.listdir(kitti_label_path)
kitti_images.sort()
kitti_labels.sort()
annotations = []
images = []
obj_count = 1
categories = [{'id':id, 'name': cat.rstrip()} for id, cat in enumerate(kitti_names_contents)]
for idx, v in enumerate(kitti_labels):
    common_file_name = v.split('.')[0]
    kitti_image =  common_file_name + '.png'
    kitti_label =  common_file_name + '.txt'
    kitti_img_totest_path = os.path.join(kitti_img_path, kitti_image)
    kitti_label_totest_path = os.path.join(kitti_label_path, kitti_label)
    filename = kitti_image
    height, width = mmcv.imread(kitti_img_totest_path).shape[:2]

    images.append(dict(
        id=idx,
        file_name=filename,
        height=height,
        width=width))

    bboxes = []
    labels = []
    masks = []
    objects = open(kitti_label_totest_path,'r')
    for _, obj in enumerate(objects):
        obj = obj.split(' ')
        assert len(obj) == 7
        category_id = int(obj[0])
        x1     = float(obj[3])
        y1     = float(obj[4])
        obj_width  = float(obj[5])
        obj_height = float(obj[6])
        area = float(format(float(obj_width * obj_height), '.4f'))
        assert area>0
        bbox=[x1, y1, obj_width, obj_height]

        data_anno = dict(
            image_id=idx,
            id=obj_count,
            category_id=category_id,
            area = area,
            bbox=bbox,
            iscrowd=0)
        annotations.append(data_anno)
        obj_count += 1

coco_format_json = dict(
    images=images,
    categories=categories,
    annotations=annotations)
# mmcv.dump(coco_format_json, kitti_label_json_path)
# kitti_label_json_path = '/nwstore/datasets/KITTI/2d_object/training/image_2_label_CoCo_format_1.json'
with open(kitti_label_json_path, "w") as f:
    json.dump(coco_format_json, f)
    f.flush()