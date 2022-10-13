# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import os
import json
import mmcv

##############################################
# modify paths according to your environment #
##############################################

# input
kitti_label_json_path = '/nwstore/datasets/KITTI/2d_object/training/image_2_label_CoCo_format.json'

# output
train_json = '/nwstore/datasets/KITTI/2d_object/training/split/image_2_label_CoCo_format_train_split.json'
val_json = '/nwstore/datasets/KITTI/2d_object/training/split/image_2_label_CoCo_format_val_split.json'
test_json = '/nwstore/datasets/KITTI/2d_object/training/split/image_2_label_CoCo_format_test_split.json'

##############
# end modify #
##############

os.makedirs(os.path.dirname(train_json), exist_ok=True)
os.makedirs(os.path.dirname(val_json), exist_ok=True)
os.makedirs(os.path.dirname(test_json), exist_ok=True)

with open(kitti_label_json_path) as f:
    gt_label_json = json.load(f)
    f.close()

train_split, val_split, test_split = 0.75, 0.1, 0.15
from sklearn.model_selection import train_test_split
X_train_val, x_test_files  = train_test_split(gt_label_json['images'], test_size=test_split, random_state=10)
X_train_files, x_val_files  = train_test_split(X_train_val, test_size=val_split*len(gt_label_json['images'])/len(X_train_val), random_state=10)

train_anns = []
for i in range(len(X_train_files)):
    img_id = X_train_files[i]['id']
    train_ann = [anns for anns in gt_label_json['annotations'] if anns['image_id'] == img_id]
    train_anns.extend(train_ann)
val_anns = []
for i in range(len(x_val_files)):
    img_id = x_val_files[i]['id']
    val_ann = [anns for anns in gt_label_json['annotations'] if anns['image_id'] == img_id]
    val_anns.extend(val_ann)
test_anns = []
for i in range(len(x_test_files)):
    img_id = x_test_files[i]['id']
    test_ann = [anns for anns in gt_label_json['annotations'] if anns['image_id'] == img_id]
    test_anns.extend(test_ann)

coco_format_json = dict(
    images=X_train_files,
    categories=gt_label_json['categories'],
    annotations=train_anns)
# mmcv.dump(coco_format_json, kitti_label_json_path)
with open(train_json, "w") as f:
    json.dump(coco_format_json, f)
    f.flush()
    f.close()

coco_format_json = dict(
    images=x_val_files,
    categories=gt_label_json['categories'],
    annotations=val_anns)
# mmcv.dump(coco_format_json, kitti_label_json_path)
with open(val_json, "w") as f:
    json.dump(coco_format_json, f)
    f.flush()
    f.close()

coco_format_json = dict(
    images=x_test_files,
    categories=gt_label_json['categories'],
    annotations=test_anns)
# mmcv.dump(coco_format_json, kitti_label_json_path)

with open(test_json, "w") as f:
    json.dump(coco_format_json, f)
    f.flush()
    f.close()