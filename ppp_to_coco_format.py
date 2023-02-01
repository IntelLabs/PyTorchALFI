import json
from torch.utils.data import Dataset
import os, natsort
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from csv import reader
import torch.nn.functional as functional
import numpy as np


def load_coco_json(json_path):
        with open(json_path) as f:
                coco_json = json.load(f)
                f.close()
        return coco_json


def load_ppp_csv(labels_path):
        # Load labels from csv file
        csv_length = csv_count(labels_path)
        with open(labels_path, 'r') as read_obj:
                csv_reader = reader(read_obj)
                pic_all = []
                cnt = 0
                pth = None
                for row in csv_reader:
                        # print(row, cnt)
                        if pth == None: #first time
                                pth = row[0]
                                # print('new imge:', pth)
                                blck = []
                                blck.append(row)
                        elif cnt == csv_length - 1: #last time
                                # pth = row[0]
                                # print('last img')
                                blck.append(row)
                                pic_all.append(blck)                    
                        else:
                                if row[0] == pth:
                                        # print('add to same img')
                                        blck.append(row)
                                else:
                                        pth = row[0]
                                        # print('new img', pth)
                                        pic_all.append(blck)
                                        blck = []
                                        blck.append(row)
                        cnt += 1

        return pic_all # List of lists


def csv_count(labels_path):
    with open(labels_path, 'r') as read_obj:
        rr = reader(read_obj, delimiter = ",")
        # data = [l for l in reader]
        row_count = sum(1 for row in rr)
    return row_count


# def labels2num(x):
#         """
#         Convert string labels to numbers. car, truck, van, motorcycle/motorbike, person, bus
#         """
#         if x == 'car' or x =='can':
#                 return 0
#         elif x == 'truck':
#                 return 1
#         elif x == 'van':
#                 return 2
#         elif x == 'motorcycle' or x == "motorbike":
#                 return 3
#         elif x == 'person':
#                 return 4
#         elif x == 'bus':
#                 return 5
#         else:
#                 print('not an expected class', x)
#                 return 1000

# def labels2num(x):
#         """
#         Convert string labels to numbers. car, truck, van, motorcycle/motorbike, person, bus
#         # Like in Kitti: class_mapping_yolov3_ppp = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
#         """
#         if x == 'car' or x =='can':
#                 return 0
#         elif x == 'truck':
#                 return 2
#         elif x == 'van':
#                 return 1
#         elif x == 'motorcycle' or x == "motorbike":
#                 return 5
#         elif x == 'person':
#                 return 3
#         elif x == 'bus':
#                 return 2
#         else:
#                 print('not an expected class', x)
#                 return 7


def labels2num(x):
        """
        Convert string labels to numbers. car, truck, van, motorcycle/motorbike, person, bus
        # Like in Coco-80: mapping as in coco.names
        """
        if x == 'car' or x =='can':
                return 2
        elif x == 'truck':
                return 7
        elif x == 'van':
                return 7
        elif x == 'motorcycle' or x == "motorbike":
                return 3
        elif x == 'person':
                return 0
        elif x == 'bus':
                return 5
        else:
                print('not an expected class', x)
                return 7

# def num2labels(x):
#         """
#         Convert number labels to strings.
#         """
#         if x == 0:
#                 return 'car'
#         elif x == 1:
#                 return 'truck'
#         elif x == 2:
#                 return 'van'
#         elif x == 3:
#                 return 'motorbike'
#         elif x == 4:
#                 return 'person'
#         elif x == 5:
#                 return 'bus'
#         else:
#                 return 'other'


# def num2labels(x):
#         """
#         Convert number labels to strings.
#         # Like in Kitti: class_mapping_yolov3_ppp = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
#         """
#         if x == 0:
#                 return 'car'
#         elif x == 1:
#                 return 'van'
#         elif x == 2:
#                 return 'truck'
#         elif x == 3:
#                 return 'pedestrian'
#         elif x == 4:
#                 return 'person_sitting'
#         elif x == 5:
#                 return 'cyclist'
#         elif x == 6:
#                 return 'tram'
#         elif x == 7:
#                 return 'misc'
#         # else:
#         #         return 'other'

# Load for comparison --------------------------------
# json_path = '/nwstore/datasets/COCO/coco2014/data/annotations/instances_val2014.json'
json_path = '/nwstore/datasets/COCO/coco2017/instances_val2017.json'
coco_json = load_coco_json(json_path)
print('loaded json')

# Load csv ---------------------------------------------
# pth = '/nwstore/datasets/Ppp_test_data/ToyDataP++/m090_001/'
# pth = '/nwstore/datasets/Ppp_test_data/ToyDataP++/s110_004/'
pth = '/nwstore/datasets/Ppp_test_data/ToyDataP++/s110_005/'
# csv_path = '/nwstore/datasets/Ppp_test_data/csv_format_labels/mp09_far.csv'
csv_path = pth + 'labels.csv'
ppp_csv = load_ppp_csv(csv_path)
print('loaded csv')



# Create ppp_json from csv ----------------------------
# in images: id
# in annotations: image_id
# in 'categories'
keys = list(coco_json.keys())
ppp_json = {x:{} for x in keys}


# Categories:
# Like in Kitti: class_mapping_yolov3_ppp = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
# cat_list = []
# cat_list.append({'supercategory': 'car', 'id': 0, 'name': 'car'})
# cat_list.append({'supercategory': 'van', 'id': 1, 'name': 'van'})
# cat_list.append({'supercategory': 'truck', 'id': 2, 'name': 'truck'})
# cat_list.append({'supercategory': 'pedestrian', 'id': 3, 'name': 'pedestrian'})
# cat_list.append({'supercategory': 'person_sitting', 'id': 4, 'name': 'person_sitting'})
# cat_list.append({'supercategory': 'cyclist', 'id': 5, 'name': 'cyclist'})
# cat_list.append({'supercategory': 'tram', 'id': 6, 'name': 'tram'})
# cat_list.append({'supercategory': 'misc', 'id': 7, 'name': 'misc'})
# Like in Coco:
cat_list = []
with open('/home/fgeissle/ranger_repo/ranger/alficore/models/yolov3/config/coco.names', 'r') as read_obj:
        csv_reader = reader(read_obj)
        cnt = 0
        for row in csv_reader:
                cat_list.append({'supercategory': row[0], 'id': cnt, 'name': row[0]}) 
                cnt += 1
# # for n in range(80):
# #         cat_list.append({'supercategory': '{}'.format(n), 'id': n, 'name': '{}'.format(n)})

ppp_json["categories"] = cat_list
print('categories added')

# Images:
# ppp_json["images"] = [{"file_name": ppp_csv[i][0][0][-9:], "id": i+1, 'height': 0, 'width': 0} for i in range(len(ppp_csv))]
ppp_json["images"] = [{"file_name": ppp_csv[i][0][0][-7:], "id": i+1, 'height': 0, 'width': 0} for i in range(len(ppp_csv))]

# pth = '/nwstore/datasets/Ppp_test_data/DataFromA9Cams/mp09_far_rosbag_0_jpg/'
for n in range(len(ppp_json["images"])):
        img_path = ppp_json["images"][n]["file_name"]
        img_path = pth + 'images/' + img_path

        im = Image.open(img_path)
        width, height = im.size
        ppp_json["images"][n]["width"] = width
        ppp_json["images"][n]["height"] = height

print('images added')


# Annotation section
annos = []
for i in range(len(ppp_csv)):
        for j in range(len(ppp_csv[i])):
                bbox = ppp_csv[i][j][1:5]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x, y, w, h = (x2-x1)/2, (y2-y1)/2, x2-x1, y2-y1
                cls = ppp_csv[i][j][5]
                # annos.append({'segmentation': [], 'area': [], 'iscrowd': 0, 'image_id': i+1, 'bbox': [x,y,w,h], 'category_id': labels2num(cls), 'id': 0})
                annos.append({'segmentation': [], 'area': 0., 'iscrowd': 0, 'image_id': i+1, 'bbox': [x1,y1,x2,y2], 'category_id': labels2num(cls), 'id': 0, 'ignore': 0})
                # in coco: x,y,w,h where x,y is the bbox center
                # in ppp: x1, y1, x2, y2 where x1,y1 is top left and x2, y2 is bottom right corner


ppp_json["annotations"] = annos
print('annotations added')




# Rewrite as json
# # Data to be written 
# new_name = "/nwstore/datasets/Ppp_test_data/ToyDataP++/m090_001/instances_ppp2022_xyxy_coco_labels_set2.json"
new_name = pth + "instances_ppp2022_xyxy_coco_labels_set3.json"
with open(new_name, "w") as outfile: 
    json.dump(ppp_json, outfile)