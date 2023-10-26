import sys
import numpy as np
import torch
import torchvision
import argparse
from pathlib import Path
from numpy import inf
import torch
import datetime
FILE = Path(__file__).resolve()

from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, resize

from torchvision.models.detection import faster_rcnn

from alficore.dataloader.objdet_baseClasses.boxes import Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.ptfiwrap_utils.build_native_model import build_native_model
from typing import Dict, List
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
from mmdet.apis.inference import inference_detector, show_result_pyplot
from mmdet.apis import init_detector
from itertools import chain

cuda_device = 1
model_name = 'mmdetection_torchvision'

class build_objdet_native_model_mmdet(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, model_name, device, dataset_name, show_result=False):
        super().__init__(model=model, device=device)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.img_size = [(0,0)]
        self.preprocess = False #not needed but function must exist due to abstract template
        self.postprocess = True
        self.show_result = show_result
        self.fps = 0

    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """
        # Preprocessing done by mmnet pipeline
        return

    def postprocess_output(self, output):
        if 'yolo' in self.model_name.lower():
            return self.postprocess_output_yolo(output)
        elif 'ssd' in self.model_name.lower():
            return self.postprocess_output_ssd(output)

    def postprocess_output_ssd(self, output):
        """
        the returning output should be stored as dictionary
        Output['instances] = fields containing pred_boxes, scores, classes
        viz. it should align to attributes as used in function instances_to_coco_json() in coco evaluation file.
        Output['instances].pred_boxes = [[2d-bb_0], [2d-bb_1], [2d-bb_2]...]
        Output['instances].scores     = [score_0, score_1, .....]
        Output['instances].classes     = [car, pedetrian, .....]

        ex: for batch size 1
        Output = [{}]
        Output['instances'] = output
        return Output
        """
        out_list = []
        iter_output = output 
        for idx, im in enumerate(iter_output):
            mask = [len(n) > 0 for n in im]
            """
            filtering only pedestrian classes
            """
            objects = np.array(im)[mask]
            model_classes = self.model.CLASSES
            if len(objects) > 0:
                obj_class_counts = [u.shape[0] for u in objects]
                objects = np.vstack(objects) #make one single list
                classes_found = np.array(model_classes)[mask]
                classes_sorted = []
                for a,b in list(zip(obj_class_counts, classes_found)):
                    for x in range(a):
                        ind = np.where(np.array(model_classes)== b)[0][0] #go from class label to index
                        classes_sorted.append(ind)
                classes_sorted = np.array(classes_sorted)
                boxes = Boxes(objects[:,:4])
                scores = objects[:,4]
                objectness = objects[:,4]
            else:
                classes_sorted = []
                scores = []
                objectness = []
                boxes = []
            out_instance = Instances(self.img_size, fields_len_check=False)

            out_instance.set("pred_boxes", boxes)
            out_instance.set("pred_classes", torch.tensor(classes_sorted).type(torch.ByteTensor))
            out_instance.set("scores", scores)
            out_instance.set("objectness", objectness)
            out_list.append({'instances': out_instance})
        # print("\n\n {}".format(self.fps))
        return out_list

    def postprocess_output_yolo(self, output):
        """
        the returning output should be stored as dictionary
        Output['instances] = fields containing pred_boxes, scores, classes
        viz. it should align to attributes as used in function instances_to_coco_json() in coco evaluation file.
        Output['instances].pred_boxes = [[2d-bb_0], [2d-bb_1], [2d-bb_2]...]
        Output['instances].scores     = [score_0, score_1, .....]
        Output['instances].classes     = [car, pedetrian, .....]

        ex: for batch size 1
        Output = [{}]
        Output['instances'] = output
        return Output
        """

        out_list = []
        iter_output = output
        for idx, im in enumerate(iter_output):
            mask = [len(n) > 0 for n in im]
            class_count = [len(n) for n in im]
            """
            filtering only pedestrian classes
            """
            objects = np.array(im)[mask]
            model_classes = self.model.CLASSES
            if len(objects) > 0:
                # obj_class_counts = [u.shape[0] for u in objects]
                objects = np.vstack(objects) #make one single list
                classes_sorted = []
                for a, b in enumerate(class_count):
                    classes_sorted.append([a]*b)
                classes_sorted = list(chain.from_iterable(classes_sorted))
                boxes = Boxes(objects[:,:4])
                scores = objects[:,4]
                objectness = False
                if objects.shape[1] > 5:
                    objectness = objects[:,5]
                    objectness = True
            else:
                classes_sorted = []
                scores = []
                objectness = []
                boxes = []
            out_instance = Instances(self.img_size, fields_len_check=False)

            out_instance.set("pred_boxes", boxes)
            out_instance.set("pred_classes", torch.tensor(classes_sorted).type(torch.ByteTensor))
            out_instance.set("scores", scores)
            if objectness:
                out_instance.set("objectness", objectness)
            out_list.append({'instances': out_instance})
        # print("\n\n {}".format(self.fps))
        return out_list
    
    def __getattr__(self, method):
        if method.startswith('__'):
            raise AttributeError(method)
        try:
            try:
                func = getattr(self.model.model, method)
            except:
                func = getattr(self.model, method)
            ## running pytorch model (self.model) inbuilt functions like eval, to(device)..etc
            ## assuming the executed method is not changing the model but rather
            ## operates on the execution level of pytorch model.
            # print('check', func)
            def wrapper(*args, **kwargs):
                if (method=='to'):
                    return self
                else:
                    return func(*args, **kwargs)
            # print('check', wrapper, method, self)
            return wrapper
        except KeyError:
            raise AttributeError(method)

    def __call__(self, input, dummy=False):
        input = pytorchFI_objDet_inputcheck(input) #wraps image with metadata


        if "file_name" in input[0].keys(): #load from original image
            img_path = [n['file_name'] for n in input]

            result = inference_detector(self.model, img_path)
            self.img_size = [(n['height'], n['width']) for n in input] #original image size, gets resclaed during inference_detector data transformation
            # test plot of first image in batch:
            if self.show_result:
                for batch in range(len(result[0])):
                    processed_result = result[0][batch]
                    processed_result = [np.concatenate((box[:, :4], np.expand_dims(box[:, -1], axis=1)), axis=1) for box in processed_result]
                    show_result_pyplot(self.model, img_path[0], processed_result, out_file = 'test/vis/demo_boxes.jpg')
        
        else: #load from given tensor (for example for dummy tensor)
            img_path = []
            self.img_size = [tuple(n['image'].shape[1:]) for n in input] #original image size
            result = inference_detector(self.model, [np.transpose(n['image'].cpu().numpy(), (1,2,0)) for n in input])

            # test plot of first image in batch:
            if self.show_result:
                for batch in range(len(result[0])):
                    # if self.enable_pre_nms:
                    show_result_pyplot(self.model, np.transpose(input[0]['image'].cpu().numpy(), (1,2,0)), result[0][batch], out_file = 'test/vis/demo_boxes.jpg')
                    # else:
                        # show_result_pyplot(self.model, np.transpose(input[0]['image'].cpu().numpy(), (1,2,0)), result[0], out_file = 'test/vis/demo_boxes.jpg')

        if self.postprocess:
            output = self.postprocess_output(result)

        output = pytorchFI_objDet_outputcheck(output)
        return output

def set_up_mmdet_model(model_name, dataset_name, device, show_result=False):

    # Retina net
    if 'retina' in model_name.lower():

        if 'coco' in dataset_name.lower():
            config_file = '~/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py'
            checkpoint_file = '~/pretrained_weights/retinanet-r50_fpn_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'

        elif 'kitti' in dataset_name.lower():
            config_file = '~/pretrained_weights/retinanet-r50_fpn_kitti/retinanet_r50_fpn_1x_kitti.py'
            checkpoint_file = '~/pretrained_weights/retinanet-r50_fpn_kitti/latest.pth'
 
    # YoloV3
    elif "yolo" in model_name.lower():

        if 'coco' in dataset_name.lower():
            config_file = '~/mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
            checkpoint_file = '~/pretrained_weights/yolo_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'

        elif 'kitti' in dataset_name.lower():
            config_file = '~/pretrained_weights/yolo_kitti/yolov3_d53_mstrain-416_273e_kitti.py'
            checkpoint_file = '~/pretrained_weights/yolo_kitti/latest.pth'

    else:
        print('Model and dataset combination not found.')
        sys.exit()


    model = build_objdet_native_model_mmdet(model=model, model_name=model_name, device=device, dataset_name=dataset_name, show_result=show_result)

    return model

def main(argv):
    opt = parse_opt()
    device = torch.device(
        "cuda:{}".format(opt.device) if torch.cuda.is_available() else "cpu")

    show_result = False
    model_name='ssd'
    model_name='yolo'


    ## set dataloader attributes
    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = opt.random_sample
    dl_attr.dl_batch_size     = opt.dl_batchsize
    dl_attr.dl_shuffle        = opt.shuffle
    dl_attr.dl_sampleN        = opt.sample_size
    dl_attr.dl_num_workers    = opt.num_workers
    dl_attr.dl_device         = device
    dl_attr.dl_dataset_name   = opt.dl_ds_name
    dl_attr.dl_img_root       = opt.dl_img_root
    dl_attr.dl_gt_json        = opt.dl_json

    fault_files = opt.fault_files

    # Model   ----------------------------------------------------------
    model = set_up_mmdet_model(model_name=model_name, dataset_name='kitti', device=device, show_result=show_result)

    mmdetection_Errormodel = TestErrorModels_ObjDet(model=model, resil_model=None, resil_name=None, model_name=model_name, config_location=opt.config_file, \
                    ranger_bounds=None, device=device,  inf_nan_monitoring=True, disable_FI=False, dl_attr=dl_attr, num_faults=0, fault_file=fault_files, resume_dir=None, copy_yml_scenario = False, \
                        result_files="~/pytorchalfi/Fault_detection/YoloV3/"
                        )
    mmdetection_Errormodel.test_rand_ObjDet_SBFs_inj()

def parse_opt():
    parser = argparse.ArgumentParser()
    ## KITTI
    parser.add_argument('--dl-json', type=str, default='~/KITTI/2d_object/training/split/image_2_label_CoCo_format_test_split.json', help='path to datasets ground truth json file')
    parser.add_argument('--dl-img-root', type=str, default='~/KITTI/2d_object/training/image_2', help='path to datasets images')

    ## BDD100K
    # parser.add_argument('--dl-json', type=str, default='~/BDD100K/bdd100k_det_val.json', help='path to datasets ground truth json file')
    # parser.add_argument('--dl-img-root', type=str, default='~/BDD100K/bdd100k/images/100k/val', help='path to datasets images')    

    parser.add_argument('--dl-ds-name', type=str, default='coco', help='dataset short name')
    parser.add_argument('--config-file', type=str, default='default_neurons.yml', help='name of default yml file - inside scenarios folder')
    parser.add_argument('--fault-files', type=str, default=None, help='directory of already existing fault files')
    parser.add_argument('--dl-batchsize', type=int, default=1, help='dataloader batch size')
    parser.add_argument('--sample-size', type=int, default=100, help='dataloader sample size') ## a. float mode: 0.0 - 1.0 -> uses percentage of the data; b. absolute mode: 1-N uses abosolute number of images in the dataloader
    parser.add_argument('--num-workers', type=int, default=1, help='dataloader number of workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--random-sample', type=bool, default=False, help='randomly sampled of len sample-size from the dataset')
    parser.add_argument('--shuffle', type=bool,  default=False, help='Shuffle the sampled data in dataloader')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    main(sys.argv)