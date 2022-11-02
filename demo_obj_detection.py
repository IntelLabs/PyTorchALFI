import sys
import numpy as np
import torch
import torchvision
import argparse
from pathlib import Path

import torch
FILE = Path(__file__).resolve()

from alficore.wrapper.test_error_models_objdet import TestErrorModels_ObjDet
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, pytorchFI_objDet_outputcheck, resize

from torchvision.models.detection import faster_rcnn

from alficore.dataloader.objdet_baseClasses.boxes import Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances
from alficore.ptfiwrap_utils.build_native_model import build_native_model
from typing import Dict, List
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr


cuda_device = 0
model_name = 'frcnn_torchvision'

class  build_objdet_native_model(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, device):
        super().__init__(model=model, device=device)
        ### img_size, preprocess and postprocess can also be inialised using kwargs which will be set in base class
        self.img_size = 416
        self.preprocess = True
        self.postprocess = True

    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """
        # images = [letterbox(x["image"], self.img_size, HWC=True, use_torch=True)[0] for x in batched_inputs]
        images = [resize(x['image'], self.img_size) for x in batched_inputs]
        images = [x/255. for x in images]
        # Convert to tensor
        images = torch.stack(images).to(self.device)
        ## normalisde the input if neccesary
        return images

    def rescale_boxes(self, boxes, current_dim, original_shape):
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
        boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
        return boxes

    def postprocess_output(self, output, input, original_shapes, current_dim):
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
        def rescale_bounding_boxes(output):
            def clip_coords(boxes, shape):
                # Clip bounding xyxy bounding boxes to image shape (height, width)
                if isinstance(boxes, torch.Tensor):  # faster individually
                    boxes[:, 0].clamp_(0, shape[1])  # x1
                    boxes[:, 1].clamp_(0, shape[0])  # y1
                    boxes[:, 2].clamp_(0, shape[1])  # x2
                    boxes[:, 3].clamp_(0, shape[0])  # y2
                else:  # np.array (faster grouped)
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

            def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
                # Rescale coords (xyxy) from img1_shape to img0_shape
                if ratio_pad is None:  # calculate from img0_shape
                    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
                    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
                else:
                    gain = ratio_pad[0][0]
                    pad = ratio_pad[1]

                coords[:, [0, 2]] -= pad[0]  # x padding
                coords[:, [1, 3]] -= pad[1]  # y padding
                coords[:, :4] /= gain
                clip_coords(coords, img0_shape)
                return coords

            if len(original_shapes):
                # boxes = Boxes(output[:,:4])P
                # boxes.rescale_boxes(current_dim=[self.img_size]*2, original_shape=original_shapes[idx])
                boxes = Boxes(scale_coords(current_dim, output, original_shapes[idx]))
            else:
                boxes = Boxes(output)
            return boxes
        raw_output = output
        Output, nms_indx = self.non_max_suppression(output)
        out_list = []
        for idx, output in enumerate(Output): # for each image in batch
            if output:
                out_instance = Instances(self.img_size, fields_len_check=False)
                out_instance.set("image_id", input[idx]["image_id"])
                boxes = rescale_bounding_boxes(output=Output[idx]['boxes'])
                out_instance.set("pred_boxes", boxes)
                out_instance.set("pred_classes", Output[idx]['labels'])
                out_instance.set("scores", Output[idx]['scores'])
            else:
                out_instance = Instances(self.img_size, fields_len_check=False)
                out_instance.set("image_id", input[idx]["image_id"])
                out_instance.set("pred_boxes", None)
                out_instance.set("pred_classes", None)
                out_instance.set("scores", None)
            # if input[idx]["image_id"]== 14473:
            #     debug_test = True
            out_list.append({'instances': out_instance})
        return out_list

    def non_max_suppression(self, prediction, conf_thres=0.5, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        def box_iou(box1, box2):
            # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
            """
            Return intersection-over-union (Jaccard index) of boxes.
            Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
            Arguments:
                box1 (Tensor[N, 4])
                box2 (Tensor[M, 4])
            Returns:
                iou (Tensor[N, M]): the NxM matrix containing the pairwise
                    IoU values for every element in boxes1 and boxes2
            """

            def box_area(box):
                # box = 4xn
                return (box[2] - box[0]) * (box[3] - box[1])

            area1 = box_area(box1.T)
            area2 = box_area(box2.T)

            # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
            inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
            return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
        min_wh, max_wh = 2, 4096
        nms_indx = []
        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        merge = False  # use merge-NMS
        out_prediction = [None]*len(prediction)
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            xc = x['scores'] > conf_thres
            indx = np.where(xc.cpu())[0]
            x['scores'] = x['scores'][xc]
            x['labels'] = x['labels'][xc]
            x['boxes'] = x['boxes'][xc]

            # If none remain process next image
            if not len(x['scores']):
                continue

            n = len(x['boxes'])  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                filterd = x['scores'].argsort(descending=True)[:max_nms]
                x = x[x['scores'].argsort(descending=True)[:max_nms]]  # sort by confidence
                x['scores'] = x['scores'][filterd]
                x['labels'] = x['labels'][filterd]
                x['boxes'] = x['boxes'][filterd]
            # Batched NMS
            c = x['labels'] * (0 if agnostic else max_wh) 
            boxes = x['boxes'] + c.unsqueeze(dim=1)
            scores = x['scores']
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(x['boxes'][i], x['boxes']) > iou_thres  # iou matrix
                weights = iou * x['scores'][None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            out_prediction[xi] = {'boxes': None, 'labels':None, 'scores':None}
            out_prediction[xi]['boxes'] = x['boxes'][i]
            out_prediction[xi]['scores'] = x['scores'][i]
            out_prediction[xi]['labels'] = x['labels'][i]
            nms_indx.append(indx[i.cpu().numpy()])
        return out_prediction, nms_indx

    def __getattr__(self, method):
        if method.startswith('__'):
            raise AttributeError(method)
        try:
        # if hasattr(self.model, method):
            
            try:
                func = getattr(self.model.model, method)
            except:
                func = getattr(self.model, method)
            ## running pytorch model (self.model) inbuilt functions like eval, to(device)..etc
            ## assuming the executed method is not changing the model but rather
            ## operates on the execution level of pytorch model.
            def wrapper(*args, **kwargs):
                if (method=='to'):
                    return self
                else:
                    return  func(*args, **kwargs)
            return wrapper
        except KeyError:
            raise AttributeError(method)

    def __call__(self, input, dummy=False):
        input = pytorchFI_objDet_inputcheck(input)
        try:
            original_shapes = [(input[i]['height'], input[i]['width']) for i in range(len(input))]
        except:
            original_shapes = []
        _input = input
        if self.preprocess:
            _input = self.preprocess_input(input)
        output = self.model(_input)
        if self.postprocess:
            output = self.postprocess_output(output, input, original_shapes, _input.shape[2:])

        output = pytorchFI_objDet_outputcheck(output)
        return output

def main(argv):
    opt = parse_opt()
    device = torch.device(
        "cuda:{}".format(opt.device) if torch.cuda.is_available() else "cpu")

    # Model   ----------------------------------------------------------
    frcnn_model = faster_rcnn.fasterrcnn_resnet50_fpn(weights=faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    frcnn_model = frcnn_model.to(device)
    frcnn_model.eval()
    # model = build_objdet_native_model(model=frcnn_model)
    model = frcnn_model

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
    wrapped_model = build_objdet_native_model(model, device)

    frcnn_Errormodel = TestErrorModels_ObjDet(model=wrapped_model, resil_model=None, resil_name=None, model_name=model_name, config_location=opt.config_file, \
                    ranger_bounds=None, device=device,  inf_nan_monitoring=True, disable_FI=False, dl_attr=dl_attr, num_faults=0, fault_file=fault_files, \
                        resume_dir=None, copy_yml_scenario = False)
    frcnn_Errormodel.test_rand_ObjDet_SBFs_inj()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dl-json', type=str, default='/nwstore/datasets/COCO/coco2017/annotations/instances_val2017.json', help='path to datasets ground truth json file')
    parser.add_argument('--dl-img-root', type=str, default='/nwstore/datasets/COCO/coco2017/val2017', help='path to datasets images')
    parser.add_argument('--dl-ds-name', type=str, default='CoCo', help='dataset short name')
    parser.add_argument('--config-file', type=str, default='default.yml', help='name of default yml file - inside scenarios folder')
    parser.add_argument('--fault-files', type=str, default=None, help='directory of already existing fault files to repeat existing experiment')
    parser.add_argument('--dl-batchsize', type=int, default=10, help='dataloader batch size')
    parser.add_argument('--sample-size', type=int, default=100, help='dataloader sample size')
    parser.add_argument('--num-workers', type=int, default=1, help='dataloader number of workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--random-sample', type=bool, default=False, help='randomly sampled of len sample-size from the dataset')
    parser.add_argument('--shuffle', type=bool,  default=False, help='Shuffle the sampled data in dataloader')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    main(sys.argv)
