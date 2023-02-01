import sys
import numpy as np
import torch
import torchvision
import argparse
from pathlib import Path

import torch
FILE = Path(__file__).resolve()
sys.path.append("/home/fgeissle/ranger_repo/ranger")
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

cuda_device = 0
model_name = 'frcnn_torchvision'

class build_objdet_native_model_mmdet(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, device, dataset_name):
        super().__init__(model=model, device=device)
        self.dataset_name = dataset_name
        self.img_size = [(0,0)]
        self.preprocess = False #not needed but function must exist due to abstract template
        self.postprocess = True

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
        for _, im in enumerate(output):
            mask = [len(n) > 0 for n in im]
            objects = np.array(im)[mask]
            if len(objects) > 0:
                obj_class_counts = [u.shape[0] for u in objects]
                objects = np.vstack(objects) #make one single list

                classes_found = np.array(self.model.CLASSES)[mask]
                classes_sorted = []
                for a,b in list(zip(obj_class_counts, classes_found)):
                    for x in range(a):
                        ind = np.where(np.array(self.model.CLASSES)== b)[0][0] #go from class label to index
                        classes_sorted.append(ind)
                
                scores = objects[:,4]
            else:
                classes_sorted = []
                scores = []

            out_instance = Instances(self.img_size)

            # # Filter by confidence:
            # mask_high_conf = np.array(scores)>=self.conf_thres
            # scores = np.array(scores)[mask_high_conf]
            # objects = np.array(objects)[mask_high_conf]
            # classes_sorted = np.array(classes_sorted)[mask_high_conf].tolist()

            boxes = Boxes(objects[:,:4])
            out_instance.set("pred_boxes", boxes)
            out_instance.set("pred_classes", torch.tensor(classes_sorted).type(torch.ByteTensor))
            out_instance.set("scores", scores)

            # if 'robo' in self.dataset_name: #filter for only classes with label 0 (persons)
            #     red_mask = out_instance.pred_classes == 0
            #     cls = out_instance.pred_classes[red_mask]
            #     bxs = out_instance.pred_boxes[red_mask]
            #     scrs = out_instance.scores[red_mask]
            #     out_instance = Instances(self.img_size)
            #     out_instance.set("pred_classes", cls)
            #     out_instance.set("pred_boxes", bxs)
            #     out_instance.set("scores", scrs)

            out_list.append({'instances': out_instance})
        return out_list

    def __getattr__(self, method):
        if method.startswith('__'):
            raise AttributeError(method)
        try:
        # if hasattr(self.model, method):
            # print(self.model, 'method:', method)
            # print(self.model.model)

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
                    return  func(*args, **kwargs)
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
            show_result_pyplot(self.model, img_path[0], result[0], out_file = 'test/vis/demo_boxes.jpg')
        
        else: #load from given tensor (for example for dummy tensor)
            img_path = []
            self.img_size = [tuple(n['image'].shape[1:]) for n in input] #original image size
            result = inference_detector(self.model, [np.transpose(n['image'].cpu().numpy(), (1,2,0)) for n in input])

            # test plot of first image in batch:
            show_result_pyplot(self.model, np.transpose(input[0]['image'].cpu().numpy(), (1,2,0)), result[0], out_file = 'test/vis/demo_boxes.jpg')
        

        if self.postprocess:
            output = self.postprocess_output(result)

        output = pytorchFI_objDet_outputcheck(output)
        return output

def set_up_mmdet_model(model_name, dataset_name, device):

    # Retina net
    if 'retina' in model_name.lower():

        if 'coco' in dataset_name.lower():
            config_file = '/home/qutub/PhD/git_repos/github_repos/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/Retina-Net/pretrained_weights/retinanet-r50_fpn_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
            
        elif 'kitti' in dataset_name.lower():
            config_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/Retina-Net/pretrained_weights/retinanet-r50_fpn_kitti/retinanet_r50_fpn_1x_kitti.py'
            # checkpoint_file = '/nwstore/florian/mm_checkpoints/retina/latest.pth' #'../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/Retina-Net/pretrained_weights/retinanet-r50_fpn_kitti/latest.pth'
    
    # Yolov3
    elif "yolo" in model_name.lower():

        if 'coco' in dataset_name.lower():
            config_file = '/home/qutub/PhD/git_repos/github_repos/mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/YoloV3/pretrained_weights/yolo_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'

        if 'kitti' in dataset_name.lower():
            config_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/YoloV3/pretrained_weights/yolo_kitti/yolov3_d53_mstrain-416_273e_kitti.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/YoloV3/pretrained_weights/yolo_kitti/latest.pth'

    elif "ssd" in model_name.lower():

        if 'coco' in dataset_name.lower():
            config_file = '/home/qutub/PhD/git_repos/github_repos/mmdetection/configs/ssd/ssd512_coco.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/SSD/pretrained_weights/ssd_512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth'

        if 'kitti' in dataset_name.lower():
            config_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/SSD/pretrained_weights/ssd_512_kitti/ssd512_kitti.py'
            checkpoint_file = '/nwstore/Computer-Vision-Models/Obj-Det/MM-Detection-pretrained-Models/SSD/pretrained_weights/ssd_512_kitti/latest.pth'

    # elif "ssd" in model_name.lower():

    else:
        print('Model and dataset combination not found.')
        sys.exit()
        
    model = init_detector(config_file, checkpoint_file, device=device) #takes care of eval, to(device)
    classes = model.CLASSES
    indices = list(range(len(classes)))
    class_dict = dict(zip(classes, indices))
    
    model = build_objdet_native_model_mmdet(model=model, device=device, dataset_name=dataset_name)

    return model, class_dict

def main(argv):
    opt = parse_opt()
    device = torch.device(
        "cuda:{}".format(opt.device) if torch.cuda.is_available() else "cpu")

    # Model   ----------------------------------------------------------
    model, class_dict = set_up_mmdet_model(model_name='yolov3', dataset_name='kitti', device=device)
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

    frcnn_Errormodel = TestErrorModels_ObjDet(model=model, resil_model=None, resil_name=None, model_name=model_name, config_location=opt.config_file, \
                    ranger_bounds=None, device=device,  ranger_detector=False, inf_nan_monitoring=True, disable_FI=True, dl_attr=dl_attr, num_faults=0, fault_file=fault_files, \
                        resume_dir=None, copy_yml_scenario = False)
    frcnn_Errormodel.test_rand_ObjDet_SBFs_inj()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dl-json', type=str, default='/nwstore/datasets/KITTI/2d_object/training/split/image_2_label_CoCo_format_test_split.json', help='path to datasets ground truth json file')
    parser.add_argument('--dl-img-root', type=str, default='/nwstore/datasets/KITTI/2d_object/training/image_2', help='path to datasets images')
    parser.add_argument('--dl-ds-name', type=str, default='kitti', help='dataset short name')
    parser.add_argument('--config-file', type=str, default='default.yml', help='name of default yml file - inside scenarios folder')
    parser.add_argument('--fault-files', type=str, default=None, help='directory of already existing fault files')
    parser.add_argument('--dl-batchsize', type=int, default=1, help='dataloader batch size')
    parser.add_argument('--sample-size', type=int, default=500, help='dataloader sample size')
    parser.add_argument('--num-workers', type=int, default=1, help='dataloader number of workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--random-sample', action='store_true', help='randomly sampled of len sample-size from the dataset')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the sampled data in dataloader')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    main(sys.argv)
