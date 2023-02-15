# pytorchalfi
Application Level Fault Injection for Pytorch

This is an experimental branch using quantile hooks to monitor activations in deep neural networks.
Please refer to the main branch for current use.


Current changes needed to enable an end-to-end example script for quantile monitoring (experimental):
- The file object_detection/det_quant_test_auto2.py in that repo gives an end-to-end example of how to run object detection models with quantile activation monitoring. Usage of the script however assumes that certain models are available from the mmdet library (Yolov3, RetinaNet, SSD are enabled). To do so one needs to :
Install (clone) mmdetection (the 2D version, different from FOCETA setup) from: https://github.com/open-mmlab/mmdetection and change path in object_detection/det_quant_test_auto2.py L. 25 to your individual location of mmdet.
- Change the paths in the set_up_mmdet_model function of the script to your individual config file locations. The config files are included in the github repo in a separate folder "model_configs".
- Make sure that in L. 320f the flags are quant_monitoring = True, ftrace_monitoring = False to enable quantile monitoring.
- A bound file can be created with the separate script object_detection/det_quant_test_auto2_bnds.py (need to do adjustments similar to above).
- Using other models or data sets will require to add a new data loader to the repo, as well an adjustment of the the model pre- and postprocessing performed in the  build_objdet_native_model_mmdetfunction.
- In mmdetection/mmdet/apis/inference.py do the following changes from L. 115f (inference_detector function):  
   if isinstance(imgs[0], np.ndarray):  
        cfg = cfg.copy()  
        # set loading pipeline type  
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'  
    else:  
        cfg = cfg.copy()  
        cfg.data.test.pipeline[0].type = 'LoadImageFromFile'  
- The Coco config files have unresolved dependencies, replace paths in the coco files of pytorchalfi/model_configs to the ones in mmdetection/configs .
