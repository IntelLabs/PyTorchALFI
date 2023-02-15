# Activation monitoring with pytorchalfi (experimental)

*NOTE* This is an experimental branch using quantile hooks to monitor activations in deep neural networks.
Please refer to the main branch for a more mature version of pytorchalfi.

Prerequisite:
Please install the mmdetection framework from https://github.com/open-mmlab/mmdetection in a separate location and change paths wherever needed to your location of mmdetection.

The following changes specific to mmdetection need to be done to enable an end-to-end example script for quantile monitoring (named object_detection/det_quant_test_auto2.py):
- Change path in object_detection/det_quant_test_auto2.py L. 25 to your individual location of mmdetection.
- Change the paths in the set_up_mmdet_model function of the script object_detection/det_quant_test_auto2.py to your individual config file locations. The config files are included in the separate folder "model_configs" or the ones already included in mmdetection can be used.
- The Coco config files provided in this folder in "model_configs" have unresolved dependencies, update paths in those files (only Coco) of pytorchalfi/model_configs to the relevant locations in mmdetection/configs. For example, instead of "./base/..." use "<path-to-mmdetection>/configs/base/...".
- Make sure that in L. 320f the flags are quant_monitoring = True, ftrace_monitoring = False to enable quantile monitoring.
- A bound file is provided for the existing setups, in general it can be created with the separate script object_detection/det_quant_test_auto2_bnds.py (need to do adjustments similar to above).
- In mmdetection/mmdet/apis/inference.py do the following changes to the if clause starting from L. 115f (in inference_detector function):  
   if isinstance(imgs[0], np.ndarray):  
        cfg = cfg.copy()  
        # set loading pipeline type  
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'  
    else:  
        cfg = cfg.copy()  
        cfg.data.test.pipeline[0].type = 'LoadImageFromFile'  

