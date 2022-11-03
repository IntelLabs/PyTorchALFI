import sys, os
sys.path.append(os.getcwd())
from alficore.evaluation.sdc_plots.obj_det_plot_metrics_all_models import obj_det_plot_metrics

def obj_det_analysis(argv):

    exp_folder_paths = {"frcnn+CoCo":{
                        "neurons":{"path" :"/../pytorchalfi/result_files/frcnn_torchvision_1_trials/neurons_injs/per_batch/objDet_20221101-035607_1_faults_[1]_bits/coco", "typ":"no_resil"},
                        # "weights":{"path" :"result_files/yolov3_ultra_50_trials_nwstore/weights_injs/per_batch/objDet_20220213-202846_1_faults_[0_32]_bits/kitti/val/sdc_eval", "typ":"no_resil"}
                        },
            #             },
            #  "Yolo+Coco":{
            #             "neurons":{"path" :"path", "typ":"ranger"}, 
            #             "weights":{"path" :"path", "typ":"no_resil"}
            #             }
            }
    obj_det_plot_metrics(exp_folder_paths=exp_folder_paths)

if __name__ == "__main__":
    obj_det_analysis(sys.argv)