import os
PATH = '/nwstore/florian/LR_detector_data_auto/2022-9-13-15-54/yolov3_ultra_1_trials/neurons_injs/per_image/test'
PATH1 = '/nwstore/florian/LR_detector_data_auto/2022-9-13-15-54/yolov3_ultra_1_trials/neurons_injs/per_image/test1'
os.makedirs(PATH, exist_ok=True)
os.rename(PATH, PATH1)