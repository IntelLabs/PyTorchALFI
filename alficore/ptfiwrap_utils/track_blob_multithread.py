import concurrent.futures
import concurrent
from copy import deepcopy
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm, trange
import multiprocessing.dummy as mp
from time import time

gt_json_file = "/home/qutub/PhD/git_repos/intel_git_repos/pfd_uet/result_files/result_files_paper/yolov3_ultra_200_trials/weights_injs/per_epoch/objDet_20220211-131505_1_faults_[0,8]_bits/lyft/val/coco_format.json"

# gt_json_file = "/home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_paper/yolov3_ultra_200_trials/weights_injs/per_epoch/objDet_20220128-111806_1_faults_[0, 8]_bits/lyft/val/coco_format.json"
trails = 200
def get_stable_pixels_v2(diff_list_blobfp, t_now, m_param, depth_trck, vic):

    assert m_param <= depth_trck
    def check_past(old_vics_maxs, m_param):
        checked = False
        for a in range(len(old_vics_maxs)):
            if a == 0 and old_vics_maxs[a] >= m_param:
                checked = True
                break
            elif a > 0 and np.array(old_vics_maxs[:a+1]).all() and np.sum(np.array(old_vics_maxs[:a+1])) >= m_param: #all occupied until a with at least 1
                checked = True
                break
        return checked

    frame_new = deepcopy(diff_list_blobfp[t_now])
    frame_new = frame_new[:,:,0]/255
    w,h = frame_new.shape
    stable_px_frame = np.zeros((w,h), dtype='int')

    # collect info about past frames
    olds = np.zeros((w,h))
    for x in range(1,depth_trck+1):
        if t_now - x < 0:
            break
        olds += diff_list_blobfp[t_now - x][:,:,0]/255

    # old_frames_u = t_now - depth_trck if t_now - depth_trck > 0 else 0
    # old_frames_t = t_now - 1 if t_now - 1 > 0 else t_now
    # if not old_frames_u == old_frames_t:
    #     olds = np.sum(diff_list_blobfp[old_frames_u: old_frames_t], axis = 0)/255

    # pixelwise check of persistency
    for i in range(w):
        for j in range(h):
            if frame_new[i,j] == 1 and olds[i,j] >= m_param: #static track update, ok
                stable_px_frame[i,j] += 1

            elif frame_new[i,j] == 1 and olds[i,j] < m_param and depth_trck > 1: #no static update, check for dynamic
                old_vics_maxs = [] #list of max in occupation in the order smallest (latest) region ... largest (oldest) region
                for u in range(1, depth_trck+1):
                    old_vic = deepcopy(olds[np.max([i-u*vic,0]):np.min([i+u*vic,w-1]),np.max([j-u*vic,0]):np.min([j+u*vic,h-1])])
                    if u > 1:
                        old_vic[np.max([i-(u-1)*vic,0]):np.min([i+(u-1)*vic,w-1]),np.max([j-(u-1)*vic,0]):np.min([j+(u-1)*vic,h-1])] = 0 #subtract inside area, note that a:a is empty selection
                    old_vics_maxs.append(np.max(old_vic)) #np.max(old_vic))

                # Determine which movement is plausible for pixel
                if check_past(old_vics_maxs, m_param):
                    stable_px_frame[i,j] += 1 #track update ok but moving

            elif frame_new[i,j] == 0 and olds[i,j] >= m_param: #track coasting
                stable_px_frame[i,j] += 1
    return stable_px_frame #stable at t_now

def get_stable_pixels(blob, dts, t_now, m_param, depth_trck, vic, track_type='fp'):
    frame_new = deepcopy(blob[t_now])
    frame_new = frame_new[:,:,0]/255
    w,h = frame_new.shape
    stable_px_frame = np.zeros((w,h), dtype='int')

    # iou between both
    olds = np.zeros((w,h))
    for x in range(1,depth_trck+1):
        if t_now - x < 0:
            break
        olds += blob[t_now - x][:,:,0]/255

    for i in range(w):
        for j in range(h):
            if frame_new[i,j] == 1 and olds[i,j] >= m_param: #track update ok
                stable_px_frame[i,j] += 1
            elif frame_new[i,j] == 1 and olds[i,j] < m_param:
                old_vic = olds[np.max([i-vic,0]):np.min([i+vic,w-1]),np.max([j-vic,0]):np.min([j+vic,h-1])]
                if np.max(old_vic) >= m_param:
                    stable_px_frame[i,j] += 1 #track update ok but moving
            elif frame_new[i,j] == 0 and olds[i,j] >= m_param: #track coasting
                stable_px_frame[i,j] += 1
    # stable_px_frame[stable_px_frame > 1] = 255
    if track_type=='fn':
        stable_px_frame = stable_px_frame * (dts[:,:,0]/255)
    return stable_px_frame #stable at t_now

viz = ['dt', 'cdt']
blob_track_type = 'dt_cdt'

# viz = ['gt', 'dt']
# blob_track_type = 'gt_dt'

# viz = ['dt', 'rcdt']
# blob_track_type = 'dt_rcdt'


track_types = ['fp', 'fn']
vicinity = [50, 50]

def run_tracking(epoch):
    print("started process for epoch: {} for tracks".format(epoch, track_types))
    vis_iso_type = None
    frames_dts = []
    frames_cdts = []
    frames_blob = []
    frames_blob_fp = []
    frames_blob_fn = []
    # frames_diff_blob_tDim = []
    # frames_diff_blob_fp_tDim = []
    # frames_diff_blob_fn_tDim = []
    # epoch = 0
    viz_path = os.path.join(os.path.dirname(gt_json_file), 'vis/images_only_bbox')
    # visualizer.save_video(viz_path=viz_path, epochs=epochs, vis_types=vis_types, vis_iso_types=[None]) ## saves prediction videos

    for img_id in range(len(os.listdir(viz_path))):
    # for img_id in range(1):
        # if viz[0] in ['gt', 'dt', 'rdt']:
        img_path = os.path.join(viz_path, str(img_id), "test_viz_{}_{}.png".format(img_id, viz[0]))
        if not os.path.exists(img_path):
            print("{} is missing".format(img_path))
            # exit()
        frames_dt = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE)
        frames_dts.append(frames_dt)
        
        if viz[1] == 'dt':
            img_path = os.path.join(viz_path, str(img_id), "test_viz_{}_{}.png".format(img_id, viz[1]))
        else:
            img_path = os.path.join(viz_path, str(img_id), str(epoch), "test_viz_{}_{}.png".format(img_id, viz[1]))
        if not os.path.exists(img_path):
            print("{} is missing".format(img_path))
            # exit()
        frames_cdt = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE)
        frames_cdts.append(frames_cdt)
        frames_blob.append(cv2.bitwise_xor(np.array(frames_cdt), np.array(frames_dt)))
        frame_blob_fp = np.array(frames_cdt) - np.array(frames_dt)
        frame_blob_fp[frame_blob_fp < 0] = 0
        frame_blob_fp[frame_blob_fp <127] = 0
        frame_blob_fp[frame_blob_fp > 127] = 255
        frames_blob_fp.append(frame_blob_fp)

        frame_blob_fn = np.array(frames_dt) - np.array(frames_cdt)
        frame_blob_fn[frame_blob_fn < 0] = 0
        frame_blob_fn[frame_blob_fn <127] = 0
        frame_blob_fn[frame_blob_fn > 127] = 255
        frames_blob_fn.append(frame_blob_fn)

    frames_dts = np.array(frames_dts)
    frames_cdts = np.array(frames_cdts)
    frames_blob = np.array(frames_blob)
    frame_blob_fp = np.array(frame_blob_fp)
    frame_blob_fn = np.array(frame_blob_fn)
        
    # frames_diff_blob_tDim = np.array(frames_diff_blob_tDim)
    # frames_diff_blob_fp_tDim = np.array(frames_diff_blob_fp_tDim)
    # frames_diff_blob_fn_tDim = np.array(frames_diff_blob_fn_tDim)


    # track_types = ['fp', 'fn', 'fp', 'fn']
    # vicinity = [5, 5, 10, 10]
    vicinity = [50, 50]
    for i, (track_type, vic) in enumerate(zip(track_types, vicinity)):
        depth_trck = 15
        m_param = 10 #< depth_trck
        if blob_track_type == 'dt_cdt':
            fp_tracking_per = os.path.join(os.path.dirname(gt_json_file),  'vis/plots_raw_data/{}/tracking_cdt_{}_{}_{}_{}.npy'.format(epoch, track_type, m_param, depth_trck, vic))
        elif blob_track_type == 'dt_rcdt':
            fp_tracking_per = os.path.join(os.path.dirname(gt_json_file),  'vis/plots_raw_data/{}/tracking_rcdt_{}_{}_{}_{}.npy'.format(epoch, track_type, m_param, depth_trck, vic))
        elif blob_track_type == 'gt_dt':
            fp_tracking_per = os.path.join(os.path.dirname(gt_json_file),  'vis/plots_raw_data/tracking_dt_{}_{}_{}_{}.npy'.format(track_type, m_param, depth_trck, vic))
        if os.path.exists(fp_tracking_per):
            print("epoch {} with track type: {}; vicinity: {} is skipped as it exists already".format(epoch, track_type, vic))
            # continue

        # vic = 5 # pixel vicinity to check for motion
        if track_type in 'fp':
            frames_blob = frames_blob_fp
        elif track_type in 'fn':
            frames_blob = frames_blob_fn
        nr_pics = len(frames_blob)
        stable_px_frames = []
        stable_pixels = []
        for t_now in trange(nr_pics, desc="epoch: {}".format(epoch)):
            # Get stable pixels
            if track_type in 'fp':
                stable_px_frame = get_stable_pixels( blob=frames_blob, dts=None, t_now=t_now, m_param=m_param, depth_trck=depth_trck, vic=vic, track_type='fp')
                stable_px_frames.append(stable_px_frame)
                stable_area = np.sum(stable_px_frame)/np.prod(stable_px_frame.shape)
            elif track_type in 'fn':
                stable_px_frame = get_stable_pixels(blob=frames_blob, dts=frames_dts[t_now], t_now=t_now, m_param=m_param, depth_trck=depth_trck, vic=vic, track_type='fn')
                stable_px_frames.append(stable_px_frame)
                stable_area = np.sum(stable_px_frame)/(np.sum(frames_dts[t_now][:,:,0]/255))
                # stable_area = np.sum(stable_px_frame)
            stable_pixels.append(stable_area)
            if t_now >16:
                DEBUG = True

        os.makedirs(os.path.dirname(fp_tracking_per), exist_ok=True)
        f = open(fp_tracking_per, 'wb')
        np.save(f, np.array(stable_pixels))
        print("epoch {} with track type: {}; vicinity: {} is complete".format(epoch, track_type, vic))

        if blob_track_type == 'dt_cdt':
            output = os.path.join(os.path.dirname(gt_json_file),  'vis/videos/{}/blob/blob_tracking_cdt_{}_{}_{}_{}.mp4'.format(epoch, track_type, m_param, depth_trck, vic))
        if blob_track_type == 'dt_rcdt':
            output = os.path.join(os.path.dirname(gt_json_file),  'vis/videos/{}/blob/blob_tracking_rcdt_{}_{}_{}_{}.mp4'.format(epoch, track_type, m_param, depth_trck, vic))            
        elif blob_track_type == 'gt_dt':
            output = os.path.join(os.path.dirname(gt_json_file),  'vis/videos/blob_tracking_dt_{}_{}_{}_{}.mp4'.format(track_type, m_param, depth_trck, vic))

        if output:
            only_bbox = True
            sample_frame = cv2.imread(img_path)
            height, width, _ = sample_frame.shape
            # if not os.path.exists(output):
            os.makedirs(os.path.dirname(output), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
            out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (50, 50)
            # fontScale
            fontScale = 1
            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2
            for img_id in range(len(stable_px_frames)):
                frame = stable_px_frames[img_id]*255
                if not only_bbox:
                    frame = cv2.putText(frame, 'frame id: {}'.format(img_id), org, font, fontScale, color, thickness, cv2.LINE_AA)
                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                out.write(frame) # Write out frame to video

                # cv2.imshow('video',frame)
                # if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                #     break

            # Release everything if job is finished
            out.release()
        # cv2.destroyAllWindows()

        # print("The output video is {}".format(output))
    # else:
    #     print("The output video {} already exists".format(output))
        # print('metric: perc of stable pixels', np.sum(stable_px_frame)/np.prod(stable_px_frame.shape))
    print("finished process for epoch: {}; file saved in {}".format(epoch, os.path.dirname(fp_tracking_per)))
    return True

# run_tracking(108)
try:
    executor = concurrent.futures.ProcessPoolExecutor(14)
    futures = [executor.submit(run_tracking, group)
            for group in range(trails)]
    concurrent.futures.wait(futures)
except KeyboardInterrupt:
    quit = True