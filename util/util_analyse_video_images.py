import sys
import os
sys.path.append("..")
from alficore.dataloader.objdet_baseClasses.catalog import MetadataCatalog
from alficore.evaluation.visualization.evaluation_visualization_visualization import Visualizer, ColorMode
from PIL import Image, TiffImagePlugin
TiffImagePlugin.DEBUG = False
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# # image = np.random.randn(416,416,3)
# trails = 25
# fault = 1
# # /home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_paper/yolov3_5_trials/weights_injs/per_batch/objDet_20211217-151348_1_faults_[0, 31]_bits
# inj_policy = 'per_epoch'
# # inj_policy = 'per_image'
# # inj_policy = 'per_batch'
# # inj_type = 'neurons'
# inj_type = 'weights'

# # uuid = 'objDet_20211216-060241_1_faults_[0, 31]_bits' ## detectron2
# # # /home/qutub/PhD/git_repos/intel_git_repos/ranger/result_files/result_files_paper/yolov3_ultra_50_trials/neurons_injs/per_image/objDet_20220111-002317_1_faults_[0, 31]_bits/coco2017/val/orig_model/epochs/0/coco_instances_results_0_epoch.json
# # uuid = 'objDet_20211218-210740_1_faults_[1]_bits'
# uuid = 'objDet_20211218-210740_1_faults_[1]_bits' # lyft
# # uuid = "objDet_20220111-002317_1_faults_[0, 31]_bits" ## coco20017
# # model_name = 'det2_fasterRCNN'
# model_name = 'yolov3_ultra'
# # dataset = 'coco2017'
# dataset = 'lyft'
# epoch = 0
# batch_size = 1

# # coco = CoCo_obj_det_dataloader(dataset_type='val', batch_size=4, sampleN=0.001, num_workers=4)
# metadata = MetadataCatalog.get('{}/{}'.format(dataset, 'val'))
# # metadata = MetadataCatalog.get('coco2017/{}'.format('val'))
# # img_ids=[139, 285, 632, 724, 776, 785, 802, 872]
# img_ids = [0,1,2,3, 4, 5, 6, 7, 8, 9, 10]
# # img_ids = None
# # epochs = list(range(25))
# epochs = [0]
# ground_truth_file = '/home/fgeissle/ranger_repo/ranger/result_files/{}_{}_trials/{}_injs/{}/{}/{}/val/coco_format.json'
# original_epoch_json_file = '/home/fgeissle/ranger_repo/ranger/result_files/{}_{}_trials/{}_injs/{}/{}/{}/val/orig_model/epochs/{}/coco_instances_results_{}_epoch.json'

# visualizer = Visualizer(metadata=metadata, instance_mode=ColorMode.IMAGE, vis_mode='offline')


# #### visualises all the predictions without tp/fp/f isolations
# visualizer.draw_instance_gt_pred_offline(img_ids=img_ids, no_imgs=None, viz_gt=False, viz_dt=True, viz_cdt=True, viz_rdt=False, viz_rcdt=False, gt_json_file=ground_truth_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset),
#     dt_json_file=original_epoch_json_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset, epoch, epoch), resil_name='ranger', epoch=epochs, only_bbox=True)
# vis_types = ['dt', 'cdt']

# # vis_iso_types = ['tp', 'fp', 'fn']
# # if only_bbox is True, predictions are visualised in the form of filled rectangle boxes on black and white image.
# # if only_bbox is False, predictions are visualised in the form of rectangle boxes on original input image
# # visualizer.vis_isolated(img_ids=None, no_imgs=None, viz_gt=False, viz_dt=True, viz_cdt=True, viz_rdt=False, viz_rcdt=False, gt_json_file=ground_truth_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset),
# #     dt_json_file=original_epoch_json_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset, 0, 0), resil_name='ranger', epochs=epochs, only_bbox=True, vis_iso_types=vis_iso_types)

# viz_path = os.path.join(os.path.dirname(ground_truth_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset)), 'vis/images_only_bbox')
# frame_list_viz = visualizer.save_video(viz_path=viz_path, epochs=epochs, vis_types=vis_types, vis_iso_types=[None], only_bbox=True) ## saves prediction videos
# # print(frame_list_viz[0], frame_list_viz[1]) #TODO:

# dt_list = frame_list_viz[0] # dt - original
# cdt_list = frame_list_viz[1] # cdt - corrupted



# def get_diff_blob(dt_list, cdt_list, n):
#     """
#     Form o img_diff:
#     # white (255): blob fp induced by corruption
#     # grey (127): blob fn induced by corruption
#     # black (0): no change by corruption, both occupied OR both free
#     """
#     img0 = dt_list[n]
#     img1 = cdt_list[n]
#     img_diff = img1 - img0 #uint8 format is circular to 256
#     # fps: 255-0 -> 255
#     # fns: 0-255 -> 1
#     img_diff[img_diff == 1] = 127
#     # print(np.sum(img_diff[img_diff > 0]), np.sum(img_diff[img_diff == 1]), np.sum(img_diff[img_diff > 1]))
#     return img0, img1, img_diff


# def split_channels(img_diff):
#     """
#     Split img_diff into blobfn and blobfp pictures (white is blob, black the rest).
#     """
#     img_blobfn = deepcopy(img_diff)
#     mask_fn = img_blobfn == 127
#     img_blobfn[mask_fn] = 255 #white
#     img_blobfn[np.logical_not(mask_fn)] = 0

#     img_blobfp = deepcopy(img_diff)
#     mask_fp = img_blobfp == 255
#     img_blobfp[mask_fp] = 255 #white
#     img_blobfp[np.logical_not(mask_fp)] = 0

#     return img_blobfn, img_blobfp


# def get_diff_blob_fp_fn(img_ids, dt_list, cdt_list):
#     # Create diff_list_blobs ------------
#     diff_list = []
#     diff_list_blobfp = []
#     diff_list_blobfn = []
#     for n in range(len(img_ids)):
#         img0, img1, img_diff = get_diff_blob(dt_list, cdt_list, n)
#         img_blobfn, img_blobfp = split_channels(img_diff)

#         # fig, ax = plt.subplots(1,5, figsize=(50,100))
#         # ax[0].imshow(img0)
#         # ax[0].set_title('orig')
#         # ax[1].imshow(img1)
#         # ax[1].set_title('corr')
#         # ax[2].imshow(img_diff)
#         # ax[2].set_title('delta')
#         # ax[3].imshow(img_blobfp)
#         # ax[3].set_title('delta fp')
#         # ax[4].imshow(img_blobfn)
#         # ax[4].set_title('delta fn')
#         # fig.savefig('test_' + str(n) + '.png')

#         diff_list.append(img_diff)
#         diff_list_blobfp.append(img_blobfp)
#         diff_list_blobfn.append(img_blobfn)

#     return diff_list, diff_list_blobfp, diff_list_blobfn


# diff_list, diff_list_blobfp, diff_list_blobfn = get_diff_blob_fp_fn(img_ids, dt_list, cdt_list)

# # Alternative: artificial example -----------------
# two consecutive frames:
diff_list_blobfp = []

frame = np.zeros((100,100,1))
frame[0:10, 0:10, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 5:15, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[10:20, 10:20, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[15:25, 15:25, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[10:20, 20:30, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)

frame = np.zeros((100,100,1))
frame[5:15, 25:35, 0] = 255
diff_list_blobfp.append(frame)


depth_trck = 3
m_param = 2 #< depth_trck. Note: if vic is <= object size can be accepted earlier
vic = 10 # pixel vicinity to check for motion, choose smaller for long tracking times m/n

def get_stable_fps(diff_list_blobfp, t_now, depth_trck, m_param, vic):

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

    # pixelwise check of persistency
    for i in range(w):
        for j in range(h):
            if frame_new[i,j] == 1 and olds[i,j] >= m_param: #static track update, ok
                stable_px_frame[i,j] += 1

            elif frame_new[i,j] == 1 and olds[i,j] < m_param and depth_trck > 1: #no static update, check for dynamic
                old_vics_maxs = [] #list of max in occupation in the order smallest (latest) region ... largest (oldest) region
                for u in range(1, depth_trck+1):
                    old_vic = deepcopy(olds[np.max([i-u*vic,0]):np.min([i+u*vic,w-1]),np.max([j-u*vic,0]):np.min([j+u*vic,h-1])])
                    # if np.max(olds)>=2:
                    #     fig, ax = plt.subplots(1,1, figsize=(100,50))
                    #     ax.imshow(old_vic*50, cmap='gray', vmin=0, vmax=255)
                    #     fig.savefig('test_temp.png')
                    if u > 1:
                        old_vic[np.max([i-(u-1)*vic,0]):np.min([i+(u-1)*vic,w-1]),np.max([j-(u-1)*vic,0]):np.min([j+(u-1)*vic,h-1])] = 0 #subtract inside area, note that a:a is empty selection
                    # if np.max(olds)>=2:
                    #     fig, ax = plt.subplots(1,1, figsize=(100,50))
                    #     ax.imshow(old_vic*50, cmap='gray', vmin=0, vmax=255)
                    #     fig.savefig('test_temp.png')

                    old_vics_maxs.append(np.max(old_vic)) #np.max(old_vic))

                # Determine which movement is plausible for pixel
                if check_past(old_vics_maxs, m_param):
                    stable_px_frame[i,j] += 1 #track update ok but moving
                # # if np.max(old_vic) >= m_param:
                # #     stable_px_frame[i,j] += 1 #track update ok but moving

            elif frame_new[i,j] == 0 and olds[i,j] >= m_param: #track coasting
                stable_px_frame[i,j] += 1

    return stable_px_frame #stable at t_now


nr_pics = 10
fig, ax = plt.subplots(1,nr_pics, figsize=(100,50))
fig2, ax2 = plt.subplots(1,nr_pics, figsize=(100,50))
for t_now in range(nr_pics):
    print('plot ', t_now)

    # Get stable pixels
    stable_px_frame = get_stable_fps(diff_list_blobfp, t_now, depth_trck, m_param, vic)
    ax[t_now].imshow(stable_px_frame*255, cmap='gray', vmin=0, vmax=255)
    ax[t_now].set_title('frame ' + str(t_now))

    # Get current blob pixels
    ax2[t_now].imshow(diff_list_blobfp[t_now], cmap='gray', vmin=0, vmax=255)
    ax2[t_now].set_title('frame ' + str(t_now))

    print('metric: perc of stable pixels', np.sum(stable_px_frame)/np.prod(stable_px_frame.shape))

fig.savefig('test_across_frames' + str(1) + '.png')
# fig2.savefig('test_across_frames_blob' + str(1) + '.png')
print()




# vis_iso_types = ['tp', 'fp', 'fn']
# if only_bbox is True, predictions are visualised in the form of filled rectangle boxes on black and white image.
# if only_bbox is False, predictions are visualised in the form of rectangle boxes on original input image
# visualizer.vis_isolated(img_ids=None, no_imgs=None, viz_gt=True, viz_dt=False, viz_cdt=False, viz_rdt=False, viz_rcdt=False, gt_json_file=ground_truth_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset),
#     dt_json_file=original_epoch_json_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset, 0, 0), resil_name='ranger', epochs=epochs, only_bbox=True, vis_iso_types=vis_iso_types)

# vis_types = ['dt', 'cdt']
# vis_types = ['gt', 'dt', 'cdt']
# vis_iso_types=  [None]
# viz_path = os.path.join(os.path.dirname(ground_truth_file.format(model_name, trails, inj_type, inj_policy, uuid, dataset)), 'vis/images_isolated_only_bbox')
# visualizer.save_video(viz_path=viz_path, epochs=epochs, vis_types=vis_types, vis_iso_types=vis_iso_types, only_bbox=True) ## saves isolated videos




# def plot_diff_blob_across_frames(dt_list, n, m):
#     img0 = dt_list[n]
#     img1 = dt_list[m]
    
#     img_diff = img1 - img0 #uint8 format is circular to 256
#     img_diff[img_diff > 0] = 255
#     fig, ax = plt.subplots(1,3, figsize=(50,100))
#     ax[0].imshow(img0)
#     ax[1].imshow(img1)
#     ax[2].imshow(img_diff)
#     fig.savefig('test_frames_' + str(n) + '_' + str(m) + '.png')
#     return img_diff
#
# plot_diff_blob_across_frames(diff_list_blobfp, 0, 1)
# plot_diff_blob_across_frames(diff_list_blobfp, 0, 2)
# plot_diff_blob_across_frames(diff_list_blobfp, 1, 2)
