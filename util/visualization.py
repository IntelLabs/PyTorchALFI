import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from torchvision.utils import make_grid
import torch
import matplotlib.gridspec as gridspec
from alficore.evaluation.visualization.visualization import simple_visualization_direct_img
from alficore.dataloader.objdet_baseClasses.boxes import BoxMode, Boxes
from alficore.dataloader.objdet_baseClasses.instances import Instances

def get_tensorlist(x):
        """
        Take all values contained in the tensor (all dimensions) and make them a flat list.
        :param x: torch tensor
        :return: list
        """
        return x.view(-1, num_flat_features_all(x)).tolist()[0]


def num_flat_features_all(x):
        """
        Multiplies ALL dimensions of input tensor, including batch nr.
        :param x: torch tensor
        :return: scalar, number of elements.
        """
        size = x.size()  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




def imshow_labels(img, pred_labels, gt_labels, col_ch, batch_size):

    nr_predict = len(pred_labels[0])

    fig, axs = plt.subplots(1, 1, figsize=(4*batch_size, 10+nr_predict))

    img = make_grid(img) # make all batch images a grid
    img = img / 2. + 0.5 # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0)) # have to rearrange because e.g. color channel is first in tensor but last in imshow
    npimg = np.clip(npimg, 0, 1) # restrict color range

    if col_ch == 1: #grayscale images
        axs.imshow(npimg[:,:,0], cmap='gray')
    else :
        axs.imshow(npimg, interpolation='nearest')

    # Text labels : ------------------------------------------
    axs.annotate('Predicted:',
                 xy=(0, 0), xycoords='axes fraction',
                 xytext=(-100, -100), fontsize=20,
                 textcoords='offset pixels')
    axs.annotate('Ground truth:',
                 xy=(0, 0), xycoords='axes fraction',
                 xytext=(-150, -100 - (nr_predict - 1)*30 - 50), fontsize=20,
                 textcoords='offset pixels')

    for h in range(batch_size):
        xshift = 100 + h * 300

        for j in range(nr_predict):
            axs.annotate(pred_labels[h][j],
                         xy=(0, 0), xycoords='axes fraction',
                         xytext=(xshift, -100 - j*30), fontsize=20,
                         textcoords='offset pixels')

        # ground truth
        axs.annotate(gt_labels[h],
                     xy=(0, 0), xycoords='axes fraction',
                     xytext=(xshift, -100 - (nr_predict - 1)*30 - 50), fontsize=20,
                     textcoords='offset pixels')


    plt.show()



def updated_pic_largest(pic, pic_largest):
    
    if pic_largest is None:
        return pic
    else: 
        pic_max = np.max(pic.get_array().data)
        pic_largest_max = np.max(pic_largest.get_array().data)
        if pic_max > pic_largest_max:
            pic_largest = pic
        # torch.max(fmap_ij)
        # np.max(pic.get_array().data)
        # if torch.max(fmap_ij) > np.max(pic_largest):
        #     pic_largest = np.array(fmap_ij)
        return pic_largest

def map_fmaps_to_fmap_array(layer_acts, layer_dim, sep_line_wdth):
    # print('before', np.shape(list_act[layer_nr]))
    fmap = np.reshape(layer_acts, layer_dim)  # N, C, H, W
    # print('after', np.shape(fmap))
    if layer_dim[0] == 1: #for fcc, the channel is 1
        fmap_oI = fmap[0]
    else:
        # fmap_oI = np.vstack([fmap[c] for c in range(layer_dim[0])]) #add all fmaps into one array of size C*W,H
        # Add little separators between feature maps
        sep_line = np.empty((sep_line_wdth, layer_dim[-1]))
        sep_line[:]=np.NaN
        fmap_oI = np.vstack([np.vstack([fmap[c], sep_line]) for c in range(layer_dim[0] - 1)])  # add up to last one and an in-between line
        fmap_oI = np.vstack([fmap_oI, fmap[layer_dim[0] - 1]])  # add last channel

    return fmap_oI



def plot_fmaps(layer, list_act, fi_list_act, layer_sizes, nr_fmaps_rearr, bnds, ylimmax, orig_pred, corr_pred, img_orig, img_corr, add_input_plot, add_output_plot, title):
    """
    Plots for one layer no-fault and fault pic of fmap side by side. Plots all channels as vstack.
    For fully connected layers there is no channel C.
    First color scale goes up to ranger bound (red). Second color scale is only red, starting from ranger bounds to 100.
    :param N: batch nr
    :param layer_nr: layer number
    :param list_act: list of all activations in absence of fault
    :param fi_list_act: list of all activations in presence of fault
    :param layer_sizes: list of all layer sizes
    :param axes: the explicit axes for the given row of the subplot. Form ax1, ax2
    :param: bnds ranger bonds for color scale
    """


    # Create main grid ------------------------------------------
    fig = plt.figure(figsize=(10, 3 * (len(layer) + int(add_output_plot) + int(add_input_plot)))) #TODO: control figure width here!
    plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.2) #, wspace=0.4 #If things get truncated increase top, bottom!

    gs0 = gridspec.GridSpec(len(layer) + int(add_output_plot)+ int(add_input_plot), 2)  # two columns
    gs_left = []
    gs_right = []

    # Reshape fcc layers into "fmaps" ----------------------------
    fmap_plot = []
    fi_fmap_plot = []
    for ls in layer:
       
        # Reshaping fcc layerss
        fmap_res = torch.reshape(torch.tensor(list_act[ls]), layer_sizes[ls])
        fmap_plot.append(fmap_res)

        fi_fmap_res = torch.reshape(torch.tensor(fi_list_act[ls]), layer_sizes[ls])
        fi_fmap_plot.append(fi_fmap_res)


    # Create fmap axes -----------------------------------------
    if add_input_plot:
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0, 0])  # left side
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0, 1])  # right side
        gs_left.append(gs00)
        gs_right.append(gs01)

    for ll in range(len(layer)):
        lay = layer[ll]
        gs00 = gridspec.GridSpecFromSubplotSpec(nr_fmaps_rearr[lay][0], nr_fmaps_rearr[lay][1],
                                                subplot_spec=gs0[ll + int(add_input_plot), 0])  # left side
        gs01 = gridspec.GridSpecFromSubplotSpec(nr_fmaps_rearr[lay][0], nr_fmaps_rearr[lay][1],
                                                subplot_spec=gs0[ll + int(add_input_plot), 1])  # right side
        gs_left.append(gs00)
        gs_right.append(gs01)

    if add_output_plot:
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1, 0])  # left side
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1, 1])  # right side
        gs_left.append(gs00)
        gs_right.append(gs01)


    # Plot fmaps ---------------------------------------------------
    all_axes_left = []
    all_axes_right = []

    
    if add_input_plot:
        # left side ----------------------------------
        ax_left = fig.add_subplot(gs_left[0][0, 0])
        all_axes_left.append(ax_left)
        mymap = customize_cmap(img_orig, 255, 255, isgray=True)  # individual cmap for each fmap!
        pic = ax_left.imshow(img_orig.permute(1,2,0), aspect='auto', cmap=mymap) #(move color channel to the back to plot) , cmap="gray"
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        # right side ----------------------------------
        ax_right = fig.add_subplot(gs_right[0][0, 0])
        all_axes_right.append(ax_right)
        fi_mymap = customize_cmap(img_corr, 255, 255, isgray=True)  # individual cmap for each fmap!
        fi_pic = ax_right.imshow(img_corr.permute(1,2,0), aspect='auto', cmap=fi_mymap) #
        ax_right.set_xticks([])
        ax_right.set_yticks([])


    for l0 in range(len(layer)):  # nr layers
        # Note: if info for all layers is present in list then use l, if only the relevant one is plotted use l0
        l = layer[l0]
        fmap = fmap_plot[l0]
        fi_fmap = fi_fmap_plot[l0]
        # print('check', l0, l, torch.max(fi_fmap))

        axes_left = []
        axes_right = []
        pic_largest = None
        fi_pic_largest = None

        for i in range(nr_fmaps_rearr[l][0]):
            for j in range(nr_fmaps_rearr[l][1]):

                # left side ----------------------------------
                ax_left = fig.add_subplot(gs_left[l0 + int(add_input_plot)][i, j])
                axes_left.append(ax_left)
                if len(fmap.shape) < 3:
                    fmap_ij = fmap
                else:
                    fmap_ij = fmap[i * nr_fmaps_rearr[l][1] + j]

                if bnds is not None:
                    act_bnd = bnds[l][1]
                else:
                    act_bnd = 10.
                # print('check', l, i, j, torch.max(fmap_ij), act_bnd, ylimmax)
                ylimmax = np.max([act_bnd, ylimmax]) #adjust ylim

                mymap = customize_cmap(fmap_ij, act_bnd, ylimmax)  # individual cmap for each fmap!
                pic = ax_left.imshow(fmap_ij,  aspect='auto', cmap=mymap) #aspect='auto',
                pic_largest = updated_pic_largest(pic, pic_largest)

                ax_left.set_xticks([])
                ax_left.set_yticks([])

                # right side ----------------------------------
                ax_right = fig.add_subplot(gs_right[l0 + int(add_input_plot)][i, j])
                axes_right.append(ax_right)
                if len(fi_fmap.shape) < 3:
                    fi_fmap_ij = fi_fmap
                else:
                    fi_fmap_ij = fi_fmap[i * nr_fmaps_rearr[l][1] + j]

                # print('check', l, i, j, torch.min(fi_fmap_ij), torch.max(fi_fmap_ij), act_bnd, ylimmax)
                fi_mymap = customize_cmap(fi_fmap_ij, act_bnd, ylimmax)  # individual cmap for each fmap!
                fi_pic = ax_right.imshow(fi_fmap_ij,  aspect='auto', cmap=fi_mymap) #aspect='auto',
                fi_pic_largest = updated_pic_largest(fi_pic, fi_pic_largest)

                ax_right.set_xticks([])
                ax_right.set_yticks([])

        all_axes_left.append(axes_left)
        all_axes_right.append(axes_right)

        fig.colorbar(pic_largest, ax=axes_left)
        fig.colorbar(fi_pic_largest, ax=axes_right)



    if add_output_plot:
        # left side ----------------------------------
        ax_left = fig.add_subplot(gs_left[-1][0, 0])
        all_axes_left.append(ax_left)
        predictions = np.array(orig_pred) - np.min(orig_pred)
        predictions = predictions/ predictions.sum()

        mymap = customize_cmap(predictions, 1.6, 2)  # individual cmap for each fmap!
        pic = ax_left.imshow(np.expand_dims(predictions, axis=0), aspect='auto', cmap=mymap)
        fig.colorbar(pic, ax=ax_left)
        ax_left.set_xticks(np.linspace(0, 9, num=10))
        ax_left.set_yticks([])

        # right side ----------------------------------
        ax_right = fig.add_subplot(gs_right[-1][0, 0])
        all_axes_right.append(ax_right)
        fi_predictions = np.array(corr_pred) - np.min(corr_pred)
        fi_predictions = fi_predictions / fi_predictions.sum()

        fi_mymap = customize_cmap(fi_predictions, 1.6, 2)  # individual cmap for each fmap! TODO: increase to remove red parts for output plot
        fi_pic = ax_right.imshow(np.expand_dims(fi_predictions, axis=0), aspect='auto', cmap=fi_mymap)
        fig.colorbar(fi_pic, ax=ax_right)
        ax_right.set_xticks(np.linspace(0, 9, num=10))
        ax_right.set_yticks([])


    # Labels:
    fig.suptitle(title, y=0.94, fontsize=18)
    # fig.suptitle('No fault' + " "*40 + 'With fault', y=0.94, fontsize=18)
    fig.text(0.07, 0.5, 'Output' + " "*22 + 'Relu2' + " "*22 + 'Relu1' + " "*23 + 'Input', va='center', rotation='vertical', fontsize=18) #0.39
    





def customize_cmap(fmap, bnd_normal, ylimmax, isgray=False):
    # Build custom color map:
    cmax = np.max(np.array(fmap))

    ylim_c1 = bnd_normal #at that number the colorbar 1 should end
    ylim_c2 = ylimmax #at that number the colorbar 2 should end
    steps_c1 = 100
    # print('cmax', cmax, ylim_c1, cmax / ylim_c1)

    # print('color max', cmax)
    if cmax <= ylim_c1:
        max_fac = cmax/ylim_c1
        if isgray:
            colors1 = plt.cm.gray(np.linspace(0, max_fac, steps_c1)) #here distribution is relative
        else:
            # colors1 = plt.cm.rainbow(np.linspace(0, max_fac, steps_c1)) #here distribution is relative
            colors1 = plt.cm.rainbow(np.linspace(0.1, max_fac, steps_c1)) #here distribution is relative #TODO test
        # colors1 = plt.cm.rainbow(np.linspace(0., 1., steps_c1))  # here distribution is relative
        colors = np.array(colors1)
    else:
        max_fac = cmax / ylim_c2
        steps_c2 = int(steps_c1*(cmax/ylim_c1 - 1))#int((cmax - ylim_c1)/cmax * ylim_c1)
        # print('steps', steps_c1, steps_c2, cmax, max_fac)
        if isgray:
            colors1 = plt.cm.gray(np.linspace(0, max_fac, steps_c1)) #here distribution is relative
        else:
            # colors1 = plt.cm.rainbow(np.linspace(0., 1, steps_c1))  # here distribution is relative
            colors1 = plt.cm.rainbow(np.linspace(0.1, 1, steps_c1))  # here distribution is relative #TODO: test
        colors2 = plt.cm.Reds(np.linspace(0.6,max_fac , steps_c2)) #np.max(0.6,max_fac)
        colors = np.vstack((colors1, colors2))

    # combine them and build a new colormap
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return mymap


def annos2instance(annos, imgsize=(1200, 1920)):
    out_instance = Instances(imgsize)
    boxes = [n["bbox"] for n in annos]

    boxes = [BoxMode.convert(box, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for box in boxes] #if boxmode is 1

    classes = [n["category_id"] for n in annos]
    if len(annos) > 0  and "score" in annos[0].keys():
        scores = [n['score'] for n in annos]
    else:
        scores = [0 for n in annos]
    out_instance.set("pred_boxes", Boxes(boxes))
    out_instance.set("pred_classes", torch.tensor(classes).type(torch.ByteTensor))
    out_instance.set("scores", scores)
    return [{"instances": out_instance}]


def plot_fmaps_objdet(layer, list_act, fi_list_act, layer_sizes, nr_fmaps_rearr, bnds, ylimmax, orig_pred, corr_pred, img_orig, img_corr, add_input_plot, add_output_plot, title, classes, figsz, y_text):
    """
    Plots for one layer no-fault and fault pic of fmap side by side. Plots all channels as vstack.
    For fully connected layers there is no channel C.
    First color scale goes up to ranger bound (red). Second color scale is only red, starting from ranger bounds to 100.
    :param N: batch nr
    :param layer_nr: layer number
    :param list_act: list of all activations in absence of fault
    :param fi_list_act: list of all activations in presence of fault
    :param layer_sizes: list of all layer sizes
    :param axes: the explicit axes for the given row of the subplot. Form ax1, ax2
    :param: bnds ranger bonds for color scale
    """

    # Create main grid ------------------------------------------
    # fig = plt.figure(figsize=(8, 3* (len(layer) + int(add_output_plot) + int(add_input_plot)))) #TODO: control figure width here! 3 for Lenet, 
    fig = plt.figure(figsize=figsz)
    plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.15, wspace=0.15) #, wspace=0.4 #If things get truncated increase top, bottom!

    gs0 = gridspec.GridSpec(len(layer) + int(add_output_plot)+ int(add_input_plot), 2)  # two columns
    gs_left = []
    gs_right = []

    # Reshape fcc layers into "fmaps" ----------------------------
    fmap_plot = []
    fi_fmap_plot = []
    for ls in layer:
       
        # Reshaping fcc layerss
        fmap_res = torch.reshape(torch.tensor(list_act[ls]), layer_sizes[ls])
        fmap_plot.append(fmap_res)

        fi_fmap_res = torch.reshape(torch.tensor(fi_list_act[ls]), layer_sizes[ls])
        fi_fmap_plot.append(fi_fmap_res)


    # Create fmap axes -----------------------------------------
    if add_input_plot:
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0, 0])  # left side
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0, 1])  # right side
        gs_left.append(gs00)
        gs_right.append(gs01)

    for ll in range(len(layer)):
        lay = layer[ll]
        gs00 = gridspec.GridSpecFromSubplotSpec(nr_fmaps_rearr[lay][0], nr_fmaps_rearr[lay][1],
                                                subplot_spec=gs0[ll + int(add_input_plot), 0])  # left side
        gs01 = gridspec.GridSpecFromSubplotSpec(nr_fmaps_rearr[lay][0], nr_fmaps_rearr[lay][1],
                                                subplot_spec=gs0[ll + int(add_input_plot), 1])  # right side
        gs_left.append(gs00)
        gs_right.append(gs01)

    if add_output_plot:
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1, 0])  # left side
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1, 1])  # right side
        gs_left.append(gs00)
        gs_right.append(gs01)


    # h = subplot(5, 2, 10);
    # % your plot command here
    # % adjust position; the offset has to be determined by visually by trial and error
    # pos = get(h, 'Position') 
    # posnew = pos; posnew(1) = posnew(1) + 0.04; set(h, 'Position', posnew)

    # gs = gridspec.GridSpec(3,1)
    # ax.set_position(gs[0:2].get_position(fig))
    # ax.set_subplotspec(gs[0:2])  
    # gs0.get_grid_positions(fig)
    pos_left_x1 = gs_left[1].get_grid_positions(fig)[2]
    # gs_left[1].set_grid_positions(pos_left_x1 + 0.03)

    # Plot fmaps ---------------------------------------------------
    all_axes_left = []
    all_axes_right = []

    
    if add_input_plot:
        # left side ----------------------------------
        ax_left = fig.add_subplot(gs_left[0][0, 0])
        all_axes_left.append(ax_left)
        mymap = customize_cmap(img_orig, 255, 255, isgray=True)  # individual cmap for each fmap!

        # # Rescale kitti image to fit in plot better
        # import cv2
        # resc_fct = 0.9
        # img_orig_resz = np.array(img_orig.permute(1,2,0))
        # img_orig_resz = cv2.resize(img_orig_resz, dsize=(int(img_orig_resz.shape[0]*resc_fct), int(img_orig_resz.shape[1]*resc_fct)), interpolation=cv2.INTER_CUBIC)
        pic = ax_left.imshow(img_orig.permute(1,2,0), cmap=mymap) #(move color channel to the back to plot) , cmap="gray"
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        # right side ----------------------------------
        ax_right = fig.add_subplot(gs_right[0][0, 0])
        all_axes_right.append(ax_right)
        fi_mymap = customize_cmap(img_corr, 255, 255, isgray=True)  # individual cmap for each fmap!
        fi_pic = ax_right.imshow(img_corr.permute(1,2,0), cmap=fi_mymap) #
        ax_right.set_xticks([])
        ax_right.set_yticks([])


    for l0 in range(len(layer)):  # nr layers
        # Note: if info for all layers is present in list then use l, if only the relevant one is plotted use l0
        l = layer[l0]
        fmap = fmap_plot[l0]
        fi_fmap = fi_fmap_plot[l0]
        # print('check', l0, l, torch.max(fi_fmap))

        axes_left = []
        axes_right = []
        pic_largest = None
        fi_pic_largest = None

        for i in range(nr_fmaps_rearr[l][0]):
            for j in range(nr_fmaps_rearr[l][1]):

                # left side ----------------------------------
                ax_left = fig.add_subplot(gs_left[l0 + int(add_input_plot)][i, j])
                axes_left.append(ax_left)
                if len(fmap.shape) < 3:
                    fmap_ij = fmap
                else:
                    fmap_ij = fmap[i * nr_fmaps_rearr[l][1] + j]

                if bnds is not None:
                    act_bnd = bnds[l][1]
                else:
                    act_bnd = 10.
                # print('check', l, i, j, torch.max(fmap_ij), act_bnd, ylimmax)
                ylimmax = np.max([act_bnd, ylimmax]) #adjust ylim

                mymap = customize_cmap(fmap_ij, act_bnd, ylimmax)  # individual cmap for each fmap!
                pic = ax_left.imshow(fmap_ij, aspect='auto', cmap=mymap) #aspect='auto', #TODO: check extent
                pic_largest = updated_pic_largest(pic, pic_largest)

                ax_left.set_xticks([])
                ax_left.set_yticks([])

                # right side ----------------------------------
                ax_right = fig.add_subplot(gs_right[l0 + int(add_input_plot)][i, j])
                axes_right.append(ax_right)
                if len(fi_fmap.shape) < 3:
                    fi_fmap_ij = fi_fmap
                else:
                    fi_fmap_ij = fi_fmap[i * nr_fmaps_rearr[l][1] + j]

                # print('check', l, i, j, torch.min(fi_fmap_ij), torch.max(fi_fmap_ij), act_bnd, ylimmax)
                fi_mymap = customize_cmap(fi_fmap_ij, act_bnd, ylimmax)  # individual cmap for each fmap!
                fi_pic = ax_right.imshow(fi_fmap_ij, aspect='auto', cmap=fi_mymap) #aspect='auto',
                fi_pic_largest = updated_pic_largest(fi_pic, fi_pic_largest)

                ax_right.set_xticks([])
                ax_right.set_yticks([])

        all_axes_left.append(axes_left)
        all_axes_right.append(axes_right)

        fig.colorbar(pic_largest, ax=axes_left)
        fig.colorbar(fi_pic_largest, ax=axes_right)



    if add_output_plot:
        # left side ----------------------------------
        ax_left = fig.add_subplot(gs_left[-1][0, 0])
        all_axes_left.append(ax_left)
        
        output_tp = annos2instance(orig_pred['tp'])
        output_fp = annos2instance(orig_pred['fp'])
        output_fn = annos2instance(orig_pred['fn'])
        img_wr = {'image': img_orig}
        label = "TP: " + str(len(orig_pred['tp'])) + ", " + "FP: " + str(len(orig_pred['fp'])) + ", " + "FN: " + str(len(orig_pred['fn']))

        vis_out = simple_visualization_direct_img(img_wr, output_tp[0]['instances'], None, None, inset_str=label, extra_boxes=output_fp[0]["instances"], extra_boxes2= output_fn[0]["instances"], classes=classes)
        ax_left.imshow(vis_out.get_image())
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        # right side ----------------------------------
        ax_right = fig.add_subplot(gs_right[-1][0, 0])
        all_axes_right.append(ax_right)
        
        output_tp = annos2instance(corr_pred['tp'])
        output_fp = annos2instance(corr_pred['fp'])
        output_fn = annos2instance(corr_pred['fn'])
        img_wr = {'image': img_corr}
        label = "TP: " + str(len(corr_pred['tp'])) + ", " + "FP: " + str(len(corr_pred['fp'])) + ", " + "FN: " + str(len(corr_pred['fn']))

        vis_out = simple_visualization_direct_img(img_wr, output_tp[0]['instances'], None, None, inset_str=label, extra_boxes=output_fp[0]["instances"], extra_boxes2= output_fn[0]["instances"], classes=classes)
        ax_right.imshow(vis_out.get_image())
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        


    # Labels:
    fig.suptitle(title, y=0.94, fontsize=22)
    # fig.suptitle('No fault' + " "*40 + 'With fault', y=0.94, fontsize=18)
    fig.text(0.07, 0.5, y_text, va='center', rotation='vertical', fontsize=22) #0.39
    

# until here refined ...



# def plot_fmap_layer(ls, layer, list_act, fi_list_act, layer_sizes, nr_fmaps_rearr, axes, bnds, ylimmax, orig_pred, corr_pred, img_orig, img_corr, add_input_plot, add_output_plot):
#     """
#     Plots for one layer no-fault and fault pic of fmap side by side. Plots all channels as vstack.
#     For fully connected layers there is no channel C.
#     First color scale goes up to ranger bound (red). Second color scale is only red, starting from ranger bounds to 100.
#     :param N: batch nr
#     :param layer_nr: layer number
#     :param list_act: list of all activations in absence of fault
#     :param fi_list_act: list of all activations in presence of fault
#     :param layer_sizes: list of all layer sizes
#     :param axes: the explicit axes for the given row of the subplot. Form ax1, ax2
#     :param: bnds ranger bonds for color scale
#     """

#     # layer_nr = layer[ls]
#     # sep_line_wdth = 1
#     # ax1, ax2 = axes[ls] #no fault, fault

#     # Create main grid ------------------------------------------
#     fig = plt.figure(figsize=(10, 3 * (len(layer) + int(add_output_plot) + int(add_input_plot)))) #TODO: control figure width here!
#     gs0 = gridspec.GridSpec(len(layer) + int(add_output_plot)+ int(add_input_plot), 2)  # two columns
#     gs_left = []
#     gs_right = []

#     # Reshape fcc layers into "fmaps" ----------------------------
#     fmap_plot = []
#     fi_fmap_plot = []
#     for ls in layer:
       
#         # Reshaping fcc layerss
#         fmap_res = torch.reshape(torch.tensor(list_act[ls]), layer_sizes[ls])
#         fmap_plot.append(fmap_res)

#         fi_fmap_res = torch.reshape(torch.tensor(fi_list_act[ls]), layer_sizes[ls])
#         fi_fmap_plot.append(fi_fmap_res)


#     # Create fmap axes -----------------------------------------
#     if add_input_plot:
#         gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0, 0])  # left side
#         gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0, 1])  # right side
#         gs_left.append(gs00)
#         gs_right.append(gs01)

#     for ll in range(len(layer)):
#         lay = layer[ll]
#         gs00 = gridspec.GridSpecFromSubplotSpec(nr_fmaps_rearr[lay][0], nr_fmaps_rearr[lay][1],
#                                                 subplot_spec=gs0[ll + int(add_input_plot), 0])  # left side
#         gs01 = gridspec.GridSpecFromSubplotSpec(nr_fmaps_rearr[lay][0], nr_fmaps_rearr[lay][1],
#                                                 subplot_spec=gs0[ll + int(add_input_plot), 1])  # right side
#         gs_left.append(gs00)
#         gs_right.append(gs01)

#     if add_output_plot:
#         gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1, 0])  # left side
#         gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1, 1])  # right side
#         gs_left.append(gs00)
#         gs_right.append(gs01)


#     # Plot fmaps ---------------------------------------------------
#     all_axes_left = []
#     all_axes_right = []

    
#     # if add_input_plot:
#     #     # left side ----------------------------------
#     #     ax_left = fig.add_subplot(gs_left[-1][0, 0])
#     #     axes_left.append(ax_left)
#     #     # predictions = torch.nn.functional.softmax(outputs[N], dim=0)  # normalized tensors
#     #     predictions = np.array(orig_pred) - np.min(orig_pred)
#     #     predictions = predictions/ predictions.sum()
#     #     # predictions += 1e-3

#     #     mymap = customize_cmap(predictions, 1.6, 2)  # individual cmap for each fmap!
#     #     pic = ax_left.imshow(np.expand_dims(predictions, axis=0), aspect='auto', cmap=mymap)
#     #     fig.colorbar(pic, ax=ax_left)
#     #     ax_left.set_xticks(np.linspace(0, 9, num=10))
#     #     ax_left.set_yticks([])

#     #     # right side ----------------------------------
#     #     ax_right = fig.add_subplot(gs_right[-1][0, 0])
#     #     axes_right.append(ax_right)
#     #     # fi_predictions = torch.nn.functional.softmax(fi_outputs[N], dim=0)  # normalized tensors
#     #     # fi_predictions = (fi_outputs[N] - torch.min(fi_outputs[N])) / fi_outputs[N].norm()
#     #     fi_predictions = np.array(corr_pred) - np.min(corr_pred)
#     #     fi_predictions = fi_predictions / fi_predictions.sum()

#     #     fi_mymap = customize_cmap(fi_predictions, 1.6, 2)  # individual cmap for each fmap! TODO: increase to remove red parts for output plot
#     #     fi_pic = ax_right.imshow(np.expand_dims(fi_predictions, axis=0), aspect='auto', cmap=fi_mymap)
#     #     fig.colorbar(fi_pic, ax=ax_right)
#     #     ax_right.set_xticks(np.linspace(0, 9, num=10))
#     #     ax_right.set_yticks([])


#     for l0 in range(len(layer)):  # nr layers
#         # Note: if info for all layers is present in list then use l, if only the relevant one is plotted use l0
#         l = layer[l0]
#         fmap = fmap_plot[l0]
#         fi_fmap = fi_fmap_plot[l0]
#         # print('check', l0, l, torch.max(fi_fmap))

#         axes_left = []
#         axes_right = []
#         pic_largest = None
#         fi_pic_largest = None

#         for i in range(nr_fmaps_rearr[l][0]):
#             for j in range(nr_fmaps_rearr[l][1]):

#                 # left side ----------------------------------
#                 ax_left = fig.add_subplot(gs_left[l0 + int(add_input_plot)][i, j])
#                 axes_left.append(ax_left)
#                 if len(fmap.shape) < 3:
#                     fmap_ij = fmap
#                 else:
#                     fmap_ij = fmap[i * nr_fmaps_rearr[l][1] + j]

#                 if bnds is not None:
#                     act_bnd = bnds[l][1]
#                 else:
#                     act_bnd = 10.
#                 # print('check', l, i, j, torch.max(fmap_ij), act_bnd, ylimmax)
#                 ylimmax = np.max([act_bnd, ylimmax]) #adjust ylim

#                 mymap = customize_cmap(fmap_ij, act_bnd, ylimmax)  # individual cmap for each fmap!
#                 pic = ax_left.imshow(fmap_ij, aspect='auto', cmap=mymap)
#                 pic_largest = updated_pic_largest(pic, pic_largest)

#                 ax_left.set_xticks([])
#                 ax_left.set_yticks([])

#                 # right side ----------------------------------
#                 ax_right = fig.add_subplot(gs_right[l0 + int(add_input_plot)][i, j])
#                 axes_right.append(ax_right)
#                 if len(fi_fmap.shape) < 3:
#                     fi_fmap_ij = fi_fmap
#                 else:
#                     fi_fmap_ij = fi_fmap[i * nr_fmaps_rearr[l][1] + j]

#                 # print('check', l, i, j, torch.min(fi_fmap_ij), torch.max(fi_fmap_ij), act_bnd, ylimmax)
#                 fi_mymap = customize_cmap(fi_fmap_ij, act_bnd, ylimmax)  # individual cmap for each fmap!
#                 fi_pic = ax_right.imshow(fi_fmap_ij, aspect='auto', cmap=fi_mymap)
#                 fi_pic_largest = updated_pic_largest(fi_pic, fi_pic_largest)

#                 ax_right.set_xticks([])
#                 ax_right.set_yticks([])

#         all_axes_left.append(axes_left)
#         all_axes_right.append(axes_right)

#         fig.colorbar(pic_largest, ax=axes_left)
#         fig.colorbar(fi_pic_largest, ax=axes_right)



#     if add_output_plot:
#         # left side ----------------------------------
#         ax_left = fig.add_subplot(gs_left[-1][0, 0])
#         axes_left.append(ax_left)
#         # predictions = torch.nn.functional.softmax(outputs[N], dim=0)  # normalized tensors
#         predictions = np.array(orig_pred) - np.min(orig_pred)
#         predictions = predictions/ predictions.sum()
#         # predictions += 1e-3

#         mymap = customize_cmap(predictions, 1.6, 2)  # individual cmap for each fmap!
#         pic = ax_left.imshow(np.expand_dims(predictions, axis=0), aspect='auto', cmap=mymap)
#         fig.colorbar(pic, ax=ax_left)
#         ax_left.set_xticks(np.linspace(0, 9, num=10))
#         ax_left.set_yticks([])

#         # right side ----------------------------------
#         ax_right = fig.add_subplot(gs_right[-1][0, 0])
#         axes_right.append(ax_right)
#         # fi_predictions = torch.nn.functional.softmax(fi_outputs[N], dim=0)  # normalized tensors
#         # fi_predictions = (fi_outputs[N] - torch.min(fi_outputs[N])) / fi_outputs[N].norm()
#         fi_predictions = np.array(corr_pred) - np.min(corr_pred)
#         fi_predictions = fi_predictions / fi_predictions.sum()

#         fi_mymap = customize_cmap(fi_predictions, 1.6, 2)  # individual cmap for each fmap! TODO: increase to remove red parts for output plot
#         fi_pic = ax_right.imshow(np.expand_dims(fi_predictions, axis=0), aspect='auto', cmap=fi_mymap)
#         fig.colorbar(fi_pic, ax=ax_right)
#         ax_right.set_xticks(np.linspace(0, 9, num=10))
#         ax_right.set_yticks([])


#     # Labels?
#     fig.suptitle('No fault' + " "*40 + 'With fault', y=0.92, fontsize=18)
#     # fig.text(0.04, 0.5, 'Layers ', va='center', rotation='vertical')

#     # # # Add layerwise fmap plots
#     # # No fault:
#     # fmap_oI = map_fmaps_to_fmap_array(list_act[layer_nr], layer_sizes[layer_nr], sep_line_wdth)
#     # mymap = customize_cmap(fmap, act_bnd, ylimmax)
#     # # pic = ax1.imshow(fmap_oI, cmap=plt.get_cmap('rainbow'), vmin=0.,vmax=10.) #set range with: vmin=0., vmax=10, norm=norm, interpolation='none'
#     # pic = ax1.imshow(fmap_oI, aspect='auto', cmap=mymap)  # set range with: vmin=0., vmax=10, norm=norm, interpolation='none'
#     # fig.colorbar(pic, ax=ax1)
#     # ax1.set_title('No fault')
#     # ax1.set_ylabel('Layer ' + str(layer_nr))

#     # # Fault:
#     # fi_fmap_oI = map_fmaps_to_fmap_array(fi_list_act[layer_nr], layer_sizes[layer_nr], sep_line_wdth)
#     # # Build custom color map:
#     # fi_mymap = customize_cmap(fi_fmap, act_bnd, ylimmax)
#     # pic = ax2.imshow(fi_fmap_oI, aspect='auto', cmap=fi_mymap) #, vmin=0, vmax = 10)
#     # fig.colorbar(pic, ax=ax2)
#     # ax2.set_title('Fault')

#     # # Remove ticks
#     # ax1.set_xticks([])
#     # ax1.set_yticks([])
#     # ax2.set_xticks([])
#     # ax2.set_yticks([])



# def plot_fmap_fromTensor(N, layer_sizes, tnsr, tnsr_fi, xlabel, bnds, ylimmax):
#     """
#     Plot a single "layer" that is given as tensor, i.e. input or output layers.
#     :param N: batch nr to plot
#     :param layer_sizes: form [(N, C, H, W)]
#     :param tnsr: tensor without faults
#     :param tnsr_fi: tensor with faults
#     :return: nothing, plots pic
#     """

#     layer = [0] #layers to plot. Layer 0 is before any ranger. Last is output

#     fig2, axes2 = plt.subplots(nrows=len(layer), ncols=2, figsize=(10, 5*len(layer))) #sharex='col', sharey='row', sharex=True
#     plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.7) #, wspace=0.4 #If things get truncated increase top, bottom!
#     if len(layer)==1:
#         axes2 = [axes2]


#     list_act_input = [get_tensorlist(tnsr)]
#     fi_list_act_input = [get_tensorlist(tnsr_fi)]

#     for ls in range(len(layer)):
#         plot_fmap_layer(N, layer[ls], list_act_input, fi_list_act_input, layer_sizes, axes2[ls], fig2, None, ylimmax)

#         # axes[ls][0].set_title("")
#         # axes[ls][1].set_title("")
#         axes2[ls][0].set_ylabel(xlabel)

#         axes2[ls][0].set_xticks([])
#         axes2[ls][0].set_yticks([])
#         axes2[ls][1].set_xticks([])
#         axes2[ls][1].set_yticks([])


#     plt.show()



# def plot_activations(list_act, bounds_max):
#     """
#     Takes the activations list and plots it.
#     :param list_act:
#     :param bounds_max list
#     :return:
#     """
#     fig, axs = plt.subplots(1, len(list_act), figsize=(17, 15))
#     axs[0].set_xlabel('Nr of activations')
#     axs[0].set_ylabel('Activation values')
#     # ymax = max(bounds_max) + 0.1
#     ymax = max([max([max(sublist) for sublist in list_act]), max(bounds_max)]) + 0.1

#     for n in range(len(list_act)):
#         layer = list_act[n]
#         # layer = fi_list_act[n]

#         axs[n].scatter(range(len(layer)), layer)
#         axs[n].plot([0, len(layer)], [bounds_max[n], bounds_max[n]], 'r')
#         title_plot = 'Layer' + str(n + 1)
#         axs[n].set_title(title_plot)
#         axs[n].set_ylim([0, ymax])

#     plt.show()


# def plot_fmap_layer_n(N, C, layer_nr, list_act, fi_list_act, layer_sizes, axes, fig):
#     """
#     Plots for one layer no-fault and fault pic of fmap side by side.
#     For fully connected layers there is no channel C.
#     :param N: batch nr
#     :param C: channel nr
#     :param layer_nr: layer number
#     :param list_act: list of all activations in absence of fault
#     :param fi_list_act: list of all activations in presence of fault
#     :param layer_sizes: list of all layer sizes
#     :param axes: the explicit axes for the given row of the subplot. Form ax1, ax2
#     """
#
#     ax1, ax2 = axes #no fault, fault
#     # ax1.set_box_aspect(1)
#     # ax2.set_box_aspect(1)
#     layer_size_nr =layer_sizes[layer_nr]
#
#     # No fault:
#     # print('before', np.shape(list_act[layer_nr]))
#     fmap = np.reshape(list_act[layer_nr], layer_size_nr)  # N, C, H, W
#     # print('after', np.shape(fmap))
#     if len(layer_size_nr)==3: #for fcc there is no C
#         fmap_oI = fmap[N]
#     else:
#         fmap_oI = fmap[N][C]
#
#     # print('fmap_reconstr no', fmap_oI)
#     # pic = ax1.imshow(fmap_oI, cmap=plt.get_cmap('rainbow'), vmin=0.,vmax=10.) #set range with: vmin=0., vmax=10, norm=norm, interpolation='none'
#     pic = ax1.imshow(fmap_oI, aspect='auto')  # set range with: vmin=0., vmax=10, norm=norm, interpolation='none'
#     fig.colorbar(pic, ax=ax1)
#     ax1.set_title('No fault')
#     ax1.set_xlabel('Layer ' + str(layer_nr))
#
#     # With fault:
#     fi_fmap = np.reshape(fi_list_act[layer_nr], layer_size_nr)  # N, C, H, W or N, H, W for fully connected layers
#     if len(layer_size_nr) == 3:  # for fcc there is no C
#         fi_fmap_oI = fi_fmap[N]
#     else:
#         fi_fmap_oI = fi_fmap[N][C]
#
#     # print('fmap_reconstr wfi', fi_fmap_oI)
#     pic = ax2.imshow(fi_fmap_oI, aspect='auto')
#     fig.colorbar(pic, ax=ax2)
#     ax2.set_title('Fault')
#     ax2.set_xlabel('Layer ' + str(layer_nr))