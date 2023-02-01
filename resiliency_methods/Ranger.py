import torch.nn as nn
import torch
import struct
import numpy as np
from numpy import all
from numba import njit, prange
import asyncio
import itertools


# https://pytorch.org/docs/stable/generated/torch.masked_select.html#torch.masked_select
# https://pytorch.org/docs/stable/tensors.html
# Use only torch operations otherwise operations are not executed (?!)


# #  Trivial case for comparison  -----------------------------------------------------------------------------------

# Renamed to Ranger_trivial
class Ranger_trivial(nn.Module): #do nothing
    def __init__(self, bnds=None):
        super(Ranger_trivial, self).__init__()
        self.Bounds = bnds

    def forward(self, x):
        return x



#  Normal Ranger -----------------------------------------------------------------------------------

class Ranger(nn.Module): #Normal Ranger (clamp)

    def __init__(self, bnds=None):
        super(Ranger, self).__init__()
        self.Bounds = bnds

    def forward(self, x):
        """
        Binds the values of tensor x with the bounds bnds.
        :return: x with bounds
        """
        if self.Bounds is None or all(self.Bounds == [None, None]) or (not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor)) or len(self.Bounds) < 2:
            # print('Ranger inactive, bounds or tensor input has wrong format.', x.type(), self.Bounds)
            return x

        bnd_low = self.Bounds[0]
        bnd_up = self.Bounds[1]

        return x.clamp(bnd_low, bnd_up)


class Ranger_Averager(nn.Module): # Averager (not mature yet)

    def __init__(self, bnds=None, perception=1):
        """Averager resilience layer

        Detects potential faults in the tensor and fixes them by averaging over an area defined
        by the perception field

        Args:
            bnds ([float], optional): Sets the bounds of acceptable values. Defaults to None.
            perception (int, optional): Size of the perceptive field to average over
                in order to replace the faulty value. Defaults to 1.
        """
        super(Ranger_Averager, self).__init__()
        self.Bounds = bnds
        self.Perception = perception
    
    def forward(self, x):

        if (self.Bounds is None) or self.Bounds == [None, None] or (not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor)) or len(self.Bounds) < 2:
            return x

        # Check if tensor contains out of bound values
        if torch.equal(x, x.clamp(self.Bounds[0], self.Bounds[1])):
            return x # If clamping does not change the tensor -> tensor has no faulty values
        else:
            [fl_n_low, fl_c_low, fl_x_low, fl_y_low] = np.where(x.cpu() < self.Bounds[0])
            [fl_n_up, fl_c_up, fl_x_up, fl_y_up] = np.where(x.cpu() > self.Bounds[1])           
            
            for n,c,h,w  in zip(fl_n_low, fl_c_low, fl_x_low, fl_y_low):
                slice = x[n,c,max(0, h-self.Perception):h+self.Perception+1, max(0,w-self.Perception):w+self.Perception+1] # Compute the average in the perception radius
                # print('slice', slice)
                # if slice.nelement() != 0: #nothing found in this neighborhood
                #     new_value = np.mean([i for i in slice.flatten().detach().cpu().numpy() if ( (i > self.Bounds[0]) and (i < self.Bounds[1]) ) ])
                #     x[n,c,h,w] = torch.as_tensor(new_value)

                mask1 = slice.le(self.Bounds[1])
                mask2 = slice.ge(self.Bounds[0])
                mask = torch.logical_or(mask1, mask2)
                x[n,c,h,w] = torch.mean(slice.masked_select(mask))

                # env = [i for i in slice.flatten().detach().cpu().numpy() if ( (i > self.Bounds[0]) and (i < self.Bounds[1]) ) ]
                # if env != []:
                #     new_value = np.mean(env)
                #     x[n,c,h,w] = torch.as_tensor(new_value)


            for n,c,h,w in zip(fl_n_up, fl_c_up, fl_x_up, fl_y_up):
                slice = x[n,c,max(0, h-self.Perception):h+self.Perception+1, max(0,w-self.Perception):w+self.Perception+1] # Compute the average in the perception pixel radius
                # print('slice', slice)
                mask1 = slice.le(self.Bounds[1])
                mask2 = slice.ge(self.Bounds[0])
                mask = torch.logical_or(mask1, mask2)
                x[n,c,h,w] = torch.mean(slice.masked_select(mask))

                # env = [i for i in slice.flatten().detach().cpu().numpy() if ( (i > self.Bounds[0]) and (i < self.Bounds[1]) ) ]
                # if env != []:
                #     new_value = np.mean(env)
                #     x[n,c,h,w] = torch.as_tensor(new_value)

        return x





# #  Clippings -----------------------------------------------------------------------------------

class Ranger_Clip(nn.Module):  # Clip all OOB values to zero (=clip2, also truncate negatives coming from flip of bit 0)
    def __init__(self, bnds=None):
        super(Ranger_Clip, self).__init__()
        self.Bounds = bnds

    def forward(self, x):
        if self.Bounds is None or all(self.Bounds == [None, None]) or (not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor)) or len(self.Bounds) < 2:
            print('Ranger inactive, bounds or tensor input has wrong format.', x.type(), self.Bounds)
            return x

        bnd_low = self.Bounds[0]
        bnd_up = self.Bounds[1]

        mask1 = x.ge(bnd_up)
        mask2 = x.le(bnd_low)
        mask = torch.logical_or(mask1, mask2)
        x = x.masked_fill(mask, 0.)

        return x



# # Backflipping -----------------------------------------------------------------------------------
# # # Version 2, go in 3 steps

class Ranger_BackFlip(nn.Module): #truncating, but motivated by bitflip
    
    def __init__(self, bnds=None):
        super(Ranger_BackFlip, self).__init__()
        self.Bounds = bnds

    def forward(self, x):
        """
        Binds the values of tensor x with the bounds bnds.
        :return: x with bounds
        """
        if self.Bounds is None or all(self.Bounds == [None, None]) or (not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor)) or len(self.Bounds) < 2:
            # print('Ranger inactive, bounds or tensor input has wrong format.', x.type(), self.Bounds)
            return x

        bnd_low = self.Bounds[0]
        bnd_up = self.Bounds[1]

        # Simplified: 
        x = self.set_back_final(x, (2**64)*bnd_up, 0.)
        x = self.set_back_final(x, 2*bnd_up, 2.)
        # x = self.set_back_final(x, 2*bnd_up, 0.)
        x = x.clamp(bnd_low, bnd_up)

        return x


    def set_back_final(self, x, thres, val):
        """
        If above thres, set back to val.
        """
        mask = x.ge(thres)
        x = x.masked_fill(mask, val)
        return x



# # Rescalings -----------------------------------------------------------------------------------
# Updated parallelized version

class Ranger_FmapRescale(nn.Module): #Rescale all OOB values per fmap

    def __init__(self, bnds=None):
        super(Ranger_FmapRescale, self).__init__()
        self.Bounds = bnds


    def forward(self, x):

        if self.Bounds is None or all(self.Bounds == [None, None]) or (not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor)) or len(self.Bounds) < 2:
            print('Ranger inactive, bounds or tensor input has wrong format.', x.type(), self.Bounds)
            return x

        # print('check shapes', len(x.shape), x.shape)
        bnd_low = self.Bounds[0] #Note: this is origin interval
        bnd_up = self.Bounds[1]
        
        if len(x.shape) > 2: #for conv layers

            wh = x.size()[2]
            fmap_max = torch.max(torch.max(x, dim=3).values, dim=2).values #2d array of maxima in batches, fmaps
            fmap_min = torch.min(torch.min(x, dim=3).values, dim=2).values #2d array of minima in batches, fmaps

            if torch.max(fmap_max) > bnd_up:
                # print('replace too large values')
                fmap_max_ext = fmap_max.unsqueeze(2).unsqueeze(3).repeat(1,1, wh, wh) #extend to 4d by repetition
                fmap_min_ext = fmap_min.unsqueeze(2).unsqueeze(3).repeat(1,1, wh, wh) #extend to 4d by repetition

                x_resc = torch.add(torch.mul(torch.div(torch.sub(x, fmap_min_ext), torch.sub(fmap_max_ext, fmap_min_ext)), torch.tensor(bnd_up - bnd_low)), torch.tensor(bnd_low))

                # replace those values that were oob by rescaled version
                mask_up = torch.gt(x, bnd_up)
                x_resc_sel = torch.masked_select(x_resc, mask_up)
                x = x.masked_scatter(mask_up, x_resc_sel)

            if torch.min(fmap_min) < bnd_low:
                # print('replace too small values')
                mask_low = torch.lt(x, bnd_low)
                x = x.masked_fill(mask_low, bnd_low)


        elif len(x.shape) == 2: #for fcc layers

            fmap_max = torch.max(x, dim=1).values #2d array of maxima in batches, fmaps
            fmap_min = torch.min(x, dim=1).values #2d array of minima in batches, fmaps

            if torch.max(fmap_max) > bnd_up:
                fmap_max_ext = fmap_max.unsqueeze(1).repeat(1, x.size()[1]) #extend to 4d by repetition
                fmap_min_ext = fmap_min.unsqueeze(1).repeat(1, x.size()[1]) #extend to 4d by repetition

                x_resc = torch.add(torch.mul(torch.div(torch.sub(x, fmap_min_ext), torch.sub(fmap_max_ext, fmap_min_ext)), torch.tensor(bnd_up - bnd_low)), torch.tensor(bnd_low))

                mask_up = torch.gt(x, bnd_up)
                x_resc_sel = torch.masked_select(x_resc, mask_up)
                x = x.masked_scatter(mask_up, x_resc_sel)
            
            if torch.min(fmap_min) < bnd_low:
                # print('replace too small values')
                mask_low = torch.lt(x, bnd_low)
                x = x.masked_fill(mask_low, bnd_low)


        return x



# Replace fmap ----------------------------------------------------------------------------------
# replace by average of all healthy fmaps


class Ranger_FmapAvg(nn.Module):  # Rescale all OOB values per fmap

    def __init__(self, bnds=None):
        super(Ranger_FmapAvg, self).__init__()
        self.Bounds = bnds


    def forward(self, x):

        if self.Bounds is None or all(self.Bounds == [None, None]) or (not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor)) or len(self.Bounds) < 2:
            print('Ranger inactive, bounds or tensor input has wrong format.', x.type(), self.Bounds)
            return x

        bnd_low = self.Bounds[0]
        bnd_up = self.Bounds[1]

        if len(x.shape) == 4:  # will work only for conv layers

            wh = x.size()[2]
            nrf = x.size()[1]

            if torch.max(x) > bnd_up or torch.min(x) < bnd_low:

                # Get mask for oob elements
                mask_up = torch.gt(x, bnd_up)
                mask_low = torch.lt(x, bnd_low)
                mask = torch.logical_or(mask_up, mask_low) #mask for all act that are oob
                
                # Get mask for oob fmaps
                fmaps_max = torch.sum(mask_up, dim=[2,3]) #2d array counting the elements in fmap that are oob (up)
                fmaps_min = torch.sum(mask_low, dim=[2,3]) #2d array counting the elements in fmap that are oob (low)

                mask_up_f = torch.gt(fmaps_max, 0)
                mask_low_f = torch.gt(fmaps_min, 0)
                mask_f = torch.logical_or(mask_up_f, mask_low_f) #mask for all fmaps that are oob 

                # Make mask numerical, inverted, and expand: mask for all healthy fmaps
                device = x.get_device()
                mask_f_ext = torch.ones(mask_f.shape).to(device)
                # if x.get_device() >= 0: #on cpu you need that other version (?)
                #     mask_f_ext.to(x.get_device())
                mask_f_ext = mask_f_ext.masked_fill(mask_f, torch.tensor(
                    0.))  # array of 1 and 0 depending on whether fmap is healthy (1) or oob (0)
                mask_f_ext = mask_f_ext.unsqueeze(2).unsqueeze(3).repeat(1, 1, wh, wh)  # expanded to full shape
                # Note: if all fmaps are healthy we are not in this if branch. If no fmap is healthy mask is all zeros.
            
                # Get average fmap and expand to all fmap channels
                # nrf_h = mask_f_ext.size()[1]
                nrf_h = nrf - torch.sum(mask_f, dim=1)  # nr of healthy fmaps, shape btchsizex1
                nrf_h = nrf_h.unsqueeze(1).unsqueeze(2).repeat(1, wh, wh)
                fmap_av = torch.div(torch.sum(torch.mul(mask_f_ext, x), dim=1),
                                    nrf_h)  # shape N, w, h since fmap dim is averaged out
                fmap_av = fmap_av.unsqueeze(1).repeat(1, nrf, 1,
                                                      1)  # new tensor that has the same av fmap for all fmap channels but diff for batches

                # print('nr of elements replaced', torch.sum(mask))
                # Replace oob values with fmap_av
                x_repl = torch.masked_select(fmap_av, mask)
                x = x.masked_scatter(mask, x_repl)


        return x

