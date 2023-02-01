#  Copyright (c) 2021 Cognition Factory
#  All rights reserved.
#
#  Any use, distribution or replication without a written permission
#  is prohibited.
#
#  The name "Cognition Factory" must not be used to endorse or promote
#  products derived from the source code without prior written permission.
#
#  You agree to indemnify, hold harmless and defend Cognition Factory from
#  and against any loss, damage, claims or lawsuits, including attorney's
#  fees that arise or result from your use the software.
#
#  THIS SOFTWARE IS PROVIDED "AS IS" AND "WITH ALL FAULTS", WITHOUT ANY
#  TECHNICAL SUPPORT OR ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
#  BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. ALSO, THERE IS NO
#  WARRANTY OF NON-INFRINGEMENT, TITLE OR QUIET ENJOYMENT. IN NO EVENT
#  SHALL COGNITION FACTORY OR ITS SUPPLIERS BE LIABLE FOR ANY DIRECT,
#  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
#  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
#  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
import numpy as np


class Averager(nn.Module): # Averager

    def __init__(self, bnds=None, perception=1):
        """Averager resilience layer

        Detects potential faults in the tensor and fixes them by averaging over an area defined
        by the perception field

        Args:
            bnds ([float], optional): Sets the bounds of acceptable values. Defaults to None.
            perception (int, optional): Size of the perceptive field to average over
                in order to replace the faulty value. Defaults to 1.
        """
        super(Averager, self).__init__()
        self.Bounds = bnds
        self.Perception = perception
    
    def forward(self, x):

        if (self.Bounds is None) or self.Bounds == [None, None] or (not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor)) or len(self.Bounds) < 2:
            return x

        # Check if tensor contains out of bound values
        if torch.equal(x, x.clamp(self.Bounds[0], self.Bounds[1])):
            return x # If clamping does not change the tensor -> tensor has no faulty values
        else:
            [fl_n_low, fl_c_low, fl_x_low, fl_y_low] = np.where(x < self.Bounds[0])
            [fl_n_up, fl_c_up, fl_x_up, fl_y_up] = np.where(x > self.Bounds[1])           
            
            for n,c,h,w  in zip(fl_n_low, fl_c_low, fl_x_low, fl_y_low):
                slice = x[n,c,max(0, h-self.Perception):h+self.Perception+1, max(0,w-self.Perception):w+self.Perception+1] # Compute the average in the perception radius
                new_value = np.mean([i for i in slice.flatten().detach().numpy() if ( (i > self.Bounds[0]) and (i < self.Bounds[1]) ) ])
                x[n,c,h,w] = torch.as_tensor(new_value)

            for n,c,h,w in zip(fl_n_up, fl_c_up, fl_x_up, fl_y_up):
                slice = x[n,c,max(0, h-self.Perception):h+self.Perception+1, max(0,w-self.Perception):w+self.Perception+1] # Compute the average in the perception pixel radius
                new_value = np.mean([i for i in slice.flatten().detach().numpy() if ( (i > self.Bounds[0]) and (i < self.Bounds[1]) ) ])
                x[n,c,h,w] = torch.as_tensor(new_value)

        return x