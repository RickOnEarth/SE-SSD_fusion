import time

import torch
from torch import nn

from det3d.models.utils import Empty, GroupNorm, Sequential

# import sys
# if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()
        self.name = 'fusion_layer'
        self._total_forward_time = 0.0
        self._total_inference_count = 0
        self.corner_points_feature = Sequential(
            nn.Conv2d(24,48,1),
            nn.ReLU(),
            nn.Conv2d(48,96,1),
            nn.ReLU(),
            nn.Conv2d(96,96,1),
            nn.ReLU(),
            nn.Conv2d(96,4,1),
        )
        self.fuse_2d_3d = Sequential(
            nn.Conv2d(4,18,1),
            nn.ReLU(),
            nn.Conv2d(18,36,1),
            nn.ReLU(),
            nn.Conv2d(36,36,1),
            nn.ReLU(),
            nn.Conv2d(36,1,1),
        )
        self.maxpool = Sequential(
            nn.MaxPool2d([200,1],1),
        )


    def forward(self,input_1, tensor_index, num_anchors):
        torch.cuda.synchronize()
        t1 = time.time()
        flag = -1
        if tensor_index[0,0] == -1:
            out_1 = torch.zeros(1, 200, num_anchors, dtype = input_1.dtype,device = input_1.device)
            out_1[:,:,:] = -9999999
            flag = 0
        else:
            x = self.fuse_2d_3d(input_1)
            out_1 = torch.zeros(1, 200, num_anchors, dtype = input_1.dtype,device = input_1.device)
            out_1[:,:,:] = -9999999
            out_1[:,tensor_index[:,0],tensor_index[:,1]] = x[0,:,0,:]
            flag = 1
        x = self.maxpool(out_1)
        #x, _ = torch.max(out_1,1)
        x = x.squeeze().reshape(1,-1,1)

        torch.cuda.synchronize()
        self._total_forward_time += time.time() - t1
        self._total_inference_count += 1
        #print("avg fusion nn time: ", self._total_forward_time / self._total_inference_count * 1000)

        delta_t = time.time() - t1
        #print("fusion layer forward_time: ", delta_t * 1000)
        return x,flag
