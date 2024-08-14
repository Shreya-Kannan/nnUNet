import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from . import layers

#Credits: https://github.com/voxelmorph/voxelmorph

class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
                 num_classes,
                 conv_op,
                 int_steps=7,
                 int_downsize=1,
                 bidir=False):
        """ 
        Parameters:
            num_classes: Number of classes
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = 1
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.int_downsize = int_downsize
        self.int_steps = int_steps

        in_channels = 1
        
        self.conv_layers = nn.Sequential(
            conv_op(num_classes, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            conv_op(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            conv_op(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            conv_op(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            conv_op(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )

        # configure unet to flow field layer
        #Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = conv_op(ndims, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure optional resize layers (downsize)
        if int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir
        self.integrate = None
        self.transformer = None

    def forward(self, x, registration=False):
        '''
        Parameters:
            x: nnunet_outputs
            registration: Return transformed image and flow. Default is False.
        '''
        # transform into flow field 
        #Only passing in the first resolution - TODO! for all. 

        registration_head_out = self.conv_layers(x)
        #print("reg head out shape",registration_head_out.shape)
        flow_field = self.flow(registration_head_out)

        # resize flow for integration
        pos_flow = flow_field #torch.Size([2, 1, 32, 208, 256])

        first_frame = x[:,:,0,:,:] #torch.Size([2, 7, 208, 256])
        first_frame = first_frame.unsqueeze(2) # Expecting torch.Size([2, 7, 1, 208, 256])
        first_frame = first_frame.expand(-1, -1, x.shape[2], -1, -1)   # Expecting torch.Size([2, 7, 32 ,208, 256])

        # configure optional integration layer for diffeomorphic warp
        inshape = pos_flow.shape[2:]
        down_shape = [int(dim / self.int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, self.int_steps) if self.int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(inshape)


        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        #first frame and flow field torch.Size([2, 7, 32, 208, 256]) torch.Size([2, 1, 32, 208, 256])
        #src shape: torch.Size([2, 1, 32, 208, 256])
        
        # warp image with flow field
        y_source = self.transformer(first_frame, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        #y_source shape: torch.Size([2, 7, 32, 208, 256])

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow