import torch
import torch.nn as nn
import torch.nn.functional as nnf

#Credits: https://github.com/voxelmorph/voxelmorph

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        frame, height, width = size
        #Ignore no. frames. Expecting shape [height, width]
        size = size[1:] 
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).unsqueeze(0).expand(-1,frame,-1,-1,-1).permute(0,2,1,3,4)
        grid = grid.type(torch.FloatTensor)  #[1, 2, no_of_frames, height, width]
        grid = grid.to('cuda')

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        #flow shape [b,1,30,_,_]
        #src shape [b,num_classes, height, width]
        num_classes = src.shape[1]
        #print("src shape:",src.shape) #During vect int: torch.Size([2, 1/7, 32, 208, 256])
        batch, channel, num_frames, height, width = flow.shape
        
        # new locations
        new_locs = self.grid + flow  #Expecting shape: [1, 2, no_of_frames, height, width]
        
        shape = flow.shape[3:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [1, 0]]   #[batch, no_of_frames, height, width, 2]

        new_locs = new_locs.reshape(batch * num_frames, height, width, 2)

        src = src.permute(0, 2, 1, 3, 4).reshape(-1, num_classes, height, width)

        output  = nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        output = output.reshape(batch, num_frames, num_classes, height, width)
        
        # Permute output to [batch, num_classes, num_frames, height, width]
        output_final = output.permute(0, 2, 1, 3, 4) 

        return output_final


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x