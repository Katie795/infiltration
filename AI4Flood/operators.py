# Operators for AI4Flood are defined here
import numpy as np
import torch

def get_weights_linear_2D(dx):
    # Isotropic Laplacian  # TODO: make this vary in dy as well
    # TODO: is this the right sign?
    w_diff = torch.tensor([[[[1/3/dx**2], # Diffusion operator
                         [1/3/dx**2], 
                         [1/3/dx**2]],

                        [[1/3/dx**2], 
                         [-8/3/dx**2], 
                         [1/3/dx**2]],

                        [[1/3/dx**2],
                         [1/3/dx**2], 
                         [1/3/dx**2]]]])

    wm1 = torch.tensor([[[[0.0], # Mass operator
                         [0.0], 
                         [0.0]],

                        [[0.0], 
                         [1.0], 
                         [0.0]],

                        [[0.0], 
                        [0.0], 
                        [0.0]]]])

    wm2 = torch.tensor([[[[0.028], # Mass operator with more smoothing
                         [0.11], 
                         [0.028]],

                        [[0.11], 
                         [0.44], 
                         [0.11]],

                        [[0.028], 
                        [0.11], 
                        [0.028]]]])
    
    w_res = torch.zeros([1,1,2,2]) 
    w_res[0,0,:,:] = 0.25

    w_diff = torch.reshape(w_diff,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
    wm1 = torch.reshape(wm1,(1,1,3,3))
    wm2 = torch.reshape(wm2,(1,1,3,3))

    diag = -np.array(w_diff.cpu(), dtype=None, copy=None)[0,0,1,1] # diagonal matrix

    bias = torch.tensor([0.0])

    return w_diff, wm1, w_res, diag, bias

def construct_diff_filter(dx):

    w_diff = torch.tensor([[[[1/3/dx**2], # Diffusion operator
                         [1/3/dx**2], 
                         [1/3/dx**2]],

                        [[1/3/dx**2], 
                         [-8/3/dx**2], 
                         [1/3/dx**2]],

                        [[1/3/dx**2],
                         [1/3/dx**2], 
                         [1/3/dx**2]]]])

    w_diff = torch.reshape(w_diff,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
    bias = torch.tensor([0.0])

    diff = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
    diff.weight.data = w_diff
    diff.bias.data = bias
    return diff


