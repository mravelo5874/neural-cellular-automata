import torch
import torch.nn.functional as func

# 3D filters
X_SOBEL_KERN = torch.tensor([
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]],
   
   [[-2., 0., 2.], 
    [-4., 0., 4.], 
    [-2., 0., 2.]],
   
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]]])
Y_SOBEL_KERN = torch.tensor([
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]],
   
   [[2., 4., 2.], 
    [0., 0., 0.], 
    [-2., -4., -2.]],
   
   [[1., 0., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]]])
Z_SOBEL_KERN = torch.tensor([
   [[1., 2., 1.], 
    [2., 4., 2.], 
    [1., 2., 1.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[-1., -2., -1.], 
    [-2., -4., -2.], 
    [-1., -2., -1.]]])
LAP_KERN = torch.tensor([
   [[2., 3., 2.], 
    [3., 6., 3.], 
    [2., 3., 2.]],
   
   [[3., 6., 3.], 
    [6.,-88., 6.], 
    [3., 6., 3.]],
   
   [[2., 3., 2.], 
    [3., 6., 3.], 
    [2., 3., 2.]]])/26.0

# * performs a convolution per filter per channel
def per_channel_conv(_x, _filters):
    batch_size, channels, height, width = _x.shape
    # * reshape x to make per-channel convolution possible + pad 1 on each side
    y = _x.reshape(batch_size*channels, 1, height, width)
    y = func.pad(y, (1, 1, 1, 1), 'circular')
    # * perform per-channel convolutions
    y = func.conv2d(y, _filters[:, None])
    y = y.reshape(batch_size, -1, height, width)
    return y

def angle_steerable_perception(_x):
    # * separate states and angle channels
    states, angle = _x[:, :-1], _x[:, -1:]
    # * compute lap, gx and gy
    lap_conv = per_channel_conv(states, LAP_KERN[None, :])
    gx = per_channel_conv(states, SOBEL_KERN[None, :])
    gy = per_channel_conv(states, SOBEL_KERN.T[None, :])
    # * compute px and py 
    _cos, _sin = angle.cos(), angle.sin()
    px = (gx*_cos)+(gy*_sin)
    py = (gy*_cos)-(gx*_sin)
    # * concat and return
    y = torch.cat([states, lap_conv, px, py], 1)
    return y

def gradient_steerable_perception(_x):
    # * compute sobel x/y convolutions
    filters = torch.stack([SOBEL_KERN, SOBEL_KERN.T])
    grad = per_channel_conv(_x, filters)
    # * extract grad and dir
    grad, dir = grad[:, :-2], grad[:, -2:]
    dir = dir / dir.norm(dim=1, keepdim=True).clip(1.0)
    gx, gy = grad[:, ::2], grad[:, 1::2]
    # * rotate gx and gy using sin/cos of dir
    _cos, _sin = dir[:, :1], dir[:, 1::2]
    rot_grad = torch.cat([gx*_cos+gy*_sin, gy*_cos-gx*_sin], 1)
    lap_conv = per_channel_conv(_x, LAP_KERN[None, :])
    # * concat and return
    y = torch.cat([_x, lap_conv, rot_grad], 1)
    return y
    
perception = {
    'STEERABLE': angle_steerable_perception,
    'GRADIENT': gradient_steerable_perception,
}