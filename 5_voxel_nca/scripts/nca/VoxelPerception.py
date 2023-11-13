import torch
import torch.nn.functional as func
from scripts.nca import VoxelUtil as util

# 3D filters
X_SOBEL_KERN = torch.tensor([
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]],
   
   [[2., 4., 2.], 
    [0., 0., 0.], 
    [-2., -4., -2.]],
   
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]]])
Y_SOBEL_KERN = torch.tensor([
   [[1., 2., 1.], 
    [2., 4., 2.], 
    [1., 2., 1.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[-1., -2., -1.], 
    [-2., -4., -2.], 
    [-1., -2., -1.]]])
Z_SOBEL_KERN = torch.tensor([
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]],
   
   [[-2., 0., 2.], 
    [-4., 0., 4.], 
    [-2., 0., 2.]],
   
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]]])
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

class VoxelPerception():
    def __init__(self, _device='cuda'):
        super().__init__()
        self.device = _device
        
    # * performs a convolution per filter per channel
    def per_channel_conv3d(self, _x, _filters):
        batch_size, channels, height, width, depth = _x.shape
        # * reshape x to make per-channel convolution possible + pad 1 on each side
        y = _x.reshape(batch_size*channels, 1, height, width, depth)
        y = func.pad(y, (1, 1, 1, 1, 1, 1), 'constant')
        # * perform per-channel convolutions
        _filters = _filters.to(self.device)
        y = func.conv3d(y, _filters[:, None])
        y = y.reshape(batch_size, -1, height, width, depth)
        return y

    def anisotropic_perception(self, _x):
        _x = _x.to(self.device)
        # * per channel convolutions
        gx = self.per_channel_conv3d(_x, X_SOBEL_KERN[None, :])
        gy = self.per_channel_conv3d(_x, Y_SOBEL_KERN[None, :])
        gz = self.per_channel_conv3d(_x, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(_x, LAP_KERN[None, :])
        return torch.cat([_x, gx, gy, gz, lap], 1)
    
    def yaw_isotropic_perception(self, _x):
        # * separate states and angle channels
        states, angle = _x[:, :-1], _x[:, -1:]
        # * calculate gx and gy
        gx = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        gy = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        # * compute px and py 
        _cos, _sin = angle.cos(), angle.sin()
        px = (gx*_cos)+(gy*_sin)
        py = (gy*_cos)-(gx*_sin)
        # * calculate gz and lap
        gz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        return torch.cat([_x, px, py, lap], 1)
    
    def quaternion_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1], _x[:, -2], _x[:, -3]
        # * per channel convolutions
        gx = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        gy = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        gz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        # * get quat values
        quat = util.euler_to_quaternion(ax.item(), ay.item(), az.item())
        b, c, x, y, z = gx.shape
        # * reshape gx, gy, and gz
        rsx = gx.reshape(b*c, x, y, z)
        rsy = gy.reshape(b*c, x, y, z)
        rsz = gz.reshape(b*c, x, y, z)
        gxyz = torch.cat([rsx, rsy, rsz], 1)
        rot = quat*gxyz*torch.conj(quat)
        
        
    perception = {
        'ANISOTROPIC': anisotropic_perception,
        'YAW_ISO': yaw_isotropic_perception,
        'QUATERNION': quaternion_perception,
    }