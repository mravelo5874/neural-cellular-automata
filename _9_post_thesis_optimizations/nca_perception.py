import torch
import torch.nn.functional as func
from enum import Enum
from util import eul2rot

class ptype(int, Enum):
    def __str__(self):
        return str(self.name)
    ANISOTROPIC: int = 0                                      
    SINGLE_AXIS_ISO: int = 1
    FULLY_ISO: int = 2
    
X_SOBEL = torch.tensor([
   [[1., 2., 1.], 
    [2., 4., 2.], 
    [1., 2., 1.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[-1., -2., -1.], 
    [-2., -4., -2.], 
    [-1., -2., -1.]]])

Y_SOBEL = torch.tensor([
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]],
   
   [[2., 4., 2.], 
    [0., 0., 0.], 
    [-2., -4., -2.]],
   
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]]])

Z_SOBEL_DOWN = torch.tensor([
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]],
   
   [[-2., 0., 2.], 
    [-4., 0., 4.], 
    [-2., 0., 2.]],
   
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]]])

LAP_KERN_27 = torch.tensor([
   [[2., 3., 2.], 
    [3., 6., 3.], 
    [2., 3., 2.]],
   
   [[3., 6., 3.], 
    [6.,-88., 6.], 
    [3., 6., 3.]],
   
   [[2., 3., 2.], 
    [3., 6., 3.], 
    [2., 3., 2.]]])/26.0

X_SOBEL_2D_XY = torch.tensor([
   [[0., 1., 0.], 
    [0., 2., 0.], 
    [0., 1., 0.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[0., -1., 0.], 
    [0., -2., 0.], 
    [0., -1., 0.]]])

Y_SOBEL_2D_XY = torch.tensor([
   [[0., 1., 0.], 
    [0., 0., 0.], 
    [0., -1.,0.]],
   
   [[0., 2., 0.], 
    [0., 0., 0.], 
    [0., -2., 0.]],
   
   [[0., 1., 0.], 
    [0., 0., 0.], 
    [0., -1.,0.]]])
    
LAP_2D_XY = torch.tensor([
   [[0., 1., 0.], 
    [0., 2., 0.], 
    [0., 1., 0.]],
   
   [[0., 2., 0.], 
    [0.,-12.,0.], 
    [0., 2., 0.]],
   
   [[0., 1., 0.], 
    [0., 2., 0.], 
    [0., 1., 0.]]])

LAP_KERN_7 = torch.tensor([
   [[0., 0., 0.], 
    [0., 1., 0.], 
    [0., 0., 0.]],
   
   [[0., 1., 0.], 
    [1.,-6., 1.], 
    [0., 1., 0.]],
   
   [[0., 0., 0.], 
    [0., 1., 0.], 
    [0., 0., 0.]]])

class nca_perception():
    
    def __init__(self):
        super().__init__()
        
    def orientation_channels(self, _ptype):
        if _ptype == ptype.ANISOTROPIC:
            return 0
        if _ptype == ptype.SINGLE_AXIS_ISO:
            return 1
        if _ptype == ptype.FULLY_ISO:
            return 3
    
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
        gx = self.per_channel_conv3d(_x, X_SOBEL[None, :])
        gy = self.per_channel_conv3d(_x, Y_SOBEL[None, :])
        gz = self.per_channel_conv3d(_x, Z_SOBEL_DOWN[None, :])
        lap = self.per_channel_conv3d(_x, LAP_KERN_27[None, :])
        return torch.cat([_x, gx, gy, gz, lap], 1)
    
    def single_axis_iso_perception(self, _x):
        # * separate states and angle channels
        states, angle = _x[:, :-1], _x[:, -1:]
        
        # * calculate gx and gy
        gx = self.per_channel_conv3d(states, X_SOBEL_2D_XY[None, :])
        gy = self.per_channel_conv3d(states, Y_SOBEL_2D_XY[None, :])
        
        # * calculate lap2d and lap3d
        lap2d = self.per_channel_conv3d(states, LAP_2D_XY[None, :])
        lap3d = self.per_channel_conv3d(states, LAP_KERN_7[None, :])
           
        # * compute px and py 
        _cos, _sin = angle.cos(), angle.sin()
        px = (gx*_cos)+(gy*_sin)
        py = (gy*_cos)-(gx*_sin)
        
        return torch.cat([states, lap2d, px, py, lap3d], 1)
    
    def fully_iso_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]

        # * per channel convolutions
        px = self.per_channel_conv3d(states, X_SOBEL[None, :])
        py = self.per_channel_conv3d(states, Y_SOBEL[None, :])
        pz = self.per_channel_conv3d(states, Z_SOBEL_DOWN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN_27[None, :])
        
        # * get perception tensors
        px = px[..., None]
        py = py[..., None]
        pz = pz[..., None]
        pxyz = torch.cat([px, py, pz], 5)
        bs, hc, sx, sy, sz, p3 = pxyz.shape
        pxyz = pxyz.reshape([bs, hc, sx*sy*sz, p3])
        pxyz = pxyz.unsqueeze(-1)
        
        # * get quat values
        bs, _, sx, sy, sz = ax.shape
        ax = ax.reshape([bs, sx*sy*sz])
        ay = ay.reshape([bs, sx*sy*sz])
        az = az.reshape([bs, sx*sy*sz])
        R_mats = eul2rot(ax, ay, az)
        
        # * rotate perception tensors
        rxyz = torch.zeros_like(pxyz)
        for i in range(hc):
            rxyz[:, i] = torch.matmul(R_mats, pxyz[:, i])
        rxyz = rxyz.reshape([bs, hc, sx, sy, sz, p3])
        
        # * extract rotated perception tensors
        rx = rxyz[:, :, :, :, :, 0]
        ry = rxyz[:, :, :, :, :, 1]
        rz = rxyz[:, :, :, :, :, 2]
        return torch.cat([states, rx, ry, rz, lap], 1)
    
    get_function = {
        ptype.ANISOTROPIC: anisotropic_perception,
        ptype.SINGLE_AXIS_ISO: single_axis_iso_perception,
        ptype.FULLY_ISO: fully_iso_perception,
    }