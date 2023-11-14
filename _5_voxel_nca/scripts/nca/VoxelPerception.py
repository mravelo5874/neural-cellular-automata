import torch
import torch.nn.functional as func
import numpy as np
import scipy.spatial.transform.Rotation as R
import pytorch3d.transforms as T

from enum import Enum
from scripts.nca import VoxelUtil as util

Perception = Enum('Perception', ['ANISOTROPIC', 'YAW_ISO', 'QUATERNION', 'FAST_QUAT', 'EULER'])

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
        return torch.cat([_x, px, py, gz, lap], 1)
    
    def quaternion_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]

        # * per channel convolutions
        px = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        py = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        pz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        # * combine perception tensors
        px = px[:, None, ...]
        py = py[:, None, ...]
        pz = pz[:, None, ...]
        p0 = torch.zeros_like(px)
        
        # * get quat values
        bs, a, sx, sy, sz = ax.shape
        ax = ax.reshape([bs, a, sx*sy*sz])
        ay = ay.reshape([bs, a, sx*sy*sz])
        az = az.reshape([bs, a, sx*sy*sz])
        quats = util.euler_to_quaternion(ax, ay, az)
        
        # * rotate perception tensors
        pxyz = torch.cat([p0, px, py, pz], 1)
        bs, p4, hc, sx, sy, sz = pxyz.shape
        pxyz = pxyz.reshape([bs, p4, hc, sx*sy*sz])
        conj = torch.conj(quats)
        rxyz = torch.zeros_like(pxyz)
        for t in range(hc):
            rxyz[:, :, t] = quats * pxyz[:, :, t] * conj
        rxyz = rxyz.reshape([bs, p4, hc, sx, sy, sz])
        
        # * extract rotated perception tensors
        rx = rxyz[:, 1]
        ry = rxyz[:, 2]
        rz = rxyz[:, 3]
        return torch.cat([_x, rx, ry, rz, lap], 1)
    
    def fast_quat_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]
        
        # * per channel convolutions
        px = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        py = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        pz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        # * combine perception tensors
        px = px[None, ...]
        py = py[None, ...]
        pz = pz[None, ...]
        pxyz = torch.cat([px, py, pz], 0)
        p3, bs, hc, sx, sy, sz = pxyz.shape
        pxyz = torch.permute(pxyz, (2, 1, 3, 4, 5, 0))
        pxyz = pxyz.reshape([hc, bs*sx*sy*sz, p3])
        
        # * get quat values
        axyz = torch.cat([ax, ay, az], 1)
        bs, a, sx, sy, sz = axyz.shape
        axyz = torch.permute(axyz, (0, 2, 3, 4, 1))
        axyz = axyz.reshape([sx*sy*sz*bs, a])
        quat = T.axis_angle_to_quaternion(axyz)
        
        # * apply quat rotations
        rxyz = torch.zeros_like(pxyz)
        for t in range(hc):
            rxyz[t] = T.quaternion_apply(quat, pxyz[t])
        rxyz = torch.permute(rxyz, (2, 0, 1))
        rxyz = rxyz.reshape([bs, p3, hc, sx, sy, sz])

        # * extract rotated perception tensors
        rx = rxyz[:, 0]
        ry = rxyz[:, 1]
        rz = rxyz[:, 2]
        return torch.cat([_x, rx, ry, rz, lap], 1)
    
    def euler_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]

        # * per channel convolutions
        px = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        py = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        pz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        # * combine perception tensors
        px = px[None, ...]
        py = py[None, ...]
        pz = pz[None, ...]
        pxyz = torch.cat([px, py, pz], 0)
        p3, bs, hc, sx, sy, sz = pxyz.shape
        pxyz = torch.permute(pxyz, (0, 2, 1, 3, 4, 5))
        pxyz = pxyz.reshape([p3, hc, bs*sx*sy*sz])
        pxyz = torch.permute(pxyz, (1, 2, 0))
 
        # * get euler values
        bs, a, sx, sy, sz = ax.shape
        ax = torch.permute(ax, (1, 0, 2, 3, 4))
        ay = torch.permute(ay, (1, 0, 2, 3, 4))
        az = torch.permute(az, (1, 0, 2, 3, 4))
        ax = ax.reshape([a, sx*sy*sz*bs])
        ay = ay.reshape([a, sx*sy*sz*bs])
        az = az.reshape([a, sx*sy*sz*bs])
        axyz = torch.cat([ax, ay, az], 0)
        axyz = torch.permute(axyz, (1, 0))

        # * perform rotations
        
        rots = R.from_euler('xyz', axyz.cpu().detach().numpy(), degrees=False)
        rxyz = np.zeros_like(pxyz.cpu().detach().numpy())
        for t in range(hc):
            rxyz[t] = rots.apply(pxyz[t].cpu().detach().numpy())
        rxyz = torch.tensor(rxyz)
        rxyz = torch.permute(rxyz, (2, 0, 1))
        rxyz = rxyz.reshape([bs, p3, hc, sx, sy, sz])

        # * extract rotated perception tensors
        rx = rxyz[:, 0]
        ry = rxyz[:, 1]
        rz = rxyz[:, 2]
        return torch.cat([_x, rx, ry, rz, lap], 1)
        
    perception = {
        Perception.ANISOTROPIC: anisotropic_perception,
        Perception.YAW_ISO: yaw_isotropic_perception,
        Perception.QUATERNION: quaternion_perception,
        Perception.FAST_QUAT: fast_quat_perception,
        Perception.EULER: euler_perception,
    }